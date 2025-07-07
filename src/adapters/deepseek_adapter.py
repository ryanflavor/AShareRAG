import functools
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from string import Template

import httpx
import yaml
from filelock import FileLock
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from src.adapters.llm_adapter import LLMAdapter, LLMResponse

logger = logging.getLogger(__name__)


def cache_response(func):
    """缓存装饰器 - 基于SQLite的高性能缓存"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 如果未启用缓存, 直接调用原函数
        if not getattr(self, "enable_cache", False) or not hasattr(
            self, "cache_file_name"
        ):
            return func(self, *args, **kwargs)

        # 构建缓存键
        key_data = {
            "func_name": func.__name__,
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ["self"]},
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # 缓存文件和锁
        cache_file = self.cache_file_name
        lock_file = cache_file + ".lock"

        # 尝试从缓存读取
        with FileLock(lock_file):
            try:
                conn = sqlite3.connect(cache_file)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        result TEXT,
                        timestamp REAL
                    )
                """)
                conn.commit()

                c.execute("SELECT result FROM cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                conn.close()

                if row is not None:
                    logger.info(f"🎯 Cache hit for {func.__name__}")
                    cached_data = json.loads(row[0])
                    
                    # If this is a generate method, reconstruct LLMResponse object
                    if func.__name__ == "generate":
                        from src.adapters.llm_adapter import LLMResponse
                        
                        if isinstance(cached_data, str):
                            # Old cache format: just the content string
                            return LLMResponse(
                                content=cached_data,
                                model=self.settings.deepseek_model,
                                usage={}
                            )
                        elif isinstance(cached_data, dict):
                            # New cache format: full object
                            return LLMResponse(
                                content=cached_data.get("content", ""),
                                model=cached_data.get("model", ""),
                                usage=cached_data.get("usage", {})
                            )
                    
                    return cached_data
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        # 缓存未命中, 调用原函数
        logger.info(f"📡 Cache miss, calling API for {func.__name__}")
        result = func(self, *args, **kwargs)

        # 存储到缓存
        with FileLock(lock_file):
            try:
                conn = sqlite3.connect(cache_file)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        result TEXT,
                        timestamp REAL
                    )
                """)
                result_str = json.dumps(result, default=str, ensure_ascii=False)
                c.execute(
                    "INSERT OR REPLACE INTO cache (key, result, timestamp) "
                    "VALUES (?, ?, ?)",
                    (key_hash, result_str, time.time()),
                )
                conn.commit()
                conn.close()
                logger.info(f"💾 Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

        return result

    return wrapper


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek专用高性能LLM适配器 - 优化网络和缓存性能"""

    def __init__(self, enable_cache: bool = True, high_throughput: bool = True):
        """
        初始化高性能LLM适配器

        Args:
            enable_cache: 是否启用本地缓存
            high_throughput: 是否使用高吞吐量HTTP配置
        """
        self.settings = Settings()
        self.enable_cache = enable_cache

        # 设置缓存
        if enable_cache:
            cache_dir = os.path.join(os.getcwd(), ".cache", "llm")
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file_name = os.path.join(cache_dir, "llm_cache.sqlite")

        # 初始化高性能HTTP客户端
        http_client = None
        if high_throughput:
            logger.info("🚀 Initializing high-performance HTTP client...")
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            http_client = httpx.Client(
                limits=limits,
                timeout=httpx.Timeout(300.0, read=300.0),  # 5分钟超时
            )
            logger.info(
                "✅ HTTP client configured with 500 connections, 100 keep-alive"
            )

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_api_base,
            http_client=http_client,
            max_retries=0,  # 我们使用tenacity进行重试
        )

        # 加载提示词
        self._load_prompts()

        # Set model name property
        self.model_name = self.settings.deepseek_model

        logger.info(
            "⚡ DeepSeekAdapter initialized with high-performance configuration"
        )

    def _load_prompts(self):
        """加载提示词配置"""
        prompts_path = Path(self.settings.prompts_path)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        with open(prompts_path, encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

        if "ner" not in self.prompts:
            raise ValueError("NER prompts not found in configuration")

        self.ner_config = self.prompts["ner"]
        self.re_config = self.prompts.get("re", None)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_api_with_retry(self, messages: list[dict], operation: str) -> str:
        """使用tenacity进行API调用重试"""
        logger.info(f"🌐 {operation} API call starting...")

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=messages,
                temperature=0.0,
                max_tokens=4000,
            )

            api_time = time.time() - start_time
            content = response.choices[0].message.content

            logger.info(f"✅ {operation} API success in {api_time:.3f}s")
            logger.debug(
                f"📊 Tokens - Input: {response.usage.prompt_tokens}, "
                f"Output: {response.usage.completion_tokens}"
            )

            return content

        except Exception as e:
            api_time = time.time() - start_time
            logger.error(f"❌ {operation} API failed after {api_time:.3f}s: {e}")
            raise

    @cache_response
    def extract_entities(
        self, text: str, include_types: bool = True
    ) -> list[dict] | list[str]:
        """
        提取命名实体(带高性能缓存)

        Args:
            text: 输入文本
            include_types: 是否返回类型化实体

        Returns:
            实体列表
        """
        if not text or not text.strip():
            return []

        try:
            # 构建消息
            messages = self._build_ner_messages(text)

            # API调用
            response_content = self._call_api_with_retry(messages, "NER")

            # 解析响应
            return self._parse_ner_response(response_content, include_types)

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def _build_ner_messages(self, text: str) -> list[dict]:
        """构建NER消息"""
        messages = []

        # 系统提示
        messages.append({"role": "system", "content": self.ner_config["system"]})

        # 示例
        if self.ner_config.get("examples"):
            for example in self.ner_config["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        # 用户查询
        template = Template(self.ner_config["template"])
        user_content = template.substitute(passage=text)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_ner_response(
        self, response: str, include_types: bool
    ) -> list[dict] | list[str]:
        """解析NER响应"""
        try:
            # 清理响应
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response)

            if isinstance(data, dict) and "named_entities" in data:
                entities = data["named_entities"]
                if not isinstance(entities, list):
                    return []

                if not entities:
                    return []

                # 处理类型化实体
                if include_types and isinstance(entities[0], dict):
                    valid_entities = []
                    for entity in entities:
                        if (
                            isinstance(entity, dict)
                            and "text" in entity
                            and "type" in entity
                            and entity["text"]
                            and entity["text"].strip()
                        ):
                            valid_entities.append(
                                {
                                    "text": str(entity["text"]).strip(),
                                    "type": entity["type"],
                                }
                            )
                    return valid_entities
                else:
                    # 返回字符串列表
                    return [str(e).strip() for e in entities if e and str(e).strip()]

            return []

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse NER response: {e}")
            return []

    @cache_response
    def extract_relations(
        self, text: str, entities: list[dict[str, str]]
    ) -> list[list[str]]:
        """
        提取关系(带高性能缓存)

        Args:
            text: 输入文本
            entities: 实体列表

        Returns:
            关系三元组列表
        """
        if not text or not text.strip() or not entities:
            return []

        if not self.re_config:
            logger.error("RE configuration not found")
            return []

        try:
            # 提取实体文本
            entity_texts = [entity["text"] for entity in entities if "text" in entity]
            if not entity_texts:
                return []

            # 构建消息
            messages = self._build_re_messages(text, entity_texts)

            # API调用
            response_content = self._call_api_with_retry(messages, "RE")

            # 解析响应
            return self._parse_re_response(response_content, entity_texts)

        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []

    def _build_re_messages(self, text: str, entity_texts: list[str]) -> list[dict]:
        """构建RE消息"""
        messages = []

        # 系统提示
        messages.append({"role": "system", "content": self.re_config["system"]})

        # 示例
        if self.re_config.get("examples"):
            for example in self.re_config["examples"]:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["assistant"]})

        # 格式化实体JSON
        entity_json = json.dumps({"named_entities": entity_texts}, ensure_ascii=False)

        # 用户查询
        template = Template(self.re_config["template"])
        user_content = template.substitute(passage=text, named_entity_json=entity_json)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _parse_re_response(
        self, response: str, entity_texts: list[str]
    ) -> list[list[str]]:
        """解析RE响应"""
        try:
            # 清理响应
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]

            data = json.loads(response)

            if isinstance(data, dict) and "triples" in data:
                raw_triples = data["triples"]
                if not isinstance(raw_triples, list):
                    return []

                # 验证和去重
                entity_set = set(entity_texts)
                valid_triples = []
                seen_triples = set()

                for triple in raw_triples:
                    if isinstance(triple, list) and len(triple) == 3:
                        triple = [str(elem).strip() for elem in triple]

                        # 至少包含一个命名实体
                        if any(elem in entity_set for elem in triple):
                            triple_tuple = tuple(triple)
                            if triple_tuple not in seen_triples:
                                seen_triples.add(triple_tuple)
                                valid_triples.append(triple)

                return valid_triples

            return []

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse RE response: {e}")
            return []

    def clear_cache(self):
        """清除缓存"""
        if self.enable_cache and hasattr(self, "cache_file_name"):
            if os.path.exists(self.cache_file_name):
                os.remove(self.cache_file_name)
                logger.info("🗑️ Cache cleared")

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        if not self.enable_cache or not hasattr(self, "cache_file_name"):
            return {"cache_enabled": False}

        if not os.path.exists(self.cache_file_name):
            return {"cache_enabled": True, "entries": 0}

        try:
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM cache")
            count = c.fetchone()[0]
            conn.close()
            return {"cache_enabled": True, "entries": count}
        except Exception:
            return {"cache_enabled": True, "entries": "unknown"}

    def get_http_stats(self) -> dict:
        """获取HTTP客户端统计信息"""
        http_client = getattr(self.client, "_client", None)
        if http_client and hasattr(http_client, "_pool"):
            pool = http_client._pool
            return {
                "max_connections": getattr(pool, "_max_connections", None),
                "max_keepalive": getattr(pool, "_max_keepalive_connections", None),
                "active_connections": len(getattr(pool, "_connections", [])),
            }
        return {"http_stats": "Not available"}

    @cache_response
    def generate(
        self,
        prompt: str,
        max_tokens: int = 5000,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text completion using DeepSeek.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated text
        """
        try:
            messages = [{"role": "user", "content": prompt}]

            # Add system message if provided
            if "system_prompt" in kwargs:
                messages.insert(
                    0, {"role": "system", "content": kwargs["system_prompt"]}
                )

            logger.info("🌐 Generation API call starting...")
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=self.settings.deepseek_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            api_time = time.time() - start_time
            content = response.choices[0].message.content

            logger.info(f"✅ Generation API success in {api_time:.3f}s")
            logger.debug(
                f"📊 Tokens - Input: {response.usage.prompt_tokens}, "
                f"Output: {response.usage.completion_tokens}"
            )

            return LLMResponse(
                content=content,
                model=self.settings.deepseek_model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )

        except Exception as e:
            logger.error(f"❌ Generation API failed: {e}")
            raise
