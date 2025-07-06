import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # Convert cached message back to list format
                cached_messages = json.loads(message)
                # Return cached result without cache hit indicator
                return cached_messages, metadata

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            # Convert message list to JSON string for storage
            message_str = json.dumps(message)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message_str, metadata_str))
            conn.commit()
            conn.close()

        # Return result without cache hit indicator
        return message, metadata

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)  
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation using OpenAI-compatible API."""
    
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "DeepSeekLLM":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__(global_config)
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url or "https://api.deepseek.com"

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        # Initialize DeepSeek client with OpenAI compatibility
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
            
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url=self.llm_base_url,
            http_client=client,
            max_retries=self.max_retries
        )

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name or "deepseek-chat"
        config_dict['llm_base_url'] = self.llm_base_url
        config_dict['generate_params'] = {
                "model": config_dict['llm_name'],
                "max_tokens": config_dict.get("max_new_tokens", 4000),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
                "stream": False,
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling DeepSeek API with:\n{params}")
        
        # 添加INFO级别的日志以便调试
        logger.info(f"🤖 DeepSeek API 调用 - 模型: {params.get('model', 'unknown')}")
        logger.info(f"📝 输入消息数量: {len(messages)}")
        if messages:
            last_msg = messages[-1] if isinstance(messages, list) else messages
            if isinstance(last_msg, dict) and 'content' in last_msg:
                content_preview = last_msg['content'][:200] + "..." if len(last_msg['content']) > 200 else last_msg['content']
                logger.info(f"💭 最后一条输入消息预览: {content_preview}")

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        # 添加响应日志
        response_preview = response_message[:300] + "..." if len(response_message) > 300 else response_message
        logger.info(f"🎯 DeepSeek API 响应预览: {response_preview}")
        logger.info(f"📊 Token使用 - 输入: {response.usage.prompt_tokens}, 输出: {response.usage.completion_tokens}")
        
        # Convert response to TextChatMessage format
        response_messages = [{"role": "assistant", "content": response_message}]
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_messages, metadata