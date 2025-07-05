#!/usr/bin/env python3
"""
Qwen3-Reranker-4B 官方方法最佳实践
使用生成式方法计算 yes/no 概率来评分
支持批处理、OOM 恢复、性能监控等生产级功能
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time
import logging
import sys
import os
import gc
from contextlib import contextmanager
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据类定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RerankerConfig:
    """重排序器配置"""

    model_name: str = "Qwen/Qwen3-Reranker-4B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_length: int = 8192
    batch_size: int = 8
    use_bf16: bool = True
    cache_dir: Optional[str] = None
    log_level: str = "INFO"


@dataclass
class RerankResult:
    """重排序结果"""

    document: str
    score: float
    original_index: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """性能指标"""

    total_documents: int
    processing_time: float
    throughput: float
    peak_memory_mb: float
    average_score: float
    score_range: Tuple[float, float]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主要实现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Qwen3RerankerOfficial:
    """
    Qwen3-Reranker-4B 官方实现
    使用生成式方法通过 yes/no 概率计算相关性分数
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
        self.logger = self._setup_logger()

        # 加载模型和分词器
        self._load_model()

        # 预定义的 prompt 模板（官方格式）
        self.prefix = '<|im_start|>system\nJudge. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        # 预计算 token ids
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")

        # 性能统计
        self.total_processed = 0
        self.total_time = 0.0

    def _setup_logger(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger("Qwen3RerankerOfficial")
        logger.setLevel(getattr(logging, self.config.log_level))

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_model(self):
        """加载模型和分词器"""
        self.logger.info(f"加载模型: {self.config.model_name}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

        # 设置 padding 配置
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 确定数据类型
        if self.config.use_bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            self.logger.info("使用 BFloat16 精度")
        else:
            dtype = self.config.dtype
            self.logger.info(f"使用 {dtype} 精度")

        # 加载模型（注意：使用 CausalLM）
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

        self.model.eval()

        # 如果没有使用 device_map，手动移到设备
        if self.config.device == "cuda" and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.config.device)

        self.logger.info("模型加载完成")
        self._log_memory_usage()

    def _format_input(self, query: str, document: str) -> str:
        """格式化单个输入"""
        instruction = f'Given a query "{query}", does the following document answer the query? "{document}"'
        return self.prefix + instruction + self.suffix

    def _log_memory_usage(self):
        """记录内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.debug(
                f"GPU 内存: 已分配 {allocated:.2f}GB / 已保留 {reserved:.2f}GB"
            )

    @contextmanager
    def _memory_efficient_mode(self):
        """内存高效模式上下文管理器"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        yield
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _process_batch(self, query: str, documents: List[str]) -> List[float]:
        """处理单个批次"""
        # 格式化输入
        batch_inputs = [self._format_input(query, doc) for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config.max_length,
        )

        # 移到正确的设备
        if self.config.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # 最后一个 token 的 logits

            # 获取 yes 和 no 的 logits
            yes_logits = logits[:, self.token_yes]
            no_logits = logits[:, self.token_no]

            # 计算 softmax 概率
            probs = torch.softmax(torch.stack([no_logits, yes_logits], dim=-1), dim=-1)
            scores = probs[:, 1].float().cpu().numpy()  # yes 的概率

        return scores.tolist()

    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None,
        top_k: Optional[int] = None,
        return_metrics: bool = False,
    ) -> Tuple[List[RerankResult], Optional[PerformanceMetrics]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            batch_size: 批处理大小（None 使用默认值）
            top_k: 返回前 k 个结果
            return_metrics: 是否返回性能指标

        Returns:
            (排序后的结果列表, 性能指标)
        """
        if not documents:
            return [], None

        batch_size = batch_size or self.config.batch_size
        start_time = time.time()
        peak_memory = 0

        self.logger.info(f"开始重排序 {len(documents)} 个文档")

        # 批量处理
        all_scores = []

        try:
            with self._memory_efficient_mode():
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i : i + batch_size]

                    # 处理批次
                    try:
                        batch_scores = self._process_batch(query, batch_docs)
                        all_scores.extend(batch_scores)

                        # 记录内存使用
                        if torch.cuda.is_available():
                            current_memory = torch.cuda.memory_allocated() / 1024**2
                            peak_memory = max(peak_memory, current_memory)

                        self.logger.debug(
                            f"处理批次 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}"
                        )

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            self.logger.warning(f"OOM 错误，尝试单文档处理")
                            # 清理内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # 逐个处理
                            for doc in batch_docs:
                                score = self._process_batch(query, [doc])[0]
                                all_scores.append(score)
                        else:
                            raise

        except Exception as e:
            self.logger.error(f"重排序失败: {str(e)}")
            raise

        # 创建结果
        results = []
        for i, (doc, score) in enumerate(zip(documents, all_scores)):
            results.append(RerankResult(document=doc, score=score, original_index=i))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 应用 top_k
        if top_k is not None and top_k < len(results):
            results = results[:top_k]

        # 计算性能指标
        elapsed_time = time.time() - start_time
        self.total_processed += len(documents)
        self.total_time += elapsed_time

        metrics = None
        if return_metrics:
            metrics = PerformanceMetrics(
                total_documents=len(documents),
                processing_time=elapsed_time,
                throughput=len(documents) / elapsed_time,
                peak_memory_mb=peak_memory,
                average_score=np.mean(all_scores),
                score_range=(min(all_scores), max(all_scores)),
            )

            self.logger.info(
                f"完成! 处理时间: {elapsed_time:.2f}s, "
                f"吞吐量: {metrics.throughput:.1f} docs/s, "
                f"分数范围: {metrics.score_range[0]:.4f}-{metrics.score_range[1]:.4f}"
            )

        return results, metrics

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        重排序带元数据的文档

        Args:
            query: 查询文本
            documents: 文档字典列表
            text_key: 文本字段的键名
            **kwargs: 传递给 rerank 的其他参数

        Returns:
            排序后的文档列表（包含原始元数据和分数）
        """
        # 提取文本
        texts = [doc[text_key] for doc in documents]

        # 重排序
        results, _ = self.rerank(query, texts, **kwargs)

        # 组合结果
        ranked_docs = []
        for i, result in enumerate(results):
            doc = documents[result.original_index].copy()
            doc["rerank_score"] = result.score
            doc["rerank_rank"] = i + 1
            ranked_docs.append(doc)

        return ranked_docs

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_documents_processed": self.total_processed,
            "total_processing_time": self.total_time,
            "average_throughput": self.total_processed / self.total_time
            if self.total_time > 0
            else 0,
            "model_name": self.config.model_name,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 测试和示例
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    """测试主函数"""
    print("🚀 Qwen3-Reranker-4B 官方方法测试")
    print("=" * 60)

    # 创建配置
    config = RerankerConfig(batch_size=8, use_bf16=True, log_level="INFO")

    # 初始化重排序器
    reranker = Qwen3RerankerOfficial(config)

    # 测试数据
    query = "Python编程语言的特点"
    documents = [
        "Python是一种解释型、面向对象的高级编程语言，语法简洁优雅",
        "Python广泛应用于数据科学、机器学习和Web开发",
        "Java是一种强类型的编程语言",
        "今天天气晴朗，适合外出游玩",
        "我喜欢吃苹果",
        "Python拥有丰富的第三方库和活跃的社区",
        "C++是一种编译型语言，性能优秀",
        "Python的动态类型系统使得开发更加灵活",
    ]

    # 执行重排序
    results, metrics = reranker.rerank(query, documents, top_k=5, return_metrics=True)

    # 显示结果
    print("\n🏆 重排序结果 (Top 5):")
    print("-" * 60)
    for i, result in enumerate(results):
        doc_preview = (
            result.document[:60] + "..."
            if len(result.document) > 60
            else result.document
        )
        print(f"{i + 1}. [分数: {result.score:.4f}] {doc_preview}")

    # 显示性能指标
    if metrics:
        print(f"\n📊 性能指标:")
        print(f"- 处理文档数: {metrics.total_documents}")
        print(f"- 处理时间: {metrics.processing_time:.2f}秒")
        print(f"- 吞吐量: {metrics.throughput:.1f} docs/秒")
        print(f"- 峰值内存: {metrics.peak_memory_mb:.1f} MB")
        print(f"- 平均分数: {metrics.average_score:.4f}")
        print(
            f"- 分数范围: {metrics.score_range[0]:.4f} - {metrics.score_range[1]:.4f}"
        )

    # 测试带元数据的文档
    print("\n\n📋 测试带元数据的文档:")
    print("-" * 60)

    docs_with_metadata = [
        {"id": 1, "text": "Python是面向对象的编程语言", "source": "wiki"},
        {"id": 2, "text": "今天股市大涨", "source": "news"},
        {"id": 3, "text": "Python在数据科学领域应用广泛", "source": "blog"},
    ]

    ranked_docs = reranker.rerank_with_metadata(query, docs_with_metadata)

    for doc in ranked_docs:
        print(
            f"ID: {doc['id']}, 分数: {doc['rerank_score']:.4f}, 来源: {doc['source']}"
        )
        print(f"内容: {doc['text']}")
        print()

    # 显示统计信息
    stats = reranker.get_stats()
    print(f"\n📈 总体统计:")
    print(f"- 总处理文档数: {stats['total_documents_processed']}")
    print(f"- 平均吞吐量: {stats['average_throughput']:.1f} docs/秒")


if __name__ == "__main__":
    main()
