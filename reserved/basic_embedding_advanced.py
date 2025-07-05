#!/usr/bin/env python3
"""
Qwen3-Embedding-4B 高级使用示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一个优雅且健壮的嵌入模型使用模板，展示了内存管理、错误处理和性能优化的最佳实践。

特性：
- 🛡️  自动内存管理和OOM恢复
- 📊 实时性能监控和统计
- 🔄 智能重试机制
- 🎯 优化的批处理策略
- 📈 详细的性能基准测试

作者: HippoRAG Team
版本: 2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import gc
import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置和数据类
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class MemoryInfo:
    """内存信息数据类"""

    gpu_used: float
    gpu_total: float
    gpu_free: float
    gpu_peak: float
    cpu_percent: float

    def __str__(self):
        return (
            f"GPU: {self.gpu_used:.2f}/{self.gpu_total:.2f}GB "
            f"(空闲: {self.gpu_free:.2f}GB, 峰值: {self.gpu_peak:.2f}GB)"
        )


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""

    processing_time: float
    texts_count: int
    embeddings_shape: tuple[int, int]
    throughput: float
    memory_before: MemoryInfo
    memory_after: MemoryInfo

    def __str__(self):
        return (
            f"处理 {self.texts_count} 个文本，耗时 {self.processing_time:.2f}秒\n"
            f"吞吐量: {self.throughput:.1f} texts/秒\n"
            f"嵌入维度: {self.embeddings_shape[1]}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内存管理工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_memory_info() -> MemoryInfo:
    """
    获取详细的内存使用信息

    Returns:
        MemoryInfo: 包含GPU和CPU内存信息的数据类
    """
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_free = gpu_total - gpu_memory
    else:
        gpu_memory = gpu_memory_max = gpu_total = gpu_free = 0.0

    cpu_percent = psutil.virtual_memory().percent

    return MemoryInfo(
        gpu_used=gpu_memory,
        gpu_total=gpu_total,
        gpu_free=gpu_free,
        gpu_peak=gpu_memory_max,
        cpu_percent=cpu_percent,
    )


def clear_memory():
    """
    清理GPU和系统内存

    这个函数会：
    1. 清空CUDA缓存
    2. 触发Python垃圾回收
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def auto_retry_on_oom(
    max_retries: int = 3, wait_time: float = 2.0, backoff_factor: float = 1.5
):
    """
    装饰器：自动处理CUDA OOM错误

    Args:
        max_retries: 最大重试次数
        wait_time: 初始等待时间（秒）
        backoff_factor: 等待时间递增因子

    Returns:
        装饰后的函数，具有自动重试能力
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_wait = wait_time

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except torch.cuda.OutOfMemoryError:
                    logging.warning(f"⚠️  CUDA OOM (尝试 {attempt + 1}/{max_retries})")

                    mem_info = get_memory_info()
                    logging.info(f"当前内存状态: {mem_info}")

                    if attempt < max_retries - 1:
                        logging.info(f"🔄 清理内存并等待 {current_wait:.1f} 秒...")
                        clear_memory()
                        time.sleep(current_wait)
                        current_wait *= backoff_factor

                        # 再次检查内存
                        new_mem_info = get_memory_info()
                        logging.info(f"清理后内存: {new_mem_info}")
                    else:
                        logging.error("❌ 达到最大重试次数，操作失败")
                        return None
                except Exception as e:
                    logging.error(f"❌ 其他错误: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        raise e
            return None

        return wrapper

    return decorator


@contextmanager
def memory_management(tag: str = ""):
    """
    上下文管理器：自动内存管理和性能跟踪

    Args:
        tag: 操作标签，用于日志记录

    Usage:
        with memory_management("模型加载"):
            model = load_model()
    """
    tag_str = f"[{tag}] " if tag else ""
    initial_memory = get_memory_info()
    logging.info(f"🚀 {tag_str}开始 - 内存状态: {initial_memory}")

    start_time = time.time()
    try:
        yield
    finally:
        clear_memory()
        elapsed_time = time.time() - start_time
        final_memory = get_memory_info()
        logging.info(f"✅ {tag_str}完成 - 耗时: {elapsed_time:.2f}秒")
        logging.info(f"🧹 {tag_str}清理后内存: {final_memory}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 模型管理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Qwen3EmbeddingManager:
    """
    Qwen3嵌入模型管理器

    提供了模型加载、嵌入生成和性能监控的完整解决方案
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        embedding_dim: int | None = None,
    ):
        """
        初始化嵌入模型管理器

        Args:
            model_name: 模型名称
            device: 计算设备 ('cuda', 'cpu' 或 None自动选择)
            embedding_dim: 嵌入维度 (None使用默认2560)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim or int(
            os.getenv("QWEN3_EMBEDDING_DIM", "2560")
        )
        self.model = None
        self.performance_history = []

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    def load_model(self, max_retries: int = 3) -> bool:
        """
        安全加载模型，带自动重试和降级机制

        Args:
            max_retries: 最大重试次数

        Returns:
            bool: 是否成功加载
        """
        with memory_management("模型加载"):
            for attempt in range(max_retries):
                try:
                    logging.info(f"🔄 加载模型 (尝试 {attempt + 1}/{max_retries})...")

                    # 构建模型参数
                    model_kwargs = {"trust_remote_code": True, "device": self.device}

                    # 如果指定了自定义维度
                    if self.embedding_dim != 2560:
                        model_kwargs["truncate_dim"] = self.embedding_dim

                    self.model = SentenceTransformer(self.model_name, **model_kwargs)

                    # 验证模型维度
                    actual_dim = self.model.get_sentence_embedding_dimension()
                    logging.info(f"✅ 模型加载成功! 嵌入维度: {actual_dim}")

                    return True

                except torch.cuda.OutOfMemoryError as e:
                    logging.error(f"⚠️  GPU内存不足: {str(e)[:100]}...")
                    clear_memory()

                    if attempt == max_retries - 1 and self.device == "cuda":
                        logging.warning("🔄 尝试切换到CPU模式...")
                        self.device = "cpu"

                except Exception as e:
                    logging.error(f"❌ 加载失败: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        return False

            return False

    @auto_retry_on_oom(max_retries=3)
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        is_query: bool = False,
    ) -> np.ndarray | None:
        """
        编码文本为嵌入向量

        Args:
            texts: 待编码的文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            is_query: 是否为查询文本（使用特殊prompt）

        Returns:
            np.ndarray: 嵌入向量矩阵，失败返回None
        """
        if not self.model:
            logging.error("❌ 模型未加载")
            return None

        # 记录开始状态
        start_time = time.time()
        memory_before = get_memory_info()

        # 编码参数
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress,
            "normalize_embeddings": True,
        }

        # 如果是查询，使用query prompt
        if is_query:
            encode_kwargs["prompt_name"] = "query"

        # 执行编码
        embeddings = self.model.encode(texts, **encode_kwargs)

        # 记录性能指标
        processing_time = time.time() - start_time
        memory_after = get_memory_info()

        metrics = PerformanceMetrics(
            processing_time=processing_time,
            texts_count=len(texts),
            embeddings_shape=embeddings.shape,
            throughput=len(texts) / processing_time,
            memory_before=memory_before,
            memory_after=memory_after,
        )

        self.performance_history.append(metrics)
        logging.info(f"📊 性能指标: {metrics}")

        return embeddings

    def compute_similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """计算嵌入向量间的余弦相似度"""
        return cosine_similarity(embeddings1, embeddings2)

    def search(
        self, query: str, documents: list[str], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        语义搜索

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回最相似的K个结果

        Returns:
            List[Dict]: 包含文档、相似度和排名的结果列表
        """
        # 编码查询
        query_embedding = self.encode_texts([query], is_query=True, show_progress=False)
        if query_embedding is None:
            return []

        # 编码文档
        doc_embeddings = self.encode_texts(
            documents, show_progress=len(documents) > 100
        )
        if doc_embeddings is None:
            return []

        # 计算相似度
        similarities = self.compute_similarity(query_embedding, doc_embeddings)[0]

        # 获取top-k结果
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(
                {
                    "rank": rank,
                    "document": documents[idx],
                    "similarity": float(similarities[idx]),
                    "index": int(idx),
                }
            )

        return results

    def get_performance_summary(self) -> dict[str, Any]:
        """获取性能统计摘要"""
        if not self.performance_history:
            return {}

        total_texts = sum(m.texts_count for m in self.performance_history)
        total_time = sum(m.processing_time for m in self.performance_history)
        avg_throughput = total_texts / total_time if total_time > 0 else 0

        return {
            "total_operations": len(self.performance_history),
            "total_texts_processed": total_texts,
            "total_processing_time": total_time,
            "average_throughput": avg_throughput,
            "peak_gpu_memory": max(
                m.memory_after.gpu_used for m in self.performance_history
            ),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 演示函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def demo_basic_usage():
    """基础使用演示"""
    print("\n" + "=" * 80)
    print("🎯 基础嵌入生成演示")
    print("=" * 80)

    # 初始化管理器
    manager = Qwen3EmbeddingManager()

    # 加载模型
    if not manager.load_model():
        print("❌ 模型加载失败")
        return

    # 测试文本
    test_texts = [
        "苹果公司是全球最大的科技公司之一",
        "Apple Inc. is one of the world's largest technology companies",
        "人工智能正在改变世界",
        "Artificial intelligence is transforming the world",
        "比亚迪是中国领先的新能源汽车制造商",
        "BYD is a leading new energy vehicle manufacturer in China",
    ]

    # 生成嵌入
    with memory_management("嵌入生成"):
        embeddings = manager.encode_texts(test_texts, batch_size=32)

        if embeddings is not None:
            print(f"\n✅ 成功生成 {len(test_texts)} 个文本的嵌入")
            print(f"📐 嵌入形状: {embeddings.shape}")

            # 计算相似度矩阵
            similarity_matrix = manager.compute_similarity(embeddings, embeddings)

            print("\n📊 相似度矩阵（部分）:")
            for i in range(min(3, len(test_texts))):
                print(f"\n文本 {i + 1}: {test_texts[i][:30]}...")
                for j in range(min(3, len(test_texts))):
                    if i != j:
                        print(f"  与文本 {j + 1}: {similarity_matrix[i][j]:.4f}")


def demo_semantic_search():
    """语义搜索演示"""
    print("\n" + "=" * 80)
    print("🔍 语义搜索演示")
    print("=" * 80)

    # 初始化管理器
    manager = Qwen3EmbeddingManager()

    if not manager.load_model():
        return

    # 文档库
    documents = [
        "苹果公司发布了新的iPhone 15系列，搭载A17 Pro芯片",
        "微软推出了最新的Surface Pro设备，配备Intel最新处理器",
        "谷歌发布了Gemini AI模型，在多项基准测试中表现优异",
        "特斯拉展示了新的Model 3改款，续航里程大幅提升",
        "英伟达发布了RTX 4090 GPU，专为AI和游戏优化",
        "比亚迪发布了海豹车型，配备刀片电池技术",
        "宁德时代推出了新的电池技术，能量密度创新高",
        "华为发布了Mate 60系列手机，支持卫星通信",
    ]

    # 测试查询
    queries = ["苹果的新产品", "AI相关的新闻", "电动汽车技术"]

    for query in queries:
        print(f"\n🔎 查询: {query}")
        print("-" * 50)

        results = manager.search(query, documents, top_k=3)

        for result in results:
            print(
                f"{result['rank']}. [{result['similarity']:.4f}] {result['document']}"
            )


def demo_batch_performance():
    """批处理性能测试"""
    print("\n" + "=" * 80)
    print("⚡ 批处理性能测试")
    print("=" * 80)

    manager = Qwen3EmbeddingManager()

    if not manager.load_model():
        return

    # 准备测试数据
    test_sizes = [50, 100, 200]
    batch_sizes = [16, 32, 64]

    results = []

    for text_count in test_sizes:
        test_texts = [
            f"测试文本 {i}: 这是用于性能测试的示例文本内容" for i in range(text_count)
        ]

        print(f"\n📝 测试 {text_count} 个文本:")

        for batch_size in batch_sizes:
            with memory_management(f"Batch-{batch_size}"):
                embeddings = manager.encode_texts(
                    test_texts, batch_size=batch_size, show_progress=False
                )

                if embeddings is not None:
                    # 获取最新的性能指标
                    latest_metric = manager.performance_history[-1]
                    print(
                        f"  批大小 {batch_size:2d}: {latest_metric.throughput:6.1f} texts/秒"
                    )

                    results.append(
                        {
                            "text_count": text_count,
                            "batch_size": batch_size,
                            "throughput": latest_metric.throughput,
                        }
                    )

    # 显示性能摘要
    print("\n📊 性能统计摘要:")
    summary = manager.get_performance_summary()
    print(f"  总操作次数: {summary.get('total_operations', 0)}")
    print(f"  总处理文本: {summary.get('total_texts_processed', 0)}")
    print(f"  平均吞吐量: {summary.get('average_throughput', 0):.1f} texts/秒")
    print(f"  GPU内存峰值: {summary.get('peak_gpu_memory', 0):.2f}GB")


def demo_dimension_comparison():
    """不同嵌入维度对比"""
    print("\n" + "=" * 80)
    print("📏 嵌入维度性能对比")
    print("=" * 80)

    dimensions = [1024, 1536, 2048, 2560]
    test_texts = ["测试文本"] * 100

    results = []

    for dim in dimensions:
        print(f"\n🔢 测试维度: {dim}")

        # 为每个维度创建新的管理器
        manager = Qwen3EmbeddingManager(embedding_dim=dim)

        with memory_management(f"维度-{dim}"):
            if manager.load_model():
                embeddings = manager.encode_texts(test_texts, show_progress=False)

                if embeddings is not None:
                    metric = manager.performance_history[-1]

                    results.append(
                        {
                            "dimension": dim,
                            "throughput": metric.throughput,
                            "memory_used": metric.memory_after.gpu_used,
                            "actual_dim": embeddings.shape[1],
                        }
                    )

                    print(f"  ✅ 吞吐量: {metric.throughput:.1f} texts/秒")
                    print(f"  💾 GPU内存: {metric.memory_after.gpu_used:.2f}GB")
                else:
                    print("  ❌ 测试失败")

            # 清理模型
            del manager

    # 显示对比结果
    if results:
        print("\n📊 维度对比总结:")
        print(f"{'维度':>6} | {'吞吐量':>12} | {'GPU内存':>10}")
        print("-" * 35)
        for r in results:
            print(
                f"{r['dimension']:>6} | {r['throughput']:>10.1f}/s | {r['memory_used']:>8.2f}GB"
            )

        # 推荐最佳维度
        best = max(results, key=lambda x: x["throughput"])
        print(
            f"\n🏆 推荐维度: {best['dimension']} (最高吞吐量: {best['throughput']:.1f} texts/秒)"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    """主函数：运行所有演示"""
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "Qwen3-Embedding-4B 高级使用示例" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")

    # 显示系统信息
    print("\n📋 系统信息:")
    print(f"  Python版本: {sys.version.split()[0]}")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(
            f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    # 运行演示
    demos = [
        ("基础使用", demo_basic_usage),
        ("语义搜索", demo_semantic_search),
        ("批处理性能", demo_batch_performance),
        ("维度对比", demo_dimension_comparison),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            print(f"\n\n{'=' * 80}")
            print(f"[{i}/{len(demos)}] {name}")
            print("=" * 80)
            demo_func()

            # 演示间清理内存
            clear_memory()
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n⏹️  用户中断")
            break
        except Exception as e:
            print(f"\n❌ 演示出错: {e}")
            import traceback

            traceback.print_exc()

    print("\n\n" + "=" * 80)
    print("🎉 所有演示完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
