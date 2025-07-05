"""
中文文档处理器
基于数据分析，当前50个样本文档平均526 tokens，最大1049 tokens，
远小于Qwen3-Embedding-4B的8192 tokens上下文限制，
因此采用简化方案：直接使用整个文档，无需分块处理
"""

import re
from datetime import datetime
from typing import Any

import tiktoken


class DocumentChunk:
    """文档块数据结构（简化版，用于兼容性）"""

    def __init__(
        self,
        chunk_text: str,
        parent_title: str,
        parent_idx: int,
        chunk_index: int = 0,
        tokens: int = 0,
        metadata: dict | None = None,
    ):
        self.chunk_text = chunk_text
        self.parent_title = parent_title
        self.parent_idx = parent_idx
        self.chunk_index = chunk_index
        self.tokens = tokens
        self.metadata = metadata or {}


class ChineseTextSplitter:
    """
    中文文档处理器（简化版）
    基于实际数据分析，当前文档大小完全适合单文档处理
    """

    def __init__(
        self,
        chunk_target_size: int = 8000,  # 提高到接近模型限制
        chunk_min_size: int = 100,  # 保持最小限制
        chunk_max_size: int = 8000,  # 匹配Qwen3上下文限制
        chunk_overlap_ratio: float = 0.0,  # 无需重叠
        encoding_name: str = "cl100k_base",
        enable_chunking: bool = False,
    ):  # 默认关闭分块
        self.chunk_target_size = chunk_target_size
        self.chunk_min_size = chunk_min_size
        self.chunk_max_size = chunk_max_size
        self.chunk_overlap_ratio = chunk_overlap_ratio
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.enable_chunking = enable_chunking

        # 处理统计
        self.processing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_tokens_per_doc": 0,
            "max_tokens": 0,
            "processing_time": None,
            "timestamp": None,
        }

    def count_tokens(self, text: str) -> int:
        """计算文本token数量"""
        return len(self.tokenizer.encode(text))

    def analyze_document_size(self, text: str, title: str) -> dict[str, Any]:
        """分析文档大小和特征"""
        token_count = self.count_tokens(text)
        char_count = len(text)

        # 检测语义边界
        sections = len(re.findall(r"\n#+\s", text))  # Markdown标题数量
        paragraphs = len(re.findall(r"\n\n+", text))  # 段落数量

        return {
            "title": title,
            "char_count": char_count,
            "token_count": token_count,
            "sections": sections,
            "paragraphs": paragraphs,
            "needs_chunking": token_count > self.chunk_max_size,
            "chunking_recommended": False,  # 基于分析，不推荐分块
        }

    def process_single_document(
        self, text: str, title: str, idx: int
    ) -> list[DocumentChunk]:
        """
        处理单个文档
        根据enable_chunking设置决定是否分块
        """
        analysis = self.analyze_document_size(text, title)

        if self.enable_chunking and analysis["needs_chunking"]:
            # 如果启用分块且文档过大，进行分块（当前数据中不会触发）
            return self._smart_split(text, title, idx)
        else:
            # 直接使用整个文档（推荐方案）
            chunk = DocumentChunk(
                chunk_text=text,
                parent_title=title,
                parent_idx=idx,
                chunk_index=0,
                tokens=analysis["token_count"],
                metadata={
                    "is_complete_document": True,
                    "char_count": analysis["char_count"],
                    "sections": analysis["sections"],
                    "paragraphs": analysis["paragraphs"],
                    "processing_mode": "whole_document",
                },
            )
            return [chunk]

    def _smart_split(self, text: str, title: str, idx: int) -> list[DocumentChunk]:
        """智能分块（备用方案，当前不需要）"""
        # 基于语义边界的分块逻辑
        boundaries = self._find_semantic_boundaries(text)
        chunks = []

        start = 0
        chunk_index = 0

        for boundary in boundaries:
            chunk_text = text[start:boundary].strip()
            if chunk_text and self.count_tokens(chunk_text) >= self.chunk_min_size:
                chunk = DocumentChunk(
                    chunk_text=chunk_text,
                    parent_title=title,
                    parent_idx=idx,
                    chunk_index=chunk_index,
                    tokens=self.count_tokens(chunk_text),
                    metadata={
                        "is_complete_document": False,
                        "processing_mode": "smart_chunking",
                    },
                )
                chunks.append(chunk)
                chunk_index += 1
                start = boundary

        # 处理最后一个chunk
        if start < len(text):
            remaining_text = text[start:].strip()
            if remaining_text:
                chunk = DocumentChunk(
                    chunk_text=remaining_text,
                    parent_title=title,
                    parent_idx=idx,
                    chunk_index=chunk_index,
                    tokens=self.count_tokens(remaining_text),
                    metadata={
                        "is_complete_document": False,
                        "processing_mode": "smart_chunking",
                    },
                )
                chunks.append(chunk)

        return chunks

    def _find_semantic_boundaries(self, text: str) -> list[int]:
        """识别语义边界位置"""
        boundaries = []

        # Markdown标题 (##, ###)
        for match in re.finditer(r"\n#+\s", text):
            boundaries.append(match.start())

        # 中文句号
        for match in re.finditer(r"[。！？]", text):
            boundaries.append(match.end())

        # 双换行（段落分隔）
        for match in re.finditer(r"\n\n+", text):
            boundaries.append(match.start())

        return sorted(set(boundaries))

    def split_documents(self, documents: list[dict[str, Any]]) -> list[str]:
        """
        处理文档列表，返回HippoRAG兼容的字符串列表
        支持两种输入格式：
        1. 标准corpus格式: [{"title": str, "text": str, "idx": int}, ...]
        2. 简单字符串列表: [str, str, ...]
        """
        start_time = datetime.now()

        processed_texts = []
        total_tokens = 0
        max_tokens = 0

        # 处理不同输入格式
        if documents and isinstance(documents[0], dict):
            # 标准corpus格式
            for doc in documents:
                chunks = self.process_single_document(
                    doc["text"],
                    doc.get("title", f"Document_{doc.get('idx', 0)}"),
                    doc.get("idx", 0),
                )

                for chunk in chunks:
                    processed_texts.append(chunk.chunk_text)
                    total_tokens += chunk.tokens
                    max_tokens = max(max_tokens, chunk.tokens)

        else:
            # 简单字符串列表
            for idx, text in enumerate(documents):
                chunks = self.process_single_document(text, f"Document_{idx}", idx)

                for chunk in chunks:
                    processed_texts.append(chunk.chunk_text)
                    total_tokens += chunk.tokens
                    max_tokens = max(max_tokens, chunk.tokens)

        # 更新统计信息
        end_time = datetime.now()
        self.processing_stats.update(
            {
                "total_documents": len(documents),
                "total_chunks": len(processed_texts),
                "avg_tokens_per_doc": total_tokens // len(documents)
                if documents
                else 0,
                "max_tokens": max_tokens,
                "processing_time": str(end_time - start_time),
                "timestamp": start_time.isoformat(),
            }
        )

        return processed_texts

    def get_processing_stats(self) -> dict[str, Any]:
        """获取处理统计信息"""
        return self.processing_stats.copy()

    def generate_analysis_report(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """生成文档分析报告"""
        analyses = []

        for doc in documents:
            analysis = self.analyze_document_size(
                doc["text"], doc.get("title", f"Document_{doc.get('idx', 0)}")
            )
            analyses.append(analysis)

        # 统计分析
        token_counts = [a["token_count"] for a in analyses]
        char_counts = [a["char_count"] for a in analyses]

        report = {
            "summary": {
                "total_documents": len(analyses),
                "avg_tokens": sum(token_counts) // len(token_counts),
                "avg_chars": sum(char_counts) // len(char_counts),
                "max_tokens": max(token_counts),
                "min_tokens": min(token_counts),
                "max_chars": max(char_counts),
                "min_chars": min(char_counts),
                "docs_needing_chunking": sum(
                    1 for a in analyses if a["needs_chunking"]
                ),
                "chunking_utilization": f"{max(token_counts) / self.chunk_max_size * 100:.1f}%",
            },
            "size_distribution": {
                "0-500_tokens": sum(1 for t in token_counts if t < 500),
                "500-1000_tokens": sum(1 for t in token_counts if 500 <= t < 1000),
                "1000-2000_tokens": sum(1 for t in token_counts if 1000 <= t < 2000),
                "2000+_tokens": sum(1 for t in token_counts if t >= 2000),
            },
            "recommendations": {
                "enable_chunking": any(a["needs_chunking"] for a in analyses),
                "optimal_chunk_size": min(self.chunk_max_size, max(token_counts) * 2),
                "processing_mode": "whole_document"
                if not any(a["needs_chunking"] for a in analyses)
                else "smart_chunking",
            },
            "document_details": analyses,
        }

        return report
