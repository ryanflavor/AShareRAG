"""Qwen3-Reranker-4B adapter for document reranking."""

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for the reranker."""

    model_name: str = "Qwen/Qwen3-Reranker-4B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    max_length: int = 8192
    batch_size: int = 8
    use_bf16: bool = True
    cache_dir: str | None = None


@dataclass
class RerankResult:
    """Result of document reranking."""

    document: str
    score: float
    original_index: int
    metadata: dict[str, Any] | None = None


class Qwen3RerankerAdapter:
    """
    Adapter for Qwen3-Reranker-4B model.
    Uses generative approach to compute relevance scores via yes/no probabilities.
    """

    def __init__(self, config: RerankerConfig | None = None):
        """Initialize the reranker adapter.

        Args:
            config: Reranker configuration
        """
        self.config = config or RerankerConfig()

        # Load model and tokenizer
        self._load_model()

        # Predefined prompt template (official format)
        self.prefix = (
            "<|im_start|>system\n"
            'Judge. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        # Pre-compute token IDs
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")
        

        # Performance statistics
        self.total_processed = 0
        self.total_time = 0.0

    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

        # Set padding configuration
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine data type
        if self.config.use_bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            logger.info("Using BFloat16 precision")
        else:
            dtype = self.config.dtype
            logger.info(f"Using {dtype} precision")

        # Load model (using CausalLM)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.config.device == "cuda" else None,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
        )

        self.model.eval()

        # If not using device_map, manually move to device
        if self.config.device == "cuda" and not hasattr(self.model, "hf_device_map"):
            self.model = self.model.to(self.config.device)

        logger.info("Model loaded successfully")
        self._log_memory_usage()

    def _format_input(self, query: str, document: str) -> str:
        """Format input for the model.

        Args:
            query: Query text
            document: Document text

        Returns:
            Formatted input string
        """
        instruction = (
            f'Given a query "{query}", does the following document '
            f'answer the query? "{document}"'
        )
        return self.prefix + instruction + self.suffix

    def _log_memory_usage(self):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(
                f"GPU memory: allocated {allocated:.2f}GB / reserved {reserved:.2f}GB"
            )

    @contextmanager
    def _memory_efficient_mode(self):
        """Context manager for memory efficient processing."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        yield
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _process_batch(self, query: str, documents: list[str]) -> list[float]:
        """Process a batch of documents.

        Args:
            query: Query text
            documents: List of document texts

        Returns:
            List of relevance scores
        """
        # Format inputs
        batch_inputs = [self._format_input(query, doc) for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.config.max_length,
        )

        # Move to correct device
        if self.config.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Get yes and no logits
            yes_logits = logits[:, self.token_yes]
            no_logits = logits[:, self.token_no]

            # Calculate softmax probabilities
            probs = torch.softmax(torch.stack([no_logits, yes_logits], dim=-1), dim=-1)
            scores = probs[:, 1].float().cpu().numpy()  # yes probability

        return scores.tolist()

    def rerank(
        self,
        query: str,
        documents: list[str],
        batch_size: int | None = None,
        top_k: int | None = None,
        return_metrics: bool = False,
    ) -> list[RerankResult]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Query text
            documents: List of document texts
            batch_size: Batch size for processing
            top_k: Return only top k results
            return_metrics: Whether to return performance metrics

        Returns:
            List of reranked results sorted by score descending
        """
        if not documents:
            return []

        batch_size = batch_size or self.config.batch_size
        start_time = time.time()

        logger.info(f"Starting reranking of {len(documents)} documents")

        # Batch processing
        all_scores = []

        try:
            with self._memory_efficient_mode():
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i : i + batch_size]

                    # Process batch
                    try:
                        batch_scores = self._process_batch(query, batch_docs)
                        all_scores.extend(batch_scores)

                        logger.debug(
                            f"Processed batch {i // batch_size + 1}/"
                            f"{(len(documents) - 1) // batch_size + 1}"
                        )

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning(
                                "OOM error, processing documents individually"
                            )
                            # Clear memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Process one by one
                            for doc in batch_docs:
                                score = self._process_batch(query, [doc])[0]
                                all_scores.append(score)
                        else:
                            raise

        except Exception as e:
            logger.error(f"Reranking failed: {e!s}")
            raise

        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(documents, all_scores, strict=False)):
            results.append(RerankResult(document=doc, score=score, original_index=i))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k
        if top_k is not None and top_k < len(results):
            results = results[:top_k]

        # Update statistics
        elapsed_time = time.time() - start_time
        self.total_processed += len(documents)
        self.total_time += elapsed_time

        logger.info(f"Reranking completed in {elapsed_time:.2f}s")

        return results

    def rerank_with_metadata(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents with metadata.

        Args:
            query: Query text
            documents: List of document dictionaries
            text_key: Key for text field in documents
            **kwargs: Additional arguments for rerank

        Returns:
            List of documents sorted by relevance with rerank scores
        """
        # Extract texts
        texts = [doc[text_key] for doc in documents]

        # Rerank
        results = self.rerank(query, texts, **kwargs)

        # Combine results
        ranked_docs = []
        for i, result in enumerate(results):
            doc = documents[result.original_index].copy()
            doc["rerank_score"] = result.score
            doc["rerank_rank"] = i + 1
            ranked_docs.append(doc)

        return ranked_docs

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_documents_processed": self.total_processed,
            "total_processing_time": self.total_time,
            "average_throughput": (
                self.total_processed / self.total_time if self.total_time > 0 else 0
            ),
            "model_name": self.config.model_name,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
        }
