"""Answer Synthesizer component for generating comprehensive answers.

This module implements the AnswerSynthesizer component that takes retrieved
and ranked documents to generate a comprehensive, coherent answer using
an LLM (DeepSeek V3).
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import yaml

from src.adapters.llm_adapter import LLMAdapter, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class AnswerSynthesizerConfig:
    """Configuration for the AnswerSynthesizer component."""

    max_input_tokens: int = 3000
    max_output_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    answer_language: str = "Chinese"  # Chinese or English
    include_citations: bool = True
    citation_format: str = "[{idx}]"  # Format for citation markers


class AnswerSynthesizer:
    """Synthesizes comprehensive answers from retrieved documents using LLM."""

    def __init__(self, config: AnswerSynthesizerConfig, llm_adapter: LLMAdapter):
        """Initialize the AnswerSynthesizer.

        Args:
            config: Configuration for the synthesizer
            llm_adapter: LLM adapter instance (e.g., DeepSeekAdapter)
        """
        self.config = config
        self.llm_adapter = llm_adapter
        self._prompts: dict[str, Any] | None = None

        # Statistics
        self._total_syntheses = 0
        self._total_tokens_used = 0
        self._total_synthesis_time = 0.0
        self._error_count = 0

        # Load prompts from config
        self._load_prompts()

        logger.info(
            f"AnswerSynthesizer initialized with config: "
            f"max_tokens={config.max_input_tokens}, "
            f"language={config.answer_language}, "
            f"citations={config.include_citations}"
        )

    async def synthesize_answer(
        self,
        query: str,
        documents: list[dict[str, Any]],
        custom_prompt: str | None = None,
        fallback_on_error: bool = False,
        include_metadata_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Synthesize an answer from retrieved documents.

        Args:
            query: The user's query
            documents: List of retrieved and ranked documents
            custom_prompt: Optional custom prompt template
            fallback_on_error: Whether to use fallback if synthesis fails
            include_metadata_fields: Additional metadata fields to include

        Returns:
            Dictionary containing:
                - answer: The synthesized answer
                - sources: The source documents used
                - synthesis_time: Time taken for synthesis
                - metadata: Additional metadata about the synthesis
        """
        start_time = time.time()

        try:
            # Handle empty documents case
            if not documents:
                answer = self._generate_no_results_answer(query)
                return {
                    "answer": answer,
                    "sources": [],
                    "synthesis_time": time.time() - start_time,
                    "metadata": {
                        "model": (
                            self.llm_adapter.model_name
                            if hasattr(self.llm_adapter, "model_name")
                            else "unknown"
                        ),
                        "document_count": 0,
                        "language": self.config.answer_language,
                        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    },
                }

            # Truncate documents to fit token limit
            truncated_docs = self._truncate_documents_to_fit(
                documents,
                max_chars=self.config.max_input_tokens * 3,  # Rough char estimate
            )

            # Format documents for prompt
            formatted_docs = self._format_documents_for_prompt(
                truncated_docs, include_metadata_fields=include_metadata_fields
            )

            # Construct prompt
            if custom_prompt:
                prompt = custom_prompt.format(query=query, documents=formatted_docs)
            else:
                prompt = self._build_synthesis_prompt(query, formatted_docs)

            # Generate answer using LLM
            try:
                # Get system prompt if available
                system_prompt = None
                if self._prompts and "fact_qa_synthesis" in self._prompts:
                    system_prompt = self._prompts["fact_qa_synthesis"].get("system")

                response = await self.llm_adapter.generate_async(
                    prompt=prompt,
                    max_tokens=self.config.max_output_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                if fallback_on_error:
                    logger.warning(f"Primary synthesis failed, using fallback: {e}")
                    response = await self._generate_fallback_answer(
                        query, truncated_docs
                    )
                    fallback_used = True
                else:
                    raise
            else:
                fallback_used = False

            # Update statistics
            self._total_syntheses += 1
            synthesis_time = time.time() - start_time
            self._total_synthesis_time += synthesis_time

            if hasattr(response, "usage") and response.usage:
                self._total_tokens_used += response.usage.get(
                    "prompt_tokens", 0
                ) + response.usage.get("completion_tokens", 0)

            # Log performance
            logger.info(
                f"Answer synthesis completed in {synthesis_time:.2f}s - "
                f"docs: {len(documents)}, "
                f"synthesis_time: {synthesis_time:.2f}s, "
                f"tokens: {response.usage if hasattr(response, 'usage') else 'N/A'}"
            )

            # Prepare result
            result = {
                "answer": response.content,
                "sources": truncated_docs,
                "synthesis_time": synthesis_time,
                "metadata": {
                    "model": (
                        response.model
                        if hasattr(response, "model")
                        else self.llm_adapter.model_name
                        if hasattr(self.llm_adapter, "model_name")
                        else "unknown"
                    ),
                    "document_count": len(truncated_docs),
                    "language": self.config.answer_language,
                    "token_usage": (
                        response.usage
                        if hasattr(response, "usage")
                        else {"prompt_tokens": 0, "completion_tokens": 0}
                    ),
                },
            }

            if fallback_used:
                result["metadata"]["fallback_used"] = True

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in answer synthesis: {e}")
            raise

    def _format_documents_for_prompt(
        self,
        documents: list[dict[str, Any]],
        include_metadata_fields: list[str] | None = None,
    ) -> str:
        """Format documents for inclusion in the prompt.

        Args:
            documents: List of documents to format
            include_metadata_fields: Additional metadata fields to include

        Returns:
            Formatted string representation of documents
        """
        formatted_parts = []

        for idx, doc in enumerate(documents, 1):
            # Document header
            if self.config.answer_language == "Chinese":
                parts = [f"文档 {idx}:"]
            else:
                parts = [f"Document {idx}:"]

            # Content
            parts.append(doc.get("content", ""))

            # Metadata
            metadata = doc.get("metadata", {})
            if metadata:
                # Source information
                if "source" in metadata:
                    source_label = (
                        "来源" if self.config.answer_language == "Chinese" else "Source"
                    )
                    parts.append(f"{source_label}: {metadata['source']}")

                # Additional metadata fields if requested
                if include_metadata_fields:
                    for field in include_metadata_fields:
                        if field in metadata:
                            parts.append(f"{field}: {metadata[field]}")

            # Relevance score
            if "score" in doc:
                score_label = (
                    "相关度"
                    if self.config.answer_language == "Chinese"
                    else "Relevance"
                )
                parts.append(f"{score_label}: {doc['score']:.2f}")

            # Combine parts
            formatted_parts.append("\n".join(parts))

        return "\n\n".join(formatted_parts)

    def _truncate_documents_to_fit(
        self, documents: list[dict[str, Any]], max_chars: int
    ) -> list[dict[str, Any]]:
        """Truncate documents to fit within token limit.

        Args:
            documents: Original documents
            max_chars: Maximum character count (approximate)

        Returns:
            Truncated list of documents
        """
        truncated = []
        current_chars = 0

        for doc in documents:
            doc_chars = len(doc.get("content", ""))
            if current_chars + doc_chars > max_chars:
                # If we haven't included any docs yet, include partial first doc
                if not truncated:
                    partial_doc = doc.copy()
                    partial_doc["content"] = doc["content"][: max_chars - 100] + "..."
                    truncated.append(partial_doc)
                break

            truncated.append(doc)
            current_chars += doc_chars

        return truncated

    def _build_synthesis_prompt(self, query: str, formatted_documents: str) -> str:
        """Build the synthesis prompt.

        Args:
            query: The user's query
            formatted_documents: Formatted document content

        Returns:
            Complete prompt for synthesis
        """
        if self._prompts and "fact_qa_synthesis" in self._prompts:
            # Use loaded prompts
            template = self._prompts["fact_qa_synthesis"]["user"]
            return template.format(
                query=query,
                documents=formatted_documents,
                language=self.config.answer_language,
                citation_instruction=self._get_citation_instruction(),
            )

        # Default prompt
        if self.config.answer_language == "Chinese":
            prompt = f"""基于以下参考文档，请为用户的查询提供一个准确、全面的答案。

用户查询：{query}

参考文档：
{formatted_documents}

要求：
1. 答案必须基于提供的文档内容，不要添加文档中没有的信息
2. 如果文档中的信息存在冲突，请指出并说明
3. 答案应该简洁明了，重点突出
{self._get_citation_instruction()}

请提供您的答案："""
        else:
            prompt = f"""Based on the following reference documents, please provide "
                f"an accurate and comprehensive answer to the user's query.

User Query: {query}"

Reference Documents:
{formatted_documents}

Requirements:
1. The answer must be based on the provided document content, \
do not add information not in the documents
2. If there are conflicts in the document information, please point them out
3. The answer should be concise and focused
{self._get_citation_instruction()}

Please provide your answer:"""

        return prompt

    def _get_citation_instruction(self) -> str:
        """Get citation instruction based on configuration."""
        if not self.config.include_citations:
            return ""

        if self.config.answer_language == "Chinese":
            return (
                f"4. 使用 {self.config.citation_format.format(idx='编号')} "
                f"格式标注信息来源"
            )
        else:
            return (
                f"4. Use {self.config.citation_format.format(idx='number')} "
                f"format to cite sources"
            )

    def _generate_no_results_answer(self, query: str) -> str:
        """Generate answer when no documents are found."""
        if self.config.answer_language == "Chinese":
            return "抱歉，未找到与您查询相关的信息。"
        else:
            return "Sorry, no information related to your query was found."

    async def _generate_fallback_answer(
        self, query: str, documents: list[dict[str, Any]]
    ) -> LLMResponse:
        """Generate a fallback answer when primary synthesis fails."""
        # Simple fallback: just concatenate document summaries
        if self.config.answer_language == "Chinese":
            content = "基于找到的文档，以下是相关信息的摘要：\n\n"
        else:
            content = (
                "Based on the documents found, here is a summary of "
                "relevant information:\n\n"
            )

        for idx, doc in enumerate(documents[:3], 1):  # Limit to top 3
            content += f"{idx}. {doc.get('content', '')[:200]}...\n"

        return LLMResponse(
            content=content,
            model="fallback",
            usage={"prompt_tokens": 100, "completion_tokens": len(content.split())},
        )

    def _load_prompts(self, prompts_path: str | None = None) -> None:
        """Load prompts from YAML file.

        Args:
            prompts_path: Path to the prompts YAML file.
                If None, uses default from settings
        """
        try:
            if prompts_path is None:
                # Get settings to find prompts path
                from config.settings import Settings

                settings = Settings()
                prompts_path = settings.prompts_path

            with open(prompts_path, encoding="utf-8") as f:
                self._prompts = yaml.safe_load(f)
            logger.info(f"Loaded prompts from {prompts_path}")
        except Exception as e:
            logger.error(f"Failed to load prompts from {prompts_path}: {e}")
            self._prompts = None

    def get_statistics(self) -> dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with statistics
        """
        avg_time = (
            self._total_synthesis_time / self._total_syntheses
            if self._total_syntheses > 0
            else 0
        )

        return {
            "total_syntheses": self._total_syntheses,
            "total_tokens_used": self._total_tokens_used,
            "average_synthesis_time": avg_time,
            "error_count": self._error_count,
        }
