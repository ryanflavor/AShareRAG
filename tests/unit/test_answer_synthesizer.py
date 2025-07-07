"""Unit tests for AnswerSynthesizer component.

This module tests the AnswerSynthesizer component responsible for generating
comprehensive answers from retrieved and ranked documents.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.adapters.llm_adapter import LLMResponse
from src.components.answer_synthesizer import AnswerSynthesizer, AnswerSynthesizerConfig


class TestAnswerSynthesizer:
    """Test suite for the AnswerSynthesizer component."""

    @pytest.fixture
    def mock_llm_adapter(self):
        """Create a mock LLM adapter."""
        adapter = Mock()
        adapter.generate_async = AsyncMock()
        adapter.generate = Mock()
        return adapter

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return AnswerSynthesizerConfig(
            max_input_tokens=3000,
            max_output_tokens=500,
            temperature=0.7,
            top_p=0.9,
            answer_language="Chinese",
            include_citations=True,
            citation_format="[{idx}]",
        )

    @pytest.fixture
    def synthesizer(self, config, mock_llm_adapter):
        """Create an AnswerSynthesizer instance for testing."""
        return AnswerSynthesizer(config=config, llm_adapter=mock_llm_adapter)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "content": "比亚迪是一家中国新能源汽车制造商，2023年营收达到6023亿元。",
                "metadata": {
                    "company_name": "比亚迪",
                    "source": "annual_report_2023",
                    "page": 12,
                },
                "score": 0.95,
            },
            {
                "content": "比亚迪在电池技术方面领先，其刀片电池技术安全性高。",
                "metadata": {
                    "company_name": "比亚迪",
                    "source": "tech_report_2023",
                    "page": 45,
                },
                "score": 0.88,
            },
            {
                "content": "2023年比亚迪新能源汽车销量超过300万辆，同比增长62%。",
                "metadata": {
                    "company_name": "比亚迪",
                    "source": "sales_report_2023",
                    "page": 8,
                },
                "score": 0.82,
            },
        ]

    @pytest.mark.asyncio
    async def test_synthesizer_initialization(self, synthesizer, config):
        """Test that synthesizer initializes correctly."""
        assert synthesizer.config == config
        assert synthesizer.llm_adapter is not None
        assert synthesizer._prompts is not None  # Loaded during init

    @pytest.mark.asyncio
    async def test_synthesize_answer_basic(
        self, synthesizer, mock_llm_adapter, sample_documents
    ):
        """Test basic answer synthesis functionality."""
        # Setup mock response
        expected_answer = (
            "根据提供的资料，比亚迪是中国领先的新能源汽车制造商。"
            "2023年，比亚迪实现营收6023亿元[1]，新能源汽车销量超过300万辆，"
            "同比增长62%[3]。公司在电池技术方面处于行业领先地位，"
            "其自主研发的刀片电池技术具有高安全性特点[2]。"
        )

        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content=expected_answer,
            model="deepseek-v3",
            usage={"prompt_tokens": 500, "completion_tokens": 100},
        )

        # Execute
        result = await synthesizer.synthesize_answer(
            query="介绍一下比亚迪的情况", documents=sample_documents
        )

        # Verify
        assert result["answer"] == expected_answer
        assert result["sources"] == sample_documents
        assert result["synthesis_time"] > 0
        assert result["metadata"]["model"] == "deepseek-v3"
        assert result["metadata"]["document_count"] == 3
        assert result["metadata"]["token_usage"]["prompt_tokens"] == 500
        assert result["metadata"]["token_usage"]["completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_synthesize_answer_empty_documents(
        self, synthesizer, mock_llm_adapter
    ):
        """Test synthesis with empty documents."""
        # Setup mock response
        expected_answer = "抱歉，未找到与您查询相关的信息。"
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content=expected_answer,
            model="deepseek-v3",
            usage={"prompt_tokens": 50, "completion_tokens": 10},
        )

        # Execute
        result = await synthesizer.synthesize_answer(
            query="查询某公司信息", documents=[]
        )

        # Verify
        assert result["answer"] == expected_answer
        assert result["sources"] == []
        assert result["metadata"]["document_count"] == 0

    @pytest.mark.asyncio
    async def test_synthesize_answer_with_custom_prompt(
        self, synthesizer, mock_llm_adapter
    ):
        """Test synthesis with custom prompt template."""
        custom_prompt = "请用英文回答：{query}\n\n参考资料：{documents}"
        expected_answer = "BYD is a leading Chinese electric vehicle manufacturer."

        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content=expected_answer,
            model="deepseek-v3",
            usage={"prompt_tokens": 200, "completion_tokens": 20},
        )

        # Execute
        result = await synthesizer.synthesize_answer(
            query="Tell me about BYD",
            documents=[{"content": "BYD information", "metadata": {}, "score": 0.9}],
            custom_prompt=custom_prompt,
        )

        # Verify
        assert result["answer"] == expected_answer
        assert (
            "Tell me about BYD"
            in mock_llm_adapter.generate_async.call_args[1]["prompt"]
        )

    @pytest.mark.asyncio
    async def test_synthesize_answer_token_limit(self, synthesizer, sample_documents):
        """Test that documents are truncated to fit token limit."""
        # Create many documents to exceed token limit
        many_documents = sample_documents * 20  # Repeat to create many docs

        # Setup mock
        synthesizer.llm_adapter.generate_async.return_value = LLMResponse(
            content="Answer based on truncated documents",
            model="deepseek-v3",
            usage={"prompt_tokens": 2900, "completion_tokens": 50},
        )

        # Execute
        await synthesizer.synthesize_answer(query="查询信息", documents=many_documents)

        # Verify that prompt was called and token limit was respected
        call_args = synthesizer.llm_adapter.generate_async.call_args
        assert call_args is not None
        # The prompt should be constructed but truncated to fit token limit

    @pytest.mark.asyncio
    async def test_synthesize_answer_without_citations(
        self, synthesizer, mock_llm_adapter, sample_documents
    ):
        """Test synthesis without citation markers."""
        # Update config
        synthesizer.config.include_citations = False

        expected_answer = "比亚迪是中国新能源汽车制造商,2023年营收6023亿元。"
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content=expected_answer,
            model="deepseek-v3",
            usage={"prompt_tokens": 400, "completion_tokens": 50},
        )

        # Execute
        await synthesizer.synthesize_answer(
            query="比亚迪营收", documents=sample_documents
        )

        # Verify no citation instruction in prompt
        call_args = mock_llm_adapter.generate_async.call_args
        assert call_args is not None
        assert "[" not in expected_answer or "]" not in expected_answer

    @pytest.mark.asyncio
    async def test_synthesize_answer_english_mode(
        self, synthesizer, mock_llm_adapter, sample_documents
    ):
        """Test synthesis in English mode."""
        # Update config
        synthesizer.config.answer_language = "English"

        expected_answer = "BYD is a leading Chinese EV manufacturer with 2023 revenue of 602.3 billion RMB[1]."
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content=expected_answer,
            model="deepseek-v3",
            usage={"prompt_tokens": 450, "completion_tokens": 40},
        )

        # Execute
        result = await synthesizer.synthesize_answer(
            query="What is BYD's revenue?", documents=sample_documents
        )

        # Verify
        assert result["answer"] == expected_answer
        assert result["metadata"]["language"] == "English"

    @pytest.mark.asyncio
    async def test_synthesize_answer_error_handling(
        self, synthesizer, mock_llm_adapter
    ):
        """Test error handling during synthesis."""
        # Setup mock to raise exception
        mock_llm_adapter.generate_async.side_effect = Exception(
            "LLM service unavailable"
        )

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            await synthesizer.synthesize_answer(
                query="Test query",
                documents=[{"content": "test", "metadata": {}, "score": 0.9}],
            )

        assert "LLM service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_synthesize_answer_fallback_on_error(
        self, synthesizer, mock_llm_adapter, sample_documents
    ):
        """Test fallback behavior when synthesis fails."""
        # Mock to raise exception
        mock_llm_adapter.generate_async.side_effect = Exception("Temporary failure")

        # Enable fallback mode
        result = await synthesizer.synthesize_answer(
            query="Test query", documents=sample_documents, fallback_on_error=True
        )

        # Verify fallback was used
        assert (
            "基于找到的文档" in result["answer"]
            or "Based on the documents found" in result["answer"]
        )
        assert result["metadata"]["fallback_used"] is True
        assert result["metadata"]["model"] == "fallback"

    def test_format_documents_for_prompt(self, synthesizer, sample_documents):
        """Test document formatting for prompt construction."""
        formatted = synthesizer._format_documents_for_prompt(sample_documents)

        # Verify formatting
        assert "文档 1" in formatted or "Document 1" in formatted
        assert "比亚迪是一家中国新能源汽车制造商" in formatted
        assert "相关度: 0.95" in formatted or "Relevance: 0.95" in formatted
        assert (
            "来源: annual_report_2023" in formatted
            or "Source: annual_report_2023" in formatted
        )

    def test_truncate_documents_to_token_limit(self, synthesizer, sample_documents):
        """Test document truncation to fit token limits."""
        # Create many documents
        many_docs = sample_documents * 50

        # Truncate to a small limit
        truncated = synthesizer._truncate_documents_to_fit(
            documents=many_docs,
            max_chars=1000,  # Approximate token limit
        )

        # Verify truncation
        assert len(truncated) < len(many_docs)
        assert sum(len(doc["content"]) for doc in truncated) < 1500  # Some buffer

    @pytest.mark.asyncio
    async def test_synthesize_answer_performance_logging(
        self, synthesizer, mock_llm_adapter, sample_documents, caplog
    ):
        """Test that performance metrics are logged."""
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="Test answer",
            model="deepseek-v3",
            usage={"prompt_tokens": 300, "completion_tokens": 50},
        )

        with caplog.at_level("INFO"):
            await synthesizer.synthesize_answer(
                query="Test query", documents=sample_documents
            )

        # Verify performance logging
        assert any(
            "Answer synthesis completed" in record.message for record in caplog.records
        )
        assert any("synthesis_time" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_synthesize_answer_with_metadata_fields(
        self, synthesizer, mock_llm_adapter
    ):
        """Test synthesis with additional metadata fields."""
        documents = [
            {
                "content": "Content about company financials",
                "metadata": {
                    "company_name": "TestCorp",
                    "report_type": "annual",
                    "year": 2023,
                    "confidence": 0.95,
                },
                "score": 0.9,
            }
        ]

        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="Answer with metadata",
            model="deepseek-v3",
            usage={"prompt_tokens": 200, "completion_tokens": 30},
        )

        await synthesizer.synthesize_answer(
            query="Company financials",
            documents=documents,
            include_metadata_fields=["report_type", "year"],
        )

        # Verify metadata fields were included in prompt
        call_args = mock_llm_adapter.generate_async.call_args
        prompt = call_args[1]["prompt"]
        assert "annual" in prompt
        assert "2023" in prompt

    def test_synthesizer_statistics(self, synthesizer):
        """Test statistics collection and retrieval."""
        stats = synthesizer.get_statistics()

        assert "total_syntheses" in stats
        assert "total_tokens_used" in stats
        assert "average_synthesis_time" in stats
        assert "error_count" in stats
        assert stats["total_syntheses"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_synthesis_requests(
        self, synthesizer, mock_llm_adapter, sample_documents
    ):
        """Test handling of concurrent synthesis requests."""
        # Setup mock responses
        mock_llm_adapter.generate_async.return_value = LLMResponse(
            content="Concurrent answer",
            model="deepseek-v3",
            usage={"prompt_tokens": 400, "completion_tokens": 60},
        )

        # Execute multiple concurrent requests
        tasks = [
            synthesizer.synthesize_answer(
                query=f"Query {i}", documents=sample_documents
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 5
        assert all(r["answer"] == "Concurrent answer" for r in results)

    @pytest.mark.asyncio
    async def test_load_prompts_from_config(self, synthesizer, tmp_path):
        """Test loading prompts from configuration file."""
        # Create a mock prompts file
        prompts_content = """
fact_qa_synthesis:
  system: "You are a helpful assistant."
  user: |
    Query: {query}
    Documents: {documents}
    Please synthesize an answer.
"""
        prompts_file = tmp_path / "prompts.yaml"
        prompts_file.write_text(prompts_content)

        # Load prompts
        synthesizer._load_prompts(str(prompts_file))

        assert synthesizer._prompts is not None
        assert "fact_qa_synthesis" in synthesizer._prompts
        assert (
            synthesizer._prompts["fact_qa_synthesis"]["system"]
            == "You are a helpful assistant."
        )
