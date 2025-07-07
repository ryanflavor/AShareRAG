"""
Integration tests for the offline data processing pipeline.
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

from src.pipeline import run_offline_pipeline


class TestOfflinePipeline:
    """Test suite for the offline pipeline orchestration."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_corpus(self, temp_data_dir):
        """Create a sample corpus file for testing."""
        corpus_path = Path(temp_data_dir) / "corpus.json"
        sample_data = [
            {"title": "中国平安公司信息", "text": "中国平安是一家领先的金融服务集团。", "idx": 0},
            {"title": "腾讯控股信息", "text": "腾讯控股在深圳成立，是科技行业的巨头。", "idx": 1},
        ]
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        return str(corpus_path)

    def test_complete_pipeline_execution(self, sample_corpus, temp_data_dir):
        """Test end-to-end pipeline execution with sample data."""
        # Configure output directory
        output_dir = Path(temp_data_dir) / "output"

        # Run the pipeline
        run_offline_pipeline(corpus_path=sample_corpus, output_dir=str(output_dir))

        # Verify outputs
        assert output_dir.exists()
        assert (output_dir / "graph").exists()
        assert (output_dir / "vector_store").exists()
        assert (output_dir / "graph" / "graph.pkl").exists()
        assert (output_dir / "graph" / "entity_metadata.json").exists()
        assert (output_dir / "graph" / "relation_metadata.json").exists()

    def test_pipeline_component_initialization(self):
        """Test that pipeline properly initializes all components."""
        with (
            patch("src.pipeline.offline_pipeline.DataIngestor") as mock_ingestor,
            patch(
                "src.pipeline.offline_pipeline.KnowledgeGraphConstructor"
            ) as mock_graph_constructor,
            patch("src.pipeline.offline_pipeline.VectorIndexer") as mock_vector_indexer,
            patch("src.pipeline.offline_pipeline.DeepSeekAdapter") as mock_llm_adapter,
            patch(
                "src.pipeline.offline_pipeline.EmbeddingService"
            ) as mock_embedding_service,
            patch("src.pipeline.offline_pipeline.VectorStorage") as mock_vector_storage,
            patch("src.pipeline.offline_pipeline.Settings") as mock_settings,
            patch("builtins.open", mock_open()),
            patch("src.pipeline.offline_pipeline.pickle.dump") as mock_pickle_dump,
        ):
            # Create a simple mock graph that can be pickled
            mock_graph = type("MockGraph", (), {})()

            # Mock return values
            mock_settings.return_value.deepseek_model = "deepseek-chat"
            mock_ingestor.return_value.load_and_preprocess_documents.return_value = []
            mock_graph_constructor.return_value.process_documents.return_value = (
                {},
                mock_graph,
            )

            # Run pipeline
            run_offline_pipeline()

            # Verify components were initialized
            mock_llm_adapter.assert_called_once()
            mock_embedding_service.assert_called_once()
            mock_vector_storage.assert_called_once()
            mock_ingestor.assert_called_once()
            mock_graph_constructor.assert_called_once_with(
                llm_adapter=mock_llm_adapter.return_value
            )
            mock_vector_indexer.assert_called_once_with(
                embedding_service=mock_embedding_service.return_value,
                vector_storage=mock_vector_storage.return_value,
            )

    def test_pipeline_component_sequence(self):
        """Test that pipeline calls components in correct order."""
        with (
            patch("src.pipeline.offline_pipeline.DataIngestor") as mock_ingestor,
            patch(
                "src.pipeline.offline_pipeline.KnowledgeGraphConstructor"
            ) as mock_graph_constructor,
            patch("src.pipeline.offline_pipeline.VectorIndexer") as mock_vector_indexer,
            patch("src.pipeline.offline_pipeline.DeepSeekAdapter"),
            patch("src.pipeline.offline_pipeline.EmbeddingService"),
            patch("src.pipeline.offline_pipeline.VectorStorage"),
            patch("src.pipeline.offline_pipeline.Settings"),
            patch("builtins.open", mock_open()),
            patch("src.pipeline.offline_pipeline.pickle.dump"),
        ):
            # Setup mock returns with Document objects (as returned by DataIngestor)
            from src.components.data_ingestor import Document
            mock_document_objs = [Document(title="Test Doc", text="test content", idx=0)]
            mock_converted_docs = [{"id": "doc_0", "text": "test content", "title": "Test Doc", "idx": 0}]
            mock_ner_re_results = {"doc_0": {"entities": [], "triples": []}}
            mock_graph = type("MockGraph", (), {})()

            mock_ingestor.return_value.load_corpus.return_value = mock_document_objs
            mock_graph_constructor.return_value.process_documents.return_value = (
                mock_ner_re_results,
                mock_graph,
            )

            # Run pipeline
            run_offline_pipeline()

            # Verify call sequence
            mock_ingestor.return_value.load_corpus.assert_called_once()
            mock_graph_constructor.return_value.process_documents.assert_called_once_with(
                mock_converted_docs
            )
            mock_vector_indexer.return_value.index_documents.assert_called_once_with(
                mock_converted_docs, mock_ner_re_results
            )

    def test_pipeline_error_handling(self):
        """Test pipeline handles component failures gracefully."""
        with patch("src.pipeline.offline_pipeline.DataIngestor") as mock_ingestor:
            # Simulate component failure
            mock_ingestor.side_effect = Exception("Component initialization failed")

            # Pipeline should raise with clear error message
            with pytest.raises(Exception) as exc_info:
                run_offline_pipeline()

            assert "Component initialization failed" in str(exc_info.value)

    def test_pipeline_with_invalid_corpus_path(self):
        """Test pipeline handling of invalid corpus path."""
        with pytest.raises(FileNotFoundError):
            run_offline_pipeline(corpus_path="non_existent_file.json")

    def test_pipeline_checkpoint_recovery(self, sample_corpus, temp_data_dir):
        """Test pipeline can recover from checkpoints after failure."""
        output_dir = Path(temp_data_dir) / "output"

        # First run - simulate failure after graph construction
        with patch(
            "src.pipeline.offline_pipeline.VectorIndexer"
        ) as mock_vector_indexer:
            mock_vector_indexer.return_value.index_documents.side_effect = Exception(
                "Vector indexing failed"
            )

            with pytest.raises(Exception):
                run_offline_pipeline(
                    corpus_path=sample_corpus, output_dir=str(output_dir)
                )

        # Verify checkpoint exists
        assert (output_dir / "graph").exists()

        # Second run - should resume from checkpoint
        run_offline_pipeline(
            corpus_path=sample_corpus,
            output_dir=str(output_dir),
            resume_from_checkpoint=True,
        )

        # Verify complete output
        assert (output_dir / "vector_store").exists()

    @pytest.mark.parametrize(
        "missing_component",
        ["DataIngestor", "KnowledgeGraphConstructor", "VectorIndexer"],
    )
    def test_pipeline_missing_component_import(self, missing_component):
        """Test pipeline handles missing component imports gracefully."""
        with patch(
            f"src.pipeline.offline_pipeline.{missing_component}",
            side_effect=ImportError(f"Cannot import {missing_component}"),
        ):
            with pytest.raises(ImportError) as exc_info:
                run_offline_pipeline()

            assert missing_component in str(exc_info.value)
