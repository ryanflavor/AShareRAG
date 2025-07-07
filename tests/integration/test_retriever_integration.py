"""Integration tests for VectorRetriever with actual components."""

from pathlib import Path

import pytest

from config.settings import Settings
from src.components.embedding_service import EmbeddingService
from src.components.retriever import VectorRetriever
from src.components.vector_storage import VectorStorage


class TestVectorRetrieverIntegration:
    """Integration tests for VectorRetriever."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings()

    @pytest.fixture
    def vector_storage(self, settings):
        """Create vector storage instance."""
        # Try to use existing populated database first
        populated_db_path = Path("output/full_dataset/vector_store")
        if populated_db_path.exists():
            storage = VectorStorage(
                db_path=populated_db_path, table_name=settings.vector_table_name
            )
        else:
            storage = VectorStorage(
                db_path=settings.vector_db_path, table_name=settings.vector_table_name
            )
        storage.connect()
        # Open existing table
        try:
            storage.table = storage.db.open_table(settings.vector_table_name)
        except Exception:
            # Table doesn't exist, skip test
            pytest.skip("Vector database table not found. Run pipeline first.")
        yield storage
        storage.close()

    @pytest.fixture
    def embedding_service(self, settings):
        """Create embedding service instance."""
        service = EmbeddingService(model_name=settings.embedding_model_name)
        service.load_model()
        return service

    @pytest.fixture
    def retriever(self, vector_storage, embedding_service, settings):
        """Create retriever instance."""
        return VectorRetriever(
            vector_storage=vector_storage,
            embedding_service=embedding_service,
            top_k=settings.retriever_top_k,
        )

    @pytest.mark.integration
    def test_retriever_with_real_data(self, retriever):
        """Test retriever with actual vector database."""
        # Test query about a known company
        query = "贵州茅台的主营业务是什么？"

        results = retriever.retrieve(query)

        # Assertions
        assert isinstance(results, list)
        if results:  # If we have data in the database
            assert all("content" in r for r in results)
            assert all("company_name" in r for r in results)
            assert all("score" in r for r in results)
            # Check if results are relevant to Moutai
            moutai_results = [r for r in results if "茅台" in r.get("company_name", "")]
            assert len(moutai_results) > 0, "Should find Moutai-related documents"

    @pytest.mark.integration
    def test_retriever_with_company_filter(self, retriever):
        """Test retriever with company filtering."""
        query = "新能源汽车的发展情况"

        # Test without filter
        all_results = retriever.retrieve(query)

        # Test with BYD filter
        byd_results = retriever.retrieve(query, company_filter="比亚迪")

        if byd_results:
            # All results should be from BYD
            assert all(r.get("company_name") == "比亚迪" for r in byd_results)
            # Should have fewer or equal results than unfiltered
            assert len(byd_results) <= len(all_results)

    @pytest.mark.integration
    def test_retriever_performance(self, retriever):
        """Test retriever performance meets requirements."""
        import time

        queries = [
            "贵州茅台的财务状况如何？",
            "比亚迪在新能源领域的竞争优势",
            "宁德时代的电池技术",
        ]

        for query in queries:
            start_time = time.time()
            results = retriever.retrieve(query)
            elapsed_time = time.time() - start_time

            # Performance assertions
            assert elapsed_time < 3.0, (
                f"Retrieval should complete within 3s, took {elapsed_time:.2f}s"
            )
            print(
                f"Query '{query}' took {elapsed_time:.2f}s, returned {len(results)} results"
            )
