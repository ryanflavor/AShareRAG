"""Unit tests for Vector Storage component."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import shutil

from src.components.vector_storage import VectorStorage


class TestVectorStorage:
    """Test suite for VectorStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_lancedb(self):
        """Create mock LanceDB components."""
        with patch("src.components.vector_storage.lancedb") as mock_lancedb:
            # Mock database and table
            mock_db = Mock()
            mock_table = Mock()

            # Configure mock behavior
            mock_lancedb.connect.return_value = mock_db
            mock_db.create_table.return_value = mock_table
            mock_db.open_table.return_value = mock_table
            mock_db.table_names.return_value = []

            # Mock table operations
            mock_table.add.return_value = None
            mock_table.search.return_value = Mock(
                limit=Mock(
                    return_value=Mock(
                        to_pandas=Mock(
                            return_value=Mock(
                                to_dict=Mock(return_value={"records": []})
                            )
                        )
                    )
                )
            )

            yield mock_lancedb, mock_db, mock_table

    def test_init(self, temp_dir):
        """Test VectorStorage initialization."""
        storage = VectorStorage(
            db_path=temp_dir, table_name="test_table", embedding_dim=1024
        )

        assert storage.db_path == temp_dir
        assert storage.table_name == "test_table"
        assert storage.embedding_dim == 1024
        assert storage.db is None
        assert storage.table is None

    def test_connect_creates_directory(self, temp_dir, mock_lancedb):
        """Test that connect creates the database directory if it doesn't exist."""
        mock_lancedb_obj, mock_db, _ = mock_lancedb
        db_path = temp_dir / "new_dir"

        storage = VectorStorage(db_path=db_path)
        storage.connect()

        assert db_path.exists()
        mock_lancedb_obj.connect.assert_called_once_with(str(db_path))

    def test_create_table_new(self, temp_dir, mock_lancedb):
        """Test creating a new table."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()

        # Test data with proper schema
        test_data = [
            {
                "id": "doc1_0",
                "text": "Test document",
                "vector": np.random.rand(2560).astype(np.float32),
                "company_name": "Test Company",
                "doc_id": "doc1",
                "chunk_index": 0,
                "entities": [{"text": "Test Company", "type": "COMPANY"}],
                "relations": [["Test Company", "located_in", "Beijing"]],
                "relations_count": 1,
                "source_file": "corpus.json",
                "processing_timestamp": "2025-01-06T10:30:00Z",
            }
        ]

        storage.create_table(test_data)

        # Verify table was created
        mock_db.create_table.assert_called_once()
        assert storage.table == mock_table

    def test_create_table_existing(self, temp_dir, mock_lancedb):
        """Test handling existing table."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb
        mock_db.table_names.return_value = ["ashare_documents"]

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()

        test_data = [
            {
                "id": "doc1_0",
                "text": "Test",
                "vector": np.random.rand(2560).astype(np.float32),
                "company_name": "Test",
                "doc_id": "doc1",
                "chunk_index": 0,
            }
        ]

        storage.create_table(test_data)

        # Should open existing table, not create new one
        mock_db.open_table.assert_called_once_with("ashare_documents")
        mock_db.create_table.assert_not_called()

    def test_add_documents(self, temp_dir, mock_lancedb):
        """Test adding documents to the table."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()
        storage.table = mock_table  # Simulate table already created

        documents = [
            {
                "id": "doc1_0",
                "text": "Document 1",
                "vector": np.random.rand(2560).astype(np.float32),
                "company_name": "Company A",
                "doc_id": "doc1",
                "chunk_index": 0,
                "entities": [{"text": "Company A", "type": "COMPANY"}],
                "relations": [],
                "relations_count": 0,
                "source_file": "corpus.json",
                "processing_timestamp": "2025-01-06T10:30:00Z",
            },
            {
                "id": "doc2_0",
                "text": "Document 2",
                "vector": np.random.rand(2560).astype(np.float32),
                "company_name": "Company B",
                "doc_id": "doc2",
                "chunk_index": 0,
                "entities": [],
                "relations": [],
                "relations_count": 0,
                "source_file": "corpus.json",
                "processing_timestamp": "2025-01-06T10:30:00Z",
            },
        ]

        storage.add_documents(documents)

        # Verify documents were added
        mock_table.add.assert_called_once()
        # Check that the data was properly formatted
        call_args = mock_table.add.call_args[0][0]
        assert len(call_args) == 2
        assert all("vector" in doc for doc in call_args)

    def test_add_documents_no_table(self, temp_dir):
        """Test error handling when adding documents without a table."""
        storage = VectorStorage(db_path=temp_dir)

        with pytest.raises(ValueError, match="Table not initialized"):
            storage.add_documents([{"id": "test"}])

    def test_search_similarity(self, temp_dir, mock_lancedb):
        """Test similarity search functionality."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        # Create mock DataFrame-like object
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "id": ["doc1_0", "doc2_0"],
                "text": ["Result 1", "Result 2"],
                "company_name": ["Company A", "Company B"],
                "doc_id": ["doc1", "doc2"],
                "chunk_index": [0, 0],
                "entities": [
                    [{"text": "Company A", "type": "COMPANY"}],
                    [{"text": "Company B", "type": "COMPANY"}],
                ],
                "relations": [[], []],
                "relations_count": [0, 0],
                "_distance": [0.1, 0.2],
            }
        )

        mock_search_chain = Mock()
        mock_table.search.return_value = mock_search_chain
        mock_search_chain.limit.return_value = Mock(
            to_pandas=Mock(return_value=mock_df)
        )

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()
        storage.table = mock_table

        # Perform search
        query_vector = np.random.rand(2560).astype(np.float32)
        results = storage.search(query_vector, top_k=2)

        # Verify search was called correctly
        mock_table.search.assert_called_once()
        mock_search_chain.limit.assert_called_once_with(2)

        # Verify results
        assert len(results) == 2
        assert results[0]["id"] == "doc1_0"
        assert results[0]["score"] == pytest.approx(0.9, rel=1e-3)  # 1 - 0.1
        assert results[1]["id"] == "doc2_0"
        assert results[1]["score"] == pytest.approx(0.8, rel=1e-3)  # 1 - 0.2

    def test_search_with_filter(self, temp_dir, mock_lancedb):
        """Test search with company filter."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        # Configure mock for filtered search
        import pandas as pd

        mock_df = pd.DataFrame(
            {"id": [], "text": [], "company_name": [], "_distance": []}
        )

        mock_search_chain = Mock()
        mock_where_chain = Mock()
        mock_table.search.return_value = mock_search_chain
        mock_search_chain.where.return_value = mock_where_chain
        mock_where_chain.limit.return_value = Mock(to_pandas=Mock(return_value=mock_df))

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()
        storage.table = mock_table

        # Perform filtered search
        query_vector = np.random.rand(2560).astype(np.float32)
        results = storage.search(query_vector, top_k=5, filter_company="Company A")

        # Verify filter was applied
        mock_search_chain.where.assert_called_once_with("company_name = 'Company A'")

    def test_search_no_table(self, temp_dir):
        """Test error handling when searching without a table."""
        storage = VectorStorage(db_path=temp_dir)

        with pytest.raises(ValueError, match="Table not initialized"):
            storage.search(np.random.rand(2560))

    def test_get_table_info(self, temp_dir, mock_lancedb):
        """Test getting table information."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        # Configure mock table info
        mock_table.count = Mock(return_value=1000)
        mock_table.__len__ = Mock(return_value=1000)

        # Create mock schema fields with properly configured name attribute
        mock_field_id = Mock()
        mock_field_id.name = "id"
        mock_field_text = Mock()
        mock_field_text.name = "text"
        mock_field_vector = Mock()
        mock_field_vector.name = "vector"
        mock_field_company = Mock()
        mock_field_company.name = "company_name"

        mock_table.schema = [
            mock_field_id,
            mock_field_text,
            mock_field_vector,
            mock_field_company,
        ]

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()
        storage.table = mock_table

        info = storage.get_table_info()

        assert info["table_name"] == "ashare_documents"
        assert info["num_rows"] == 1000
        assert info["embedding_dim"] == 2560
        assert info["schema"] == ["id", "text", "vector", "company_name"]

    def test_typed_entity_handling(self, temp_dir, mock_lancedb):
        """Test handling of typed entities from Story 1.2.1."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()
        storage.table = mock_table

        # Document with typed entities
        documents = [
            {
                "id": "doc1_0",
                "text": "Test document",
                "vector": np.random.rand(2560).astype(np.float32),
                "company_name": "Test Company",
                "doc_id": "doc1",
                "chunk_index": 0,
                "entities": [
                    {"text": "Test Company", "type": "COMPANY"},
                    {"text": "AI Product", "type": "PRODUCT"},
                    {"text": "Beijing", "type": "LOCATION"},
                ],
                "relations": [["Test Company", "develops", "AI Product"]],
                "relations_count": 1,
                "source_file": "corpus.json",
                "processing_timestamp": "2025-01-06T10:30:00Z",
            }
        ]

        storage.add_documents(documents)

        # Verify entities were properly stored
        call_args = mock_table.add.call_args[0][0]
        assert call_args[0]["entities"] == documents[0]["entities"]
        assert len(call_args[0]["entities"]) == 3
        assert all("type" in entity for entity in call_args[0]["entities"])

    def test_batch_add_documents(self, temp_dir, mock_lancedb):
        """Test adding documents in batches."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        storage = VectorStorage(db_path=temp_dir, batch_size=50)
        storage.connect()
        storage.table = mock_table

        # Create 150 documents to test batching
        documents = []
        for i in range(150):
            documents.append(
                {
                    "id": f"doc{i}_0",
                    "text": f"Document {i}",
                    "vector": np.random.rand(2560).astype(np.float32),
                    "company_name": f"Company {i}",
                    "doc_id": f"doc{i}",
                    "chunk_index": 0,
                    "entities": [],
                    "relations": [],
                    "relations_count": 0,
                    "source_file": "corpus.json",
                    "processing_timestamp": "2025-01-06T10:30:00Z",
                }
            )

        storage.add_documents(documents)

        # Should be called 3 times (50 + 50 + 50)
        assert mock_table.add.call_count == 3

    def test_error_handling_and_retry(self, temp_dir, mock_lancedb):
        """Test error handling and retry logic."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        # Simulate connection error then success
        mock_lancedb_obj.connect.side_effect = [Exception("Connection failed"), mock_db]

        storage = VectorStorage(db_path=temp_dir)

        # First attempt should fail, second should succeed
        with pytest.raises(Exception):
            storage.connect()

        storage.connect()  # Should succeed on retry
        assert storage.db == mock_db

    def test_close_connection(self, temp_dir, mock_lancedb):
        """Test closing database connection."""
        mock_lancedb_obj, mock_db, mock_table = mock_lancedb

        storage = VectorStorage(db_path=temp_dir)
        storage.connect()
        storage.table = mock_table

        storage.close()

        assert storage.db is None
        assert storage.table is None
