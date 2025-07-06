"""Vector Storage component using LanceDB."""

import logging
from pathlib import Path
from typing import Any

import lancedb
import numpy as np


class VectorStorage:
    """Vector storage using LanceDB for efficient similarity search."""

    def __init__(
        self,
        db_path: Path = Path("./output/vector_store"),
        table_name: str = "ashare_documents",
        embedding_dim: int = 2560,
        batch_size: int = 100,
    ):
        """
        Initialize vector storage.

        Args:
            db_path: Path to LanceDB database
            table_name: Name of the table
            embedding_dim: Dimension of embeddings
            batch_size: Batch size for adding documents
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.db = None
        self.table = None

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def connect(self):
        """Connect to LanceDB database."""
        try:
            # Create directory if it doesn't exist
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.db = lancedb.connect(str(self.db_path))
            self.logger.info(f"Connected to LanceDB at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to LanceDB: {e}")
            raise

    def create_table(self, initial_data: list[dict[str, Any]]):
        """
        Create or open table with schema based on initial data.

        Args:
            initial_data: Initial data to define schema
        """
        if self.db is None:
            raise ValueError("Database not connected. Call connect() first.")

        # Check if table exists
        existing_tables = self.db.table_names()

        if self.table_name in existing_tables:
            self.logger.info(f"Opening existing table: {self.table_name}")
            self.table = self.db.open_table(self.table_name)
        else:
            self.logger.info(f"Creating new table: {self.table_name}")
            # Create table with initial data
            self.table = self.db.create_table(
                self.table_name, data=initial_data, mode="create"
            )

    def add_documents(self, documents: list[dict[str, Any]]):
        """
        Add documents with embeddings to the table.

        Args:
            documents: List of documents with embeddings and metadata
        """
        if self.table is None:
            raise ValueError("Table not initialized. Call create_table() first.")

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]

            try:
                # Add batch to table
                self.table.add(batch)
                self.logger.info(
                    f"Added batch of {len(batch)} documents (total: {i + len(batch)}/{len(documents)})"
                )
            except Exception as e:
                self.logger.error(f"Failed to add batch starting at index {i}: {e}")
                raise

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_company: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_company: Optional company name filter

        Returns:
            List of search results with metadata
        """
        if self.table is None:
            raise ValueError("Table not initialized. Call create_table() first.")

        try:
            # Build search query
            search_query = self.table.search(query_vector)

            # Apply filter if specified
            if filter_company:
                search_query = search_query.where(f"company_name = '{filter_company}'")

            # Execute search
            results = search_query.limit(top_k).to_pandas()

            # Convert to list of dicts with scores
            search_results = []
            for _, row in results.iterrows():
                result = row.to_dict()
                # Convert distance to similarity score (1 - distance)
                if "_distance" in result:
                    result["score"] = 1.0 - result["_distance"]
                    del result["_distance"]
                search_results.append(result)

            return search_results

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []

    def get_table_info(self) -> dict[str, Any]:
        """
        Get information about the table.

        Returns:
            Dictionary with table information
        """
        if self.table is None:
            raise ValueError("Table not initialized.")

        return {
            "table_name": self.table_name,
            "num_rows": len(self.table),
            "embedding_dim": self.embedding_dim,
            "schema": [field.name for field in self.table.schema],
        }

    def close(self):
        """Close database connection."""
        self.db = None
        self.table = None
        self.logger.info("Closed LanceDB connection")
