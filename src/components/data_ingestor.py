import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Represents a document from the corpus."""

    title: str
    text: str
    idx: int


class DataIngestor:
    """Handles loading and processing of corpus data."""

    def load_corpus(self, corpus_path: str | Path) -> list[Document]:
        """
        Load corpus data from a JSON file.

        Args:
            corpus_path: Path to the corpus.json file

        Returns:
            List of Document objects

        Raises:
            FileNotFoundError: If the corpus file doesn't exist
            json.JSONDecodeError: If the JSON is malformed
            ValueError: If required fields are missing or have invalid types
        """
        corpus_path = Path(corpus_path)

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        try:
            with open(corpus_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse JSON from {corpus_path}", e.doc, e.pos
            ) from e

        documents = []
        for i, item in enumerate(data):
            # Validate required fields
            if not isinstance(item, dict):
                raise ValueError(f"Document at index {i} is not a dictionary")

            required_fields = {"title", "text", "idx"}
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(
                    f"Document at index {i} is missing required field(s): "
                    f"{missing_fields}"
                )

            # Validate field types
            if not isinstance(item["title"], str):
                raise ValueError(
                    f"Document at index {i}: 'title' must be a string, "
                    f"got {type(item['title']).__name__}"
                )

            if not isinstance(item["text"], str):
                raise ValueError(
                    f"Document at index {i}: 'text' must be a string, "
                    f"got {type(item['text']).__name__}"
                )

            if not isinstance(item["idx"], int):
                raise ValueError(
                    f"Document at index {i}: 'idx' must be an integer, "
                    f"got {type(item['idx']).__name__}. "
                    f"Invalid type for 'idx' field."
                )

            # Create Document object
            # Note: We don't split the text - entire text field = one chunk
            documents.append(
                Document(title=item["title"], text=item["text"], idx=item["idx"])
            )

        return documents
