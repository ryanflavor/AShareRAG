import json
from pathlib import Path

import pytest

from src.components.data_ingestor import DataIngestor, Document


class TestDataIngestor:
    """Test the data ingestion component."""

    def test_load_valid_corpus(self, temp_corpus_file):
        """Test loading a valid corpus.json file."""
        ingestor = DataIngestor()
        documents = ingestor.load_corpus(temp_corpus_file)

        assert len(documents) == 2
        assert isinstance(documents[0], Document)
        assert isinstance(documents[1], Document)

        # Check first document
        assert documents[0].title == "公司A"
        assert "公司A全称" in documents[0].text
        assert "电子产品制造" in documents[0].text
        assert documents[0].idx == 0

        # Check second document
        assert documents[1].title == "公司B"
        assert "公司B全称" in documents[1].text
        assert "软件开发" in documents[1].text
        assert documents[1].idx == 1

    def test_text_not_split(self, temp_corpus_file):
        """Test that document text is not split into chunks."""
        ingestor = DataIngestor()
        documents = ingestor.load_corpus(temp_corpus_file)

        # Each document should have the complete text as a single chunk
        for doc in documents:
            # The text should contain all the original content
            assert "# " in doc.text  # Contains markdown headers
            assert "\n\n" in doc.text  # Contains paragraph breaks
            assert "## " in doc.text  # Contains subheaders

    def test_load_missing_file(self):
        """Test error handling for missing file."""
        ingestor = DataIngestor()

        with pytest.raises(FileNotFoundError):
            ingestor.load_corpus("nonexistent_file.json")

    def test_load_malformed_json(self, tmp_path):
        """Test error handling for malformed JSON."""
        # Create a malformed JSON file
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{invalid json content")

        ingestor = DataIngestor()

        with pytest.raises(json.JSONDecodeError):
            ingestor.load_corpus(str(malformed_file))

    def test_missing_required_fields(self, tmp_path):
        """Test error handling for documents missing required fields."""
        # Create JSON with missing fields
        invalid_data = [
            {"title": "Test Company"},  # Missing text and idx
            {"text": "Some text", "idx": 0},  # Missing title
        ]

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text(json.dumps(invalid_data, ensure_ascii=False))

        ingestor = DataIngestor()

        with pytest.raises(ValueError) as exc_info:
            ingestor.load_corpus(str(invalid_file))

        assert "missing required field" in str(exc_info.value).lower()

    def test_invalid_field_types(self, tmp_path):
        """Test error handling for invalid field types."""
        # Create JSON with invalid field types
        invalid_data = [
            {"title": "Test Company", "text": "Some text", "idx": "not_a_number"}
        ]

        invalid_file = tmp_path / "invalid_types.json"
        invalid_file.write_text(json.dumps(invalid_data, ensure_ascii=False))

        ingestor = DataIngestor()

        with pytest.raises(ValueError) as exc_info:
            ingestor.load_corpus(str(invalid_file))

        assert "invalid type" in str(exc_info.value).lower()

    def test_load_empty_corpus(self, tmp_path):
        """Test loading an empty corpus file."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]")

        ingestor = DataIngestor()
        documents = ingestor.load_corpus(str(empty_file))

        assert documents == []

    def test_document_dataclass(self):
        """Test the Document dataclass."""
        doc = Document(title="Test Company", text="Test content", idx=42)

        assert doc.title == "Test Company"
        assert doc.text == "Test content"
        assert doc.idx == 42

        # Test immutability (dataclasses are mutable by default,
        # but we should make them frozen)
        with pytest.raises(AttributeError):
            doc.title = "New Title"

    def test_load_from_path_object(self, temp_corpus_file):
        """Test that Path objects are accepted as input."""
        ingestor = DataIngestor()
        documents = ingestor.load_corpus(Path(temp_corpus_file))

        assert len(documents) == 2

    def test_preserve_unicode(self, tmp_path):
        """Test that Chinese characters are properly preserved."""
        unicode_data = [
            {
                "title": "中文公司名称",
                "text": (
                    "# 测试公司\n\n这是一段包含中文字符的文本。\n\n"
                    "## 业务范围\n人工智能、机器学习"
                ),
                "idx": 0,
            }
        ]

        unicode_file = tmp_path / "unicode.json"
        unicode_file.write_text(
            json.dumps(unicode_data, ensure_ascii=False), encoding="utf-8"
        )

        ingestor = DataIngestor()
        documents = ingestor.load_corpus(str(unicode_file))

        assert documents[0].title == "中文公司名称"
        assert "测试公司" in documents[0].text
        assert "人工智能、机器学习" in documents[0].text
