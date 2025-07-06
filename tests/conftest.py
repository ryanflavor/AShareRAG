import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def test_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env") as f:
        f.write("DEEPSEEK_API_KEY=test_deepseek_key\n")
        f.write("EMBEDDING_MODEL_NAME=test/embedding-model\n")
        f.write("RERANKER_MODEL_NAME=test/reranker-model\n")
        f.write("LLM_MODEL_NAME=test-llm\n")
        f.write("BATCH_SIZE=16\n")
        f.write("MAX_WORKERS=2\n")
        f.write("LOG_LEVEL=DEBUG\n")
        f.write("GRAPH_STORAGE_PATH=test/graph\n")
        f.write("VECTOR_STORAGE_PATH=test/vector\n")
        f.write("API_HOST=127.0.0.1\n")
        f.write("API_PORT=8001\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def sample_corpus_data():
    """Sample corpus data for testing."""
    return [
        {
            "title": "公司A",
            "text": "# 公司A全称\n\n## 公司简称\n公司A\n\n## 主营业务\n电子产品制造",
            "idx": 0,
        },
        {
            "title": "公司B",
            "text": "# 公司B全称\n\n## 公司简称\n公司B\n\n## 主营业务\n软件开发",
            "idx": 1,
        },
    ]


@pytest.fixture
def temp_corpus_file(sample_corpus_data):
    """Create a temporary corpus.json file for testing."""
    import json

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(sample_corpus_data, f, ensure_ascii=False, indent=2)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)
