"""Unit tests for prompts configuration."""

from pathlib import Path

import yaml


def test_ner_prompt_exists():
    """Test that NER prompt configuration exists."""
    prompts_file = Path("config/prompts.yaml")
    assert prompts_file.exists(), "prompts.yaml file should exist"

    with open(prompts_file, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    assert "ner" in prompts, "NER prompt section should exist"
    assert "system" in prompts["ner"], "NER should have system prompt"
    assert "examples" in prompts["ner"], "NER should have examples"
    assert "template" in prompts["ner"], "NER should have template"


def test_ner_prompt_structure():
    """Test NER prompt has correct structure."""
    with open("config/prompts.yaml", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    ner_prompt = prompts["ner"]

    # Check system prompt
    assert isinstance(ner_prompt["system"], str)
    assert "命名实体" in ner_prompt["system"]
    assert "JSON" in ner_prompt["system"]

    # Check examples
    assert isinstance(ner_prompt["examples"], list)
    assert len(ner_prompt["examples"]) > 0

    example = ner_prompt["examples"][0]
    assert "user" in example
    assert "assistant" in example

    # Check template
    assert ner_prompt["template"] == "${passage}"


def test_ner_one_shot_example():
    """Test that one-shot example is properly structured."""
    with open("config/prompts.yaml", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    example = prompts["ner"]["examples"][0]

    # Check user input contains expected company data
    user_input = example["user"]
    assert "综艺股份" in user_input
    assert "600770" in user_input
    assert "信息科技板块" in user_input

    # Check assistant output is valid JSON with entities
    assistant_output = example["assistant"]
    import json

    parsed_output = json.loads(assistant_output)

    assert "named_entities" in parsed_output
    assert isinstance(parsed_output["named_entities"], list)
    assert len(parsed_output["named_entities"]) > 0

    # Check that entities are dictionaries with text and type
    first_entity = parsed_output["named_entities"][0]
    assert isinstance(first_entity, dict)
    assert "text" in first_entity
    assert "type" in first_entity

    # Check that 综艺股份 is in the entities
    entity_texts = [e["text"] for e in parsed_output["named_entities"]]
    assert "综艺股份" in entity_texts
