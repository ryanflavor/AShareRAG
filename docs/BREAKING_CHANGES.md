# Breaking Changes

## Version 1.2.1 - Enhanced NER with Entity Types

### Overview
The Named Entity Recognition (NER) output format has been enhanced to include entity type classification. This is a breaking change that affects all consumers of the NER output.

### Changes

#### Before (v1.2.0)
```json
{
  "named_entities": ["综艺股份", "600770", "南京天悦", "助听器芯片"]
}
```

#### After (v1.2.1)
```json
{
  "named_entities": [
    {"text": "综艺股份", "type": "COMPANY"},
    {"text": "600770", "type": "COMPANY_CODE"},
    {"text": "南京天悦", "type": "SUBSIDIARY"},
    {"text": "助听器芯片", "type": "TECHNOLOGY"}
  ]
}
```

### Migration Guide

#### For API Consumers
Update your code to handle the new dictionary format:

```python
# Old code
for entity in entities:
    print(entity)  # prints "综艺股份"

# New code  
for entity in entities:
    print(f"{entity['text']} ({entity['type']})")  # prints "综艺股份 (COMPANY)"
```

#### Backwards Compatibility
The `LLMAdapter.extract_entities()` method includes an `include_types` parameter for backwards compatibility:

```python
# Get typed entities (default behavior)
typed_entities = adapter.extract_entities(text, include_types=True)

# Get string entities (backwards compatible)
string_entities = adapter.extract_entities(text, include_types=False)
```

### Supported Entity Types
The following entity types are supported:
- `COMPANY`: Company or group entity
- `SUBSIDIARY`: Explicitly mentioned subsidiary
- `AFFILIATE`: Explicitly mentioned affiliate or joint venture
- `BUSINESS_SEGMENT`: Major business segment or department
- `CORE_BUSINESS`: Description of core business or business model
- `PRODUCT`: Specific product, service, or brand name
- `TECHNOLOGY`: Technology, technology field, or key capability
- `INDUSTRY_APPLICATION`: Target application field or industry for products
- `COMPANY_CODE`: Company stock code

### Performance Impact
- Minimal performance impact: ~0.01ms additional parsing time per entity extraction
- No significant memory overhead

### Affected Components
- `src.adapters.llm_adapter.LLMAdapter`
- `src.components.knowledge_graph_constructor.KnowledgeGraphConstructor`
- All consumers of the NER output