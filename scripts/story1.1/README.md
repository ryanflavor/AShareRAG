# Scripts Directory

This directory contains utility scripts for testing and validation.

## Query Intent Router Validation Test

**File**: `test_query_intent_router_validation.py`

### Purpose
Comprehensive validation test for the Query Intent Router implementation (Story 2.1).

### Features
- Tests keyword-based intent detection performance (< 100ms requirement)
- Tests LLM fallback classification (15-20s acceptable)
- Tests cache performance optimization (< 500ms for cached queries)
- Validates logic correctness for all scenarios
- Real DeepSeek V3 API integration

### Prerequisites
Ensure the `.env` file contains:
```
DEEPSEEK_API_KEY=your_api_key_here
```

### Usage
```bash
uv run python scripts/test_query_intent_router_validation.py
```

### Output
- Console output showing test progress and results
- JSON report saved to `output/test_results/story_2.1_validation_YYYYMMDD_HHMMSS.json`

### Test Results
The test validates all acceptance criteria from Story 2.1:
- AC2: Chinese keywords trigger relationship_discovery
- AC3: Default to fact_qa when no keywords
- AC10: Edge cases handled gracefully
- AC11: Performance requirements met
  - Keyword detection < 100ms
  - LLM classification 15-20s acceptable
  - Cached queries < 500ms

### Expected Results
All tests should pass with:
- 100% accuracy on intent classification
- Keyword detection averaging ~0.07ms
- LLM classification averaging ~7s
- Cache providing >100,000x speedup