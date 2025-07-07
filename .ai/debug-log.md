# AI Debug Log

## 2025-01-07 - LLM Integration Test Implementation

### Task: Design test case with real LLM API and cache toggle

Created comprehensive integration test for Story 2.1 Query Intent Detection:

1. **Main Test Script**: `/scripts/test_llm_integration_with_cache.py`
   - Tests real DeepSeek V3 API integration
   - Cache performance comparison (enabled/disabled)
   - Multiple cache sizes (128, 512, 1024)
   - Comprehensive test cases covering all scenarios
   - Performance metrics collection and reporting
   - JSON output with detailed results

2. **Cache Demo Script**: `/scripts/test_cache_performance_demo.py`
   - Simple focused demonstration of cache impact
   - Shows dramatic performance improvement with cache
   - Visual indicators for cache hits/misses
   - Performance summary with metrics

3. **Documentation**: `/scripts/README_llm_integration_test.md`
   - Usage instructions
   - Test case descriptions
   - Performance expectations
   - Troubleshooting guide

### Test Design:

**Test Cases Cover:**
- Keyword-based detection (Chinese: 相似, 关联, 类似, 竞品, 上下游)
- Fact-based queries (default behavior)
- Ambiguous queries requiring LLM
- Edge cases (empty, special chars, long queries)

**Performance Validation:**
- Keyword detection: < 100ms requirement
- LLM classification: 15-20s acceptable
- Cache performance: < 500ms for cached queries

**Cache Toggle Implementation:**
- Tests with cache disabled (baseline)
- Tests with cache enabled (various sizes)
- Measures cache hit rates and performance impact
- Compares runtime with/without cache

### Key Findings:
- Router uses `IntentRouter` class (not `QueryIntentRouter`)
- Configuration via `IntentDetectionConfig` object
- Synchronous API (not async)
- Returns dict with intent_type, confidence, metadata
- Cache provides significant performance boost on repeated queries

# AI Debug Log

## 2025-01-07 - Story 2.1: Query Intent Detection

### Task 1: Create intent detection module structure
- Created new module structure at `/src/intent_detection/`
- Created `__init__.py` with module exports
- Created `types.py` with IntentType enum, QueryIntent dataclass, and IntentDetectionConfig
- Created `keywords.py` with comprehensive keyword sets for fact-based and relationship queries
- Includes both English and Chinese keywords for better detection accuracy
- Module structure follows project conventions with proper type definitions

### Task 2: Implement keyword-based intent detection
- Created `detector.py` with KeywordIntentDetector class
- Implemented keyword matching that handles both English and Chinese text
- Chinese text matching uses substring search (no word boundaries)
- English text uses both substring and regex pattern matching
- Confidence calculation based on number and strength of keyword matches
- Single strong keyword: 0.6-0.85 confidence
- Multiple keywords: up to 0.95 confidence  
- Default to fact_qa with 0.5 confidence when no keywords match
- All tests pass for keyword detection including edge cases

### Task 3: Add LLM-based intent detection fallback
- Added query_intent_classification prompt to config/prompts.yaml
- Created IntentClassificationAdapter that wraps DeepSeek API
- Uses OpenAI client configured for DeepSeek endpoint
- Implements caching using SQLite (similar to existing adapters)
- Created LLMIntentDetector that uses the adapter
- Handles errors gracefully, returning UNKNOWN intent
- Supports configurable timeout with ThreadPoolExecutor
- All LLM detector tests pass including error and timeout scenarios

### Task 4: Implement routing logic and main interface
- Reviewed existing router.py - routing logic is already implemented
- IntentRouter class properly routes queries:
  - fact_qa → vector_retriever 
  - relationship_discovery → hybrid_retriever
- Returns QueryRouteResult with intent_type, confidence, hint, and metadata
- Unit tests already cover the routing logic comprehensively
- Need to create integration tests for downstream component interactions
- Created comprehensive integration tests at `/tests/integration/test_query_intent_routing.py`
- Tests cover routing to VectorRetriever and HybridRetriever
- All 12 integration tests pass successfully

### Task 5: Add logging and monitoring
- Enhanced IntentRouter with comprehensive logging:
  - Added StructuredLogger class for JSON logging
  - Added PerformanceMonitor class for metrics tracking
  - Logs classification decisions with query, intent, confidence, method, latency
  - Tracks metrics: total queries, detection counts, latencies, cache hits
  - Structured error logging with error type and message
- Fixed datetime deprecation warnings by using timezone.utc
- All logging tests pass successfully

### Task 6: Handle edge cases and error scenarios  
- Added comprehensive edge case tests:
  - Empty queries (various formats including None)
  - Special characters and injection attempts
  - Unicode edge cases (emojis, RTL, zalgo text)
  - Very long queries (up to 10k chars)
  - Mixed language queries
  - Malformed queries
  - Concurrent query handling
- Implemented _validate_and_sanitize_query method:
  - Handles None and non-string inputs
  - Truncates very long queries to 10k chars
  - Removes zero-width characters
  - Normalizes whitespace
  - Logs suspicious patterns (but still processes)
- All 8 edge case tests pass successfully

### Task 7: Performance optimization and testing
- Installed pytest-benchmark for performance testing
- Created comprehensive performance tests at `/tests/performance/test_query_intent_performance.py`
- Performance results:
  - Cached queries: ~12 microseconds (0.012ms)
  - Non-cached queries: ~38 microseconds (0.038ms)  
  - Long queries (10k chars): ~390 microseconds (0.39ms)
  - All well under the 100ms target!
- LRU cache already implemented in router using functools.lru_cache
- Cache efficiency tests show >85% hit rate for repeated queries
- Optimization validation tests ensure cache returns same results as non-cached
- Fixed failing unit tests for LLM fallback and low confidence scenarios
- All 29 unit tests pass

### Task 8: Documentation and final integration
- Added missing docstrings to __init__ methods in detector.py and router.py
- Fixed class docstring formatting in types.py using ruff
- Created comprehensive component documentation at `/docs/components/query_intent_router.md`
- Ran code formatting with `uv run ruff format`
- Fixed all linting issues with `uv run ruff check --fix`
- Fixed remaining B904 error (raise from None)
- Updated modern Python type hints (dict[str, Any] instead of Dict)
- All code quality checks pass
- Component is fully documented and ready for use