# Query Intent Router Component

## Overview

The Query Intent Router is a critical component in the AShareRAG system that analyzes user queries and routes them to appropriate processing flows based on detected intent. It serves as the gateway between user input and the retrieval system, ensuring queries are handled by the most suitable processing pipeline.

## Purpose

The router distinguishes between two primary query types:
- **Fact-based Q&A**: Queries seeking specific factual information about entities
- **Relationship Discovery**: Queries exploring relationships, similarities, or connections between entities

## Architecture

### Component Location
- **Module**: `/src/intent_detection/`
- **Main Class**: `IntentRouter` in `/src/intent_detection/router.py`

### Key Classes

#### IntentRouter
The main routing component that coordinates intent detection and query routing.

```python
from src.intent_detection import IntentRouter, IntentDetectionConfig

config = IntentDetectionConfig(
    keyword_threshold=0.6,
    use_llm_fallback=False,
    cache_enabled=True
)
router = IntentRouter(config=config)
result = router.route_query("找出与腾讯相似的公司")
```

#### KeywordIntentDetector
Fast keyword-based detection for primary classification.

#### LLMIntentDetector
Optional LLM-based detection for ambiguous queries.

## Features

### 1. Dual Detection Strategy
- **Primary**: Keyword-based detection (<100ms)
- **Fallback**: LLM-based classification (15-20s, optional)

### 2. Language Support
Supports both Chinese and English queries with specialized keyword sets:
- Chinese relationship keywords: 相似, 关联, 类似, 竞品, 上下游
- English relationship keywords: similar, related, competitor, compare

### 3. Performance Optimization
- LRU caching for repeated queries
- Compiled regex patterns for efficient matching
- Lazy initialization of LLM detector

### 4. Comprehensive Logging
- Structured JSON logging for all classification decisions
- Performance metrics tracking
- Error logging with context

### 5. Robust Error Handling
- Input validation and sanitization
- Graceful handling of edge cases
- Fallback to fact_qa on errors

## Configuration

### IntentDetectionConfig
```python
@dataclass
class IntentDetectionConfig:
    keyword_threshold: float = 0.6      # Minimum confidence for keyword detection
    llm_threshold: float = 0.7          # Minimum confidence for LLM detection
    use_llm_fallback: bool = False      # Enable LLM fallback
    cache_enabled: bool = False         # Enable query caching
    cache_ttl: int = 3600              # Cache TTL in seconds
    timeout: float = 5.0               # LLM timeout in seconds
```

### Prompts Configuration
Intent classification prompts are stored in `/config/prompts.yaml`:

```yaml
query_intent_classification:
  system: |
    You are a query intent classifier for a financial knowledge base system...
  user: |
    Query: {query}
    Classify this query's intent.
```

## Usage Examples

### Basic Usage
```python
from src.intent_detection import IntentRouter

# Create router with default config
router = IntentRouter()

# Route a query
result = router.route_query("What is Alibaba's revenue?")
print(f"Intent: {result['intent_type']}")  # "fact_qa"
print(f"Route to: {result['route_to']}")   # "vector_retriever"
```

### With Custom Configuration
```python
from src.intent_detection import IntentRouter, IntentDetectionConfig

# Enable caching and LLM fallback
config = IntentDetectionConfig(
    use_llm_fallback=True,
    cache_enabled=True,
    keyword_threshold=0.7
)
router = IntentRouter(config=config)

# Route with advanced features
result = router.route_query("分析市场趋势")
```

### Integration with Downstream Components
```python
# In FastAPI server
@app.post("/api/query")
async def process_query(request: QueryRequest):
    router = IntentRouter(config=settings.INTENT_CONFIG)
    route_result = router.route_query(request.query)
    
    if route_result["intent_type"] == "relationship_discovery":
        return await hybrid_retriever.retrieve(request.query)
    else:
        return await vector_retriever.retrieve(request.query)
```

## Performance Characteristics

### Latency Benchmarks
- Keyword detection: ~20-40μs
- Cached queries: ~12μs
- Long queries (10k chars): ~390μs
- LLM classification: 15-20s (when enabled)

### Scalability
- Thread-safe for concurrent access
- LRU cache with configurable size (default: 1000)
- Efficient memory usage with bounded cache

## Error Handling

The router handles various error scenarios gracefully:

1. **Empty/None queries**: Returns fact_qa with low confidence
2. **Special characters**: Sanitizes input, logs suspicious patterns
3. **Very long queries**: Truncates to 10k characters
4. **LLM timeouts**: Falls back to keyword detection
5. **Invalid encoding**: Handles unicode edge cases

## Monitoring and Observability

### Metrics Tracked
- Total queries processed
- Detection method distribution
- Average/P50/P95/P99 latencies
- Cache hit rate
- Error count

### Logging
All routing decisions are logged with:
- Query text (truncated)
- Detected intent type
- Confidence score
- Detection method used
- Processing latency
- Matched keywords

## Testing

### Unit Tests
Located in `/tests/unit/test_query_intent_router.py`
- Keyword detection accuracy
- Edge case handling
- Performance requirements
- Logging functionality

### Integration Tests
Located in `/tests/integration/test_query_intent_routing.py`
- Routing to downstream components
- End-to-end pipeline testing
- Error propagation

### Performance Tests
Located in `/tests/performance/test_query_intent_performance.py`
- Latency benchmarks
- Cache efficiency
- Concurrent access

## Best Practices

1. **Configuration**
   - Disable LLM fallback for latency-sensitive applications
   - Enable caching for high-traffic scenarios
   - Adjust thresholds based on your data

2. **Integration**
   - Initialize router once and reuse
   - Handle routing results defensively
   - Monitor performance metrics

3. **Maintenance**
   - Review and update keyword lists periodically
   - Monitor classification accuracy
   - Adjust confidence thresholds as needed

## Future Enhancements

1. **Multi-intent Support**: Handle queries with multiple intents
2. **Custom Intent Types**: Support domain-specific intent categories
3. **ML-based Classification**: Train lightweight classifier on historical data
4. **Dynamic Keyword Learning**: Learn new keywords from user feedback
5. **A/B Testing Support**: Route percentage of traffic to experimental flows