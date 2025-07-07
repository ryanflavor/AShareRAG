"""Intent detection implementations."""

import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from .keywords import normalize_keywords
from .types import IntentDetectionConfig, IntentType, QueryIntent


class KeywordIntentDetector:
    """Keyword-based intent detection."""

    def __init__(self, config: IntentDetectionConfig | None = None):
        """Initialize keyword intent detector.

        Args:
            config: Optional configuration for intent detection.
        """
        self.config = config or IntentDetectionConfig()
        self.keywords = normalize_keywords()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.fact_patterns = []
        self.relationship_patterns = []

        # Compile patterns for multi-word keywords
        for kw in self.keywords["fact_qa"]:
            if " " in kw:
                pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                self.fact_patterns.append((pattern, kw))

        for kw in self.keywords["relationship"]:
            if " " in kw:
                pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                self.relationship_patterns.append((pattern, kw))

    def detect(self, query: str) -> QueryIntent:
        """Detect intent based on keyword matching."""
        if not query:
            return QueryIntent(
                query=query,
                intent_type=IntentType.FACT_QA,
                confidence=0.5,
                detection_method="keyword",
                keywords_matched=[],
            )

        query_lower = query.lower()
        fact_keywords = self._find_keywords(query_lower, "fact_qa")
        relationship_keywords = self._find_keywords(query_lower, "relationship")

        # Calculate confidence based on keyword matches
        fact_score = len(fact_keywords) * 0.1
        relationship_score = len(relationship_keywords) * 0.1

        # Boost scores for certain strong indicators
        strong_relationship_indicators = {
            "关系",
            "关联",
            "相似",
            "类似",
            "竞品",
            "上下游",
            "relationship",
            "similar",
            "competitor",
            "compare",
        }
        strong_boost = 0
        for kw in relationship_keywords:
            if kw in strong_relationship_indicators:
                strong_boost += 0.15
        relationship_score += strong_boost

        # Determine intent type
        if relationship_score > fact_score and (
            relationship_score >= 0.1 or strong_boost > 0
        ):
            intent_type = IntentType.RELATIONSHIP_DISCOVERY
            # Base confidence 0.6, plus bonus for keywords (cap at different levels)
            if len(relationship_keywords) == 1:
                confidence = min(0.6 + relationship_score, 0.85)
            else:
                confidence = min(0.6 + relationship_score, 0.95)
            keywords = relationship_keywords
        elif fact_score >= 0.1:
            intent_type = IntentType.FACT_QA
            # Base confidence 0.6, plus bonus for keywords
            confidence = min(0.6 + fact_score, 0.95)
            keywords = fact_keywords
        else:
            intent_type = IntentType.FACT_QA  # Default
            confidence = 0.5
            keywords = []

        return QueryIntent(
            query=query,
            intent_type=intent_type,
            confidence=confidence,
            detection_method="keyword",
            keywords_matched=keywords,
        )

    def _find_keywords(self, query_lower: str, category: str) -> list[str]:
        """Find matching keywords in query."""
        matched = []

        # For Chinese/CJK characters, we need to check substring matches
        # as they don't have word boundaries like English
        for kw in self.keywords[category]:
            if " " not in kw:
                # For single keywords, check if it's contained in the query
                if kw in query_lower:
                    matched.append(kw)

        # Check multi-word patterns (mostly for English phrases)
        patterns = (
            self.fact_patterns if category == "fact_qa" else self.relationship_patterns
        )
        for pattern, keyword in patterns:
            if pattern.search(query_lower):
                matched.append(keyword)

        return list(set(matched))  # Remove duplicates


class LLMIntentDetector:
    """LLM-based intent detection."""

    def __init__(self, llm_adapter=None, config: IntentDetectionConfig | None = None):
        """Initialize LLM intent detector.

        Args:
            llm_adapter: Adapter for LLM-based classification.
            config: Optional configuration for intent detection.
        """
        self.config = config or IntentDetectionConfig()
        self.llm_adapter = llm_adapter
        self.executor = ThreadPoolExecutor(max_workers=1)

    def detect(self, query: str) -> QueryIntent:
        """Detect intent using LLM classification."""
        try:
            # Run LLM query with timeout
            future = self.executor.submit(self._query_llm, query)
            result = future.result(timeout=self.config.timeout)

            # Parse LLM response
            intent_str = result.get("intent", "unknown").lower()
            if "relationship" in intent_str:
                intent_type = IntentType.RELATIONSHIP_DISCOVERY
            elif "fact" in intent_str or "qa" in intent_str:
                intent_type = IntentType.FACT_QA
            else:
                intent_type = IntentType.UNKNOWN

            confidence = float(result.get("confidence", 0.0))

            return QueryIntent(
                query=query,
                intent_type=intent_type,
                confidence=confidence,
                detection_method="llm",
                keywords_matched=[],
                metadata={"reasoning": result.get("reasoning", "")},
            )

        except FutureTimeoutError:
            raise TimeoutError(
                f"LLM query timed out after {self.config.timeout}s"
            ) from None
        except Exception as e:
            # Return unknown intent on error
            return QueryIntent(
                query=query,
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                detection_method="llm_error",
                keywords_matched=[],
                metadata={"error": str(e)},
            )

    def _query_llm(self, query: str) -> dict[str, Any]:
        """Query LLM for intent classification."""
        if not self.llm_adapter:
            raise ValueError("LLM adapter not configured")

        # Use the actual adapter's method to classify intent
        return self.llm_adapter.classify_intent(query)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
