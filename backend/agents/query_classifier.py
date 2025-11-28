# backend/agents/query_classifier.py
"""
Query Classification & Intent Router

Classifies queries into:
- Document Search: Needs to search indexed documents
- General Knowledge: Can be answered directly by LLM
- System/Meta: Questions about the system
- Comparison: Comparing multiple documents
- Summarization: Multi-document summarization
- Clarification Needed: Ambiguous queries
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import re
import httpx
from loguru import logger
import json


class QueryIntent(Enum):
    """Query intent types"""
    DOCUMENT_SEARCH = "document_search"
    GENERAL_KNOWLEDGE = "general_knowledge"
    SYSTEM_META = "system_meta"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass
class ClassificationResult:
    """Result of query classification"""
    intent: QueryIntent
    confidence: float
    sub_intent: Optional[str] = None
    entities: List[str] = None
    filters: Optional[Dict] = None
    clarification_questions: List[str] = None
    reasoning: str = ""

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.clarification_questions is None:
            self.clarification_questions = []


class QueryClassifier:
    """
    Intelligent query classifier with LLM-based fallback

    Uses a hybrid approach:
    1. Rule-based classification for common patterns (fast)
    2. LLM-based classification for complex queries (accurate)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']

        # Document-related keywords
        self.doc_keywords = [
            "find", "search", "show", "get", "document", "file", "invoice",
            "contract", "report", "spreadsheet", "image", "pdf", "where is"
        ]

        # General knowledge keywords
        self.general_keywords = [
            "what is", "who is", "how to", "explain", "define", "tell me about",
            "why does", "how does", "when did"
        ]

        # Comparison keywords
        self.comparison_keywords = [
            "compare", "difference", "versus", "vs", "better", "contrast",
            "similarities", "which one"
        ]

        # Summarization keywords
        self.summary_keywords = [
            "summarize", "summary", "overview", "recap", "all documents about",
            "everything about", "compile", "aggregate"
        ]

    async def classify(self, query: str, context: Optional[Dict] = None) -> ClassificationResult:
        """
        Classify query into intent type

        Args:
            query: User query
            context: Optional session context for better classification

        Returns:
            ClassificationResult with intent and metadata
        """
        # Try rule-based first (fast path)
        rule_result = self._rule_based_classify(query, context)

        # If confidence is high, return immediately
        if rule_result.confidence > 0.8:
            logger.info(f"Rule-based classification: {rule_result.intent.value} (confidence: {rule_result.confidence})")
            return rule_result

        # Otherwise, use LLM for better accuracy
        llm_result = await self._llm_classify(query, context)

        # Combine results (LLM has final say)
        return llm_result

    def _rule_based_classify(self, query: str, context: Optional[Dict]) -> ClassificationResult:
        """Fast rule-based classification"""
        q_lower = query.lower()

        # Check for system/meta queries
        if any(kw in q_lower for kw in ["how does this work", "how do i", "what can you", "can you help"]):
            return ClassificationResult(
                intent=QueryIntent.SYSTEM_META,
                confidence=0.9,
                reasoning="System help query"
            )

        # Check for comparison
        if any(kw in q_lower for kw in self.comparison_keywords):
            return ClassificationResult(
                intent=QueryIntent.COMPARISON,
                confidence=0.85,
                reasoning="Comparison keywords detected"
            )

        # Check for summarization
        if any(kw in q_lower for kw in self.summary_keywords):
            return ClassificationResult(
                intent=QueryIntent.SUMMARIZATION,
                confidence=0.85,
                reasoning="Summarization keywords detected"
            )

        # Check for document search
        if any(kw in q_lower for kw in self.doc_keywords):
            # Extract document type filters
            filters = self._extract_filters(query)
            return ClassificationResult(
                intent=QueryIntent.DOCUMENT_SEARCH,
                confidence=0.8,
                filters=filters,
                reasoning="Document search keywords detected"
            )

        # Check for general knowledge
        if any(kw in q_lower for kw in self.general_keywords):
            return ClassificationResult(
                intent=QueryIntent.GENERAL_KNOWLEDGE,
                confidence=0.75,
                reasoning="General knowledge question pattern"
            )

        # Default to document search with lower confidence
        return ClassificationResult(
            intent=QueryIntent.DOCUMENT_SEARCH,
            confidence=0.5,
            reasoning="Default classification"
        )

    async def _llm_classify(self, query: str, context: Optional[Dict]) -> ClassificationResult:
        """
        LLM-based classification for complex queries

        Uses structured output with Ollama
        """
        context_str = ""
        if context:
            recent_queries = context.get("recent_queries", [])
            if recent_queries:
                context_str = f"\nRecent queries: {', '.join(recent_queries[-3:])}"

        prompt = f"""Classify the following search query into one of these categories:

1. DOCUMENT_SEARCH: User wants to find specific documents in their collection
2. GENERAL_KNOWLEDGE: User wants general information (not about their documents)
3. SYSTEM_META: User asking about how the system works
4. COMPARISON: User wants to compare multiple documents
5. SUMMARIZATION: User wants summary of multiple documents
6. ANALYSIS: User wants insights across documents
7. CLARIFICATION_NEEDED: Query is ambiguous and needs clarification

Query: "{query}"{context_str}

Respond in JSON format:
{{
    "intent": "DOCUMENT_SEARCH|GENERAL_KNOWLEDGE|SYSTEM_META|COMPARISON|SUMMARIZATION|ANALYSIS|CLARIFICATION_NEEDED",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "entities": ["extracted", "entities"],
    "filters": {{"file_type": [".pdf"], "document_type": "invoice"}},
    "clarification_questions": ["if ambiguous"]
}}"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.1}
                    }
                )
                result = response.json()

                # Parse JSON response
                classification = json.loads(result.get('response', '{}'))

                intent_str = classification.get('intent', 'DOCUMENT_SEARCH')
                try:
                    intent = QueryIntent[intent_str]
                except KeyError:
                    intent = QueryIntent.DOCUMENT_SEARCH

                return ClassificationResult(
                    intent=intent,
                    confidence=float(classification.get('confidence', 0.7)),
                    entities=classification.get('entities', []),
                    filters=classification.get('filters'),
                    clarification_questions=classification.get('clarification_questions', []),
                    reasoning=classification.get('reasoning', 'LLM classification')
                )

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fallback to rule-based
            return self._rule_based_classify(query, context)

    def _extract_filters(self, query: str) -> Optional[Dict]:
        """
        Extract document filters from query

        Returns:
            Dict with filters for document search
        """
        q_lower = query.lower()
        filters = {}

        # File type filters
        file_type_map = {
            "pdf": [".pdf"],
            "word": [".docx", ".doc"],
            "excel": [".xlsx", ".xls"],
            "spreadsheet": [".xlsx", ".xls", ".csv"],
            "image": [".png", ".jpg", ".jpeg", ".gif", ".bmp"],
            "photo": [".png", ".jpg", ".jpeg"],
            "picture": [".png", ".jpg", ".jpeg"],
        }

        for keyword, extensions in file_type_map.items():
            if keyword in q_lower:
                filters["file_type"] = extensions
                break

        # Document type filters
        doc_type_map = {
            "invoice": "invoice",
            "contract": "contract",
            "report": "report",
            "receipt": "invoice",
            "agreement": "contract"
        }

        for keyword, doc_type in doc_type_map.items():
            if keyword in q_lower:
                filters["document_type"] = doc_type
                break

        # Time filters (simple extraction)
        time_patterns = [
            (r"last month", "last_month"),
            (r"this month", "this_month"),
            (r"last week", "last_week"),
            (r"this week", "this_week"),
            (r"today", "today"),
            (r"yesterday", "yesterday"),
            (r"last year", "last_year"),
            (r"2024", "2024"),
            (r"2023", "2023")
        ]

        for pattern, time_filter in time_patterns:
            if re.search(pattern, q_lower):
                filters["time_range"] = time_filter
                break

        return filters if filters else None

    def needs_clarification(self, result: ClassificationResult) -> bool:
        """
        Determine if query needs clarification

        Returns True if:
        - Intent is CLARIFICATION_NEEDED
        - Confidence is very low
        - Has clarification questions
        """
        return (
            result.intent == QueryIntent.CLARIFICATION_NEEDED or
            result.confidence < 0.4 or
            len(result.clarification_questions) > 0
        )

    async def suggest_clarifications(self, query: str) -> List[str]:
        """
        Generate clarification questions for ambiguous query

        Returns:
            List of clarification questions
        """
        prompt = f"""The user asked: "{query}"

This query is ambiguous. Generate 2-3 clarifying questions to better understand what they're looking for.

Questions should be specific and actionable.

Respond in JSON format:
{{
    "questions": ["Question 1?", "Question 2?", "Question 3?"]
}}"""

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.3}
                    }
                )
                result = response.json()
                data = json.loads(result.get('response', '{}'))
                return data.get('questions', [])

        except Exception as e:
            logger.error(f"Clarification generation failed: {e}")
            return [
                "Are you looking for a specific type of document?",
                "Do you remember any keywords or dates associated with it?"
            ]

    def extract_entities(self, query: str) -> List[str]:
        """
        Extract named entities from query

        Simple extraction for now - could use NER model
        """
        # Capitalize words (simple heuristic for names/entities)
        words = query.split()
        entities = []

        for word in words:
            # If word is capitalized (not at start), likely an entity
            if word[0].isupper() and word not in ["I", "A", "The", "This", "That"]:
                entities.append(word)

        # Extract dates (YYYY-MM-DD, YYYY, etc.)
        date_pattern = r'\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b'
        dates = re.findall(date_pattern, query)
        entities.extend(dates)

        return entities
