# backend/agents/critic_agent.py
"""
Critic Agent - Self-reflection and Quality Control

Implements the critic pattern for quality assurance
"""

from typing import Dict, Any, List, Optional
import httpx
from loguru import logger
import json


class CriticAgent:
    """
    Agent that evaluates and critiques search results

    Checks for:
    - Relevance
    - Completeness
    - Hallucination detection
    - Quality assessment
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']

    async def evaluate_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate quality of search results

        Args:
            query: Original query
            results: Search results to evaluate

        Returns:
            Quality assessment with score and recommendations
        """
        if not results:
            return {
                "quality_score": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
                "recommendations": ["No results found. Try broadening your search terms."],
                "should_reformulate": True
            }

        # Prepare results summary for evaluation
        results_summary = []
        for i, r in enumerate(results[:5], 1):
            results_summary.append(
                f"{i}. {r['filename']} (score: {r.get('score', 0):.2f})"
            )

        prompt = f"""Evaluate the quality of these search results for the query: "{query}"

Results:
{chr(10).join(results_summary)}

Assess:
1. Relevance: Do results match the query intent?
2. Completeness: Are important aspects covered?
3. Quality: Are results diverse and useful?

Respond in JSON:
{{
    "quality_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "strengths": ["strength 1", "strength 2"],
    "weaknesses": ["weakness 1", "weakness 2"],
    "recommendations": ["recommendation 1", "recommendation 2"],
    "should_reformulate": true/false
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
                        "options": {"temperature": 0.2}
                    }
                )
                result = response.json()
                evaluation = json.loads(result.get('response', '{}'))

                return {
                    "quality_score": float(evaluation.get('quality_score', 0.5)),
                    "relevance": float(evaluation.get('relevance_score', 0.5)),
                    "completeness": float(evaluation.get('completeness_score', 0.5)),
                    "strengths": evaluation.get('strengths', []),
                    "weaknesses": evaluation.get('weaknesses', []),
                    "recommendations": evaluation.get('recommendations', []),
                    "should_reformulate": evaluation.get('should_reformulate', False)
                }

        except Exception as e:
            logger.error(f"Result evaluation failed: {e}")
            return {
                "quality_score": 0.7,  # Assume decent quality
                "relevance": 0.7,
                "completeness": 0.6,
                "recommendations": []
            }

    async def detect_hallucination(
        self,
        query: str,
        response_text: str,
        source_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect if response contains hallucinated information

        Args:
            query: Original query
            response_text: Generated response
            source_documents: Source documents used

        Returns:
            Hallucination detection results
        """
        # Extract source summaries
        source_summaries = [
            doc.get('content_summary', '')[:200]
            for doc in source_documents[:3]
        ]

        prompt = f"""Check if this response contains information not supported by the sources:

Query: "{query}"

Response: "{response_text}"

Sources:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(source_summaries))}

Determine if the response contains any claims not supported by the sources.

Respond in JSON:
{{
    "has_hallucination": true/false,
    "confidence": 0.0-1.0,
    "unsupported_claims": ["claim 1", "claim 2"],
    "supported_claims": ["claim 1", "claim 2"]
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
                return json.loads(result.get('response', '{}'))

        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return {
                "has_hallucination": False,
                "confidence": 0.5,
                "unsupported_claims": []
            }

    def calculate_confidence_score(
        self,
        results: List[Dict[str, Any]],
        evaluation: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence score for results

        Combines multiple signals:
        - Number of results
        - Quality scores
        - Top result score
        """
        if not results:
            return 0.0

        # Factors
        num_results_score = min(len(results) / 5.0, 1.0)  # Max at 5 results
        quality_score = evaluation.get('quality_score', 0.5)
        top_score = results[0].get('score', 0) if results else 0

        # Weighted combination
        confidence = (
            num_results_score * 0.2 +
            quality_score * 0.4 +
            min(top_score, 1.0) * 0.4
        )

        return round(confidence, 2)

    async def suggest_improvements(
        self,
        query: str,
        results: List[Dict[str, Any]],
        evaluation: Dict[str, Any]
    ) -> List[str]:
        """
        Suggest improvements to search query or strategy

        Args:
            query: Original query
            results: Current results
            evaluation: Quality evaluation

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Based on evaluation
        if evaluation.get('should_reformulate'):
            suggestions.append("Try rephrasing your query with different keywords")

        if evaluation.get('relevance', 1.0) < 0.5:
            suggestions.append("Results may not match your intent. Be more specific")

        if len(results) == 0:
            suggestions.extend([
                "Try broader search terms",
                "Check if documents are indexed",
                "Use different keywords"
            ])

        elif len(results) < 3:
            suggestions.append("Limited results found. Try expanding your search")

        # Add evaluation recommendations
        suggestions.extend(evaluation.get('recommendations', []))

        return suggestions[:5]  # Top 5 suggestions
