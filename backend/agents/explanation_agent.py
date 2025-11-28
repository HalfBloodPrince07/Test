# backend/agents/explanation_agent.py
"""Explanation Agent - Explains search results and reasoning"""

from typing import Dict, Any, List
import httpx
from loguru import logger
import json


class ExplanationAgent:
    """Agent specialized in explaining search results and reasoning"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']

    async def explain_ranking(
        self,
        query: str,
        document: Dict[str, Any],
        rank: int
    ) -> str:
        """
        Explain why a document ranked at specific position

        Args:
            query: Search query
            document: The document
            rank: Ranking position

        Returns:
            Explanation text
        """
        prompt = f"""Explain why this document ranked #{rank} for the query "{query}":

Document: {document['filename']}
Content: {document.get('content_summary', '')[:300]}
Score: {document.get('score', 0):.3f}

Provide a brief, user-friendly explanation of why this document matches the query.
Focus on specific matching aspects."""

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2}
                    }
                )
                result = response.json()
                return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Ranking explanation failed: {e}")
            return f"This document matches your query with a relevance score of {document.get('score', 0):.2f}"

    async def highlight_matches(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> List[str]:
        """
        Identify and extract matching sections from document

        Returns:
            List of relevant excerpts
        """
        content = document.get('content_summary', document.get('content', ''))[:1000]

        prompt = f"""Find the most relevant sections from this document that match the query "{query}":

{content}

Extract 2-3 short excerpts (1-2 sentences each) that best match the query.

Respond in JSON:
{{
    "excerpts": ["excerpt 1", "excerpt 2", "excerpt 3"]
}}"""

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
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
                data = json.loads(result.get('response', '{}'))
                return data.get('excerpts', [])

        except Exception as e:
            logger.error(f"Match highlighting failed: {e}")
            return []

    def explain_score_components(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Break down relevance score into components

        Args:
            document: Document with scoring metadata

        Returns:
            Score breakdown
        """
        score = document.get('score', 0)

        # Simplified breakdown (actual would come from search engine)
        return {
            "total_score": round(score, 3),
            "semantic_similarity": round(score * 0.7, 3),
            "keyword_match": round(score * 0.3, 3),
            "explanation": f"Score of {score:.2f} indicates {'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low'} relevance"
        }
