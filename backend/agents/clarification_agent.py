# backend/agents/clarification_agent.py
"""
Clarification Agent

Handles ambiguous queries by:
- Detecting ambiguity
- Generating follow-up questions
- Refining user intent through dialogue
"""

from typing import Dict, Any, List, Optional
import httpx
from loguru import logger
import json


class ClarificationAgent:
    """Agent specialized in handling ambiguous queries"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']

    async def detect_ambiguity(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect if query is ambiguous

        Returns:
            Dict with ambiguity score and specific issues
        """
        prompt = f"""Analyze if this query is ambiguous or unclear:

Query: "{query}"

Consider:
1. Is the intent clear?
2. Is the scope well-defined?
3. Are there multiple possible interpretations?

Respond in JSON:
{{
    "is_ambiguous": true/false,
    "ambiguity_score": 0.0-1.0,
    "issues": ["specific ambiguity issues"],
    "possible_interpretations": ["interpretation 1", "interpretation 2"]
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
                        "options": {"temperature": 0.2}
                    }
                )
                result = response.json()
                return json.loads(result.get('response', '{}'))

        except Exception as e:
            logger.error(f"Ambiguity detection failed: {e}")
            return {
                "is_ambiguous": False,
                "ambiguity_score": 0.0,
                "issues": []
            }

    async def generate_clarifying_questions(
        self,
        query: str,
        ambiguity_info: Dict[str, Any],
        max_questions: int = 3
    ) -> List[str]:
        """
        Generate clarifying questions to resolve ambiguity

        Args:
            query: Original ambiguous query
            ambiguity_info: Output from detect_ambiguity
            max_questions: Maximum questions to generate

        Returns:
            List of clarifying questions
        """
        issues = ambiguity_info.get('issues', [])
        interpretations = ambiguity_info.get('possible_interpretations', [])

        prompt = f"""The user asked: "{query}"

Ambiguity issues: {', '.join(issues)}
Possible interpretations: {', '.join(interpretations)}

Generate {max_questions} specific clarifying questions to resolve the ambiguity.
Questions should be:
- Specific and actionable
- Help narrow down what the user wants
- Easy to answer with a simple response

Respond in JSON:
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
                        "options": {"temperature": 0.4}
                    }
                )
                result = response.json()
                data = json.loads(result.get('response', '{}'))
                return data.get('questions', [])[:max_questions]

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return [
                "Could you be more specific about what you're looking for?",
                "What type of documents are you interested in?",
                "Do you have a time frame in mind?"
            ][:max_questions]

    async def refine_query(
        self,
        original_query: str,
        clarification_answer: str
    ) -> str:
        """
        Refine query based on clarification answer

        Args:
            original_query: Original ambiguous query
            clarification_answer: User's answer to clarifying question

        Returns:
            Refined query with better clarity
        """
        prompt = f"""Refine this query based on the user's clarification:

Original query: "{original_query}"
User clarification: "{clarification_answer}"

Generate a more specific, refined query that incorporates the clarification.
Keep it natural and concise.

Respond with just the refined query (no JSON, just text):"""

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3}
                    }
                )
                result = response.json()
                refined = result.get('response', '').strip()

                # Clean up any quotes
                refined = refined.strip('"\'')

                return refined if refined else original_query

        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            # Fallback: combine original + clarification
            return f"{original_query} {clarification_answer}"

    async def suggest_alternatives(
        self,
        query: str,
        num_alternatives: int = 3
    ) -> List[str]:
        """
        Suggest alternative phrasings for the query

        Helps users explore different search approaches
        """
        prompt = f"""Generate {num_alternatives} alternative phrasings for this search query:

Original: "{query}"

Alternatives should:
- Mean the same thing but use different words
- Be more specific or approach from different angle
- Help find relevant documents

Respond in JSON:
{{
    "alternatives": ["Alternative 1", "Alternative 2", "Alternative 3"]
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
                        "options": {"temperature": 0.5}
                    }
                )
                result = response.json()
                data = json.loads(result.get('response', '{}'))
                return data.get('alternatives', [])

        except Exception as e:
            logger.error(f"Alternative generation failed: {e}")
            return []
