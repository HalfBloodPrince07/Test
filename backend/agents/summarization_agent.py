# backend/agents/summarization_agent.py
"""Summarization Agent - Multi-document summarization"""

from typing import Dict, Any, List
import httpx
from loguru import logger
import json


class SummarizationAgent:
    """Agent specialized in multi-document summarization"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']

    async def summarize_documents(
        self,
        documents: List[Dict[str, Any]],
        summary_type: str = "comprehensive"
    ) -> str:
        """
        Generate summary of multiple documents

        Args:
            documents: List of documents to summarize
            summary_type: "comprehensive", "brief", or "bullet_points"

        Returns:
            Generated summary
        """
        if not documents:
            return "No documents to summarize."

        # Extract content
        doc_texts = []
        for i, doc in enumerate(documents[:10], 1):
            summary = doc.get('content_summary', doc.get('content', ''))[:400]
            doc_texts.append(f"**{doc['filename']}**: {summary}")

        combined = "\n\n".join(doc_texts)

        summary_instructions = {
            "comprehensive": "Provide a detailed summary covering all key points",
            "brief": "Provide a concise 2-3 sentence summary",
            "bullet_points": "Provide summary as bullet points"
        }

        instruction = summary_instructions.get(summary_type, summary_instructions["comprehensive"])

        prompt = f"""Summarize these {len(documents)} documents:

{combined}

{instruction}.

Return the summary as plain text."""

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
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
                return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Error summarizing documents: {str(e)}"

    async def hierarchical_summary(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate hierarchical summary for large document sets

        Summarizes in tiers for better organization
        """
        if len(documents) <= 5:
            summary = await self.summarize_documents(documents)
            return {"summary": summary, "tiers": 1}

        # For larger sets, create hierarchical structure
        # This is a simplified version
        tier1_summary = await self.summarize_documents(documents[:5], "comprehensive")
        tier2_summary = await self.summarize_documents(documents[5:10], "comprehensive")

        # Combine tier summaries
        combined_prompt = f"""Combine these summaries into a cohesive overview:

Summary 1: {tier1_summary}

Summary 2: {tier2_summary}

Provide a unified summary."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": combined_prompt,
                        "stream": False,
                        "options": {"temperature": 0.2}
                    }
                )
                result = response.json()
                final_summary = result.get('response', '').strip()

                return {
                    "summary": final_summary,
                    "tier1": tier1_summary,
                    "tier2": tier2_summary,
                    "tiers": 2
                }

        except Exception as e:
            logger.error(f"Hierarchical summarization failed: {e}")
            return {"error": str(e)}
