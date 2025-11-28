# backend/agents/analysis_agent.py
"""
Analysis Agent

Performs cross-document analysis:
- Document comparison
- Aggregation across results
- Trend detection
- Insight generation
"""

from typing import Dict, Any, List, Optional
import httpx
from loguru import logger
import json
from datetime import datetime


class AnalysisAgent:
    """Agent specialized in analyzing multiple documents"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']

    async def compare_documents(
        self,
        documents: List[Dict[str, Any]],
        comparison_criteria: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple documents

        Args:
            documents: List of documents to compare
            comparison_criteria: What to compare (e.g., "content", "dates", "values")

        Returns:
            Comparison results with similarities and differences
        """
        if len(documents) < 2:
            return {"error": "Need at least 2 documents to compare"}

        # Extract document summaries
        doc_summaries = []
        for i, doc in enumerate(documents[:3], 1):  # Limit to top 3
            summary = doc.get('content_summary', doc.get('content', ''))[:500]
            doc_summaries.append(f"Document {i} ({doc['filename']}): {summary}")

        summaries_text = "\n\n".join(doc_summaries)

        prompt = f"""Compare these documents{' based on: ' + comparison_criteria if comparison_criteria else ''}:

{summaries_text}

Provide:
1. Key similarities
2. Key differences
3. Unique aspects of each document

Respond in JSON:
{{
    "similarities": ["similarity 1", "similarity 2"],
    "differences": ["difference 1", "difference 2"],
    "unique_aspects": {{
        "doc1": ["aspect 1"],
        "doc2": ["aspect 1"]
    }},
    "summary": "brief comparison summary"
}}"""

        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
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
                return json.loads(result.get('response', '{}'))

        except Exception as e:
            logger.error(f"Document comparison failed: {e}")
            return {"error": str(e)}

    async def aggregate_data(
        self,
        documents: List[Dict[str, Any]],
        aggregation_type: str
    ) -> Dict[str, Any]:
        """
        Aggregate data across documents

        Args:
            documents: Documents to aggregate
            aggregation_type: Type of aggregation (e.g., "total expenses", "count by type")

        Returns:
            Aggregated results
        """
        # Extract relevant content
        doc_info = []
        for doc in documents:
            doc_info.append({
                "filename": doc['filename'],
                "type": doc.get('document_type', 'unknown'),
                "summary": doc.get('content_summary', '')[:200]
            })

        prompt = f"""Aggregate data from these documents for: "{aggregation_type}"

Documents:
{json.dumps(doc_info, indent=2)}

Provide aggregated insights:

Respond in JSON:
{{
    "aggregation_result": "aggregated value or summary",
    "breakdown": {{"category": "value"}},
    "insights": ["insight 1", "insight 2"]
}}"""

        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
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
            logger.error(f"Data aggregation failed: {e}")
            return {"error": str(e)}

    async def detect_trends(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect trends across documents over time

        Args:
            documents: Time-series documents

        Returns:
            Trend analysis
        """
        # Sort documents by date if available
        dated_docs = []
        for doc in documents:
            # Try to extract date from metadata or filename
            doc_date = doc.get('created_date') or doc.get('modified_date')
            if doc_date:
                dated_docs.append({
                    "date": doc_date,
                    "filename": doc['filename'],
                    "summary": doc.get('content_summary', '')[:200]
                })

        if not dated_docs:
            return {"error": "No dated documents for trend analysis"}

        # Sort by date
        dated_docs.sort(key=lambda x: x['date'])

        prompt = f"""Analyze trends in these documents over time:

{json.dumps(dated_docs[:10], indent=2)}

Identify:
1. Temporal patterns
2. Evolving themes
3. Changes over time

Respond in JSON:
{{
    "trends": ["trend 1", "trend 2"],
    "patterns": ["pattern 1", "pattern 2"],
    "insights": ["insight 1", "insight 2"]
}}"""

        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
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
                return json.loads(result.get('response', '{}'))

        except Exception as e:
            logger.error(f"Trend detection failed: {e}")
            return {"error": str(e)}

    async def generate_insights(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[str]:
        """
        Generate insights from document set

        Args:
            documents: Documents to analyze
            query: Original query for context

        Returns:
            List of insights
        """
        doc_summaries = [
            f"{doc['filename']}: {doc.get('content_summary', '')[:150]}"
            for doc in documents[:5]
        ]

        prompt = f"""Based on the query "{query}", generate insights from these documents:

{chr(10).join(doc_summaries)}

Provide 3-5 key insights that would be valuable to the user.

Respond in JSON:
{{
    "insights": ["insight 1", "insight 2", "insight 3"]
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
                        "options": {"temperature": 0.4}
                    }
                )
                result = response.json()
                data = json.loads(result.get('response', '{}'))
                return data.get('insights', [])

        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []
