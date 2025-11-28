# backend/agents/__init__.py
"""
Specialized Agent System for LocalLens

Multi-agent architecture with:
- Query Classification
- Intent Routing
- Specialized task agents
- Self-reflection and quality control
"""

from .query_classifier import QueryClassifier, QueryIntent
from .clarification_agent import ClarificationAgent
from .analysis_agent import AnalysisAgent
from .summarization_agent import SummarizationAgent
from .explanation_agent import ExplanationAgent
from .critic_agent import CriticAgent

__all__ = [
    "QueryClassifier",
    "QueryIntent",
    "ClarificationAgent",
    "AnalysisAgent",
    "SummarizationAgent",
    "ExplanationAgent",
    "CriticAgent"
]
