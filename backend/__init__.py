# backend/__init__.py - LocalLens Backend Package

"""
LocalLens Backend Package v2.0

Enhanced semantic search with:
- Hybrid search (Vector + BM25)
- Query expansion
- Conversational AI responses
- A2A agent orchestration
"""

__version__ = "2.0.0"
__author__ = "LocalLens Team"

from backend.api import app
from backend.opensearch_client import OpenSearchClient
from backend.ingestion import IngestionPipeline
from backend.reranker import CrossEncoderReranker
from backend.watcher import FileWatcher
from backend.mcp_tools import MCPToolRegistry
from backend.a2a_agent import (
    A2AAgent,
    IngestionAgent,
    SearchAgent,
    ConversationAgent,
    OrchestratorAgent
)

__all__ = [
    "app",
    "OpenSearchClient",
    "IngestionPipeline",
    "CrossEncoderReranker",
    "FileWatcher",
    "MCPToolRegistry",
    "A2AAgent",
    "IngestionAgent",
    "SearchAgent",
    "ConversationAgent",
    "OrchestratorAgent"
]
