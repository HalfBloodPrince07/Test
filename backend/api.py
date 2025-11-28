# backend/api.py - Enhanced FastAPI Server with Conversational Responses

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, AsyncGenerator
import yaml
from pathlib import Path
import asyncio
from loguru import logger
import sys
import json
from datetime import datetime

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Import local modules
from backend.opensearch_client import OpenSearchClient
from backend.ingestion import IngestionPipeline
from backend.reranker import CrossEncoderReranker
from backend.watcher import FileWatcher
from backend.mcp_tools import MCPToolRegistry
from backend.a2a_agent import (
    SearchAgent, 
    IngestionAgent, 
    ConversationAgent,
    OrchestratorAgent
)

# Load Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI(
    title="LocalLens API",
    description="Semantic Search with Conversational AI",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
opensearch_client: Optional[OpenSearchClient] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
reranker: Optional[CrossEncoderReranker] = None
watcher: Optional[FileWatcher] = None
mcp_registry: Optional[MCPToolRegistry] = None
search_agent: Optional[SearchAgent] = None
ingestion_agent: Optional[IngestionAgent] = None
conversation_agent: Optional[ConversationAgent] = None
orchestrator_agent: Optional[OrchestratorAgent] = None

# Status tracking for streaming
status_updates: Dict[str, List[Dict]] = {}
ingestion_status: Dict[str, List[Dict]] = {}


# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    stream_status: bool = True


class IndexRequest(BaseModel):
    directory: str
    watch_mode: bool = False


class SearchResponse(BaseModel):
    status: str
    message: str
    query: str
    intent: str
    results: List[Dict[str, Any]]
    count: int
    search_time: float


# Status Streaming
async def status_callback(update: Dict[str, Any]):
    """Callback for status updates from agents"""
    task_id = update.get('task_id', 'default')
    if task_id not in status_updates:
        status_updates[task_id] = []
    status_updates[task_id].append(update)


async def ingestion_status_callback(update: Dict[str, Any]):
    """Callback for ingestion status updates"""
    task_id = update.get('task_id', 'default')
    if task_id not in ingestion_status:
        ingestion_status[task_id] = []
    ingestion_status[task_id].append(update)
    logger.debug(f"Ingestion status update: {update}")


async def stream_status(task_id: str) -> AsyncGenerator[str, None]:
    """Stream status updates as Server-Sent Events"""
    last_index = 0
    timeout = 30  # 30 second timeout
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        if task_id in status_updates:
            updates = status_updates[task_id]
            while last_index < len(updates):
                update = updates[last_index]
                yield f"data: {json.dumps(update)}\n\n"
                last_index += 1
                
                # Check if task is complete
                if update.get('status') in ['completed', 'failed']:
                    return
        
        await asyncio.sleep(0.1)
    
    yield f"data: {json.dumps({'status': 'timeout'})}\n\n"


# Health Check
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        opensearch_status = await opensearch_client.ping() if opensearch_client else False
        return {
            "status": "healthy" if opensearch_status else "degraded",
            "opensearch": "connected" if opensearch_status else "disconnected",
            "model": config['ollama']['model'],
            "hybrid_search": config['search']['hybrid']['enabled'],
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


# --- ENHANCED SEARCH ENDPOINT ---
@app.post("/search")
async def search(request: SearchRequest):
    """
    Enhanced semantic search with:
    - Conversational responses
    - Intent detection
    - Hybrid search (Vector + BM25)
    - Query expansion
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Search query: {request.query}")
        
        # 1. Intent Detection & Filter Building
        filters, intent, intent_desc = await _detect_intent(request.query)
        
        # 2. Generate query embedding
        query_embedding = await ingestion_pipeline.generate_embedding(request.query)
        
        # 3. Perform search (hybrid or vector-only)
        recall_top_k = config['search']['recall_top_k']
        
        if request.use_hybrid and config['search']['hybrid']['enabled']:
            logger.info("📚 Using hybrid search (Vector + BM25)")
            candidates = await opensearch_client.hybrid_search(
                query=request.query,
                query_vector=query_embedding,
                top_k=recall_top_k,
                filters=filters
            )
        else:
            candidates = await opensearch_client.vector_search(
                query_embedding,
                top_k=recall_top_k,
                filters=filters
            )
        
        logger.info(f"Retrieved {len(candidates)} candidates")
        
        # 4. Cross-encoder reranking
        if candidates:
            reranked = await reranker.rerank(
                query=request.query,
                documents=candidates,
                top_k=request.top_k
            )
            
            # 5. Generate conversational response
            response_message = _generate_response_message(
                query=request.query,
                results=reranked,
                intent=intent,
                intent_desc=intent_desc
            )
            
            search_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": response_message,
                "query": request.query,
                "intent": intent,
                "results": reranked,
                "count": len(reranked),
                "search_time": round(search_time, 3)
            }
        else:
            search_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": f"😕 I couldn't find any documents matching '{request.query}'. Try different keywords or check if documents are indexed.",
                "query": request.query,
                "intent": intent,
                "results": [],
                "count": 0,
                "search_time": round(search_time, 3)
            }
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _detect_intent(query: str) -> tuple:
    """
    Enhanced intent detection with more categories
    Returns: (filters, intent_type, intent_description)
    """
    q_lower = query.lower()
    
    # File type categories
    IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"]
    DOC_EXTS = [".pdf", ".docx", ".doc", ".txt", ".md"]
    SPREADSHEET_EXTS = [".xlsx", ".xls", ".csv"]
    
    # Document type keywords
    INTENT_PATTERNS = {
        "image": {
            "keywords": ["image", "photo", "picture", "pic", "screenshot", "diagram", 
                        "chart", "graph", "visual", "logo", "drawing"],
            "filter": {"terms": {"file_type": IMAGE_EXTS}},
            "desc": "images and visual content"
        },
        "document": {
            "keywords": ["document", "doc", "pdf", "file", "report", "article", 
                        "paper", "memo", "letter", "contract"],
            "filter": {"terms": {"file_type": DOC_EXTS}},
            "desc": "documents and text files"
        },
        "spreadsheet": {
            "keywords": ["spreadsheet", "excel", "csv", "table", "data", "sheet", 
                        "budget", "financial", "numbers"],
            "filter": {"terms": {"file_type": SPREADSHEET_EXTS}},
            "desc": "spreadsheets and data files"
        },
        "invoice": {
            "keywords": ["invoice", "bill", "receipt", "payment", "purchase"],
            "filter": {"term": {"document_type": "invoice"}},
            "desc": "invoices and receipts"
        },
        "report": {
            "keywords": ["report", "summary", "analysis", "review", "assessment"],
            "filter": {"term": {"document_type": "report"}},
            "desc": "reports and analyses"
        },
        "contract": {
            "keywords": ["contract", "agreement", "legal", "terms", "nda"],
            "filter": {"term": {"document_type": "contract"}},
            "desc": "contracts and agreements"
        }
    }
    
    # Check each intent pattern
    for intent_type, pattern in INTENT_PATTERNS.items():
        if any(kw in q_lower for kw in pattern["keywords"]):
            logger.info(f"🎯 Intent Detected: {intent_type.upper()}")
            return pattern["filter"], intent_type, pattern["desc"]
    
    # No specific intent - general search
    return None, "general", "all documents"


def _generate_response_message(
    query: str,
    results: List[Dict],
    intent: str,
    intent_desc: str
) -> str:
    """Generate a natural, conversational response message with personality"""

    count = len(results)

    if count == 0:
        # More empathetic no-results messages
        no_result_messages = [
            f"I've searched through all your {intent_desc}, but I couldn't find anything that matches '{query}'. Would you like to try different keywords?",
            f"Hmm, I didn't find any {intent_desc} related to '{query}'. Maybe try rephrasing your search?",
            f"I looked everywhere in your {intent_desc}, but nothing matched '{query}'. Let's try a broader search term!"
        ]
        import random
        return random.choice(no_result_messages)

    # Extract key topic from query
    topic = _extract_topic(query)

    # Build response based on count and intent - more conversational
    if count == 1:
        return f"Perfect! I found exactly what you're looking for - 1 document about **{topic}**:"
    elif count == 2:
        return f"Great! I found 2 documents that match your query about **{topic}**. Let me show you both:"
    elif count <= 5:
        return f"Nice! I discovered {count} relevant documents about **{topic}**. Here they are, ranked by relevance:"
    else:
        top_doc = results[0]['filename']
        return f"Excellent! I found {count} documents matching your search. The best match appears to be **{top_doc}**. Here are the top results:"


def _extract_topic(query: str) -> str:
    """Extract main topic from query for response"""
    stop_words = {
        'find', 'search', 'show', 'get', 'the', 'a', 'an', 'my', 'about', 
        'for', 'with', 'where', 'is', 'are', 'all', 'any', 'please', 'can',
        'you', 'me', 'i', 'want', 'need', 'looking', 'help', 'documents',
        'files', 'document', 'file'
    }
    
    words = query.lower().split()
    topic_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if topic_words:
        return ' '.join(topic_words[:4])
    return query


# Streaming Search Endpoint
@app.post("/search/stream")
async def search_stream(request: SearchRequest):
    """
    Search with streaming status updates.
    Returns Server-Sent Events for real-time status.
    """
    task_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    status_updates[task_id] = []
    
    # Start search in background
    asyncio.create_task(_perform_search_with_status(request, task_id))
    
    return StreamingResponse(
        stream_status(task_id),
        media_type="text/event-stream"
    )


async def _perform_search_with_status(request: SearchRequest, task_id: str):
    """Perform search with detailed thinking steps and status updates"""
    try:
        # Step 1: Understanding the query
        status_updates[task_id].append({
            "step": "analyzing",
            "message": "🤔 Let me understand what you're looking for...",
            "progress": 0.05,
            "thinking": "Analyzing query structure and intent"
        })
        await asyncio.sleep(0.2)

        # Step 2: Detect intent
        filters, intent, intent_desc = await _detect_intent(request.query)

        status_updates[task_id].append({
            "step": "intent",
            "message": f"🎯 I see you're looking for {intent_desc}",
            "intent": intent,
            "progress": 0.15,
            "thinking": f"Detected search intent: {intent}"
        })
        await asyncio.sleep(0.2)

        # Step 3: Generate semantic embedding
        status_updates[task_id].append({
            "step": "embedding",
            "message": "🧠 Converting your query into semantic vectors...",
            "progress": 0.25,
            "thinking": "Generating embeddings using nomic-embed-text model"
        })

        query_embedding = await ingestion_pipeline.generate_embedding(request.query)

        status_updates[task_id].append({
            "step": "embedding_done",
            "message": "✓ Query encoded into 768-dimensional vector space",
            "progress": 0.35,
            "thinking": "Embedding generated successfully"
        })
        await asyncio.sleep(0.2)

        # Step 4: Hybrid Search
        status_updates[task_id].append({
            "step": "searching",
            "message": "📚 Searching with hybrid vector + keyword matching...",
            "progress": 0.45,
            "thinking": "Combining semantic similarity and BM25 keyword search"
        })

        candidates = await opensearch_client.hybrid_search(
            query=request.query,
            query_vector=query_embedding,
            top_k=config['search']['recall_top_k'],
            filters=filters
        )

        status_updates[task_id].append({
            "step": "search_done",
            "message": f"✓ Found {len(candidates)} potential matches",
            "progress": 0.60,
            "thinking": f"Retrieved {len(candidates)} candidates from index"
        })
        await asyncio.sleep(0.2)

        # Step 5: Reranking with cross-encoder
        if candidates:
            status_updates[task_id].append({
                "step": "reranking",
                "message": f"⚡ Re-ranking {len(candidates)} results with AI...",
                "progress": 0.75,
                "thinking": "Using cross-encoder model for precise relevance scoring"
            })

            reranked = await reranker.rerank(
                query=request.query,
                documents=candidates,
                top_k=request.top_k
            )

            status_updates[task_id].append({
                "step": "rerank_done",
                "message": f"✓ Ranked top {len(reranked)} most relevant results",
                "progress": 0.90,
                "thinking": f"Reranking complete, top score: {reranked[0]['score']:.3f}"
            })
        else:
            reranked = []

        await asyncio.sleep(0.1)

        # Complete
        response_message = _generate_response_message(
            request.query, reranked, intent, intent_desc
        )
        
        status_updates[task_id].append({
            "step": "completed",
            "message": response_message,
            "status": "completed",
            "results": reranked,
            "count": len(reranked),
            "intent": intent,
            "progress": 1.0
        })
        
    except Exception as e:
        status_updates[task_id].append({
            "step": "error",
            "message": f"❌ Search failed: {str(e)}",
            "status": "failed",
            "error": str(e)
        })


# Indexing Endpoint
@app.post("/index")
async def start_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """Start indexing a directory with status feedback"""
    try:
        directory = Path(request.directory)
        if not directory.exists():
            raise HTTPException(status_code=404, detail="Directory not found")

        task_id = f"index_{directory.name}_{int(asyncio.get_event_loop().time())}"

        # Initialize ingestion status for this task
        ingestion_status[task_id] = []

        # Start ingestion in background
        background_tasks.add_task(
            ingestion_pipeline.process_directory,
            directory,
            task_id
        )

        # Start file watcher if requested
        if request.watch_mode:
            global watcher
            watcher = FileWatcher(config, ingestion_pipeline, opensearch_client)
            background_tasks.add_task(watcher.start, directory)

        return {
            "status": "success",
            "message": f"📂 Starting to index {directory.name}. This may take a moment...",
            "task_id": task_id,
            "directory": str(directory),
            "watch_mode": request.watch_mode
        }
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingestion Status Stream Endpoint
@app.get("/ingestion/status/{task_id}")
async def stream_ingestion_status(task_id: str):
    """Stream real-time ingestion status updates"""
    async def event_stream():
        last_index = 0
        timeout = 300  # 5 minute timeout
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            if task_id in ingestion_status:
                updates = ingestion_status[task_id]
                while last_index < len(updates):
                    update = updates[last_index]
                    yield f"data: {json.dumps(update)}\n\n"
                    last_index += 1

                    # Check if task is complete
                    if update.get('status') == 'completed':
                        yield f"data: {json.dumps({'status': 'done'})}\n\n"
                        return

            await asyncio.sleep(0.5)

        yield f"data: {json.dumps({'status': 'timeout'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Get current ingestion status (polling fallback)
@app.get("/ingestion/status")
async def get_ingestion_status():
    """Get current ingestion status for all active tasks"""
    active_tasks = {}
    for task_id, updates in ingestion_status.items():
        if updates:
            latest = updates[-1]
            if latest.get('status') != 'completed':
                active_tasks[task_id] = latest

    return {
        "status": "success",
        "active_tasks": active_tasks,
        "count": len(active_tasks)
    }


# Statistics Endpoint
@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = await opensearch_client.get_stats() if opensearch_client else {}
        return {
            "status": "success",
            "total_documents": stats.get("count", 0),
            "total_vectors": stats.get("count", 0),
            "watcher_active": 1 if watcher and watcher.is_running else 0,
            "hybrid_search_enabled": config['search']['hybrid']['enabled'],
            "avg_search_time": 0.15
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"status": "error", "message": str(e), "total_documents": 0}


# Cluster Visualization Endpoint
@app.get("/clusters")
async def get_clusters():
    """Get semantic clusters for visualization"""
    try:
        cluster_data = await opensearch_client.get_cluster_data()
        if not cluster_data:
            return {}
        return {"status": "success", **cluster_data}
    except Exception as e:
        logger.error(f"Cluster error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# A2A Message Endpoint
@app.post("/a2a/message")
async def handle_a2a_message(message: Dict[str, Any]):
    """Handle incoming A2A messages"""
    try:
        recipient = message.get('recipient', 'orchestrator')
        
        agent_map = {
            'orchestrator': orchestrator_agent,
            'search': search_agent,
            'ingestion': ingestion_agent,
            'conversation': conversation_agent
        }
        
        agent = agent_map.get(recipient)
        if agent:
            return await agent.handle_message(message)
        else:
            return {"error": f"Unknown recipient: {recipient}"}
            
    except Exception as e:
        logger.error(f"A2A message error: {e}")
        return {"error": str(e)}


# MCP Tool Endpoint
@app.post("/mcp/tools/{tool_name}")
async def execute_mcp_tool(tool_name: str, params: Dict[str, Any]):
    """Execute MCP tool"""
    try:
        result = await mcp_registry.execute(tool_name, params)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup Event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global opensearch_client, ingestion_pipeline, reranker
    global mcp_registry, search_agent, ingestion_agent
    global conversation_agent, orchestrator_agent
    
    logger.info("🚀 Starting LocalLens API v2.0...")
    
    # Initialize components
    opensearch_client = OpenSearchClient(config)
    ingestion_pipeline = IngestionPipeline(config, opensearch_client, ingestion_status_callback)
    reranker = CrossEncoderReranker(config)
    mcp_registry = MCPToolRegistry()
    
    # Initialize agents
    search_agent = SearchAgent(config)
    ingestion_agent = IngestionAgent(config)
    conversation_agent = ConversationAgent(config)
    orchestrator_agent = OrchestratorAgent(config)
    
    # Initialize OpenSearch index
    await opensearch_client.create_index()
    
    # Register MCP tools
    mcp_registry.register_tool("search_documents", search)
    mcp_registry.register_tool("index_directory", start_indexing)
    
    logger.info("✅ LocalLens API ready!")


# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down LocalLens API...")
    
    if watcher:
        watcher.stop()
    if opensearch_client:
        await opensearch_client.close()
    if ingestion_pipeline:
        await ingestion_pipeline.close()
    
    # Close agents
    for agent in [search_agent, ingestion_agent, conversation_agent, orchestrator_agent]:
        if agent:
            await agent.close()
    
    logger.info("LocalLens API stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
