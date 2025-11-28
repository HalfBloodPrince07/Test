# API Integration Example

This guide shows exactly how to integrate the new memory and agent systems into your existing `backend/api.py`.

## Step-by-Step Integration

### Step 1: Add Imports (Top of backend/api.py)

```python
# Add these imports after existing imports
from backend.memory import MemoryManager
from backend.agents import (
    QueryClassifier,
    ClarificationAgent,
    AnalysisAgent,
    SummarizationAgent,
    ExplanationAgent,
    CriticAgent
)
from backend.orchestration import EnhancedOrchestrator
from typing import Optional
import uuid
```

### Step 2: Add Global Variables

```python
# Add to global components section (around line 54)
memory_manager: Optional[MemoryManager] = None
enhanced_orchestrator: Optional[EnhancedOrchestrator] = None

# Agent instances (optional - orchestrator manages them)
query_classifier: Optional[QueryClassifier] = None
```

### Step 3: Update Startup Event

Replace the existing `startup_event()` function with this enhanced version:

```python
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup with enhanced features"""
    global opensearch_client, ingestion_pipeline, reranker
    global memory_manager, enhanced_orchestrator, query_classifier
    global mcp_registry, search_agent, ingestion_agent
    global conversation_agent, orchestrator_agent

    logger.info("🚀 Starting LocalLens API v2.0 with Agentic Features...")

    # === EXISTING COMPONENTS ===
    opensearch_client = OpenSearchClient(config)
    ingestion_pipeline = IngestionPipeline(config, opensearch_client, ingestion_status_callback)
    reranker = CrossEncoderReranker(config)
    mcp_registry = MCPToolRegistry()

    # Initialize existing agents
    search_agent = SearchAgent(config)
    ingestion_agent = IngestionAgent(config)
    conversation_agent = ConversationAgent(config)
    orchestrator_agent = OrchestratorAgent(config)

    # Initialize OpenSearch index
    await opensearch_client.create_index()

    # === NEW COMPONENTS ===
    logger.info("Initializing enhanced memory and agent systems...")

    # 1. Initialize Memory Manager
    try:
        memory_manager = MemoryManager(
            redis_url=config.get("memory", {}).get("session", {}).get("redis_url", "redis://localhost:6379"),
            database_url=config.get("memory", {}).get("user_profile", {}).get("database_url", "sqlite+aiosqlite:///locallens_memory.db")
        )
        await memory_manager.initialize()
        logger.info("✅ Memory Manager initialized")
    except Exception as e:
        logger.warning(f"Memory Manager initialization failed: {e}. Continuing without memory features.")
        memory_manager = None

    # 2. Initialize Query Classifier
    query_classifier = QueryClassifier(config)
    logger.info("✅ Query Classifier initialized")

    # 3. Create search function wrapper for orchestrator
    async def search_function(query: str, filters: Optional[Dict] = None, weights: Optional[Dict] = None):
        """Wrapper function for search to pass to orchestrator"""
        # Generate embedding
        query_embedding = await ingestion_pipeline.generate_embedding(query)

        # Use provided weights or defaults
        if weights:
            # Could override config weights here
            pass

        # Perform hybrid search
        candidates = await opensearch_client.hybrid_search(
            query=query,
            query_vector=query_embedding,
            top_k=config['search']['recall_top_k'],
            filters=filters
        )

        # Rerank
        if candidates:
            results = await reranker.rerank(
                query=query,
                documents=candidates,
                top_k=config['search']['rerank_top_k']
            )
            return results

        return []

    # 4. Initialize Enhanced Orchestrator
    if memory_manager:
        try:
            enhanced_orchestrator = EnhancedOrchestrator(
                config=config,
                memory_manager=memory_manager,
                search_function=search_function
            )
            logger.info("✅ Enhanced Orchestrator initialized with LangGraph")
        except Exception as e:
            logger.warning(f"Orchestrator initialization failed: {e}")
            enhanced_orchestrator = None

    # Register MCP tools (existing)
    mcp_registry.register_tool("search_documents", search)
    mcp_registry.register_tool("index_directory", start_indexing)

    logger.info("✅ LocalLens API ready with enhanced agentic features!")
```

### Step 4: Create Helper Function for Session Management

```python
def get_or_create_session(request_session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    if request_session_id:
        return request_session_id
    return str(uuid.uuid4())[:16]

def get_user_id(request_user_id: Optional[str] = None) -> str:
    """Get user ID from request or use default"""
    return request_user_id or "anonymous"
```

### Step 5: Update SearchRequest Model

```python
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    stream_status: bool = True
    # NEW: Add session and user tracking
    session_id: Optional[str] = None
    user_id: Optional[str] = None
```

### Step 6: Create Enhanced Search Endpoint

Add this NEW endpoint (or replace existing `/search`):

```python
@app.post("/search/enhanced")
async def enhanced_search(request: SearchRequest):
    """
    Enhanced search with memory, agents, and orchestration

    This is the NEW intelligent search endpoint that uses:
    - Memory for context and learning
    - Agents for specialized tasks
    - Orchestrator for workflow management
    """
    import time
    start_time = time.time()

    try:
        # Get or create session
        session_id = get_or_create_session(request.session_id)
        user_id = get_user_id(request.user_id)

        logger.info(f"🔍 Enhanced search: '{request.query}' (user: {user_id}, session: {session_id})")

        # Use orchestrator if available, otherwise fallback
        if enhanced_orchestrator and memory_manager:
            result = await enhanced_orchestrator.process_query(
                user_id=user_id,
                session_id=session_id,
                query=request.query
            )

            # Add session info to response
            result["session_id"] = session_id
            result["user_id"] = user_id

            return result

        else:
            # Fallback to original search
            logger.warning("Enhanced features not available, using standard search")
            return await search(request)

    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 7: Add Memory Endpoints

```python
@app.get("/memory/summary")
async def get_memory_summary(
    user_id: str,
    session_id: str
):
    """Get user memory summary"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        summary = await memory_manager.get_memory_summary(
            user_id=user_id,
            session_id=session_id
        )
        return {"status": "success", "summary": summary}

    except Exception as e:
        logger.error(f"Memory summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/preferences")
async def get_user_preferences(user_id: str):
    """Get personalized user preferences"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        prefs = await memory_manager.get_user_preferences(user_id)
        return {"status": "success", "preferences": prefs}

    except Exception as e:
        logger.error(f"Preferences error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        await memory_manager.session_memory.clear_session(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}

    except Exception as e:
        logger.error(f"Session clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 8: Add Specialized Agent Endpoints

```python
@app.post("/analyze/compare")
async def compare_documents(doc_ids: List[str], user_id: Optional[str] = None):
    """Compare multiple documents"""
    if not enhanced_orchestrator:
        raise HTTPException(status_code=503, detail="Analysis features not available")

    try:
        # Fetch documents by IDs
        documents = []
        for doc_id in doc_ids:
            # Query OpenSearch for document
            doc = await opensearch_client.get_document(doc_id)
            if doc:
                documents.append(doc)

        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 documents to compare")

        # Use analysis agent
        from backend.agents import AnalysisAgent
        analysis_agent = AnalysisAgent(config)

        comparison = await analysis_agent.compare_documents(documents)

        return {
            "status": "success",
            "comparison": comparison,
            "document_count": len(documents)
        }

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/summarize")
async def summarize_multiple(
    doc_ids: List[str],
    summary_type: str = "comprehensive"
):
    """Summarize multiple documents"""
    try:
        # Fetch documents
        documents = []
        for doc_id in doc_ids:
            doc = await opensearch_client.get_document(doc_id)
            if doc:
                documents.append(doc)

        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")

        # Use summarization agent
        from backend.agents import SummarizationAgent
        summarization_agent = SummarizationAgent(config)

        summary = await summarization_agent.summarize_documents(
            documents=documents,
            summary_type=summary_type
        )

        return {
            "status": "success",
            "summary": summary,
            "document_count": len(documents),
            "summary_type": summary_type
        }

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Step 9: Update Shutdown Event

```python
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down LocalLens API...")

    # Existing cleanup
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

    # NEW: Close memory manager
    if memory_manager:
        await memory_manager.close()
        logger.info("Memory Manager closed")

    logger.info("LocalLens API stopped")
```

## Testing the Integration

### 1. Test Enhanced Search

```bash
curl -X POST http://localhost:8000/search/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "find invoices from construction project",
    "user_id": "test_user_1",
    "session_id": "test_session_1"
  }'
```

Expected response:
```json
{
  "status": "success",
  "response_message": "I found 5 relevant documents...",
  "results": [...],
  "intent": "document_search",
  "quality_evaluation": {
    "quality_score": 0.85,
    "relevance": 0.90
  },
  "session_id": "test_session_1",
  "user_id": "test_user_1"
}
```

### 2. Test Memory Summary

```bash
curl http://localhost:8000/memory/summary?user_id=test_user_1&session_id=test_session_1
```

Expected response:
```json
{
  "status": "success",
  "summary": {
    "current_session": {
      "topic": "construction invoices",
      "recent_queries": ["find invoices from construction project"]
    },
    "user_statistics": {
      "total_queries": 1,
      "total_documents_accessed": 0
    }
  }
}
```

### 3. Test Document Comparison

```bash
curl -X POST http://localhost:8000/analyze/compare \
  -H "Content-Type: application/json" \
  -d '["doc_id_1", "doc_id_2"]'
```

### 4. Test Summarization

```bash
curl -X POST http://localhost:8000/analyze/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "doc_ids": ["doc_1", "doc_2", "doc_3"],
    "summary_type": "comprehensive"
  }'
```

## Streamlit Integration

Update `app.py` to use enhanced search:

```python
# In app.py, update search call:

# OLD:
# result = await api.search(query, top_k, use_hybrid)

# NEW:
async def enhanced_search_call(query: str, top_k: int, session_id: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{api.base_url}/search/enhanced",
            json={
                "query": query,
                "top_k": top_k,
                "session_id": session_id,
                "user_id": st.session_state.get("user_id", "anonymous")
            }
        )
        return response.json()

# Initialize session ID in session state
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:16]

# Use enhanced search
result = await enhanced_search_call(
    query=query,
    top_k=top_k,
    session_id=st.session_state.session_id
)
```

## Verification Checklist

After integration:

- [ ] API starts without errors
- [ ] Memory database file created (`locallens_memory.db`)
- [ ] `/health` endpoint works
- [ ] `/search/enhanced` returns results
- [ ] `/memory/summary` returns session data
- [ ] Subsequent searches show memory accumulation
- [ ] Quality evaluation appears in responses
- [ ] Intent classification works
- [ ] Session ID persists across requests

## Troubleshooting

### "MemoryManager initialization failed"

**Cause**: Redis not running or SQLite permission issue

**Fix**:
1. Check Redis: `redis-cli ping`
2. Or change config to use in-memory:
   ```yaml
   memory:
     session:
       backend: "memory"
   ```

### "Orchestrator not available"

**Cause**: LangGraph not installed

**Fix**:
```bash
pip install langgraph langchain langchain-core langchain-community
```

### "Search still uses old endpoint"

**Cause**: Frontend not updated

**Fix**: Update `app.py` as shown above

---

**Ready to integrate!** Start with Step 1 and work through sequentially.
