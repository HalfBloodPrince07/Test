# LocalLens V2 - Setup & Deployment Guide

## Quick Start (5 Minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start OpenSearch (Docker)
docker-compose up -d

# 3. Start Ollama with model
ollama pull qwen3-vl:4b

# 4. Initialize memory database (optional but recommended)
python -c "
from backend.memory import UserProfileManager
import asyncio

async def init():
    mgr = UserProfileManager()
    await mgr.initialize()
    print('✅ Memory database initialized!')

asyncio.run(init())
"

# 5. Start the API server
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# 6. Start the Streamlit UI (in another terminal)
streamlit run app.py
```

Access LocalLens at: `http://localhost:8501`

---

## Detailed Setup

### Prerequisites

#### Required
- **Python 3.9+**
- **Ollama** (for LLM operations)
- **OpenSearch** or **Elasticsearch**

#### Optional (Enhanced Features)
- **Redis** (for session memory - falls back to in-memory if unavailable)
- **Docker** (for easy OpenSearch deployment)

### Step 1: System Dependencies

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI, Streamlit (web frameworks)
- OpenSearch client
- Sentence Transformers (embeddings)
- LangGraph, LangChain (multi-agent orchestration)
- Redis, SQLAlchemy (memory systems)
- OpenTelemetry, Prometheus (observability)

#### Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com](https://ollama.com/download/windows)

**Pull required model:**
```bash
ollama pull qwen3-vl:4b
```

#### Install Redis (Optional)

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**Windows:**
Use Redis via WSL or Docker:
```bash
docker run -d -p 6379:6379 redis:latest
```

**Verify Redis:**
```bash
redis-cli ping  # Should return "PONG"
```

### Step 2: OpenSearch Setup

#### Option A: Docker (Recommended)

Use the provided `docker-compose.yml`:

```bash
docker-compose up -d
```

This starts OpenSearch with:
- Single-node cluster
- Security enabled (username: admin, password: LocalLens@1234)
- Port 9200
- Dashboard on port 5601

#### Option B: Manual Installation

Download from [opensearch.org](https://opensearch.org/downloads.html)

Configure `config/opensearch.yml`:
```yaml
network.host: 0.0.0.0
discovery.type: single-node
plugins.security.disabled: false
```

### Step 3: Initialize Memory System

The memory system uses SQLite for user profiles (no separate installation needed).

Initialize the database:

```python
# Run this Python script to create tables
from backend.memory import UserProfileManager, MemoryManager
import asyncio

async def initialize_memory():
    # Initialize user profile database
    profile_mgr = UserProfileManager()
    await profile_mgr.initialize()
    print("✅ User profile database created")

    # Initialize full memory manager (includes Redis connection test)
    memory = MemoryManager()
    await memory.initialize()
    print("✅ Memory Manager initialized")
    await memory.close()

asyncio.run(initialize_memory())
```

Save as `init_memory.py` and run:
```bash
python init_memory.py
```

### Step 4: Configure LocalLens

Edit `config.yaml` to match your environment:

```yaml
# Ollama settings
ollama:
  base_url: "http://localhost:11434"  # Change if Ollama runs elsewhere
  model: "qwen3-vl:4b"

# OpenSearch
opensearch:
  host: "localhost"
  port: 9200
  index_name: "locallens_index"
  auth:
    username: "admin"
    password: "LocalLens@1234"  # CHANGE THIS in production!

# Memory (Redis)
memory:
  session:
    backend: "redis"  # Use "memory" if Redis not available
    redis_url: "redis://localhost:6379"

# Enable/disable agents
agents:
  classifier:
    enabled: true
  clarification:
    enabled: true
  analysis:
    enabled: true
  summarization:
    enabled: true
  explanation:
    enabled: true
  critic:
    enabled: true
```

### Step 5: Start Services

#### Terminal 1: API Server

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Flags:
- `--reload`: Auto-restart on code changes (development)
- `--workers 4`: Multiple workers (production)

Verify: http://localhost:8000/health

#### Terminal 2: Streamlit Frontend

```bash
streamlit run app.py --server.port 8501
```

Verify: http://localhost:8501

### Step 6: Index Your Documents

1. Open LocalLens UI (http://localhost:8501)
2. Enter directory path (e.g., `C:\Users\YourName\Documents`)
3. Enable "Watch Mode" for auto-indexing new files
4. Click "Start Indexing"

Watch the ingestion status widget in bottom-right corner.

---

## Architecture Overview

### Services

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser                             │
│              http://localhost:8501                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Streamlit Frontend (app.py)                    │
│  - Chat interface                                           │
│  - Real-time status updates                                 │
│  - Result visualization                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP API calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            FastAPI Backend (backend/api.py)                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       Enhanced Orchestrator (LangGraph)             │   │
│  │  - Query classification                             │   │
│  │  - Agent coordination                               │   │
│  │  - Workflow management                              │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │                                        │
│  ┌─────────────────┴───────────────────────────────────┐   │
│  │           Specialized Agents                        │   │
│  │  - Clarification Agent                              │   │
│  │  - Analysis Agent                                   │   │
│  │  - Summarization Agent                              │   │
│  │  - Explanation Agent                                │   │
│  │  - Critic Agent (QA)                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Memory Manager                           │   │
│  │  - Session Memory (Redis)                           │   │
│  │  - User Profiles (SQLite)                           │   │
│  │  - Procedural Learning                              │   │
│  └──┬──────────────────────────────────────────────────┘   │
│     │                                                       │
└─────┼───────────────────────────────────────────────────────┘
      │
      │ Connects to external services
      │
      ├─► OpenSearch (9200) - Vector + Keyword Search
      ├─► Ollama (11434) - LLM Operations
      ├─► Redis (6379) - Session Memory [OPTIONAL]
      └─► SQLite - User Profiles (local file)
```

### Data Flow

1. **User Query** → Streamlit → FastAPI
2. **Orchestrator** loads memory context and user preferences
3. **Classifier** determines intent (document search, general knowledge, etc.)
4. **Router** dispatches to appropriate workflow:
   - Document Search → Search → Rerank → Explain → Quality Check
   - General Knowledge → LLM Direct Answer
   - Comparison → Search → Analysis Agent
   - Summarization → Search → Summarization Agent
5. **Critic Agent** evaluates result quality
6. **Memory Manager** records interaction for learning
7. **Response** returned to user via Streamlit

---

## Usage Examples

### Basic Document Search

```
User: "Find invoices from last month"

System Flow:
1. Classifier → DOCUMENT_SEARCH (invoice intent)
2. Extract filters: {document_type: "invoice", time_range: "last_month"}
3. Hybrid search (Vector + BM25)
4. Rerank top 50 → top 5
5. Explanation agent explains each result
6. Critic agent evaluates quality
7. Memory records search patterns

Response: "I found 3 invoice documents from last month: ..."
```

### General Knowledge Question

```
User: "What is machine learning?"

System Flow:
1. Classifier → GENERAL_KNOWLEDGE
2. Route to Ollama directly (no document search)
3. Generate answer
4. Return

Response: "Machine learning is a subset of artificial intelligence..."
```

### Document Comparison

```
User: "Compare contract A and contract B"

System Flow:
1. Classifier → COMPARISON
2. Search for both contracts
3. Analysis Agent compares documents
4. Extract similarities and differences
5. Return structured comparison

Response:
Similarities:
- Both are service contracts
- Same duration (12 months)

Differences:
- Contract A: $5000/month, Contract B: $7000/month
- Contract A includes support, Contract B does not
```

### Ambiguous Query Handling

```
User: "Show me that thing"

System Flow:
1. Classifier → CLARIFICATION_NEEDED (low confidence)
2. Clarification Agent generates questions:
   - "What type of document are you looking for?"
   - "Do you remember any keywords?"
3. User responds → Query refined → Search

Response: "I need clarification:
• What type of document are you looking for?
• Do you remember any keywords?"
```

---

## Memory & Learning

### Session Memory (Short-term)

Stores last 10 conversation turns per session. Enables:

- **Follow-up questions**: "Show me more like that"
- **Context continuity**: Remembers what "that" refers to
- **Topic tracking**: Knows if user is exploring invoices vs contracts

**Storage**: Redis (or in-memory fallback)
**TTL**: 1 hour (configurable)

### User Profiles (Long-term)

Tracks per user:
- Search history with intents and clicks
- Most accessed documents
- Frequently searched topics
- Usage patterns (peak hours, common intents)

**Storage**: SQLite database (`locallens_memory.db`)
**Persistence**: Permanent

### Procedural Learning

Learns optimal strategies per user:
- Best hybrid search weights (vector vs keyword)
- Whether reranking improves results
- Successful query reformulations
- Click position bias

**Storage**: In-memory with TTL
**Effect**: Search automatically adapts to each user

### Example: Personalization in Action

**User A** (developer):
- Frequently searches code files (.py, .js)
- Clicks top result 80% of time
- Peak usage: 9am-5pm

**Learned behavior**:
- Boost code file types in results
- Reranking is critical (high top-result click rate)
- Prefer keyword matching (searches for function names)

**User B** (researcher):
- Searches academic PDFs
- Explores many results (clicks positions 1-7)
- Peak usage: evenings

**Learned behavior**:
- Boost PDF documents
- Reranking less critical (explores broadly)
- Prefer semantic search (conceptual queries)

---

## Monitoring & Debugging

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "opensearch": "connected",
  "model": "qwen3-vl:4b",
  "hybrid_search": true
}
```

### Memory Status

Check user memory state:

```python
from backend.memory import MemoryManager
import asyncio

async def check_memory():
    memory = MemoryManager()
    await memory.initialize()

    summary = await memory.get_memory_summary(
        user_id="user123",
        session_id="session456"
    )

    print(summary)

asyncio.run(check_memory())
```

### Logs

Logs are output to stderr by default.

View real-time:
```bash
tail -f logs/locallens.log  # If logging to file
```

Log levels (in `config.yaml`):
```yaml
observability:
  logging:
    level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Agent Tracing

Enable detailed agent logging:

```yaml
observability:
  logging:
    agent_logging: true
    memory_logging: true
```

This shows:
- Which agents are invoked
- Decision reasoning
- Execution times
- Memory operations

---

## Performance Tuning

### Indexing Speed

```yaml
ingestion:
  max_workers: 4  # Increase for faster indexing (uses more CPU)
  chunk_size: 800  # Smaller = more chunks = slower but better precision
```

### Search Performance

```yaml
search:
  recall_top_k: 50  # Higher = better recall, slower
  rerank_top_k: 5   # Number of final results

performance:
  enable_caching: true
  cache_ttl: 3600
```

### Memory System

```yaml
memory:
  session:
    window_size: 10  # Fewer turns = less memory, less context
```

### Concurrent Requests

```yaml
performance:
  max_concurrent_requests: 10
  request_timeout: 120
```

For production, run with multiple Uvicorn workers:

```bash
uvicorn backend.api:app --workers 4 --host 0.0.0.0 --port 8000
```

---

## Troubleshooting

### Redis Connection Failed

**Error**: `Redis connection failed: Connection refused`

**Solution**:
1. Check if Redis is running: `redis-cli ping`
2. If not available, edit `config.yaml`:
   ```yaml
   memory:
     session:
       backend: "memory"  # Fallback to in-memory
   ```

### Ollama Model Not Found

**Error**: `Model 'qwen3-vl:4b' not found`

**Solution**:
```bash
ollama pull qwen3-vl:4b
ollama list  # Verify
```

### OpenSearch Connection Error

**Error**: `ConnectionError: Connection to opensearch failed`

**Solution**:
1. Check OpenSearch is running: `curl -k https://localhost:9200`
2. Verify credentials in `config.yaml`
3. Check Docker: `docker ps | grep opensearch`

### Out of Memory

**Error**: `MemoryError` or slow performance

**Solution**:
1. Reduce batch size:
   ```yaml
   models:
     cross_encoder:
       batch_size: 16  # Reduce from 32
   ```

2. Disable GPU if causing issues:
   ```yaml
   performance:
     enable_gpu: false
   ```

### Slow Searches

**Symptoms**: Searches take > 5 seconds

**Solutions**:
1. Reduce recall candidates:
   ```yaml
   search:
     recall_top_k: 20  # Reduce from 50
   ```

2. Enable caching:
   ```yaml
   performance:
     enable_caching: true
   ```

3. Check OpenSearch health:
   ```bash
   curl -k https://admin:LocalLens@1234@localhost:9200/_cluster/health
   ```

---

## Security Considerations

### Production Deployment

**DO NOT use default passwords!**

Change in `config.yaml`:
```yaml
opensearch:
  auth:
    username: "admin"
    password: "STRONG_PASSWORD_HERE"  # CHANGE THIS!
```

### Secure Redis

Add password protection:

```bash
# redis.conf
requirepass YOUR_STRONG_PASSWORD
```

Update config:
```yaml
memory:
  session:
    redis_url: "redis://:YOUR_STRONG_PASSWORD@localhost:6379"
```

### HTTPS/TLS

For production, use HTTPS:

```bash
uvicorn backend.api:app \
  --host 0.0.0.0 \
  --port 443 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

### User Authentication

Currently, LocalLens does not have user auth. For multi-user production:

1. Add FastAPI OAuth2/JWT authentication
2. Update API endpoints to require auth tokens
3. Streamlit can pass tokens via headers

(This is a future enhancement - see roadmap)

---

## Backup & Restore

### Backup User Data

```bash
# SQLite database
cp locallens_memory.db locallens_memory_backup_$(date +%Y%m%d).db

# OpenSearch indices
curl -X PUT "localhost:9200/_snapshot/my_backup" \
  -H 'Content-Type: application/json' \
  -d '{"type": "fs", "settings": {"location": "/backups"}}'
```

### Restore

```bash
# SQLite
cp locallens_memory_backup_20250129.db locallens_memory.db

# OpenSearch snapshot restore
curl -X POST "localhost:9200/_snapshot/my_backup/snapshot_1/_restore"
```

---

## Next Steps

After setup:

1. **Index your documents** via the UI
2. **Try example queries** to test intelligence
3. **Review memory dashboard** to see learning
4. **Customize agents** in `config.yaml`
5. **Monitor performance** via logs and health endpoint
6. **Enable observability** (OpenTelemetry, Prometheus) for production

For advanced usage, see: `IMPLEMENTATION_SUMMARY.md`

For development: `DEVELOPMENT_GUIDE.md` (TODO)

---

## Support & Contribution

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` folder
- **Community**: (Add Discord/forum link if available)

---

**Version**: 2.0-alpha
**Last Updated**: 2025-11-29
