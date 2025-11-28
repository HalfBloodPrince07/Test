# LocalLens V2 - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start Required Services

#### Start OpenSearch (Docker)
```bash
docker-compose up -d
```

#### Start Ollama
```bash
# If not already running
ollama serve

# Pull the model (in another terminal)
ollama pull qwen3-vl:4b
```

#### Start Redis (Optional - for session memory)
```bash
# macOS/Linux
redis-server

# Windows (Docker)
docker run -d -p 6379:6379 redis:latest

# Verify
redis-cli ping  # Should return "PONG"
```

### Step 3: Initialize Memory System

```bash
python init_memory.py
```

You should see:
```
✅ User profile database created (locallens_memory.db)
✅ Memory Manager initialized successfully
✅ Session memory test successful
```

### Step 4: Start LocalLens

#### Terminal 1: Start API
```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

Wait for:
```
✅ Memory Manager initialized
✅ Query Classifier initialized
✅ Enhanced Orchestrator initialized with LangGraph
✅ LocalLens API ready with enhanced agentic features!
```

#### Terminal 2: Start UI
```bash
streamlit run app.py
```

### Step 5: Open and Use

1. Open browser: **http://localhost:8501**
2. Click "🚀 Start Indexing" to index your documents
3. Try a search: "find invoices from last month"
4. See the magic! ✨

---

## 🧪 Test the Enhanced Features

### Test 1: Basic Search with Memory
```
1. Search: "machine learning papers"
2. Check sidebar "Memory & Session" - you'll see session ID
3. Search again: "neural networks"
4. The system remembers your context!
```

### Test 2: Quality Indicators
```
1. Search any query
2. Look for quality score in results
3. See suggestions if results are low quality
```

### Test 3: API Endpoints

**Health check:**
```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "opensearch": "connected",
  "model": "qwen3-vl:4b",
  "memory_system": "available",
  "enhanced_orchestrator": "available"
}
```

**Enhanced search:**
```bash
curl -X POST http://localhost:8000/search/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "find invoices",
    "user_id": "test_user",
    "session_id": "test_session_123"
  }'
```

**Memory summary:**
```bash
curl "http://localhost:8000/memory/summary?user_id=test_user&session_id=test_session_123"
```

---

## 🔧 Troubleshooting

### "Memory Manager initialization failed"

**Cause**: Redis not running

**Solution**:
- Option 1: Start Redis (see Step 2)
- Option 2: System will automatically use in-memory fallback (works fine!)

### "Orchestrator initialization failed"

**Cause**: LangGraph not installed

**Solution**:
```bash
pip install langgraph langchain langchain-core langchain-community
```

### "OpenSearch connection error"

**Cause**: OpenSearch not running

**Solution**:
```bash
docker-compose up -d
# Wait 30 seconds for startup
curl -k https://admin:LocalLens@1234@localhost:9200
```

### "Model not found"

**Cause**: Ollama model not pulled

**Solution**:
```bash
ollama pull qwen3-vl:4b
ollama list  # Verify
```

---

## 🎯 Features to Try

### 1. **Conversational Search**
```
You: "Find construction documents"
→ System classifies intent, searches, explains results

You: "Compare the top two"
→ System remembers context, compares documents
```

### 2. **Ambiguity Handling**
```
You: "Show me that report"
→ System: "Which report are you referring to?"
```

### 3. **Quality Control**
```
Every search shows:
- Quality score (0-100%)
- Confidence level
- Improvement suggestions
```

### 4. **Learning & Adaptation**
```
After 5-10 searches:
- System learns your preferences
- Optimizes search weights
- Boosts document types you prefer
```

### 5. **Session Memory**
```
All within same session:
- Context is maintained
- Related queries linked
- Follow-up questions work
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Enable/disable features
agents:
  classifier: {enabled: true}
  clarification: {enabled: true}
  analysis: {enabled: true}

# Memory settings
memory:
  session:
    backend: "redis"  # or "memory"
    window_size: 10

# Search tuning
search:
  hybrid:
    vector_weight: 0.7
    bm25_weight: 0.3
```

---

## 📊 Monitoring

### Check System Status

In Streamlit sidebar:
- Click "🔄 Refresh Status"
- See all component statuses

### View Memory

In sidebar "🧠 Memory & Session":
- Session ID
- Query count
- Create new session

### API Health

```bash
# Quick health check
curl http://localhost:8000/health | jq

# Check logs
# API logs show in Terminal 1
# UI logs show in Terminal 2
```

---

## 🎓 Next Steps

1. **Index Your Documents**: Use sidebar to index your actual documents
2. **Try Complex Queries**: "Compare all Q4 reports"
3. **Check Memory**: See how system learns from your searches
4. **Read Full Docs**: See `SETUP_GUIDE.md` for advanced features

---

## 💡 Tips

1. **Enhanced Search ON**: Keep the "🤖 Enhanced Search" checkbox enabled in sidebar
2. **Session Management**: Use "🔄 New Session" to start fresh
3. **Quality Matters**: Pay attention to quality scores - system learns from this
4. **Be Specific**: More specific queries = better results

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | This file - get started fast |
| `SETUP_GUIDE.md` | Complete installation guide |
| `IMPLEMENTATION_SUMMARY.md` | Technical architecture |
| `INTEGRATION_EXAMPLE.md` | API integration details |
| `COMPLETION_STATUS.md` | Implementation status |

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] API starts without errors
- [ ] Health check returns "healthy"
- [ ] Memory system shows "available"
- [ ] Streamlit UI loads
- [ ] Can index a directory
- [ ] Search returns results
- [ ] Session ID appears in sidebar
- [ ] Enhanced search works

---

**Ready to search intelligently!** 🚀

For issues or questions, see `SETUP_GUIDE.md` troubleshooting section.
