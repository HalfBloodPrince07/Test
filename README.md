# 🔍 LocalLens v2.0 - Enhanced Semantic Document Search

An AI-powered semantic search engine for your local documents with **conversational responses**, **hybrid search**, and **agent orchestration**.

## ✨ What's New in v2.0

### 1. 🔀 Hybrid Search (Vector + BM25)
Combines dense vector search with traditional keyword matching for better retrieval:
- **Vector Search**: Finds semantically similar content
- **BM25 Search**: Catches exact keyword matches
- **Reciprocal Rank Fusion**: Combines scores optimally

```python
# Example: Search combines both methods
results = await opensearch_client.hybrid_search(
    query="construction invoice",
    query_vector=embedding,
    top_k=50
)
```

### 2. 💬 Conversational Responses
The agent now talks back like an assistant:

```
User: "Find invoices from the construction project"

LocalLens: "🔍 Found 3 documents matching your search. The most relevant is **Invoice_2024_Construction.pdf**:"

1. Invoice_2024_Construction.pdf (Score: 0.892)
   Invoice for construction materials dated March 2024...

2. Construction_Report.docx (Score: 0.756)
   Monthly report summarizing construction progress...
```

### 3. 🎯 Enhanced Intent Detection
Automatically detects what type of document you're looking for:

| Query | Detected Intent | Filter Applied |
|-------|----------------|----------------|
| "show me images of diagrams" | `image` | `.png, .jpg, .jpeg` |
| "find the contract" | `contract` | `document_type: contract` |
| "budget spreadsheets" | `spreadsheet` | `.xlsx, .csv` |

### 4. 🤖 A2A Agent Orchestration
Multi-agent system with proper coordination:

```
OrchestratorAgent
    ├── ConversationAgent (query understanding, response generation)
    ├── SearchAgent (semantic + hybrid search, reranking)
    └── IngestionAgent (document processing, indexing)
```

### 5. 📝 Improved Prompts

**Document Summarization:**
```
Analyze this document and provide:
1. **Summary**: 2-3 sentences capturing main purpose
2. **Keywords**: 5-10 searchable terms
3. **Entities**: Names, organizations, dates
```

**Image Captioning:**
```
Analyze this image comprehensively:
1. Main Subject
2. Objects & Elements
3. Text Content (transcribe if visible)
4. Visual Style
5. Context Clues
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                      │
│                   (Conversational UI)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Orchestrator│  │Conversation │  │   Search    │          │
│  │   Agent     │◄─┤   Agent     │  │   Agent     │          │
│  └──────┬──────┘  └─────────────┘  └──────┬──────┘          │
│         │                                  │                 │
│         │         ┌─────────────┐          │                 │
│         └────────►│ Ingestion   │◄─────────┘                 │
│                   │   Agent     │                            │
│                   └─────────────┘                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
┌───▼───┐          ┌──────▼──────┐       ┌──────▼──────┐
│Ollama │          │ OpenSearch  │       │Cross-Encoder│
│(Embed)│          │(Hybrid k-NN)│       │ (Reranker)  │
└───────┘          └─────────────┘       └─────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Services
```bash
docker-compose up -d
```

### 3. Pull Required Models (Ollama)
```bash
ollama pull nomic-embed-text
ollama pull qwen3-vl:4b
```

### 4. Run Backend
```bash
uvicorn backend.api:app --reload
```

### 5. Run Frontend
```bash
streamlit run app.py
```

## 📖 API Endpoints

### Search
```bash
POST /search
{
    "query": "find construction invoices",
    "top_k": 5,
    "use_hybrid": true
}
```

**Response:**
```json
{
    "status": "success",
    "message": "🔍 Found 3 documents about construction invoices:",
    "intent": "invoice",
    "results": [...],
    "search_time": 0.234
}
```

### Streaming Search (SSE)
```bash
POST /search/stream
```
Returns Server-Sent Events with status updates:
```
data: {"step": "analyzing", "message": "🔍 Analyzing your query...", "progress": 0.1}
data: {"step": "searching", "message": "📚 Searching through your documents...", "progress": 0.5}
data: {"step": "completed", "status": "completed", "results": [...]}
```

### Index Directory
```bash
POST /index
{
    "directory": "/path/to/documents",
    "watch_mode": true
}
```

## ⚙️ Configuration

Key settings in `config.yaml`:

```yaml
# Hybrid Search Weights
search:
  hybrid:
    enabled: true
    vector_weight: 0.7  # Semantic similarity
    bm25_weight: 0.3    # Keyword matching
  
  query_expansion:
    enabled: true       # Generate alternative queries

# Agent Status Messages
agent:
  status_messages:
    analyzing: "🔍 Analyzing your query..."
    searching: "📚 Searching through your documents..."
    reranking: "⚡ Ranking results by relevance..."
```

## 🧪 Example Queries

| Query | What Happens |
|-------|-------------|
| "Find all images with charts" | Filters to image files, searches for chart-related content |
| "What spreadsheets contain budget data?" | Filters to xlsx/csv, searches for budget keywords |
| "Show me contracts from 2024" | Filters to contract document type, uses date context |
| "construction site visit report" | General search with semantic + keyword matching |

## 📁 Project Structure

```
locallens_improved/
├── app.py                    # Streamlit frontend (conversational UI)
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── docker-compose.yml        # Docker services
│
└── backend/
    ├── __init__.py
    ├── api.py               # FastAPI with conversational responses
    ├── opensearch_client.py # Hybrid search implementation
    ├── ingestion.py         # Enhanced prompts & keyword extraction
    ├── reranker.py          # Cross-encoder with MMR diversity
    ├── a2a_agent.py         # Agent orchestration system
    ├── watcher.py           # Real-time file monitoring
    └── mcp_tools.py         # MCP tool registry
```

## 🔧 Key Improvements Summary

| Component | v1.0 | v2.0 |
|-----------|------|------|
| Search | Vector only | Hybrid (Vector + BM25) |
| Response | Raw results | Conversational messages |
| Intent | Basic | Multi-category detection |
| Prompts | Simple | Structured with keywords |
| Agents | Basic | Full A2A orchestration |
| Reranking | Score only | Score + MMR diversity |
| Status | None | Real-time streaming |

## 📝 License

MIT License - Built with ❤️ using Streamlit, FastAPI, and OpenSearch
