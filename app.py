# app.py - Enhanced Streamlit Frontend with Conversational UI

import streamlit as st
import httpx
import asyncio
import plotly.graph_objects as go
import yaml
from pathlib import Path
from typing import List, Dict, Any
import time
import subprocess
import platform
import json

# Page Configuration
st.set_page_config(
    page_title="LocalLens - AI Document Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Configuration
# Load Configuration
# CHANGE: Add encoding="utf-8" here
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# API Client
class LocalLensAPI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def health_check(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.json()
            except Exception as e:
                return {"status": "error", "message": str(e)}
    
    async def search(self, query: str, top_k: int = 5, use_hybrid: bool = True) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={
                        "query": query, 
                        "top_k": top_k,
                        "use_hybrid": use_hybrid
                    }
                )
                return response.json()
            except Exception as e:
                return {"status": "error", "message": str(e)}
    
    async def search_stream(self, query: str, top_k: int = 5):
        """Stream search with status updates"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/search/stream",
                    json={"query": query, "top_k": top_k, "use_hybrid": True}
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            yield data
            except Exception as e:
                yield {"status": "error", "message": str(e)}
    
    async def start_indexing(self, directory: str, watch_mode: bool = False) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/index",
                    json={"directory": directory, "watch_mode": watch_mode}
                )
                return response.json()
            except Exception as e:
                return {"status": "error", "message": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{self.base_url}/stats")
                return response.json()
            except Exception:
                return {}
    
    async def get_clusters(self) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(f"{self.base_url}/clusters")
                return response.json()
            except Exception:
                return {}

    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/ingestion/status")
                return response.json()
            except Exception:
                return {"status": "error", "active_tasks": {}, "count": 0}


api = LocalLensAPI()


# Custom CSS
st.markdown("""
<style>
    /* Chat-like search interface */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background: #f0f2f6;
        color: #1f2937;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 90%;
    }
    
    .status-message {
        background: #fef3c7;
        color: #92400e;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .search-result {
        background: white;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #21808d;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .search-result:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .result-title {
        font-size: 16px;
        font-weight: 600;
        color: #1f2937;
    }
    
    .result-score {
        display: inline-block;
        background: linear-gradient(135deg, #21808d 0%, #1d7480 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .result-summary {
        color: #4b5563;
        font-size: 14px;
        line-height: 1.6;
        margin: 10px 0;
    }
    
    .result-path {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 12px;
        color: #9ca3af;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .intent-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #21808d 0%, #1d7480 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 32px;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 13px;
        opacity: 0.9;
    }
    
    /* Progress animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .thinking {
        animation: pulse 1.5s ease-in-out infinite;
    }

    /* Floating status widget - Minimalist Design */
    .status-widget {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding: 8px 16px;
        min-width: 180px;
        max-width: 220px;
        z-index: 1000;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .status-widget:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateY(-2px);
        max-width: 320px;
        padding: 12px 20px;
    }

    .status-widget.processing {
        border: 1.5px solid #f59e0b;
    }

    .status-widget.completed {
        border: 1.5px solid #10b981;
    }

    .status-widget.error {
        border: 1.5px solid #ef4444;
    }

    .status-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }

    .status-icon {
        font-size: 16px;
        line-height: 1;
    }

    .status-title {
        font-weight: 500;
        font-size: 13px;
        color: #374151;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex: 1;
    }

    .status-percentage {
        font-size: 12px;
        font-weight: 600;
        color: #6b7280;
        min-width: 38px;
        text-align: right;
    }

    .status-body {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 6px;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease;
    }

    .status-widget:hover .status-body {
        max-height: 50px;
    }

    .status-progress {
        width: 100%;
        height: 3px;
        background: #f3f4f6;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 6px;
    }

    .status-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #21808d, #1d7480);
        transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: progressShine 2s ease-in-out infinite;
    }

    @keyframes progressShine {
        0% { opacity: 0.9; }
        50% { opacity: 1; }
        100% { opacity: 0.9; }
    }

    .status-widget.completed .status-progress-bar {
        background: linear-gradient(90deg, #10b981, #059669);
    }

    .status-widget.error .status-progress-bar {
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }

    .status-file {
        font-size: 10px;
        color: #9ca3af;
        font-family: 'Monaco', 'Menlo', monospace;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-height: 0;
        transition: max-height 0.3s ease;
    }

    .status-widget:hover .status-file {
        max-height: 20px;
    }

    /* Real-time progress bar for search */
    .progress-container {
        width: 100%;
        height: 4px;
        background: #f3f4f6;
        border-radius: 2px;
        overflow: hidden;
        margin: 10px 0;
    }

    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #21808d, #1d7480);
        transition: width 0.3s ease;
        animation: shimmer 1.5s infinite;
    }

    @keyframes shimmer {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }

    /* Auto-scroll to input */
    .search-anchor {
        scroll-margin-top: 100px;
    }
</style>

<script>
// Auto-scroll to search input after results
function scrollToSearch() {
    const searchInput = document.querySelector('[data-testid="stTextInput"]');
    if (searchInput) {
        searchInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}
</script>
""", unsafe_allow_html=True)


# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'show_ingestion_status' not in st.session_state:
    st.session_state.show_ingestion_status = False
if 'last_stats_update' not in st.session_state:
    st.session_state.last_stats_update = time.time()


# Sidebar Configuration
with st.sidebar:
    st.title("🔍 LocalLens")
    st.caption("AI-Powered Document Search")
    
    st.divider()
    
    # Directory Selection
    st.subheader("📁 Document Source")
    target_dir = st.text_input(
        "Directory to Index",
        value=str(Path.home() / "Documents"),
        help="Select the directory containing your documents"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        watch_mode = st.checkbox("🔄 Watch", value=False, help="Auto-index new files")
    with col2:
        use_hybrid = st.checkbox("🔀 Hybrid", value=True, help="Use hybrid search")
    
    if st.button("🚀 Start Indexing", use_container_width=True):
        with st.spinner("Starting..."):
            result = asyncio.run(api.start_indexing(target_dir, watch_mode))
            if result.get("status") == "success":
                st.success(result.get("message", "Indexing started!"))
                st.session_state.show_ingestion_status = True
                time.sleep(1)
                st.rerun()  # Refresh to show status widget
            else:
                st.error(result.get("message", "Failed to start indexing"))
    
    st.divider()
    
    # Search Settings
    st.subheader("⚙️ Search Settings")
    top_k = st.slider("Results to show", 1, 10, 5)
    
    st.divider()
    
    # System Status
    st.subheader("📊 System Status")
    
    if st.button("🔄 Refresh Status"):
        with st.spinner("Checking..."):
            health = asyncio.run(api.health_check())
            if health.get("status") == "healthy":
                st.success("✅ All systems operational")
                st.caption(f"Model: {health.get('model', 'N/A')}")
                st.caption(f"Hybrid Search: {'✓' if health.get('hybrid_search') else '✗'}")
            else:
                st.error(f"❌ {health.get('message', 'Connection error')}")


# Main Content
st.title("🔍 LocalLens")
st.markdown("*Search your documents using natural language*")

# Statistics Dashboard
st.subheader("📊 Overview")
col1, col2, col3, col4 = st.columns(4)

try:
    stats = asyncio.run(api.get_stats())
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('total_documents', 0)}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('total_vectors', 0)}</div>
            <div class="stat-label">Vectors</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        hybrid_status = "✓" if stats.get('hybrid_search_enabled') else "✗"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{hybrid_status}</div>
            <div class="stat-label">Hybrid Search</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('avg_search_time', 0.15):.2f}s</div>
            <div class="stat-label">Avg Search</div>
        </div>
        """, unsafe_allow_html=True)

except Exception:
    st.info("📊 Connect to backend to see statistics")

st.divider()

# Search Interface
st.subheader("💬 Ask LocalLens")

# Chat-like interface
chat_container = st.container()

# Create anchor point for auto-scroll
search_input_anchor = st.empty()

# Search input
query = st.text_input(
    "What would you like to find?",
    placeholder="e.g., Find the invoice from last month's construction project",
    label_visibility="collapsed",
    key="search_input"
)

# Add invisible anchor element
search_input_anchor.markdown('<div id="search-input-anchor"></div>', unsafe_allow_html=True)

# Search button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_clicked = st.button("🔍 Search", use_container_width=True)
with col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

# Perform search with streaming
if query and (search_clicked or query != st.session_state.last_query):
    st.session_state.last_query = query

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    # Show status updates with streaming
    status_placeholder = st.empty()
    thinking_placeholder = st.empty()
    results_placeholder = st.empty()

    async def perform_search_with_streaming():
        """Perform search with real-time streaming updates"""
        statuses = []
        thinking_steps = []

        try:
            # Use the streaming endpoint
            async for update in api.search_stream(query, top_k):
                if "step" in update:
                    step = update.get("step")
                    message = update.get("message", "")
                    progress = update.get("progress", 0)

                    # Show thinking process
                    thinking_steps.append(message)
                    thinking_placeholder.markdown(f"""
                    <div class="status-message">
                        <span class="thinking">{message}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Update status
                    if step in ["analyzing", "searching", "reranking"]:
                        status_placeholder.markdown(f"""
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {int(progress * 100)}%"></div>
                        </div>
                        """, unsafe_allow_html=True)

                    await asyncio.sleep(0.1)

                elif update.get("status") == "completed":
                    # Clear status messages
                    status_placeholder.empty()
                    thinking_placeholder.empty()
                    return update

            # Fallback to non-streaming if stream fails
            return await api.search(query, top_k, use_hybrid)

        except Exception as e:
            logger.error(f"Search streaming error: {e}")
            # Fallback to regular search
            return await api.search(query, top_k, use_hybrid)

    # Execute search
    start_time = time.time()
    results = asyncio.run(perform_search_with_streaming())
    search_time = time.time() - start_time
    
    status_placeholder.empty()
    
    if results.get("status") == "success":
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": results.get("message", "Here are your results:"),
            "results": results.get("results", []),
            "intent": results.get("intent", "general"),
            "search_time": results.get("search_time", search_time)
        })
    else:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"❌ {results.get('message', 'Search failed')}",
            "results": [],
            "error": True
        })

    st.rerun()


# Display chat history
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                🔍 {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        
        elif msg["role"] == "assistant":
            # Show assistant message
            intent = msg.get('intent', 'general')
            search_time = msg.get('search_time', 0)
            
            st.markdown(f"""
            <div class="assistant-message">
                <div style="margin-bottom: 10px;">
                    {msg['content']}
                </div>
                <div style="display: flex; gap: 10px; font-size: 12px; color: #6b7280;">
                    <span class="intent-badge">{intent}</span>
                    <span>⏱️ {search_time:.2f}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show results
            results = msg.get('results', [])
            if results:
                for idx, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    score_pct = min(score * 100, 100) if score <= 1 else score
                    
                    st.markdown(f"""
                    <div class="search-result">
                        <div class="result-header">
                            <span class="result-title">{idx}. {result['filename']}</span>
                            <span class="result-score">Score: {score:.3f}</span>
                        </div>
                        <div class="result-summary">
                            {result.get('content_summary', 'No summary available.')[:300]}...
                        </div>
                        <div class="result-path">
                            📂 {result['file_path']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Open file button - use unique hash from file path and position
                    import hashlib
                    unique_key = hashlib.md5(f"{result['file_path']}_{idx}_{len(st.session_state.chat_history)}_{time.time()}".encode()).hexdigest()[:16]

                    if st.button(f"📂 Open", key=f"open_{unique_key}"):
                        file_path = result['file_path']
                        try:
                            if platform.system() == 'Windows':
                                subprocess.Popen(['start', file_path], shell=True)
                            elif platform.system() == 'Darwin':
                                subprocess.Popen(['open', file_path])
                            else:
                                subprocess.Popen(['xdg-open', file_path])
                        except Exception as e:
                            st.error(f"Could not open file: {e}")

st.divider()

# Example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    Try these natural language queries:
    
    - **"Find invoices from construction project"** - Searches for invoice documents
    - **"Show me all images with diagrams"** - Filters to image files
    - **"What spreadsheets contain budget data?"** - Finds Excel/CSV files
    - **"Find contracts or agreements"** - Searches for legal documents
    - **"Show screenshots from last week"** - Looks for screenshot images
    
    The AI will automatically detect your intent and filter results accordingly!
    """)

# Cluster Visualization
with st.expander("📊 Document Clusters"):
    st.markdown("*Visualize how your documents are semantically grouped*")
    
    if st.button("Generate Visualization"):
        with st.spinner("Generating clusters..."):
            try:
                cluster_data = asyncio.run(api.get_clusters())
                
                if cluster_data and cluster_data.get("x"):
                    fig = go.Figure()
                    
                    unique_clusters = sorted(list(set(cluster_data['cluster_ids'])))
                    
                    for cluster_id in unique_clusters:
                        indices = [i for i, x in enumerate(cluster_data['cluster_ids']) if x == cluster_id]
                        
                        fig.add_trace(go.Scatter(
                            x=[cluster_data['x'][i] for i in indices],
                            y=[cluster_data['y'][i] for i in indices],
                            mode='markers',
                            name=f'Cluster {cluster_id}',
                            text=[cluster_data['labels'][i] for i in indices],
                            hoverinfo='text',
                            marker=dict(size=12, opacity=0.8)
                        ))
                    
                    fig.update_layout(
                        title="Document Semantic Clusters",
                        xaxis_title="Dimension 1",
                        yaxis_title="Dimension 2",
                        height=500,
                        hovermode='closest',
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No cluster data available. Index some documents first.")
            
            except Exception as e:
                st.error(f"❌ Visualization error: {str(e)}")

# Floating Ingestion Status Widget
status_placeholder = st.empty()

# Check for active ingestion tasks
try:
    ingestion_status_data = asyncio.run(api.get_ingestion_status())
    active_tasks = ingestion_status_data.get("active_tasks", {})

    if active_tasks:
        # Get the first active task
        task_id, task_info = list(active_tasks.items())[0]
        status = task_info.get("status", "processing")
        message = task_info.get("message", "Processing files...")
        current_file = task_info.get("current_file", "")
        processed = task_info.get("processed", 0)
        total_files = task_info.get("total_files", 0)

        # Calculate progress percentage
        progress_pct = (processed / total_files * 100) if total_files > 0 else 0

        # Determine status class
        status_class = "processing"
        icon = "⚙️"
        if status == "completed":
            status_class = "completed"
            icon = "✅"
        elif status == "error":
            status_class = "error"
            icon = "❌"
        elif status == "started":
            status_class = "processing"
            icon = "🔍"

        # Render the minimalist status widget with auto-refresh
        status_placeholder.markdown(f"""
        <div class="status-widget {status_class}" id="status-widget">
            <div class="status-header">
                <span class="status-icon">{icon}</span>
                <span class="status-title">Processing</span>
                <span class="status-percentage">{int(progress_pct)}%</span>
            </div>
            <div class="status-progress">
                <div class="status-progress-bar" style="width: {progress_pct}%"></div>
            </div>
            <div class="status-body">
                {processed} of {total_files} files processed
            </div>
            {f'<div class="status-file">{current_file}</div>' if current_file else ''}
        </div>

        <script>
        (function() {{
            // Real-time SSE updates for ingestion progress
            if ({str(status in ["processing", "started"]).lower()}) {{
                const taskId = '{task_id}';
                let lastProgress = {int(progress_pct)};

                // Try SSE first for real-time updates
                const eventSource = new EventSource(`http://localhost:8000/ingestion/status/${{taskId}}`);

                eventSource.onmessage = function(event) {{
                    try {{
                        const data = JSON.parse(event.data);

                        if (data.status === 'done' || data.status === 'completed') {{
                            eventSource.close();
                            setTimeout(() => window.location.reload(), 2000);
                            return;
                        }}

                        const processed = data.processed || 0;
                        const totalFiles = data.total_files || 1;
                        const progressPct = (processed / totalFiles * 100);
                        const currentFile = data.current_file || '';

                        // Only update if progress changed
                        if (Math.abs(progressPct - lastProgress) >= 0.5) {{
                            lastProgress = progressPct;

                            const widget = document.getElementById('status-widget');
                            if (widget) {{
                                const percentage = widget.querySelector('.status-percentage');
                                const progressBar = widget.querySelector('.status-progress-bar');
                                const statusBody = widget.querySelector('.status-body');
                                const statusFile = widget.querySelector('.status-file');

                                if (percentage) percentage.textContent = Math.round(progressPct) + '%';
                                if (progressBar) {{
                                    progressBar.style.width = progressPct + '%';
                                    progressBar.style.transition = 'width 0.5s ease';
                                }}
                                if (statusBody) statusBody.textContent = processed + ' of ' + totalFiles + ' files processed';
                                if (statusFile) statusFile.textContent = currentFile;
                            }}
                        }}
                    }} catch (error) {{
                        console.error('SSE parse error:', error);
                    }}
                }};

                eventSource.onerror = function(error) {{
                    console.error('SSE connection error, falling back to polling');
                    eventSource.close();

                    // Fallback to polling
                    let pollCount = 0;
                    const poll = async () => {{
                        if (pollCount++ >= 60) return; // 2 min max

                        try {{
                            const response = await fetch('http://localhost:8000/ingestion/status');
                            const data = await response.json();
                            const activeTasks = data.active_tasks || {{}};

                            if (Object.keys(activeTasks).length > 0) {{
                                const taskInfo = activeTasks[Object.keys(activeTasks)[0]];
                                const processed = taskInfo.processed || 0;
                                const totalFiles = taskInfo.total_files || 1;
                                const progressPct = (processed / totalFiles * 100);
                                const currentFile = taskInfo.current_file || '';

                                const widget = document.getElementById('status-widget');
                                if (widget) {{
                                    const percentage = widget.querySelector('.status-percentage');
                                    const progressBar = widget.querySelector('.status-progress-bar');
                                    const statusBody = widget.querySelector('.status-body');
                                    const statusFile = widget.querySelector('.status-file');

                                    if (percentage) percentage.textContent = Math.round(progressPct) + '%';
                                    if (progressBar) progressBar.style.width = progressPct + '%';
                                    if (statusBody) statusBody.textContent = processed + ' of ' + totalFiles + ' files processed';
                                    if (statusFile) statusFile.textContent = currentFile;
                                }}

                                setTimeout(poll, 2000);
                            }} else {{
                                window.location.reload();
                            }}
                        }} catch (err) {{
                            console.error('Polling failed:', err);
                            setTimeout(poll, 2000);
                        }}
                    }};

                    setTimeout(poll, 1000);
                }};
            }}
        }})();
        </script>
        """, unsafe_allow_html=True)

except Exception as e:
    pass  # Silently fail if status check fails

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 12px;">
    LocalLens v2.0 | AI-Powered Semantic Search<br>
    Built with Streamlit, FastAPI, and OpenSearch
</div>

<script>
// Auto-scroll to search input after chat updates (like Claude UI)
(function() {
    // Wait for page to fully load
    window.addEventListener('load', function() {
        setTimeout(function() {
            scrollToSearchInput();
        }, 300);
    });

    // Also trigger on any mutation (when chat history updates)
    const observer = new MutationObserver(function(mutations) {
        // Only scroll if there are chat messages
        const chatMessages = document.querySelectorAll('.user-message, .assistant-message');
        if (chatMessages.length > 0) {
            setTimeout(scrollToSearchInput, 200);
        }
    });

    // Observe the entire document for changes
    setTimeout(function() {
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }, 100);

    function scrollToSearchInput() {
        // Find the search input
        const searchInput = document.querySelector('[data-testid="stTextInput"] input');
        const anchor = document.getElementById('search-input-anchor');

        if (searchInput) {
            // Scroll to the input field
            searchInput.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
                inline: 'nearest'
            });

            // Focus the input (like Claude)
            setTimeout(() => {
                searchInput.focus();
            }, 500);
        } else if (anchor) {
            // Fallback to anchor
            anchor.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    }

    // Expose globally for manual triggering
    window.scrollToSearchInput = scrollToSearchInput;
})();
</script>
""", unsafe_allow_html=True)
