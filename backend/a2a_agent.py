# backend/a2a_agent.py - Enhanced Agent-to-Agent Communication

from typing import Dict, Any, Optional, List, Callable
import httpx
from loguru import logger
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
import asyncio
from dataclasses import dataclass, field


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageType(Enum):
    """A2A message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    STATUS_UPDATE = "status_update"


@dataclass
class A2AMessage:
    """Enhanced A2A protocol message"""
    method: str
    params: Dict[str, Any]
    msg_type: MessageType = MessageType.REQUEST
    id: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id or f"msg_{hash(str(self.params))}",
            "type": self.msg_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class Task:
    """Represents a task in the orchestration system"""
    id: str
    name: str
    agent: str
    params: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class A2AAgent:
    """Enhanced base A2A Agent with better communication"""
    
    def __init__(self, name: str, endpoint: str, config: Dict[str, Any]):
        self.name = name
        self.endpoint = endpoint
        self.config = config
        self.client = httpx.AsyncClient(timeout=60.0)
        self._handlers: Dict[str, Callable] = {}
        self._status_callback: Optional[Callable] = None
        
        logger.info(f"A2A Agent initialized: {name}")
    
    def set_status_callback(self, callback: Callable):
        """Set callback for status updates (for UI feedback)"""
        self._status_callback = callback
    
    async def emit_status(self, message: str, progress: Optional[float] = None):
        """Emit status update for UI feedback"""
        if self._status_callback:
            await self._status_callback({
                "agent": self.name,
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            })
        logger.info(f"[{self.name}] {message}")
    
    def register_handler(self, method: str, handler: Callable):
        """Register a method handler"""
        self._handlers[method] = handler
        logger.debug(f"Registered handler for method: {method}")
    
    async def send_message(
        self,
        to: str,
        method: str,
        params: Dict[str, Any],
        msg_type: MessageType = MessageType.REQUEST,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Send A2A message to another agent"""
        
        message = A2AMessage(
            method=method,
            params=params,
            msg_type=msg_type,
            sender=self.name,
            recipient=to,
            metadata=metadata or {}
        )
        
        # Get target endpoint
        if to in self.config['a2a']['agents']:
            target_endpoint = self.config['a2a']['agents'][to]['endpoint']
        else:
            logger.error(f"Unknown agent: {to}")
            return {"error": f"Unknown agent: {to}"}
        
        try:
            response = await self.client.post(
                f"{target_endpoint}/a2a/message",
                json=message.to_dict()
            )
            return response.json()
        except Exception as e:
            logger.error(f"A2A message error: {e}")
            return {"error": str(e)}
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming A2A message"""
        method = message.get('method')
        params = message.get('params', {})
        
        if method in self._handlers:
            try:
                result = await self._handlers[method](params)
                return {
                    "jsonrpc": "2.0",
                    "id": message.get('id'),
                    "result": result
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": message.get('id'),
                    "error": {"code": -32000, "message": str(e)}
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": message.get('id'),
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Return agent card for discovery"""
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "capabilities": self.get_capabilities(),
            "version": "2.0.0",
            "status": "active"
        }
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return list(self._handlers.keys())
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class OrchestratorAgent(A2AAgent):
    """
    Orchestrator agent that coordinates tasks between agents
    and provides conversational responses
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            "OrchestratorAgent",
            config['a2a']['agents']['orchestrator']['endpoint'],
            config
        )
        self.tasks: Dict[str, Task] = {}
        self.conversation_history: List[Dict] = []
        
        # Register handlers
        self.register_handler("orchestrate_search", self._orchestrate_search)
        self.register_handler("orchestrate_index", self._orchestrate_index)
        self.register_handler("get_status", self._get_status)
    
    async def _orchestrate_search(self, params: Dict) -> Dict[str, Any]:
        """Orchestrate a search workflow"""
        query = params.get('query', '')
        top_k = params.get('top_k', 5)
        
        task_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create task chain
        tasks = [
            Task(
                id=f"{task_id}_understand",
                name="Query Understanding",
                agent="conversation",
                params={"query": query}
            ),
            Task(
                id=f"{task_id}_search",
                name="Semantic Search",
                agent="search",
                params={"query": query, "top_k": top_k},
                dependencies=[f"{task_id}_understand"]
            ),
            Task(
                id=f"{task_id}_respond",
                name="Response Generation",
                agent="conversation",
                params={"query": query},
                dependencies=[f"{task_id}_search"]
            )
        ]
        
        # Execute task chain
        results = await self._execute_task_chain(tasks)
        return results
    
    async def _orchestrate_index(self, params: Dict) -> Dict[str, Any]:
        """Orchestrate an indexing workflow"""
        directory = params.get('directory', '')
        
        task_id = f"index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = Task(
            id=task_id,
            name="Document Indexing",
            agent="ingestion",
            params={"directory": directory}
        )
        
        await self.emit_status(f"📁 Starting indexing of {directory}")
        result = await self._execute_task(task)
        
        return result
    
    async def _execute_task_chain(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute a chain of tasks respecting dependencies"""
        results = {}
        
        for task in tasks:
            # Wait for dependencies
            for dep_id in task.dependencies:
                if dep_id not in results:
                    logger.warning(f"Missing dependency: {dep_id}")
            
            # Execute task
            result = await self._execute_task(task)
            results[task.id] = result
            
            # Store result for dependent tasks
            task.result = result
        
        return results
    
    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        await self.emit_status(f"⚙️ {task.name}...")
        
        try:
            result = await self.send_message(
                to=task.agent,
                method=task.name.lower().replace(' ', '_'),
                params=task.params
            )
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task failed: {task.name} - {e}")
            return {"error": str(e)}
    
    async def _get_status(self, params: Dict) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "active_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "agents": list(self.config['a2a']['agents'].keys())
        }


class ConversationAgent(A2AAgent):
    """
    Agent that handles user interactions and generates
    natural language responses
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            "ConversationAgent",
            config['a2a']['agents']['conversation']['endpoint'],
            config
        )
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']
        self.status_messages = config['agent']['status_messages']
        self.templates = config['agent']['templates']
        
        # Register handlers
        self.register_handler("query_understanding", self._understand_query)
        self.register_handler("response_generation", self._generate_response)
        self.register_handler("format_results", self._format_results)
    
    async def _understand_query(self, params: Dict) -> Dict[str, Any]:
        """Understand user query intent and extract key information"""
        query = params.get('query', '')
        
        prompt = f"""Analyze this search query and extract:
1. Main intent (what the user is looking for)
2. Key entities/terms
3. Document type hints (if any)
4. Time references (if any)

Query: "{query}"

Respond in JSON format:
{{"intent": "...", "entities": [...], "doc_type": "...", "time_ref": "..."}}"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                
                # Parse JSON response
                try:
                    understanding = json.loads(result.get('response', '{}'))
                except json.JSONDecodeError:
                    understanding = {"intent": query, "entities": []}
                
                return {
                    "query": query,
                    "understanding": understanding
                }
        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            return {"query": query, "understanding": {"intent": query}}
    
    async def _generate_response(self, params: Dict) -> Dict[str, Any]:
        """Generate natural language response for search results"""
        query = params.get('query', '')
        results = params.get('results', [])
        intent = params.get('intent', 'general')
        
        if not results:
            return {
                "message": self.templates['no_results'].format(query=query),
                "results": []
            }
        
        # Extract topic from query
        topic = self._extract_topic(query)
        
        intro = self.templates['search_intro'].format(
            count=len(results),
            topic=topic
        )
        
        return {
            "message": intro,
            "results": results,
            "intent": intent
        }
    
    async def _format_results(self, params: Dict) -> Dict[str, Any]:
        """Format search results for display"""
        results = params.get('results', [])
        query = params.get('query', '')
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append({
                "rank": i,
                "filename": result.get('filename', 'Unknown'),
                "file_path": result.get('file_path', ''),
                "score": round(result.get('score', 0), 3),
                "summary": result.get('content_summary', 'No summary available'),
                "document_type": result.get('document_type', 'document'),
                "relevance_explanation": self._explain_relevance(query, result)
            })
        
        return {"formatted_results": formatted}
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query"""
        # Remove common question words
        stop_words = ['find', 'search', 'show', 'get', 'the', 'a', 'an', 'my', 'about', 'for', 'with']
        words = query.lower().split()
        topic_words = [w for w in words if w not in stop_words]
        return ' '.join(topic_words[:5]) if topic_words else query
    
    def _explain_relevance(self, query: str, result: Dict) -> str:
        """Generate a brief explanation of why this result is relevant"""
        summary = result.get('content_summary', '').lower()
        query_terms = set(query.lower().split())
        
        matching_terms = [term for term in query_terms if term in summary]
        
        if matching_terms:
            return f"Matches: {', '.join(matching_terms[:3])}"
        return "Semantic match"
    
    def get_status_message(self, status_type: str) -> str:
        """Get appropriate status message"""
        return self.status_messages.get(status_type, "Processing...")


class IngestionAgent(A2AAgent):
    """Enhanced agent for document ingestion"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            "IngestionAgent",
            config['a2a']['agents']['ingestion']['endpoint'],
            config
        )
        
        # Register handlers
        self.register_handler("index_directory", self._index_directory)
        self.register_handler("process_file", self._process_file)
        self.register_handler("get_status", self._get_status)
    
    async def _index_directory(self, params: Dict) -> Dict[str, Any]:
        """Index a directory"""
        from backend.ingestion import IngestionPipeline
        from backend.opensearch_client import OpenSearchClient
        
        directory = params.get('directory', '')
        
        await self.emit_status(f"📂 Scanning directory: {directory}")
        
        opensearch = OpenSearchClient(self.config)
        pipeline = IngestionPipeline(self.config, opensearch)
        
        try:
            await pipeline.process_directory(Path(directory), "a2a_task")
            
            await self.emit_status("✅ Indexing completed!")
            
            # Notify search agent
            await self.send_message(
                to="search",
                method="index_complete",
                params={"directory": directory},
                msg_type=MessageType.NOTIFICATION
            )
            
            return {
                "status": "completed",
                "directory": directory
            }
        finally:
            await opensearch.close()
            await pipeline.close()
    
    async def _process_file(self, params: Dict) -> Dict[str, Any]:
        """Process a single file"""
        file_path = params.get('file_path', '')
        
        await self.emit_status(f"📄 Processing: {Path(file_path).name}")
        
        # Implementation would call ingestion pipeline
        return {"status": "processed", "file": file_path}
    
    async def _get_status(self, params: Dict) -> Dict[str, Any]:
        """Get ingestion status"""
        return {"status": "active", "pending_files": 0}


class SearchAgent(A2AAgent):
    """Enhanced agent for semantic search"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            "SearchAgent",
            config['a2a']['agents']['search']['endpoint'],
            config
        )
        
        # Register handlers
        self.register_handler("semantic_search", self._search)
        self.register_handler("hybrid_search", self._hybrid_search)
        self.register_handler("index_complete", self._handle_index_complete)
    
    async def _search(self, params: Dict) -> Dict[str, Any]:
        """Perform semantic search"""
        query = params.get('query', '')
        top_k = params.get('top_k', 5)
        filters = params.get('filters')
        
        await self.emit_status("🔍 Searching through documents...")
        
        from backend.opensearch_client import OpenSearchClient
        from backend.ingestion import IngestionPipeline
        from backend.reranker import CrossEncoderReranker
        
        opensearch = OpenSearchClient(self.config)
        ingestion = IngestionPipeline(self.config, opensearch)
        reranker = CrossEncoderReranker(self.config)
        
        try:
            # Generate query embedding
            query_embedding = await ingestion.generate_embedding(query)
            
            await self.emit_status("⚡ Ranking results by relevance...")
            
            # Perform hybrid search
            if self.config['search']['hybrid']['enabled']:
                candidates = await opensearch.hybrid_search(
                    query=query,
                    query_vector=query_embedding,
                    top_k=self.config['search']['recall_top_k'],
                    filters=filters
                )
            else:
                candidates = await opensearch.vector_search(
                    query_embedding,
                    top_k=self.config['search']['recall_top_k'],
                    filters=filters
                )
            
            # Rerank
            results = await reranker.rerank(query, candidates, top_k)
            
            await self.emit_status("✨ Preparing your results...")
            
            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results)
            }
        finally:
            await opensearch.close()
            await ingestion.close()
    
    async def _hybrid_search(self, params: Dict) -> Dict[str, Any]:
        """Perform hybrid search with query expansion"""
        query = params.get('query', '')
        top_k = params.get('top_k', 5)
        
        from backend.ingestion import IngestionPipeline
        from backend.opensearch_client import OpenSearchClient
        
        opensearch = OpenSearchClient(self.config)
        ingestion = IngestionPipeline(self.config, opensearch)
        
        try:
            # Expand query if enabled
            expanded_queries = []
            if self.config['search']['query_expansion']['enabled']:
                await self.emit_status("🧠 Analyzing query...")
                expanded_queries = await ingestion.expand_query(query)
            
            # Search with original and expanded queries
            all_results = []
            
            # Original query
            query_embedding = await ingestion.generate_embedding(query)
            results = await opensearch.hybrid_search(
                query=query,
                query_vector=query_embedding,
                top_k=self.config['search']['recall_top_k']
            )
            all_results.extend(results)
            
            # Expanded queries
            for exp_query in expanded_queries[:2]:
                exp_embedding = await ingestion.generate_embedding(exp_query)
                exp_results = await opensearch.hybrid_search(
                    query=exp_query,
                    query_vector=exp_embedding,
                    top_k=10
                )
                all_results.extend(exp_results)
            
            # Deduplicate and return
            seen_ids = set()
            unique_results = []
            for r in all_results:
                if r['id'] not in seen_ids:
                    seen_ids.add(r['id'])
                    unique_results.append(r)
            
            return {
                "status": "success",
                "results": unique_results[:top_k * 2],  # Return more for reranking
                "expanded_queries": expanded_queries
            }
        finally:
            await opensearch.close()
            await ingestion.close()
    
    async def _handle_index_complete(self, params: Dict) -> Dict[str, Any]:
        """Handle index completion notification"""
        directory = params.get('directory', '')
        logger.info(f"Index completed notification received: {directory}")
        return {"status": "acknowledged"}
