# backend/memory/session_memory.py
"""
Short-Term Memory (Session Context)
Implements sliding window buffer with Redis backend
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import hashlib
from loguru import logger

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory fallback")


class SessionMemory:
    """
    Short-term memory manager with sliding window buffer

    Features:
    - Maintains last N conversation turns
    - Tracks query intent and retrieved documents
    - Session-based context with TTL
    - Conversation threading
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        window_size: int = 10,
        ttl_seconds: int = 3600
    ):
        self.redis_url = redis_url
        self.window_size = window_size
        self.ttl_seconds = ttl_seconds
        self.redis_client: Optional[aioredis.Redis] = None

        # In-memory fallback
        self._memory_store: Dict[str, List[Dict]] = {}
        self._use_redis = REDIS_AVAILABLE

    async def connect(self):
        """Initialize Redis connection"""
        if self._use_redis:
            try:
                self.redis_client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("✅ Connected to Redis for session memory")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory fallback")
                self._use_redis = False

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}:memory"

    def _get_context_key(self, session_id: str) -> str:
        """Generate Redis key for session context"""
        return f"session:{session_id}:context"

    async def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        results: List[Dict],
        intent: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a conversation turn to session memory

        Args:
            session_id: Unique session identifier
            query: User query
            response: System response
            results: Search results
            intent: Detected intent
            metadata: Additional metadata
        """
        turn = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "results": results,
            "intent": intent,
            "metadata": metadata or {}
        }

        if self._use_redis and self.redis_client:
            await self._add_turn_redis(session_id, turn)
        else:
            await self._add_turn_memory(session_id, turn)

    async def _add_turn_redis(self, session_id: str, turn: Dict):
        """Add turn using Redis"""
        key = self._get_session_key(session_id)

        # Append turn
        await self.redis_client.rpush(key, json.dumps(turn))

        # Trim to window size
        await self.redis_client.ltrim(key, -self.window_size, -1)

        # Set TTL
        await self.redis_client.expire(key, self.ttl_seconds)

    async def _add_turn_memory(self, session_id: str, turn: Dict):
        """Add turn using in-memory store"""
        if session_id not in self._memory_store:
            self._memory_store[session_id] = []

        self._memory_store[session_id].append(turn)

        # Trim to window size
        if len(self._memory_store[session_id]) > self.window_size:
            self._memory_store[session_id] = self._memory_store[session_id][-self.window_size:]

    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get conversation history for session

        Args:
            session_id: Session identifier
            limit: Maximum turns to retrieve (None = all within window)

        Returns:
            List of conversation turns
        """
        if self._use_redis and self.redis_client:
            return await self._get_history_redis(session_id, limit)
        else:
            return await self._get_history_memory(session_id, limit)

    async def _get_history_redis(self, session_id: str, limit: Optional[int]) -> List[Dict]:
        """Get history from Redis"""
        key = self._get_session_key(session_id)

        if limit:
            items = await self.redis_client.lrange(key, -limit, -1)
        else:
            items = await self.redis_client.lrange(key, 0, -1)

        return [json.loads(item) for item in items]

    async def _get_history_memory(self, session_id: str, limit: Optional[int]) -> List[Dict]:
        """Get history from in-memory store"""
        history = self._memory_store.get(session_id, [])
        if limit:
            return history[-limit:]
        return history

    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session context (work context tracking)

        Returns:
            Dict with current context information:
            - topic: Current topic of interest
            - document_types: Types being explored
            - recent_queries: Last few queries
        """
        history = await self.get_history(session_id, limit=5)

        if not history:
            return {
                "topic": None,
                "document_types": [],
                "recent_queries": [],
                "intents": []
            }

        # Extract context
        recent_queries = [turn["query"] for turn in history]
        intents = [turn["intent"] for turn in history]

        # Detect topic from most common terms
        all_queries = " ".join(recent_queries).lower()

        # Simple topic extraction (could be enhanced with NLP)
        from collections import Counter
        words = [w for w in all_queries.split() if len(w) > 3]
        common_words = Counter(words).most_common(3)
        topic = " ".join([w for w, _ in common_words]) if common_words else None

        # Get document types from results
        doc_types = set()
        for turn in history:
            for result in turn.get("results", []):
                doc_type = result.get("document_type", "unknown")
                doc_types.add(doc_type)

        return {
            "topic": topic,
            "document_types": list(doc_types),
            "recent_queries": recent_queries,
            "intents": list(set(intents))
        }

    async def update_context(self, session_id: str, context: Dict[str, Any]):
        """Update session context explicitly"""
        if self._use_redis and self.redis_client:
            key = self._get_context_key(session_id)
            await self.redis_client.setex(
                key,
                self.ttl_seconds,
                json.dumps(context)
            )
        # In-memory version could store in _memory_store with special key

    async def clear_session(self, session_id: str):
        """Clear session memory"""
        if self._use_redis and self.redis_client:
            await self.redis_client.delete(self._get_session_key(session_id))
            await self.redis_client.delete(self._get_context_key(session_id))
        else:
            self._memory_store.pop(session_id, None)

        logger.info(f"Cleared session memory for {session_id}")

    async def get_related_turns(
        self,
        session_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Find related conversation turns based on query similarity

        This enables conversation threading - linking related queries
        """
        history = await self.get_history(session_id)

        if not history:
            return []

        # Simple keyword matching (could use embeddings for better results)
        query_words = set(query.lower().split())

        scored_turns = []
        for turn in history:
            turn_words = set(turn["query"].lower().split())
            overlap = len(query_words & turn_words)
            if overlap > 0:
                scored_turns.append((overlap, turn))

        # Sort by score and return top_k
        scored_turns.sort(reverse=True, key=lambda x: x[0])
        return [turn for _, turn in scored_turns[:top_k]]

    def generate_session_id(self, user_id: Optional[str] = None) -> str:
        """Generate unique session ID"""
        base = f"{user_id or 'anonymous'}:{datetime.now().isoformat()}"
        return hashlib.md5(base.encode()).hexdigest()[:16]
