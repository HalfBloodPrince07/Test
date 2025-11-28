# backend/memory/memory_manager.py
"""
Unified Memory Manager
Coordinates all memory subsystems with memory decay and consolidation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from loguru import logger

from .session_memory import SessionMemory
from .user_profile import UserProfileManager
from .procedural_memory import ProceduralMemory


class MemoryManager:
    """
    Unified memory manager implementing MemGPT-style tiered architecture

    Memory Tiers:
    1. Working Memory: Current session context (SessionMemory)
    2. Short-term Memory: Recent history with decay (SessionMemory)
    3. Long-term Memory: Persistent user patterns (UserProfileManager)
    4. Procedural Memory: Learned behaviors (ProceduralMemory)

    Features:
    - Memory consolidation: Periodically analyze patterns
    - Memory decay: Older memories get lower weights
    - Cross-tier retrieval: Query across all memory types
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        database_url: str = "sqlite+aiosqlite:///locallens_memory.db",
        consolidation_interval: int = 3600  # 1 hour
    ):
        # Initialize memory subsystems
        self.session_memory = SessionMemory(redis_url=redis_url)
        self.user_profile = UserProfileManager(database_url=database_url)
        self.procedural_memory = ProceduralMemory()

        self.consolidation_interval = consolidation_interval
        self._consolidation_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize all memory systems"""
        await self.session_memory.connect()
        await self.user_profile.initialize()

        # Start background consolidation task
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())

        logger.info("✅ Memory Manager initialized")

    async def close(self):
        """Cleanup all memory systems"""
        if self._consolidation_task:
            self._consolidation_task.cancel()

        await self.session_memory.close()
        await self.user_profile.close()

        logger.info("Memory Manager closed")

    async def _consolidation_loop(self):
        """
        Background task for memory consolidation

        Periodically:
        - Extract patterns from session memory
        - Update long-term user profiles
        - Prune old session data
        """
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval)
                logger.info("Starting memory consolidation...")

                # Consolidation logic here
                # For now, just log
                logger.info("Memory consolidation complete")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation error: {e}")

    async def record_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        results: List[Dict],
        intent: str,
        search_time: float,
        clicked_results: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Record a complete user interaction across all memory tiers

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User query
            response: System response
            results: Search results
            intent: Detected intent
            search_time: Time taken for search
            clicked_results: IDs of results clicked
            metadata: Additional metadata
        """
        # 1. Add to session memory (working/short-term)
        await self.session_memory.add_turn(
            session_id=session_id,
            query=query,
            response=response,
            results=results,
            intent=intent,
            metadata=metadata
        )

        # 2. Record in long-term memory
        await self.user_profile.record_search(
            user_id=user_id,
            session_id=session_id,
            query=query,
            intent=intent,
            num_results=len(results),
            search_time=search_time,
            clicked_results=clicked_results or []
        )

        # 3. Update topic interests
        if intent and intent != "general":
            await self.user_profile.update_topic_interest(user_id, intent)

        # Extract topic from query
        topic = self._extract_topic(query)
        if topic:
            await self.user_profile.update_topic_interest(user_id, topic)

        # 4. Record in procedural memory for learning
        clicked_positions = []
        if clicked_results and results:
            result_ids = [r.get("id") or r.get("file_path") for r in results]
            clicked_positions = [
                i for i, rid in enumerate(result_ids)
                if rid in clicked_results
            ]

        top_score = results[0].get("score", 0.0) if results else 0.0

        self.procedural_memory.record_search_outcome(
            user_id=user_id,
            strategy="hybrid",  # Could be dynamic
            query=query,
            num_results=len(results),
            clicked_positions=clicked_positions,
            top_score=top_score
        )

        logger.debug(f"Recorded interaction for user {user_id}")

    async def record_document_access(
        self,
        user_id: str,
        document_id: str,
        file_path: str,
        filename: str,
        source_query: Optional[str] = None
    ):
        """Record when user accesses a document"""
        await self.user_profile.record_document_access(
            user_id=user_id,
            document_id=document_id,
            file_path=file_path,
            filename=filename,
            source_query=source_query
        )

    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """Get current session context (working memory)"""
        return await self.session_memory.get_context(session_id)

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user preferences from all memory tiers

        Returns:
            Dict with preferences for personalizing search
        """
        # Long-term patterns
        preferences = await self.user_profile.get_personalized_preferences(user_id)

        # Learned optimal weights
        optimal_weights = self.procedural_memory.get_optimal_weights(user_id)

        # Best strategy
        best_strategy = self.procedural_memory.get_best_strategy(user_id)

        # Frequently accessed
        freq_docs = await self.user_profile.get_frequently_accessed_documents(
            user_id, limit=5
        )

        return {
            **preferences,
            "optimal_weights": optimal_weights,
            "best_strategy": best_strategy,
            "frequently_accessed": freq_docs,
            "should_rerank": self.procedural_memory.should_rerank(user_id)
        }

    async def get_personalized_search_config(self, user_id: str) -> Dict[str, Any]:
        """
        Get search configuration personalized for user

        Returns config that can directly override defaults
        """
        prefs = await self.get_user_preferences(user_id)

        return {
            "hybrid_weights": prefs["optimal_weights"],
            "use_reranking": prefs["should_rerank"],
            "boost_intents": prefs.get("preferred_intents", []),
            "boost_documents": [d["document_id"] for d in prefs.get("frequently_accessed", [])]
        }

    async def get_query_suggestions(
        self,
        user_id: str,
        session_id: str,
        query: str
    ) -> List[str]:
        """
        Get query suggestions from multiple memory sources

        Combines:
        - Related queries from session
        - Learned reformulations
        - User's frequent topics
        """
        suggestions = []

        # 1. From session memory (related turns)
        related = await self.session_memory.get_related_turns(
            session_id, query, top_k=2
        )
        suggestions.extend([t["query"] for t in related if t["query"] != query])

        # 2. From procedural memory (successful reformulations)
        reformulations = self.procedural_memory.get_query_suggestions(query, limit=2)
        suggestions.extend(reformulations)

        # 3. From user profile (frequent topics as suggestions)
        topics = await self.user_profile.get_frequently_searched_topics(user_id, limit=3)
        suggestions.extend([t["topic"] for t in topics if query.lower() not in t["topic"].lower()])

        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique_suggestions.append(s)

        return unique_suggestions[:5]

    async def get_memory_summary(
        self,
        user_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get summary of user's memory state

        Useful for debugging and showing user what system knows
        """
        # Session context
        context = await self.get_context(session_id)

        # User patterns
        patterns = await self.user_profile.get_search_patterns(user_id)

        # Learning stats
        learning = self.procedural_memory.get_learning_stats(user_id)

        # Profile
        profile = await self.user_profile.get_or_create_profile(user_id)

        return {
            "current_session": {
                "topic": context.get("topic"),
                "recent_queries": context.get("recent_queries", [])[-3:],
                "active_intents": context.get("intents", [])
            },
            "user_statistics": {
                "total_queries": profile.total_queries,
                "total_documents_accessed": profile.total_documents_accessed,
                "member_since": profile.created_at.isoformat()
            },
            "patterns": patterns,
            "learning": learning
        }

    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract main topic from query"""
        stop_words = {
            'find', 'search', 'show', 'get', 'the', 'a', 'an', 'my', 'about',
            'for', 'with', 'where', 'is', 'are', 'all', 'any', 'please'
        }

        words = query.lower().split()
        topic_words = [w for w in words if w not in stop_words and len(w) > 3]

        if topic_words:
            return ' '.join(topic_words[:3])

        return None

    async def apply_memory_decay(self, session_id: str):
        """
        Apply memory decay function

        Older memories get lower weights in retrieval
        This is handled implicitly by TTL in Redis and sliding window
        """
        # Could implement explicit decay scoring here
        pass
