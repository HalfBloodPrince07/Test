# backend/memory/user_profile.py
"""
Long-Term Memory (User Profile)
Implements episodic and semantic memory with SQLite
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, Text,
    select, func, and_, or_
)
from loguru import logger
import json

Base = declarative_base()


class UserProfile(Base):
    """User profile with preferences and patterns"""
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Preferences
    preferences = Column(JSON, default=dict)

    # Statistics
    total_queries = Column(Integer, default=0)
    total_documents_accessed = Column(Integer, default=0)


class SearchHistory(Base):
    """Search history for pattern analysis"""
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)

    query = Column(Text)
    intent = Column(String)
    num_results = Column(Integer)
    clicked_results = Column(JSON, default=list)  # Track which results were clicked

    # Performance metrics
    search_time = Column(Float)
    satisfaction_score = Column(Float, nullable=True)  # Could be inferred or explicit


class DocumentAccess(Base):
    """Track document access patterns"""
    __tablename__ = "document_access"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    document_id = Column(String, index=True)
    file_path = Column(String)
    filename = Column(String)

    accessed_at = Column(DateTime, default=datetime.now, index=True)
    access_count = Column(Integer, default=1)
    source_query = Column(Text, nullable=True)  # Query that led to this document


class TopicInterest(Base):
    """Track user interests and frequently searched topics"""
    __tablename__ = "topic_interests"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True)
    topic = Column(String, index=True)
    interest_score = Column(Float, default=1.0)  # Increments with each search
    last_searched = Column(DateTime, default=datetime.now)
    search_count = Column(Integer, default=1)


class UserProfileManager:
    """
    Manages long-term user profiles and patterns

    Features:
    - Episodic memory: Search history, document access
    - Semantic memory: Topics, preferences, patterns
    - Memory consolidation: Periodic pattern analysis
    - Adaptive personalization
    """

    def __init__(self, database_url: str = "sqlite+aiosqlite:///locallens_memory.db"):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """Create database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ User profile database initialized")

    async def close(self):
        """Close database connections"""
        await self.engine.dispose()

    async def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        async with self.async_session() as session:
            result = await session.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()

            if not profile:
                profile = UserProfile(user_id=user_id, preferences={})
                session.add(profile)
                await session.commit()
                await session.refresh(profile)
                logger.info(f"Created new user profile: {user_id}")

            return profile

    async def record_search(
        self,
        user_id: str,
        session_id: str,
        query: str,
        intent: str,
        num_results: int,
        search_time: float,
        clicked_results: Optional[List[str]] = None
    ):
        """Record search in history"""
        async with self.async_session() as session:
            search = SearchHistory(
                user_id=user_id,
                session_id=session_id,
                query=query,
                intent=intent,
                num_results=num_results,
                search_time=search_time,
                clicked_results=clicked_results or []
            )
            session.add(search)

            # Update user stats
            result = await session.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()
            if profile:
                profile.total_queries += 1
                profile.last_active = datetime.now()

            await session.commit()

    async def record_document_access(
        self,
        user_id: str,
        document_id: str,
        file_path: str,
        filename: str,
        source_query: Optional[str] = None
    ):
        """Record document access"""
        async with self.async_session() as session:
            # Check if already accessed
            result = await session.execute(
                select(DocumentAccess).where(
                    and_(
                        DocumentAccess.user_id == user_id,
                        DocumentAccess.document_id == document_id
                    )
                )
            )
            access = result.scalar_one_or_none()

            if access:
                access.access_count += 1
                access.accessed_at = datetime.now()
            else:
                access = DocumentAccess(
                    user_id=user_id,
                    document_id=document_id,
                    file_path=file_path,
                    filename=filename,
                    source_query=source_query
                )
                session.add(access)

            # Update user stats
            result = await session.execute(
                select(UserProfile).where(UserProfile.user_id == user_id)
            )
            profile = result.scalar_one_or_none()
            if profile:
                profile.total_documents_accessed += 1

            await session.commit()

    async def update_topic_interest(self, user_id: str, topic: str):
        """Update interest in a topic"""
        async with self.async_session() as session:
            result = await session.execute(
                select(TopicInterest).where(
                    and_(
                        TopicInterest.user_id == user_id,
                        TopicInterest.topic == topic
                    )
                )
            )
            interest = result.scalar_one_or_none()

            if interest:
                interest.search_count += 1
                interest.interest_score += 0.5
                interest.last_searched = datetime.now()
            else:
                interest = TopicInterest(
                    user_id=user_id,
                    topic=topic
                )
                session.add(interest)

            await session.commit()

    async def get_frequently_searched_topics(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get user's frequently searched topics"""
        async with self.async_session() as session:
            result = await session.execute(
                select(TopicInterest)
                .where(TopicInterest.user_id == user_id)
                .order_by(TopicInterest.interest_score.desc())
                .limit(limit)
            )
            topics = result.scalars().all()

            return [
                {
                    "topic": t.topic,
                    "score": t.interest_score,
                    "search_count": t.search_count,
                    "last_searched": t.last_searched.isoformat()
                }
                for t in topics
            ]

    async def get_frequently_accessed_documents(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get most accessed documents by user"""
        async with self.async_session() as session:
            result = await session.execute(
                select(DocumentAccess)
                .where(DocumentAccess.user_id == user_id)
                .order_by(DocumentAccess.access_count.desc())
                .limit(limit)
            )
            docs = result.scalars().all()

            return [
                {
                    "document_id": d.document_id,
                    "filename": d.filename,
                    "file_path": d.file_path,
                    "access_count": d.access_count,
                    "last_accessed": d.accessed_at.isoformat()
                }
                for d in docs
            ]

    async def get_search_patterns(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze search patterns for personalization

        Returns:
        - Peak usage hours
        - Most common intents
        - Average results clicked
        - Click-through rate
        """
        async with self.async_session() as session:
            # Get recent searches (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)

            result = await session.execute(
                select(SearchHistory)
                .where(
                    and_(
                        SearchHistory.user_id == user_id,
                        SearchHistory.timestamp >= thirty_days_ago
                    )
                )
            )
            searches = result.scalars().all()

            if not searches:
                return {
                    "peak_hours": [],
                    "common_intents": [],
                    "avg_clicks": 0,
                    "click_through_rate": 0
                }

            # Extract patterns
            hours = [s.timestamp.hour for s in searches]
            from collections import Counter
            hour_counts = Counter(hours)
            peak_hours = [h for h, _ in hour_counts.most_common(3)]

            intents = [s.intent for s in searches]
            intent_counts = Counter(intents)
            common_intents = [i for i, _ in intent_counts.most_common(5)]

            # Click metrics
            total_clicks = sum(len(s.clicked_results) for s in searches)
            avg_clicks = total_clicks / len(searches) if searches else 0

            searches_with_clicks = sum(1 for s in searches if s.clicked_results)
            ctr = searches_with_clicks / len(searches) if searches else 0

            return {
                "peak_hours": peak_hours,
                "common_intents": common_intents,
                "avg_clicks": round(avg_clicks, 2),
                "click_through_rate": round(ctr, 2),
                "total_searches": len(searches)
            }

    async def get_personalized_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get personalized preferences for search optimization

        Returns preferences that can be used to:
        - Adjust hybrid search weights
        - Boost certain document types
        - Filter by time ranges
        """
        patterns = await self.get_search_patterns(user_id)
        topics = await self.get_frequently_searched_topics(user_id, limit=5)

        return {
            "preferred_intents": patterns["common_intents"],
            "favorite_topics": [t["topic"] for t in topics],
            "active_hours": patterns["peak_hours"],
            "engagement_score": patterns["click_through_rate"]
        }
