#!/usr/bin/env python3
"""
Initialize LocalLens Memory System

Run this script once to create the user profile database.
"""

import asyncio
from backend.memory import UserProfileManager, MemoryManager
from loguru import logger


async def initialize_memory():
    """Initialize memory system databases"""
    logger.info("Initializing LocalLens Memory System...")

    # 1. Initialize user profile database
    try:
        profile_mgr = UserProfileManager()
        await profile_mgr.initialize()
        logger.info("✅ User profile database created (locallens_memory.db)")
        await profile_mgr.close()
    except Exception as e:
        logger.error(f"❌ User profile initialization failed: {e}")
        return False

    # 2. Test memory manager initialization
    try:
        memory = MemoryManager()
        await memory.initialize()
        logger.info("✅ Memory Manager initialized successfully")

        # Test session memory
        test_session = "test_session_123"
        await memory.session_memory.add_turn(
            session_id=test_session,
            query="test query",
            response="test response",
            results=[],
            intent="test"
        )
        logger.info("✅ Session memory test successful")

        # Cleanup test
        await memory.session_memory.clear_session(test_session)

        await memory.close()
    except Exception as e:
        logger.warning(f"⚠️  Memory Manager test failed: {e}")
        logger.warning("This is normal if Redis is not running. System will use in-memory fallback.")

    logger.info("")
    logger.info("="*60)
    logger.info("Memory System Initialization Complete!")
    logger.info("="*60)
    logger.info("")
    logger.info("✅ SQLite database: locallens_memory.db created")
    logger.info("ℹ️  Redis: Optional (system will fallback to in-memory if unavailable)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start the API: uvicorn backend.api:app --reload")
    logger.info("  2. Start the UI: streamlit run app.py")
    logger.info("  3. Open http://localhost:8501")
    logger.info("")

    return True


if __name__ == "__main__":
    success = asyncio.run(initialize_memory())
    exit(0 if success else 1)
