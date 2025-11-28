# backend/memory/__init__.py
"""
Agentic Memory System for LocalLens

This module implements a tiered memory architecture:
- Short-term: Session context (sliding window buffer)
- Long-term: User profiles and patterns
- Procedural: Tool usage learning and optimization
"""

from .session_memory import SessionMemory
from .user_profile import UserProfileManager
from .procedural_memory import ProceduralMemory
from .memory_manager import MemoryManager

__all__ = [
    "SessionMemory",
    "UserProfileManager",
    "ProceduralMemory",
    "MemoryManager"
]
