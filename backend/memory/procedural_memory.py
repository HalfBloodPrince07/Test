# backend/memory/procedural_memory.py
"""
Procedural Memory (Tool Usage Learning)
Learns which search strategies work best for each user
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from loguru import logger
from cachetools import TTLCache


class ProceduralMemory:
    """
    Learns and adapts based on successful patterns

    Features:
    - Track search strategy effectiveness
    - Learn optimal hybrid search weights per user
    - Store successful query reformulations
    - Adapt reranking preferences based on click-through rates
    """

    def __init__(self, cache_ttl: int = 3600):
        # In-memory caches with TTL
        self.strategy_performance: Dict[str, Dict] = defaultdict(
            lambda: {
                "success_count": 0,
                "total_count": 0,
                "avg_score": 0.0,
                "avg_clicks": 0.0
            }
        )

        # User-specific learned weights
        self.user_weights: TTLCache = TTLCache(maxsize=1000, ttl=cache_ttl)

        # Successful query reformulations
        self.query_reformulations: Dict[str, List[str]] = defaultdict(list)

        # Click pattern learning
        self.click_patterns: Dict[str, Dict] = defaultdict(
            lambda: {
                "position_clicks": defaultdict(int),
                "total_searches": 0
            }
        )

    def record_search_outcome(
        self,
        user_id: str,
        strategy: str,
        query: str,
        num_results: int,
        clicked_positions: List[int],
        top_score: float
    ):
        """
        Record outcome of a search to learn from it

        Args:
            user_id: User identifier
            strategy: Search strategy used (e.g., "hybrid", "vector_only")
            query: Original query
            num_results: Number of results returned
            clicked_positions: Positions of results that were clicked (0-indexed)
            top_score: Top relevance score
        """
        key = f"{user_id}:{strategy}"

        # Update strategy performance
        perf = self.strategy_performance[key]
        perf["total_count"] += 1

        # Success = got results with clicks
        if clicked_positions:
            perf["success_count"] += 1

        # Update avg score (running average)
        perf["avg_score"] = (
            (perf["avg_score"] * (perf["total_count"] - 1) + top_score)
            / perf["total_count"]
        )

        # Update avg clicks
        perf["avg_clicks"] = (
            (perf["avg_clicks"] * (perf["total_count"] - 1) + len(clicked_positions))
            / perf["total_count"]
        )

        # Record click positions
        click_data = self.click_patterns[user_id]
        click_data["total_searches"] += 1
        for pos in clicked_positions:
            click_data["position_clicks"][pos] += 1

        logger.debug(f"Recorded search outcome for {user_id}: {strategy}")

    def get_best_strategy(self, user_id: str) -> str:
        """
        Get best performing search strategy for user

        Returns:
            Strategy name with highest success rate
        """
        user_strategies = {
            key.split(":")[1]: perf
            for key, perf in self.strategy_performance.items()
            if key.startswith(f"{user_id}:")
        }

        if not user_strategies:
            return "hybrid"  # Default

        # Find strategy with best success rate
        best = max(
            user_strategies.items(),
            key=lambda x: x[1]["success_count"] / max(x[1]["total_count"], 1)
        )

        return best[0]

    def learn_hybrid_weights(
        self,
        user_id: str,
        vector_weight: float,
        bm25_weight: float,
        success: bool
    ):
        """
        Learn optimal hybrid search weights for user

        Uses simple reinforcement: successful weights are averaged in
        """
        if user_id not in self.user_weights:
            self.user_weights[user_id] = {
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "sample_count": 0
            }

        weights = self.user_weights[user_id]

        if success:
            # Update weights with moving average
            alpha = 0.1  # Learning rate
            weights["vector_weight"] = (
                (1 - alpha) * weights["vector_weight"] + alpha * vector_weight
            )
            weights["bm25_weight"] = (
                (1 - alpha) * weights["bm25_weight"] + alpha * bm25_weight
            )
            weights["sample_count"] += 1

        logger.debug(f"Updated weights for {user_id}: {weights}")

    def get_optimal_weights(self, user_id: str) -> Dict[str, float]:
        """
        Get learned optimal weights for user

        Returns:
            Dict with vector_weight and bm25_weight
        """
        if user_id in self.user_weights:
            weights = self.user_weights[user_id]
            return {
                "vector_weight": weights["vector_weight"],
                "bm25_weight": weights["bm25_weight"]
            }

        # Default weights
        return {"vector_weight": 0.7, "bm25_weight": 0.3}

    def record_query_reformulation(
        self,
        original_query: str,
        reformulated_query: str,
        success: bool
    ):
        """
        Record successful query reformulations

        This helps with query expansion
        """
        if success:
            key = original_query.lower().strip()
            if reformulated_query not in self.query_reformulations[key]:
                self.query_reformulations[key].append(reformulated_query)

            # Keep only top 5 reformulations
            self.query_reformulations[key] = self.query_reformulations[key][:5]

    def get_query_suggestions(self, query: str, limit: int = 3) -> List[str]:
        """
        Get successful reformulations for query

        Returns:
            List of suggested alternative queries
        """
        key = query.lower().strip()
        suggestions = self.query_reformulations.get(key, [])
        return suggestions[:limit]

    def get_click_position_bias(self, user_id: str) -> Dict[int, float]:
        """
        Get user's click position bias

        This helps understand if user prefers top results or explores deeper

        Returns:
            Dict mapping position to click probability
        """
        click_data = self.click_patterns.get(user_id)

        if not click_data or click_data["total_searches"] == 0:
            return {}

        # Calculate click probability per position
        total_searches = click_data["total_searches"]
        position_probs = {
            pos: count / total_searches
            for pos, count in click_data["position_clicks"].items()
        }

        return position_probs

    def should_rerank(self, user_id: str) -> bool:
        """
        Determine if reranking is beneficial for this user

        If user always clicks top result, reranking has high value
        If user explores many results, reranking less critical
        """
        click_bias = self.get_click_position_bias(user_id)

        if not click_bias:
            return True  # Default to reranking

        # Calculate concentration on top-3
        top_3_prob = sum(click_bias.get(i, 0) for i in range(3))

        # If >70% of clicks in top-3, reranking is very important
        return top_3_prob > 0.7

    def get_learning_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get learning statistics for user

        Useful for debugging and transparency
        """
        strategies = {
            key.split(":")[1]: perf
            for key, perf in self.strategy_performance.items()
            if key.startswith(f"{user_id}:")
        }

        weights = self.get_optimal_weights(user_id)
        click_bias = self.get_click_position_bias(user_id)

        return {
            "learned_weights": weights,
            "strategy_performance": strategies,
            "click_position_bias": dict(click_bias),
            "prefers_reranking": self.should_rerank(user_id)
        }
