# backend/reranker.py - Enhanced Cross-Encoder Reranking

from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from loguru import logger
import numpy as np


class CrossEncoderReranker:
    """
    Enhanced two-stage retrieval reranker with:
    - Cross-encoder scoring
    - Score normalization
    - Diversity-aware reranking (MMR)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_name = config['models']['cross_encoder']['name']
        
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        
        self.max_length = config['models']['cross_encoder'].get('max_length', 512)
        logger.info("Cross-encoder loaded and ready")
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        diversity_weight: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder with optional diversity.
        
        Args:
            query: Search query
            documents: Candidate documents from retrieval
            top_k: Number of results to return
            diversity_weight: Weight for MMR diversity (0 = pure relevance)
        
        Returns:
            Reranked documents with normalized scores
        """
        
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = []
        for doc in documents:
            # Combine summary with keywords for better matching
            text = doc.get('content_summary', '')
            keywords = doc.get('keywords', '')
            if keywords:
                text = f"{text}\nKeywords: {keywords}"
            
            # Truncate if too long
            text = text[:self.max_length * 4]  # Rough char limit
            pairs.append([query, text])
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Normalize scores to 0-1 range using sigmoid
        normalized_scores = self._normalize_scores(scores)
        
        # Apply diversity-aware reranking if requested
        if diversity_weight > 0:
            selected_indices = self._mmr_rerank(
                normalized_scores, 
                documents, 
                top_k, 
                diversity_weight
            )
        else:
            # Simple sorting by score
            selected_indices = np.argsort(normalized_scores)[::-1][:top_k]
        
        # Build result list
        reranked = []
        for idx in selected_indices:
            doc = documents[idx].copy()
            doc['score'] = float(normalized_scores[idx])
            doc['raw_score'] = float(scores[idx])
            doc['reranked'] = True
            
            # Remove vector embedding from response (too large)
            doc.pop('vector_embedding', None)
            
            reranked.append(doc)
        
        logger.info(f"Reranked {len(documents)} → {len(reranked)} documents")
        
        return reranked
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to 0-1 range using sigmoid transformation.
        This handles the varying score ranges from cross-encoders.
        """
        # Sigmoid normalization
        normalized = 1 / (1 + np.exp(-scores))
        return normalized
    
    def _mmr_rerank(
        self,
        scores: np.ndarray,
        documents: List[Dict],
        top_k: int,
        diversity_weight: float
    ) -> List[int]:
        """
        Maximal Marginal Relevance (MMR) reranking for diversity.
        
        Balances relevance with diversity to avoid returning
        very similar documents.
        """
        selected = []
        candidates = list(range(len(documents)))
        
        # Select documents iteratively
        while len(selected) < top_k and candidates:
            best_score = -float('inf')
            best_idx = None
            
            for idx in candidates:
                # Relevance score
                relevance = scores[idx]
                
                # Diversity penalty (similarity to already selected)
                if selected:
                    max_similarity = max(
                        self._doc_similarity(documents[idx], documents[s])
                        for s in selected
                    )
                    diversity_penalty = diversity_weight * max_similarity
                else:
                    diversity_penalty = 0
                
                # MMR score
                mmr_score = relevance - diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)
        
        return selected
    
    def _doc_similarity(self, doc1: Dict, doc2: Dict) -> float:
        """
        Estimate similarity between two documents.
        Uses simple Jaccard similarity on keywords.
        """
        kw1 = set(doc1.get('keywords', '').lower().split(', '))
        kw2 = set(doc2.get('keywords', '').lower().split(', '))
        
        if not kw1 or not kw2:
            return 0.0
        
        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)
        
        return intersection / union if union > 0 else 0.0
    
    async def explain_ranking(
        self,
        query: str,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Explain why a document ranks high for a query.
        Useful for debugging and transparency.
        """
        text = document.get('content_summary', '')
        
        # Get score
        score = self.model.predict([[query, text]])[0]
        
        # Find matching terms
        query_terms = set(query.lower().split())
        doc_terms = set(text.lower().split())
        matching_terms = query_terms & doc_terms
        
        # Extract key phrases that might be relevant
        keywords = document.get('keywords', '').split(', ')
        relevant_keywords = [
            kw for kw in keywords 
            if any(qt in kw.lower() for qt in query_terms)
        ]
        
        return {
            "score": float(score),
            "normalized_score": float(1 / (1 + np.exp(-score))),
            "matching_terms": list(matching_terms),
            "relevant_keywords": relevant_keywords,
            "explanation": self._generate_explanation(
                score, matching_terms, relevant_keywords
            )
        }
    
    def _generate_explanation(
        self,
        score: float,
        matching_terms: set,
        relevant_keywords: List[str]
    ) -> str:
        """Generate human-readable explanation"""
        if score > 5:
            relevance = "highly relevant"
        elif score > 2:
            relevance = "moderately relevant"
        elif score > 0:
            relevance = "somewhat relevant"
        else:
            relevance = "marginally relevant"
        
        explanation = f"This document is {relevance} to your query"
        
        if matching_terms:
            explanation += f" (matches: {', '.join(list(matching_terms)[:5])})"
        
        if relevant_keywords:
            explanation += f" with related topics: {', '.join(relevant_keywords[:3])}"
        
        return explanation + "."
