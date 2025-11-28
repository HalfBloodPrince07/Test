# backend/opensearch_client.py - Enhanced OpenSearch Client with Hybrid Search

from opensearchpy import AsyncOpenSearch
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenSearchClient:
    """
    Enhanced OpenSearch client with:
    - Hybrid search (Vector + BM25)
    - Better filtering
    - Improved scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index_name = config['opensearch']['index_name']
        
        # Initialize client with authentication
        self.client = AsyncOpenSearch(
            hosts=[{
                'host': config['opensearch']['host'],
                'port': config['opensearch']['port']
            }],
            http_auth=(
                config['opensearch']['auth']['username'],
                config['opensearch']['auth']['password']
            ),
            use_ssl=config['opensearch'].get('use_ssl', True),
            verify_certs=config['opensearch'].get('verify_certs', False),
            ssl_show_warn=False,
            timeout=30
        )
        
        # Hybrid search settings
        self.hybrid_enabled = config['search']['hybrid']['enabled']
        self.vector_weight = config['search']['hybrid']['vector_weight']
        self.bm25_weight = config['search']['hybrid']['bm25_weight']
        
        logger.info(f"OpenSearch client initialized (Hybrid: {self.hybrid_enabled})")

    async def create_index(self):
        """Create index with k-NN and BM25 settings for hybrid search"""
        mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                },
                "analysis": {
                    "analyzer": {
                        "content_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "filename": {
                        "type": "text",
                        "analyzer": "content_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "file_path": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "content_summary": {
                        "type": "text",
                        "analyzer": "content_analyzer"
                    },
                    # Full content for BM25 search
                    "content_full": {
                        "type": "text",
                        "analyzer": "content_analyzer"
                    },
                    # Extracted keywords for boosting
                    "keywords": {
                        "type": "text",
                        "analyzer": "content_analyzer"
                    },
                    "vector_embedding": {
                        "type": "knn_vector",
                        "dimension": self.config['models']['embedding']['dimension'],
                        "method": {
                            "name": "hnsw",
                            "space_type": "innerproduct",
                            "engine": "faiss",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "cluster_id": {"type": "integer"},
                    "created_at": {"type": "date"},
                    "last_modified": {"type": "date"},
                    # Metadata for better filtering
                    "document_type": {"type": "keyword"},  # invoice, report, image, etc.
                    "entities": {"type": "keyword"},  # Extracted named entities
                    "word_count": {"type": "integer"}
                }
            }
        }
        
        try:
            exists = await self.client.indices.exists(index=self.index_name)
            if not exists:
                await self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created index: {self.index_name}")
            else:
                logger.info(f"Index already exists: {self.index_name}")
        except Exception as e:
            logger.error(f"Index creation error: {e}")
            raise

    async def document_exists(self, doc_id: str) -> bool:
        """Check if a document already exists"""
        try:
            return await self.client.exists(index=self.index_name, id=doc_id)
        except Exception:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def index_document(self, doc: Dict[str, Any]):
        """Index a single document"""
        try:
            await self.client.index(
                index=self.index_name,
                id=doc['id'],
                body=doc,
                refresh=True
            )
            logger.debug(f"Indexed document: {doc['filename']}")
        except Exception as e:
            logger.error(f"Document indexing error: {e}")
            raise

    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 50,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and BM25 text matching.
        Uses Reciprocal Rank Fusion (RRF) to combine scores.
        """
        
        # Vector search query
        vector_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector_embedding": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            },
            "_source": {
                "excludes": ["vector_embedding"]
            }
        }
        
        # BM25 text search query
        bm25_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content_summary^3",
                                    "content_full^2",
                                    "filename^2",
                                    "keywords^4"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        },
                        {
                            "match_phrase": {
                                "content_summary": {
                                    "query": query,
                                    "boost": 2
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": {
                "excludes": ["vector_embedding"]
            }
        }
        
        # Apply filters if provided
        if filters:
            vector_query["query"] = {
                "bool": {
                    "must": [vector_query["query"]],
                    "filter": filters
                }
            }
            bm25_query["query"]["bool"]["filter"] = filters
        
        try:
            # Execute both searches
            vector_response = await self.client.search(
                index=self.index_name, body=vector_query
            )
            bm25_response = await self.client.search(
                index=self.index_name, body=bm25_query
            )
            
            # Combine results using Reciprocal Rank Fusion
            combined = self._reciprocal_rank_fusion(
                vector_response['hits']['hits'],
                bm25_response['hits']['hits'],
                k=60  # RRF constant
            )
            
            return combined[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to vector-only search
            return await self.vector_search(query_vector, top_k, filters)

    def _reciprocal_rank_fusion(
        self,
        vector_hits: List[Dict],
        bm25_hits: List[Dict],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        RRF score = sum(1 / (k + rank))
        """
        scores = {}
        docs = {}
        
        # Score vector results
        for rank, hit in enumerate(vector_hits):
            doc_id = hit['_id']
            rrf_score = self.vector_weight / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in docs:
                docs[doc_id] = {
                    'id': doc_id,
                    'vector_score': hit['_score'],
                    **hit['_source']
                }
        
        # Score BM25 results
        for rank, hit in enumerate(bm25_hits):
            doc_id = hit['_id']
            rrf_score = self.bm25_weight / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in docs:
                docs[doc_id] = {
                    'id': doc_id,
                    'bm25_score': hit['_score'],
                    **hit['_source']
                }
            else:
                docs[doc_id]['bm25_score'] = hit['_score']
        
        # Sort by combined RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids:
            doc = docs[doc_id]
            doc['score'] = scores[doc_id]
            doc['hybrid'] = True
            results.append(doc)
        
        return results

    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 25,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Fallback vector-only search"""
        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector_embedding": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            },
            "_source": {
                "excludes": ["vector_embedding"]
            }
        }
        
        if filters:
            query["query"] = {
                "bool": {
                    "must": [query["query"]],
                    "filter": filters
                }
            }
            
        try:
            response = await self.client.search(index=self.index_name, body=query)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    **hit['_source']
                })
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def delete_document(self, doc_id: str):
        """Delete document from index"""
        try:
            await self.client.delete(index=self.index_name, id=doc_id, refresh=True)
            logger.debug(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Document deletion error: {e}")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = await self.client.count(index=self.index_name)
            return {"count": stats['count']}
        except Exception:
            return {"count": 0}

    async def get_cluster_data(self) -> Optional[Dict[str, Any]]:
        """Get cluster visualization data"""
        try:
            response = await self.client.search(
                index=self.index_name,
                body={
                    "size": 1000,
                    "query": {"match_all": {}},
                    "_source": ["filename", "vector_embedding", "cluster_id"]
                }
            )
            
            if not response['hits']['hits']:
                return None
                
            vectors = []
            labels = []
            cluster_ids = []
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                vectors.append(source['vector_embedding'])
                labels.append(source['filename'])
                cluster_ids.append(source.get('cluster_id', 0))
                
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(np.array(vectors))
            
            return {
                "x": coords_2d[:, 0].tolist(),
                "y": coords_2d[:, 1].tolist(),
                "labels": labels,
                "cluster_ids": cluster_ids
            }
        except Exception as e:
            logger.error(f"Cluster data error: {e}")
            return None

    async def ping(self) -> bool:
        """Check connection"""
        try:
            await self.client.ping()
            return True
        except:
            return False

    async def close(self):
        """Close client connection"""
        await self.client.close()
