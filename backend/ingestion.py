# backend/ingestion.py - Enhanced Document Processing & Embedding Pipeline

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import asyncio
from datetime import datetime
from loguru import logger
import httpx
import base64
import re

# File processors
from pypdf import PdfReader
from docx import Document
import pandas as pd


class IngestionPipeline:
    """
    Enhanced document ingestion pipeline with:
    - Improved chunking strategies
    - Better prompts for summarization
    - Keyword extraction
    - Entity extraction
    """
    
    def __init__(self, config: Dict[str, Any], opensearch_client, status_callback=None):
        self.config = config
        self.opensearch = opensearch_client
        self.ollama_url = config['ollama']['base_url']
        self.model = config['ollama']['model']

        # Chunking settings
        self.chunk_size = config['ingestion']['chunk_size']
        self.chunk_overlap = config['ingestion']['chunk_overlap']

        # HTTP client for Ollama
        self.client = httpx.AsyncClient(timeout=120.0)

        # Status callback for real-time updates
        self.status_callback = status_callback

        logger.info("Enhanced ingestion pipeline initialized")

    async def process_directory(self, directory: Path, task_id: str):
        """Process all files in directory"""
        logger.info(f"Processing directory: {directory}")
        supported_exts = self.config['watcher']['supported_extensions']
        files = []

        for ext in supported_exts:
            files.extend(directory.rglob(f"*{ext}"))

        total_files = len(files)
        logger.info(f"Found {total_files} files to process")

        # Notify status callback
        if self.status_callback:
            await self.status_callback({
                "task_id": task_id,
                "status": "started",
                "message": f"Found {total_files} files to process",
                "total_files": total_files,
                "processed": 0
            })

        batch_size = self.config['watcher']['batch_size']
        for i in range(0, total_files, batch_size):
            batch = files[i:i + batch_size]
            await self._process_batch(batch, task_id, i, total_files)
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1}")

        # Notify completion
        if self.status_callback:
            await self.status_callback({
                "task_id": task_id,
                "status": "completed",
                "message": f"✓ Successfully processed {total_files} files",
                "total_files": total_files,
                "processed": total_files
            })

    async def _process_batch(self, files: List[Path], task_id: str = None, offset: int = 0, total_files: int = 0):
        """Process batch of files concurrently"""
        tasks = [self.process_file(file_path) for file_path in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, (file_path, result) in enumerate(zip(files, results)):
            current_count = offset + idx + 1

            if isinstance(result, Exception):
                logger.error(f"Failed to process {file_path}: {result}")
                if self.status_callback and task_id:
                    await self.status_callback({
                        "task_id": task_id,
                        "status": "processing",
                        "message": f"Error processing: {file_path.name}",
                        "current_file": file_path.name,
                        "processed": current_count,
                        "total_files": total_files
                    })
            elif self.status_callback and task_id:
                # Notify progress for each file (including skipped ones)
                status_msg = f"Skipped: {file_path.name}" if result.get("status") == "skipped" else f"Processed: {file_path.name}"
                await self.status_callback({
                    "task_id": task_id,
                    "status": "processing",
                    "message": status_msg,
                    "current_file": file_path.name,
                    "processed": current_count,
                    "total_files": total_files
                })

    async def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single file through the enhanced pipeline"""
        start_time = datetime.now()

        try:
            # Calculate unique ID
            doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()

            # Check if exists (skip logic)
            if await self.opensearch.document_exists(doc_id):
                logger.info(f"✓ Skipping existing file: {file_path.name}")
                return {"status": "skipped", "id": doc_id}

            logger.info(f"⚙️  Processing file: {file_path.name} (type: {file_path.suffix})")

            # 1. Extract content
            logger.debug(f"  [1/5] Extracting content from {file_path.name}")
            content = await self._extract_content(file_path)
            content_size = len(str(content.get('content', '')))
            logger.debug(f"  ✓ Extracted {content_size} characters")

            # 2. Generate enhanced summary with entities and keywords
            logger.debug(f"  [2/5] Generating summary for {file_path.name}")
            summary_data = await self._generate_enhanced_summary(content, file_path)
            logger.debug(f"  ✓ Summary generated (length: {len(summary_data['summary'])})")
            logger.debug(f"  ✓ Keywords: {summary_data.get('keywords', 'N/A')[:100]}")

            # 3. Generate embedding from summary
            logger.debug(f"  [3/5] Generating embedding for {file_path.name}")
            embedding = await self.generate_embedding(summary_data['summary'])

            # Check if we got a valid embedding (not zero vector)
            is_zero_vector = all(v == 0.0 for v in embedding)
            if is_zero_vector:
                logger.error(f"  ✗ Failed to generate valid embedding for {file_path.name} - got zero vector!")
            else:
                logger.debug(f"  ✓ Embedding generated ({len(embedding)} dimensions)")

            # 4. Determine document type
            logger.debug(f"  [4/5] Classifying document type for {file_path.name}")
            doc_type = self._classify_document(file_path, content)
            logger.debug(f"  ✓ Document type: {doc_type}")

            # 5. Create document object
            document = {
                "id": doc_id,
                "filename": file_path.name,
                "file_path": str(file_path.absolute()),
                "file_type": file_path.suffix.lower(),
                "document_type": doc_type,
                "content_summary": summary_data['summary'],
                "content_full": content.get('content', '')[:10000],  # Store truncated full content
                "keywords": summary_data.get('keywords', ''),
                "entities": summary_data.get('entities', []),
                "vector_embedding": embedding,
                "cluster_id": 0,
                "word_count": len(content.get('content', '').split()),
                "created_at": datetime.now().isoformat(),
                "last_modified": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat()
            }

            # 6. Index in OpenSearch
            logger.debug(f"  [5/5] Indexing {file_path.name} in OpenSearch")
            await self.opensearch.index_document(document)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Successfully indexed: {file_path.name} (took {elapsed:.2f}s)")

            return document

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"✗ Error processing {file_path.name} after {elapsed:.2f}s: {type(e).__name__}: {e}")
            raise

    def _classify_document(self, file_path: Path, content: Dict) -> str:
        """Classify document type based on content and filename"""
        filename_lower = file_path.name.lower()
        suffix = file_path.suffix.lower()
        
        # Image types
        if suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            if 'screenshot' in filename_lower:
                return 'screenshot'
            elif 'diagram' in filename_lower or 'chart' in filename_lower:
                return 'diagram'
            return 'image'
        
        # Document types based on filename patterns
        if 'invoice' in filename_lower:
            return 'invoice'
        elif 'receipt' in filename_lower:
            return 'receipt'
        elif 'report' in filename_lower:
            return 'report'
        elif 'contract' in filename_lower or 'agreement' in filename_lower:
            return 'contract'
        elif 'resume' in filename_lower or 'cv' in filename_lower:
            return 'resume'
        elif 'presentation' in filename_lower:
            return 'presentation'
        
        # Spreadsheet types
        if suffix in ['.xlsx', '.csv']:
            return 'spreadsheet'
        
        # Default based on extension
        if suffix == '.pdf':
            return 'pdf_document'
        elif suffix == '.docx':
            return 'word_document'
        elif suffix in ['.txt', '.md']:
            return 'text_document'
        
        return 'document'

    async def _extract_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content based on file type"""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            return {"type": "text", "content": text}
                
        elif suffix == '.pdf':
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return {"type": "text", "content": text.strip()}
            except Exception as e:
                logger.warning(f"PDF extraction failed: {e}")
                return {"type": "text", "content": f"PDF file: {file_path.name}"}
                
        elif suffix == '.docx':
            try:
                doc = Document(file_path)
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                text = "\n".join(paragraphs)
                return {"type": "text", "content": text}
            except Exception as e:
                logger.warning(f"DOCX extraction failed: {e}")
                return {"type": "text", "content": f"DOCX file: {file_path.name}"}
                
        elif suffix in ['.xlsx', '.csv']:
            try:
                if suffix == '.xlsx':
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                    
                text = f"Spreadsheet: {file_path.name}\n"
                text += f"Shape: {len(df)} rows × {len(df.columns)} columns\n"
                text += f"Columns: {', '.join(str(c) for c in df.columns)}\n\n"
                text += "Sample data:\n"
                text += df.head(20).to_string()
                return {"type": "spreadsheet", "content": text, "dataframe": df}
            except Exception as e:
                logger.warning(f"Spreadsheet extraction failed: {e}")
                return {"type": "text", "content": f"Spreadsheet file: {file_path.name}"}
                
        elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            return {"type": "image", "path": str(file_path)}
            
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    async def _generate_enhanced_summary(
        self,
        content: Dict[str, Any],
        file_path: Path
    ) -> Dict[str, Any]:
        """Generate enhanced summary with keywords and entities"""
        
        if content['type'] == 'image':
            return await self._process_image(content, file_path)
        else:
            return await self._process_text(content, file_path)

    async def _process_image(self, content: Dict, file_path: Path, max_retries: int = 3) -> Dict[str, Any]:
        """Process image with enhanced captioning prompt and retry logic"""

        for attempt in range(max_retries):
            try:
                # Read and encode image
                logger.debug(f"Attempt {attempt + 1}/{max_retries}: Reading image {file_path.name}")
                with open(content['path'], 'rb') as f:
                    image_data = f.read()

                if len(image_data) == 0:
                    logger.error(f"Image file is empty: {file_path.name}")
                    break

                image_base64 = base64.b64encode(image_data).decode('utf-8')
                logger.debug(f"Image encoded to base64, size: {len(image_base64)} chars")

                prompt = """Analyze this image comprehensively and provide a detailed description.

Your description should include:
1. **Main Subject**: What is the primary focus of this image?
2. **Objects & Elements**: List all visible objects, people, or elements
3. **Text Content**: If there's any text visible, transcribe it exactly
4. **Visual Style**: Colors, composition, quality (photo, screenshot, diagram, etc.)
5. **Context Clues**: What does this image appear to be about or used for?
6. **Notable Details**: Any logos, brands, dates, or identifiable information

Provide a comprehensive description (3-5 sentences) that would help someone find this image through a text search. Be specific and detailed."""

                logger.debug(f"Sending image to Ollama model: {self.model}")
                response = await self.client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "images": [image_base64],
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 500
                        }
                    },
                    timeout=120.0  # Longer timeout for vision models
                )

                response.raise_for_status()
                result = response.json()

                # Validate response
                if 'response' not in result:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: No 'response' field in Ollama result")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        break

                summary = result.get('response', '').strip()

                # Check if we got a meaningful response
                if not summary or len(summary) < 10:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: Received empty or too short caption: '{summary}'")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        summary = f"Image file: {file_path.name}"

                logger.info(f"Successfully generated image caption for {file_path.name} (length: {len(summary)})")

                # Extract keywords from image description
                keywords = await self._extract_keywords(summary)

                return {
                    "summary": summary,
                    "keywords": keywords,
                    "entities": []
                }

            except httpx.HTTPStatusError as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: HTTP error during image captioning: {e.response.status_code}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except httpx.TimeoutException:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Timeout during image captioning")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Image captioning failed: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue

        # All retries failed - return fallback
        logger.warning(f"Using fallback description for image: {file_path.name}")
        return {
            "summary": f"Image file: {file_path.name}. Unable to generate detailed description.",
            "keywords": file_path.stem.replace('_', ' ').replace('-', ' '),
            "entities": []
        }

    async def _process_text(self, content: Dict, file_path: Path) -> Dict[str, Any]:
        """Process text document with enhanced summarization"""
        text = content.get('content', '')
        
        # Handle empty or very short content
        if len(text.strip()) < 50:
            return {
                "summary": f"Document: {file_path.name}",
                "keywords": file_path.stem.replace('_', ' ').replace('-', ' '),
                "entities": []
            }
        
        # Truncate for prompt
        max_length = 3000
        truncated = text[:max_length] + ("..." if len(text) > max_length else "")
        
        prompt = f"""You are a document analysis assistant. Analyze this document and provide:

1. **Summary**: A concise 2-3 sentence summary capturing the document's main purpose and key information.

2. **Keywords**: 5-10 important keywords or phrases that would help find this document in a search.

3. **Entities**: Any important names, organizations, dates, locations, or other specific entities mentioned.

---
DOCUMENT CONTENT:
{truncated}
---

Respond in this exact format:
SUMMARY: [Your 2-3 sentence summary]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
ENTITIES: [entity1, entity2, entity3, ...]"""

        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 600
                    }
                }
            )
            result = response.json()
            response_text = result.get('response', '')
            
            # Parse structured response
            parsed = self._parse_summary_response(response_text)
            
            # Fallback if parsing fails
            if not parsed['summary']:
                parsed['summary'] = text[:500]
            
            return parsed
            
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return {
                "summary": text[:500],
                "keywords": file_path.stem.replace('_', ' ').replace('-', ' '),
                "entities": []
            }

    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured summary response"""
        result = {
            "summary": "",
            "keywords": "",
            "entities": []
        }
        
        # Extract summary
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?=KEYWORDS:|ENTITIES:|$)', response, re.DOTALL | re.IGNORECASE)
        if summary_match:
            result['summary'] = summary_match.group(1).strip()
        
        # Extract keywords
        keywords_match = re.search(r'KEYWORDS:\s*(.+?)(?=ENTITIES:|$)', response, re.DOTALL | re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # Clean up brackets and format
            keywords_text = re.sub(r'[\[\]]', '', keywords_text)
            result['keywords'] = keywords_text
        
        # Extract entities
        entities_match = re.search(r'ENTITIES:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
        if entities_match:
            entities_text = entities_match.group(1).strip()
            entities_text = re.sub(r'[\[\]]', '', entities_text)
            entities = [e.strip() for e in entities_text.split(',') if e.strip()]
            result['entities'] = entities[:20]  # Limit to 20 entities
        
        return result

    async def _extract_keywords(self, text: str) -> str:
        """Extract keywords from text using simple NLP"""
        # Simple keyword extraction - remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'as',
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
            'between', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'image', 'shows',
            'appears', 'visible', 'contains', 'includes', 'features'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter and count
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Get top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:15]]
        
        return ', '.join(keywords)

    async def generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding using Nomic-Embed with retry logic"""

        # Clean and validate input text
        text = text.strip()
        if not text:
            logger.warning("Empty text provided for embedding. Using zero vector.")
            return [0.0] * self.config['models']['embedding']['dimension']

        # Truncate to model's token limit
        text = text[:8000]  # Nomic has ~8k token limit

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": "nomic-embed-text",
                        "prompt": text
                    },
                    timeout=60.0
                )

                # Check HTTP status
                response.raise_for_status()

                result = response.json()
                embedding = result.get('embedding')

                # Validate embedding
                if embedding is None:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: Ollama returned null embedding")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait before retry
                        continue
                    else:
                        logger.error("All retry attempts exhausted. Using zero vector.")
                        return [0.0] * self.config['models']['embedding']['dimension']

                if not isinstance(embedding, list) or len(embedding) == 0:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: Invalid embedding format: {type(embedding)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.error("All retry attempts exhausted. Using zero vector.")
                        return [0.0] * self.config['models']['embedding']['dimension']

                # Verify dimension matches expected size
                expected_dim = self.config['models']['embedding']['dimension']
                if len(embedding) != expected_dim:
                    logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}")
                    # Pad or truncate if necessary
                    if len(embedding) < expected_dim:
                        embedding.extend([0.0] * (expected_dim - len(embedding)))
                    else:
                        embedding = embedding[:expected_dim]

                logger.debug(f"Successfully generated embedding with dimension {len(embedding)}")
                return embedding

            except httpx.HTTPStatusError as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: HTTP error: {e.response.status_code} - {e.response.text}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
            except httpx.TimeoutException:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Request timeout")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Embedding generation failed: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue

        # All retries failed
        logger.error("All embedding generation attempts failed. Using zero vector.")
        return [0.0] * self.config['models']['embedding']['dimension']

    async def expand_query(self, query: str) -> List[str]:
        """Expand query with related terms for better recall"""
        prompt = f"""Given this search query, generate 2-3 alternative phrasings or related search terms that might help find relevant documents.

Query: "{query}"

Return only the alternative queries, one per line, without numbering or explanation."""

        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 100
                    }
                }
            )
            result = response.json()
            expansions = result.get('response', '').strip().split('\n')
            expansions = [e.strip() for e in expansions if e.strip() and len(e.strip()) > 3]
            return expansions[:3]
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()