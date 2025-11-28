# backend/watcher.py - Enhanced Real-time File System Monitoring

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
from loguru import logger
from collections import defaultdict
import time
import hashlib


class FileWatcher:
    """
    Enhanced real-time file system watcher with:
    - Debouncing to prevent duplicate processing
    - Batch processing for efficiency
    - Status callbacks for UI updates
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        ingestion_pipeline, 
        opensearch_client,
        status_callback: Optional[callable] = None
    ):
        self.config = config
        self.ingestion = ingestion_pipeline
        self.opensearch = opensearch_client
        self.status_callback = status_callback
        self.observer = Observer()
        self.is_running = False
        
        # Debouncing
        self.debounce_seconds = config['watcher']['debounce_seconds']
        self.pending_files = defaultdict(float)
        
        logger.info("Enhanced file watcher initialized")
    
    async def emit_status(self, message: str, file_path: Optional[str] = None):
        """Emit status update for UI feedback"""
        if self.status_callback:
            await self.status_callback({
                "type": "watcher",
                "message": message,
                "file": file_path,
                "timestamp": time.time()
            })
        logger.info(f"[Watcher] {message}")
    
    async def start(self, directory: Path):
        """Start watching directory"""
        
        # Capture the running event loop
        loop = asyncio.get_running_loop()
        
        handler = EnhancedDebounceHandler(
            self.ingestion,
            self.opensearch,
            self.debounce_seconds,
            self.config['watcher']['supported_extensions'],
            loop,
            self.status_callback
        )
        
        self.observer.schedule(handler, str(directory), recursive=True)
        self.observer.start()
        self.is_running = True
        
        await self.emit_status(f"👀 Now watching: {directory}")
        
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop watching"""
        # --- FIX: Safe shutdown check ---
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        # --------------------------------
            
        self.is_running = False
        logger.info("File watcher stopped")


class EnhancedDebounceHandler(FileSystemEventHandler):
    """
    Enhanced event handler with:
    - Debouncing
    - Batch collection
    - Async processing
    - Status updates
    """
    
    def __init__(
        self, 
        ingestion, 
        opensearch, 
        debounce_seconds: float, 
        supported_exts: List[str], 
        loop,
        status_callback: Optional[callable] = None
    ):
        self.ingestion = ingestion
        self.opensearch = opensearch
        self.debounce_seconds = debounce_seconds
        self.supported_exts = [ext.lower() for ext in supported_exts]
        self.loop = loop
        self.status_callback = status_callback
        
        self.pending = {}
        self.processing = set()
    
    def _is_supported(self, path: Path) -> bool:
        """Check if file type is supported"""
        return path.suffix.lower() in self.supported_exts
    
    def _is_temp_file(self, path: Path) -> bool:
        """Check if file is a temporary file"""
        name = path.name
        return (
            name.startswith('.') or
            name.startswith('~') or
            name.endswith('.tmp') or
            name.endswith('.temp') or
            '~$' in name  # Office temp files
        )
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation"""
        if not event.is_directory:
            path = Path(event.src_path)
            if self._is_supported(path) and not self._is_temp_file(path):
                logger.info(f"📄 New file detected: {path.name}")
                self._schedule_processing(path, "created")
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification"""
        if not event.is_directory:
            path = Path(event.src_path)
            if self._is_supported(path) and not self._is_temp_file(path):
                # Only process if not already pending
                if str(path) not in self.processing:
                    logger.info(f"✏️ File modified: {path.name}")
                    self._schedule_processing(path, "modified")
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion"""
        if not event.is_directory:
            path = Path(event.src_path)
            if self._is_supported(path):
                logger.info(f"🗑️ File deleted: {path.name}")
                
                asyncio.run_coroutine_threadsafe(
                    self._delete_from_index(path),
                    self.loop
                )
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename"""
        if not event.is_directory:
            src_path = Path(event.src_path)
            dest_path = Path(event.dest_path)
            
            # Delete old entry
            if self._is_supported(src_path):
                asyncio.run_coroutine_threadsafe(
                    self._delete_from_index(src_path),
                    self.loop
                )
            
            # Index new location
            if self._is_supported(dest_path) and not self._is_temp_file(dest_path):
                logger.info(f"📦 File moved: {src_path.name} → {dest_path.name}")
                self._schedule_processing(dest_path, "moved")
    
    def _schedule_processing(self, path: Path, event_type: str):
        """Schedule file for processing with debounce"""
        path_str = str(path)
        self.pending[path_str] = {
            "time": time.time(),
            "event": event_type
        }
        
        asyncio.run_coroutine_threadsafe(
            self._process_after_debounce(path),
            self.loop
        )
    
    async def _process_after_debounce(self, path: Path):
        """Process file after debounce period"""
        await asyncio.sleep(self.debounce_seconds)
        
        path_str = str(path)
        if path_str not in self.pending:
            return
        
        pending_info = self.pending[path_str]
        last_update = pending_info["time"]
        
        # Check if no new updates in debounce period
        if time.time() - last_update >= self.debounce_seconds:
            del self.pending[path_str]
            self.processing.add(path_str)
            
            try:
                # Emit status
                if self.status_callback:
                    await self._emit_status(f"⏳ Processing: {path.name}")
                
                # Process file
                await self.ingestion.process_file(path)
                
                if self.status_callback:
                    await self._emit_status(f"✅ Indexed: {path.name}")
                
                logger.info(f"✅ Processed: {path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {path.name}: {e}")
                if self.status_callback:
                    await self._emit_status(f"❌ Failed: {path.name}")
            
            finally:
                self.processing.discard(path_str)
    
    async def _delete_from_index(self, path: Path):
        """Delete document from OpenSearch"""
        doc_id = hashlib.md5(str(path.absolute()).encode()).hexdigest()
        
        try:
            await self.opensearch.delete_document(doc_id)
            
            if self.status_callback:
                await self._emit_status(f"🗑️ Removed from index: {path.name}")
            
            logger.info(f"🗑️ Deleted from index: {path.name}")
            
        except Exception as e:
            logger.error(f"Error deleting {path.name}: {e}")
    
    async def _emit_status(self, message: str):
        """Emit status callback"""
        if self.status_callback:
            try:
                await self.status_callback({
                    "type": "watcher",
                    "message": message,
                    "timestamp": time.time()
                })
            except Exception:
                pass  # Ignore callback errors