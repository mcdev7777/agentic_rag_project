import argparse
import asyncio
from datetime import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

from agent.db_utils import close_database, initialize_database, db_pool, execute_init_sql
from agent.models import IngestionConfig, IngestionResult
# Import the PDF extractor
from .extract_files import create_pdf_extractor, PDFExtractionConfig
from .chunker import ChunkingConfig, DocumentChunk, create_chunker
from agent.providers import get_embedding_model

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents with table/image processing into vector DB"""

    def __init__(self, config: IngestionConfig, documents_folder: str = "documents", 
                clean_before_ingest: bool = False, sql_schema_path: str = "sql/schema.sql"):
        """
        Initialize ingestion pipeline.
        
        Args:
            config: Ingestion configuration
            documents_folder: Folder containing PDF documents
            clean_before_ingest: Whether to clean existing data before ingestion
            sql_schema_path: Path to SQL schema file
        """
        
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.sql_schema_path = sql_schema_path
        
        # Configure  PDF extraction
        self.extractor_config = PDFExtractionConfig(
            enable_ocr=False,
            images_scale=1.0,
            include_images=True,
            include_tables=True,
        )

        # Configure chunking
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )

        # Create PDF extractor
        self.extractor = create_pdf_extractor(self.extractor_config)
        # Create chunker
        self.chunker = create_chunker(self.chunker_config)
        self._initialized = False
        
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing  ingestion pipeline...")
        
        # Initialize database connections
        await initialize_database()
        await execute_init_sql(self.sql_schema_path)

        self._initialized = True
        logger.info("ingestion pipeline initialized")
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            await close_database()
            self._initialized = False
        
    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")
        
        # Clean PostgreSQL
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        
        logger.info("Cleaned PostgreSQL database")

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document with  image processing.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            Ingestion result
        """
        start_time = datetime.now()
        
        # Extract document content with  extraction
        document_content, document_metadata = self.extractor.extract_pdf_content(file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)
        document_title = document_metadata.get("title", document_source)

        logger.info(f"Processing document: {document_title}")
        logger.info(f"Found {document_metadata.get('pictures', 0)} images and {document_metadata.get('tables', 0)} tables")

        # Chunk the main document content
        main_chunks = self.chunker.chunk_content(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata
        )
        
        if not main_chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        logger.info(f"Total chunks created: {len(main_chunks)}")
        
        # Generate embeddings for all chunks
        embedded_chunks = await self.aembed_chunks(
            chunks=main_chunks,
            model=get_embedding_model()
        )
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Save to PostgreSQL Pgvector
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata
        )
        
        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(main_chunks),
            processing_time_ms=processing_time,
        )
    
    async def ingest_documents(self, progress_callback: Optional[callable] = None) -> List[IngestionResult]:
        """
        Ingest all documents from the documents folder with  processing.
        
        Args:
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()
        
        # Clean existing data if requested
        if self.clean_before_ingest:
            await self._clean_databases()

        # Find all PDF files
        pdf_files = self._find_pdfs_in_directory(self.documents_folder)

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_folder}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        results = []

        for i, file_path in enumerate(pdf_files):
            try:
                logger.info(f"Processing file {i+1}/{len(pdf_files)}: {file_path}")

                result = await self._ingest_single_document(file_path)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(pdf_files))
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    processing_time_ms=0,
                ))
        
        # Log summary
        total_chunks = sum(r.chunks_created for r in results)

        logger.info(f" ingestion complete: {len(results)} documents, {total_chunks} chunks")


        return results

    async def aembed_chunks(self, chunks: List[DocumentChunk], model: str = "text-embedding-3-small") -> List[DocumentChunk]:
        """Generate embeddings for chunks (LangChain handles batching internally)."""
        embeddings = OpenAIEmbeddings(model=model)

        # Tüm chunk içeriklerini al
        texts = [chunk.content for chunk in chunks]

        vectors = await embeddings.aembed_documents(texts)

        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            embedded_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "embedding_model": model,
                    "embedding_generated_at": datetime.now().isoformat(),
                },
            )
            embedded_chunk.embedding = vector
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def _find_pdfs_in_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """
        Find all PDF files in a directory.

        Args:
            directory: Directory path to search
            recursive: Whether to search subdirectories

        Returns:
            List of PDF file paths
        """
        directory_path = Path(directory)

        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found or not a directory: {directory_path}")

        if recursive:
            pdf_files = list(directory_path.rglob("*.pdf"))
        else:
            pdf_files = list(directory_path.glob("*.pdf"))

        pdf_paths = [str(pdf.resolve()) for pdf in pdf_files if pdf.is_file()]
        logger.info(f"Found {len(pdf_paths)} PDF files in {directory_path}")

        return pdf_paths

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to PostgreSQL with  metadata."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata)
                )
                
                document_id = document_result["id"]
                
                # Insert chunks with content type tracking
                for chunk in chunks:
                    embedding_data = None
                    if hasattr(chunk, 'embedding') and chunk.embedding:
                        # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
                        embedding_data = '[' + ','.join(map(str, chunk.embedding)) + ']'

                    # Metadata for different content types
                    metadata = {
                        **chunk.metadata,
                        "chunk_type": chunk.metadata.get("content_type", "text")
                    }

                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(metadata),
                        chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.content.split())
                    )
                
                return document_id


async def main():
    """Main function for running  ingestion."""
    parser = argparse.ArgumentParser(description=" Document ingestion with table/image processing")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean existing data before ingestion")
    parser.add_argument("--chunk-size", type=int, default=850, help="Chunk size for splitting documents")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic chunking")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sql-schema-path", "-sql", default="sql/schema.sql", help="Path to SQL schema file")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic,
    )
    
    # Create and run  pipeline
    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        sql_schema_path=args.sql_schema_path
    )


    if args.no_images:
        pipeline.extractor_config.include_images = False
    if args.no_tables:
        pipeline.extractor_config.include_tables = False

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")
    
    try:
        start_time = datetime.now()
        
        results = await pipeline.ingest_documents(progress_callback)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print  summary
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Images extracted: {pipeline.extractor_config.include_images}")
        print(f"Tables extracted: {pipeline.extractor_config.include_tables}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print("="*60)
        
        # Print per-document summary
        for result in results:
            if result.chunks_created > 0:
                logger.info(f"{result.title}: {result.chunks_created} chunks ({result.processing_time_ms/1000:.1f}s)")
            else:
                logger.warning(f"{result.title}: Failed to process")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f" ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())