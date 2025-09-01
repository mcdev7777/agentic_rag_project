import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Disable symlink warnings

@dataclass
class PDFExtractionConfig:
    """ PDF extraction configuration with GPU support"""
    enable_ocr: bool = True
    images_scale: float = 2.0
    include_images: bool = True
    include_tables: bool = True 

class PDFExtractor:
    """ PDF content extractor using Docling with GPU support, without table extraction."""
    
    def __init__(self, config: PDFExtractionConfig = None):
        self.config = config or PDFExtractionConfig()
        self.setup_converter()
    
    def setup_converter(self):
        """Setup Docling document converter with options."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.enable_ocr
        pipeline_options.do_picture_description = self.config.include_images
        pipeline_options.do_table_structure = self.config.include_tables
        pipeline_options.images_scale = self.config.images_scale
        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        except Exception as e:
            logging.error(f"Failed to initialize docling: {e}")

    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract content from a single PDF file."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting content from: {pdf_path.name}")
        start_time = time.time()
                
        result = self.converter.convert(str(pdf_path))
        end_time = time.time()
        
        doc = result.document
        content_text = doc.export_to_markdown()
        
        metadata = {
            "source": str(pdf_path),
            "title": pdf_path.stem,
            "processing_time": round(end_time - start_time, 2),
            "pages": len(doc.pages),
            "texts": len(doc.texts),
            "pictures": len(doc.pictures),
            "tables": len(doc.tables),
            "extraction_method": "docling",
            "content_type": "pdf"
        }        
        return content_text, metadata
    

def create_pdf_extractor(config: PDFExtractionConfig = None) -> PDFExtractor:
    """Create PDF extractor instance"""
    return PDFExtractor(config)
