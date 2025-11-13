"""Tests for OCR extractor."""

import pytest
from pathlib import Path
from bookdatamaker.ocr import OCRExtractor


class TestOCRExtractor:
    """Test OCR extraction functionality."""

    def test_split_paragraphs_double_newline(self):
        """Test splitting text by double newlines."""
        extractor = OCRExtractor("fake-key")
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        
        paragraphs = extractor.split_into_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert paragraphs[0] == "Paragraph 1"
        assert paragraphs[1] == "Paragraph 2"
        assert paragraphs[2] == "Paragraph 3"

    def test_split_paragraphs_single_newline(self):
        """Test splitting text by single newlines when no double newlines."""
        extractor = OCRExtractor("fake-key")
        text = "Line 1\nLine 2\nLine 3"
        
        paragraphs = extractor.split_into_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert paragraphs[0] == "Line 1"

    def test_split_paragraphs_empty(self):
        """Test splitting empty text."""
        extractor = OCRExtractor("fake-key")
        text = ""
        
        paragraphs = extractor.split_into_paragraphs(text)
        
        assert len(paragraphs) == 0
