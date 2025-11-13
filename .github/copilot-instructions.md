# Book Data Maker - Copilot Instructions

## Project Overview
This is a Python CLI application for extracting text from documents using DeepSeek OCR and generating datasets in Parquet format.

## Key Features
- DeepSeek OCR text extraction
- MCP (Model Context Protocol) server for paragraph navigation
- LLM integration for dataset generation
- Two separate tasks: extraction and generation
- OpenAI API URL customization via CLI
- Processing status indicators

## Development Guidelines
- Use Python 3.10+ with type hints
- Follow PEP 8 style guidelines
- Use async/await for API calls
- Implement proper error handling and logging
- Keep modules focused and testable

## Project Structure
- `src/bookdatamaker/` - Main application code
  - `cli.py` - CLI interface
  - `ocr/` - OCR extraction module
  - `mcp/` - MCP server implementation
  - `llm/` - LLM integration
  - `dataset/` - Dataset generation
- `tests/` - Test files
- `pyproject.toml` - Project configuration
