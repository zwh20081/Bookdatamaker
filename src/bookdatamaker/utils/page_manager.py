"""Page manager for loading and navigating document pages."""

from pathlib import Path
from typing import Dict, List, Optional


class PageManager:
    """Manage document pages in memory with line/column navigation support."""

    def __init__(self, pages: Dict[int, str]) -> None:
        """Initialize page manager.

        Args:
            pages: Dictionary mapping page numbers to page content
        """
        self.pages = pages
        self.page_numbers = sorted(pages.keys())
        self.current_page = self.page_numbers[0] if self.page_numbers else 1
        
        # Build line index for efficient line-based navigation
        self._build_line_index()

    def _build_line_index(self) -> None:
        """Build an index mapping global line numbers to (page, local_line) and paragraph index."""
        self.line_to_page: Dict[int, tuple[int, int]] = {}  # global_line -> (page_num, local_line)
        self.page_line_ranges: Dict[int, tuple[int, int]] = {}  # page_num -> (start_line, end_line)
        self.lines: List[str] = []  # All lines in order
        self.line_to_paragraph: Dict[int, int] = {}  # global_line -> paragraph_num
        self.paragraph_line_ranges: Dict[int, tuple[int, int]] = {}  # paragraph_num -> (start_line, end_line)
        
        global_line = 0
        paragraph_num = 0
        
        for page_num in self.page_numbers:
            content = self.pages[page_num]
            page_lines = content.split('\n')
            
            start_line = global_line
            paragraph_start = global_line
            
            for local_line, line_text in enumerate(page_lines):
                self.line_to_page[global_line] = (page_num, local_line)
                self.lines.append(line_text)
                
                # Track paragraphs (new paragraph on empty line)
                if line_text.strip() == "":
                    # End current paragraph
                    if paragraph_start < global_line:
                        self.paragraph_line_ranges[paragraph_num] = (paragraph_start, global_line - 1)
                        for line_idx in range(paragraph_start, global_line):
                            self.line_to_paragraph[line_idx] = paragraph_num
                        paragraph_num += 1
                    paragraph_start = global_line + 1
                
                global_line += 1
            
            # End paragraph at end of page
            if paragraph_start < global_line:
                self.paragraph_line_ranges[paragraph_num] = (paragraph_start, global_line - 1)
                for line_idx in range(paragraph_start, global_line):
                    self.line_to_paragraph[line_idx] = paragraph_num
                paragraph_num += 1
                paragraph_start = global_line
            
            end_line = global_line - 1
            self.page_line_ranges[page_num] = (start_line, end_line)
        
        self.total_lines = global_line
        self.total_paragraphs = paragraph_num

    @classmethod
    def from_directory(cls, directory: Path) -> "PageManager":
        """Load all pages from directory into memory.

        Args:
            directory: Directory containing page_XXX/ subdirectories with result.mmd files

        Returns:
            PageManager instance with all pages loaded
        """
        pages = {}

        # Load from page_XXX subdirectories with result.mmd files
        for page_dir in sorted(directory.glob("page_*")):
            if not page_dir.is_dir():
                continue
                
            try:
                # Extract page number from directory name (e.g., page_001 -> 1)
                page_num_str = page_dir.name.split("_")[1]
                page_num = int(page_num_str)

                # Read result.mmd file
                result_file = page_dir / "result.mmd"
                if result_file.exists():
                    content = result_file.read_text(encoding="utf-8")
                    pages[page_num] = content
                else:
                    print(f"Warning: No result.mmd found in {page_dir}")

            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse page number from {page_dir}: {e}")

        if not pages:
            raise ValueError(f"No valid page directories found in {directory}")

        return cls(pages)

    @classmethod
    def from_combined_file(cls, combined_file: Path) -> "PageManager":
        """Load pages from combined.txt with page markers or paragraphs.

        Supports two formats:
        1. Page-based: [PAGE_001] markers
        2. Paragraph-based: Double newline separated paragraphs

        Args:
            combined_file: Path to combined.txt file

        Returns:
            PageManager instance with all pages loaded
        """
        content = combined_file.read_text(encoding="utf-8")
        pages = {}
        current_page_num = None
        current_page_content = []

        # Try to parse as page-based format
        for line in content.split("\n"):
            # Check for page marker: [PAGE_001] or [page_001]
            if line.startswith("[") and line.endswith("]"):
                # Save previous page if exists
                if current_page_num is not None:
                    pages[current_page_num] = "\n".join(current_page_content).strip()

                # Parse new page number
                marker = line[1:-1].lower()  # Remove brackets
                if marker.startswith("page_"):
                    try:
                        current_page_num = int(marker.split("_")[1])
                        current_page_content = []
                    except (IndexError, ValueError):
                        pass
            else:
                if current_page_num is not None:
                    current_page_content.append(line)

        # Save last page
        if current_page_num is not None:
            pages[current_page_num] = "\n".join(current_page_content).strip()

        # If no pages found, treat as paragraph-based format
        if not pages:
            # Split by double newlines to get paragraphs
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            # Treat each paragraph as a "page"
            for i, para in enumerate(paragraphs, start=1):
                pages[i] = para

        if not pages:
            raise ValueError(f"No valid pages or paragraphs found in {combined_file}")

        return cls(pages)

    def get_page(self, page_num: int) -> Optional[str]:
        """Get content of a specific page.

        Args:
            page_num: Page number

        Returns:
            Page content or None if page doesn't exist
        """
        return self.pages.get(page_num)

    def get_current_page(self) -> str:
        """Get current page content."""
        return self.pages.get(self.current_page, "")

    def get_current_page_number(self) -> int:
        """Get current page number."""
        return self.current_page

    def get_page_info(self, page_num: Optional[int] = None) -> Dict[str, any]:
        """Get complete information about a page.

        Args:
            page_num: Page number (uses current page if None)

        Returns:
            Dictionary with page information including number, content, total pages
        """
        target_page = page_num if page_num is not None else self.current_page
        
        if target_page not in self.pages:
            return {
                "page_number": target_page,
                "content": None,
                "error": "Page not found",
            }
        
        # Get page index for navigation info
        page_idx = self.page_numbers.index(target_page)
        
        return {
            "page_number": target_page,
            "page_index": page_idx,
            "total_pages": len(self.pages),
            "content": self.pages[target_page],
            "has_previous": page_idx > 0,
            "has_next": page_idx < len(self.pages) - 1,
            "previous_page_number": self.page_numbers[page_idx - 1] if page_idx > 0 else None,
            "next_page_number": self.page_numbers[page_idx + 1] if page_idx < len(self.pages) - 1 else None,
        }

    def next_page(self, steps: int = 1) -> Optional[str]:
        """Move to next page(s).

        Args:
            steps: Number of pages to move forward

        Returns:
            New current page content or None if at end
        """
        current_idx = self.page_numbers.index(self.current_page)
        new_idx = min(current_idx + steps, len(self.page_numbers) - 1)
        self.current_page = self.page_numbers[new_idx]
        return self.get_current_page()

    def previous_page(self, steps: int = 1) -> Optional[str]:
        """Move to previous page(s).

        Args:
            steps: Number of pages to move backward

        Returns:
            New current page content or None if at beginning
        """
        current_idx = self.page_numbers.index(self.current_page)
        new_idx = max(current_idx - steps, 0)
        self.current_page = self.page_numbers[new_idx]
        return self.get_current_page()

    def jump_to_page(self, page_num: int) -> Optional[str]:
        """Jump to specific page.

        Args:
            page_num: Target page number

        Returns:
            Page content or None if page doesn't exist
        """
        if page_num in self.pages:
            self.current_page = page_num
            return self.get_current_page()
        return None

    def get_page_range(self, start: int, end: int) -> Dict[int, str]:
        """Get content of a range of pages.

        Args:
            start: Start page number (inclusive)
            end: End page number (inclusive)

        Returns:
            Dictionary mapping page numbers to content
        """
        return {
            page_num: content
            for page_num, content in self.pages.items()
            if start <= page_num <= end
        }

    def get_context(self, page_num: int, before: int = 1, after: int = 1) -> Dict[str, any]:
        """Get page with surrounding context.

        Args:
            page_num: Target page number
            before: Number of pages before
            after: Number of pages after

        Returns:
            Dictionary with current page and surrounding pages
        """
        current_idx = self.page_numbers.index(page_num)
        start_idx = max(0, current_idx - before)
        end_idx = min(len(self.page_numbers), current_idx + after + 1)

        return {
            "current_page": page_num,
            "total_pages": len(self.pages),
            "content": self.pages[page_num],
            "previous_pages": {
                pn: self.pages[pn]
                for pn in self.page_numbers[start_idx:current_idx]
            },
            "next_pages": {
                pn: self.pages[pn]
                for pn in self.page_numbers[current_idx + 1 : end_idx]
            },
        }

    def get_all_pages(self) -> Dict[int, str]:
        """Get all pages.

        Returns:
            Dictionary mapping page numbers to content
        """
        return self.pages.copy()

    def get_total_pages(self) -> int:
        """Get total number of pages."""
        return len(self.pages)

    def get_total_lines(self) -> int:
        """Get total number of lines across all pages."""
        return self.total_lines

    def get_line(self, line_num: int) -> Optional[str]:
        """Get content of a specific line (0-indexed).

        Args:
            line_num: Global line number (0-indexed)

        Returns:
            Line content or None if line doesn't exist
        """
        if 0 <= line_num < self.total_lines:
            return self.lines[line_num]
        return None

    def get_line_range(self, start_line: int, end_line: int) -> List[str]:
        """Get content of a range of lines (inclusive).

        Args:
            start_line: Start line number (0-indexed, inclusive)
            end_line: End line number (0-indexed, inclusive)

        Returns:
            List of line contents
        """
        start_line = max(0, start_line)
        end_line = min(self.total_lines - 1, end_line)
        
        if start_line > end_line:
            return []
        
        return self.lines[start_line : end_line + 1]

    def get_line_with_context(
        self, line_num: int, before: int = 3, after: int = 3
    ) -> Dict[str, any]:
        """Get line with surrounding context.

        Args:
            line_num: Target line number (0-indexed)
            before: Number of lines before
            after: Number of lines after

        Returns:
            Dictionary with line content and context
        """
        if line_num < 0 or line_num >= self.total_lines:
            return {
                "line_number": line_num,
                "content": None,
                "error": "Line number out of range",
            }

        # Get page information for this line
        page_num, local_line = self.line_to_page[line_num]

        start = max(0, line_num - before)
        end = min(self.total_lines - 1, line_num + after)

        return {
            "line_number": line_num,
            "page_number": page_num,
            "local_line_number": local_line,
            "total_lines": self.total_lines,
            "content": self.lines[line_num],
            "before_lines": self.lines[start:line_num],
            "after_lines": self.lines[line_num + 1 : end + 1],
            "column_count": len(self.lines[line_num]),
        }

    def get_column_range(
        self, line_num: int, start_col: int, end_col: int
    ) -> Optional[str]:
        """Get a range of columns from a specific line.

        Args:
            line_num: Line number (0-indexed)
            start_col: Start column (0-indexed, inclusive)
            end_col: End column (0-indexed, inclusive)

        Returns:
            Substring or None if line doesn't exist
        """
        line = self.get_line(line_num)
        if line is None:
            return None

        start_col = max(0, start_col)
        end_col = min(len(line) - 1, end_col)

        if start_col > end_col:
            return ""

        return line[start_col : end_col + 1]

    def search_text(
        self, query: str, case_sensitive: bool = False, max_results: int = 100
    ) -> List[Dict[str, any]]:
        """Search for text across all pages.

        Args:
            query: Search query
            case_sensitive: Whether to perform case-sensitive search
            max_results: Maximum number of results to return

        Returns:
            List of matches with line numbers and context
        """
        results = []
        search_query = query if case_sensitive else query.lower()

        for line_num, line_text in enumerate(self.lines):
            search_text = line_text if case_sensitive else line_text.lower()

            if search_query in search_text:
                page_num, local_line = self.line_to_page[line_num]

                # Find column position
                col_start = search_text.index(search_query)

                results.append(
                    {
                        "line_number": line_num,
                        "page_number": page_num,
                        "local_line_number": local_line,
                        "column_start": col_start,
                        "column_end": col_start + len(query) - 1,
                        "content": line_text,
                        "match": line_text[col_start : col_start + len(query)],
                    }
                )

                if len(results) >= max_results:
                    break

        return results

    def get_page_line_info(self, page_num: int) -> Optional[Dict[str, int]]:
        """Get line range information for a specific page.

        Args:
            page_num: Page number

        Returns:
            Dictionary with start_line, end_line, line_count or None
        """
        if page_num not in self.page_line_ranges:
            return None

        start_line, end_line = self.page_line_ranges[page_num]
        return {
            "page_number": page_num,
            "start_line": start_line,
            "end_line": end_line,
            "line_count": end_line - start_line + 1,
        }

    def get_paragraph_number(self, line_num: int) -> Optional[int]:
        """Get paragraph number for a specific line.

        Args:
            line_num: Global line number (0-indexed)

        Returns:
            Paragraph number or None if line doesn't exist
        """
        return self.line_to_paragraph.get(line_num)

    def get_paragraph_info(self, paragraph_num: int) -> Optional[Dict[str, any]]:
        """Get information about a specific paragraph.

        Args:
            paragraph_num: Paragraph number (0-indexed)

        Returns:
            Dictionary with paragraph info or None
        """
        if paragraph_num not in self.paragraph_line_ranges:
            return None

        start_line, end_line = self.paragraph_line_ranges[paragraph_num]
        content_lines = self.lines[start_line : end_line + 1]
        content = "\n".join(content_lines)

        # Get page info for the start line
        page_num, _ = self.line_to_page[start_line]

        return {
            "paragraph_number": paragraph_num,
            "start_line": start_line,
            "end_line": end_line,
            "line_count": end_line - start_line + 1,
            "page_number": page_num,
            "content": content,
        }

    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics about the document.

        Returns:
            Dictionary with various statistics
        """
        total_chars = sum(len(line) for line in self.lines)
        non_empty_lines = sum(1 for line in self.lines if line.strip())

        return {
            "total_pages": len(self.pages),
            "total_lines": self.total_lines,
            "total_paragraphs": self.total_paragraphs,
            "total_characters": total_chars,
            "non_empty_lines": non_empty_lines,
            "empty_lines": self.total_lines - non_empty_lines,
            "average_line_length": total_chars / self.total_lines if self.total_lines > 0 else 0,
            "pages": {
                page_num: {
                    "start_line": info[0],
                    "end_line": info[1],
                    "line_count": info[1] - info[0] + 1,
                }
                for page_num, info in self.page_line_ranges.items()
            },
            "paragraphs": {
                para_num: {
                    "start_line": info[0],
                    "end_line": info[1],
                    "line_count": info[1] - info[0] + 1,
                }
                for para_num, info in self.paragraph_line_ranges.items()
            },
        }
