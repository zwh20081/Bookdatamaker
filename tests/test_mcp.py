"""Tests for MCP server."""

import pytest
from bookdatamaker.mcp import ParagraphNavigator


class TestParagraphNavigator:
    """Test paragraph navigation."""

    def setup_method(self):
        """Set up test paragraphs."""
        self.paragraphs = [
            "First paragraph",
            "Second paragraph",
            "Third paragraph",
            "Fourth paragraph",
            "Fifth paragraph",
        ]
        self.navigator = ParagraphNavigator(self.paragraphs)

    def test_initial_position(self):
        """Test initial position is 0."""
        assert self.navigator.current_index == 0
        assert self.navigator.get_current() == "First paragraph"

    def test_move_forward(self):
        """Test moving forward."""
        result = self.navigator.move_forward()
        assert self.navigator.current_index == 1
        assert result == "Second paragraph"

    def test_move_forward_multiple(self):
        """Test moving forward multiple steps."""
        result = self.navigator.move_forward(2)
        assert self.navigator.current_index == 2
        assert result == "Third paragraph"

    def test_move_backward(self):
        """Test moving backward."""
        self.navigator.jump_to(2)
        result = self.navigator.move_backward()
        assert self.navigator.current_index == 1
        assert result == "Second paragraph"

    def test_move_backward_multiple(self):
        """Test moving backward multiple steps."""
        self.navigator.jump_to(4)
        result = self.navigator.move_backward(2)
        assert self.navigator.current_index == 2
        assert result == "Third paragraph"

    def test_jump_to(self):
        """Test jumping to specific index."""
        result = self.navigator.jump_to(3)
        assert self.navigator.current_index == 3
        assert result == "Fourth paragraph"

    def test_boundary_forward(self):
        """Test forward boundary."""
        self.navigator.move_forward(10)
        assert self.navigator.current_index == 4
        assert self.navigator.get_current() == "Fifth paragraph"

    def test_boundary_backward(self):
        """Test backward boundary."""
        self.navigator.move_backward(10)
        assert self.navigator.current_index == 0
        assert self.navigator.get_current() == "First paragraph"

    def test_get_context(self):
        """Test getting context."""
        self.navigator.jump_to(2)
        context = self.navigator.get_context(before=1, after=1)
        
        assert context["current_index"] == 2
        assert context["total_paragraphs"] == 5
        assert context["current"] == "Third paragraph"
        assert context["previous"] == ["Second paragraph"]
        assert context["next"] == ["Fourth paragraph"]

    def test_get_context_at_start(self):
        """Test getting context at start."""
        context = self.navigator.get_context(before=2, after=2)
        
        assert context["current"] == "First paragraph"
        assert context["previous"] == []
        assert len(context["next"]) == 2

    def test_get_context_at_end(self):
        """Test getting context at end."""
        self.navigator.jump_to(4)
        context = self.navigator.get_context(before=2, after=2)
        
        assert context["current"] == "Fifth paragraph"
        assert len(context["previous"]) == 2
        assert context["next"] == []
