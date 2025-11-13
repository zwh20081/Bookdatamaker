"""Status indicators for processing feedback."""

from enum import Enum
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


class Status(Enum):
    """Processing status types."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class StatusIndicator:
    """Display status indicators for long-running operations."""

    def __init__(self) -> None:
        """Initialize status indicator."""
        self.console = Console()
        self.progress: Optional[Progress] = None
        self.current_task: Optional[TaskID] = None

    def __enter__(self) -> "StatusIndicator":
        """Context manager entry."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)

    def start_task(self, description: str, total: Optional[int] = None) -> TaskID:
        """Start a new task.

        Args:
            description: Task description
            total: Total number of steps (None for indeterminate)

        Returns:
            Task ID
        """
        if not self.progress:
            raise RuntimeError("StatusIndicator must be used as context manager")

        self.current_task = self.progress.add_task(description, total=total)
        return self.current_task

    def update_task(
        self,
        task_id: TaskID,
        advance: int = 1,
        description: Optional[str] = None,
    ) -> None:
        """Update task progress.

        Args:
            task_id: Task ID to update
            advance: Number of steps to advance
            description: New description (optional)
        """
        if not self.progress:
            return

        if description:
            self.progress.update(task_id, advance=advance, description=description)
        else:
            self.progress.advance(task_id, advance=advance)

    def complete_task(self, task_id: TaskID, description: Optional[str] = None) -> None:
        """Mark task as completed.

        Args:
            task_id: Task ID to complete
            description: Optional completion message
        """
        if not self.progress:
            return

        if description:
            self.progress.update(task_id, completed=True, description=description)
        else:
            self.progress.update(task_id, completed=True)

    def print_success(self, message: str) -> None:
        """Print success message.

        Args:
            message: Success message
        """
        self.console.print(f"[green]✓[/green] {message}")

    def print_error(self, message: str) -> None:
        """Print error message.

        Args:
            message: Error message
        """
        self.console.print(f"[red]✗[/red] {message}")

    def print_warning(self, message: str) -> None:
        """Print warning message.

        Args:
            message: Warning message
        """
        self.console.print(f"[yellow]⚠[/yellow] {message}")

    def print_info(self, message: str) -> None:
        """Print info message.

        Args:
            message: Info message
        """
        self.console.print(f"[blue]ℹ[/blue] {message}")
