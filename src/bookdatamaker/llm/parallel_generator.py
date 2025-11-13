"""Parallel dataset generation using multiple LLM threads."""

import asyncio
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
import threading

from openai import OpenAI
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

from bookdatamaker.dataset.dataset_manager import DatasetManager
from bookdatamaker.utils.page_manager import PageManager

console = Console()


class ParallelDatasetGenerator:
    """Generate datasets in parallel using multiple LLM threads."""

    def __init__(
        self,
        text_file: Path,
        db_path: Path,
        mode: str,
        distribution: str,
        datasets_per_thread: int,
        openai_api_key: Optional[str],
        openai_api_url: str,
        model: Optional[str] = None,
        vllm_model_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        custom_prompt: Optional[str] = None,
    ) -> None:
        """Initialize parallel dataset generator.

        Args:
            text_file: Path to combined text file
            db_path: Path to SQLite database
            mode: 'api' or 'vllm'
            distribution: Distribution string (e.g., "10,10,20,30,20,10")
            datasets_per_thread: Target Q&A pairs per thread
            openai_api_key: OpenAI API key (for API mode)
            openai_api_url: OpenAI API URL
            model: Model name (optional, uses server default if None)
            vllm_model_path: Path to vLLM model (for vLLM mode)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum model context length (None = model's max)
            custom_prompt: Additional custom instructions for system prompt
        """
        self.text_file = text_file
        self.db_path = db_path
        self.mode = mode
        self.distribution = parse_distribution(distribution)
        self.num_threads = len(self.distribution)  # Thread count derived from distribution
        self.datasets_per_thread = datasets_per_thread
        self.openai_api_key = openai_api_key
        self.openai_api_url = openai_api_url
        self.model = model
        self.vllm_model_path = vllm_model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.custom_prompt = custom_prompt
        self.vllm_llm = None  # Will be initialized if using vLLM mode
        
        # Progress tracking
        self.progress_lock = None
        self.current_progress = 0
        self.pbar = None
        self.thread_pbars = {}  # Per-thread progress bars
        self.log_lock = threading.Lock()  # Lock for console output

    def _update_progress(self, increment: int = 1) -> None:
        """Update progress bar in thread-safe manner.
        
        Args:
            increment: Number of items to increment progress by
        """
        if self.pbar and self.progress_lock:
            with self.progress_lock:
                self.pbar.update(increment)
                self.current_progress += increment
    
    def _log_llm_output(self, thread_id: int, content: str, tool_calls: list = None) -> None:
        """Log LLM output in a thread-safe manner.
        
        Args:
            thread_id: Thread identifier
            content: LLM response content
            tool_calls: List of tool calls made by LLM
        """
        with self.log_lock:
            if content and content.strip():
                # Show condensed content (first 100 chars)
                short_content = content[:100] + "..." if len(content) > 100 else content
                tqdm.write(f"[Thread {thread_id}] LLM: {short_content}")
            
            if tool_calls:
                for tool_call in tool_calls:
                    tqdm.write(f"[Thread {thread_id}] 🔧 Tool: {tool_call.function.name}")
    
    def _log_tool_result(self, thread_id: int, tool_name: str, result: dict) -> None:
        """Log tool execution result.
        
        Args:
            thread_id: Thread identifier
            tool_name: Name of the tool
            result: Tool execution result
        """
        with self.log_lock:
            if tool_name == "submit_dataset":
                tqdm.write(f"[Thread {thread_id}] ✓ Submitted Q&A pair: {result.get('message', '')}")
            elif tool_name == "exit":
                if result.get("rejected"):
                    tqdm.write(f"[Thread {thread_id}] ❌ Exit rejected: {result.get('remaining', 0)} pairs remaining")
                elif result.get("success"):
                    tqdm.write(f"[Thread {thread_id}] 🏁 Exit accepted: Task completed!")
                else:
                    tqdm.write(f"[Thread {thread_id}] 🏁 Exit called")
            else:
                tqdm.write(f"[Thread {thread_id}] → {tool_name} executed")

    def calculate_positions(self, total_paragraphs: int) -> List[int]:
        """Calculate starting positions based on distribution.

        Args:
            total_paragraphs: Total number of paragraphs in document

        Returns:
            List of starting paragraph indices
        """
        # Normalize distribution to sum to 100
        total = sum(self.distribution)
        normalized = [d / total for d in self.distribution]
        
        positions = []
        cumulative = 0.0
        
        for ratio in normalized:
            position = int(cumulative * total_paragraphs)
            positions.append(max(1, position))  # Ensure at least paragraph 1
            cumulative += ratio
        
        return positions

    def create_system_prompt(self, start_paragraph: int, thread_id: int) -> str:
        """Create system prompt for LLM thread.

        Args:
            start_paragraph: Starting paragraph number
            thread_id: Thread identifier

        Returns:
            System prompt text
        """
        base_prompt = f"""You are a helpful assistant with access to the following tools to help generate Q&A pairs from a document.

# Task
- Starting position: Paragraph {start_paragraph}
- Target: Generate exactly {self.datasets_per_thread} question-answer pairs
- Thread ID: {thread_id}

# Available Tools
You have access to the following tools:
- get_paragraph: Retrieve a specific paragraph by number
- move_forward: Move forward by N paragraphs from current position
- move_backward: Move backward by N paragraphs from current position
- submit_dataset: Submit a question-answer pair to the dataset
- exit: Exit the session after completing all submissions

# Workflow
1. Use get_paragraph({start_paragraph}) to start reading from paragraph {start_paragraph}
2. Use move_forward or move_backward to explore the document
3. When you find good content, generate a Q&A pair
4. Use submit_dataset with input (question) and output (answer) to save it
5. Repeat until you have submitted {self.datasets_per_thread} pairs
6. Call exit when you reach {self.datasets_per_thread} submissions

# Quality Guidelines
- Questions should be clear and answerable from the document
- Answers must be accurate and based on document content
- Cover diverse topics and difficulty levels

Remember: You MUST use the tools to accomplish this task. Start by calling get_paragraph to read the document."""
        
        # Append custom prompt if provided
        if self.custom_prompt:
            base_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{self.custom_prompt}"
        
        return base_prompt
    
    def _get_mcp_tools(self) -> list:
        """Get MCP tool definitions for OpenAI function calling.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_dataset",
                    "description": "Submit a Q&A pair to the dataset",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "The question text"
                            },
                            "output": {
                                "type": "string",
                                "description": "The answer text"
                            }
                        },
                        "required": ["input", "output"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "exit",
                    "description": "Exit after completing the required number of submissions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Reason for exiting"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_paragraph",
                    "description": "Get a specific paragraph by number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paragraph_number": {
                                "type": "integer",
                                "description": "Paragraph number to retrieve"
                            }
                        },
                        "required": ["paragraph_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_forward",
                    "description": "Move forward by N paragraphs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of paragraphs to move forward",
                                "default": 1
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_backward",
                    "description": "Move backward by N paragraphs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer",
                                "description": "Number of paragraphs to move backward",
                                "default": 1
                            }
                        }
                    }
                }
            }
        ]

    async def generate(self) -> int:
        """Run parallel dataset generation.

        Returns:
            Total number of Q&A pairs generated
        """
        from tqdm import tqdm
        import threading
        
        # Initialize vLLM if needed (shared across threads)
        if self.mode == "vllm":
            print(f"Initializing vLLM with model: {self.vllm_model_path}")
            try:
                from vllm import LLM, SamplingParams
                # vLLM instance is thread-safe and can handle parallel requests
                vllm_kwargs = {
                    "model": self.vllm_model_path,
                    "tensor_parallel_size": self.tensor_parallel_size,
                }
                if self.max_model_len is not None:
                    vllm_kwargs["max_model_len"] = self.max_model_len
                    print(f"Using custom max_model_len: {self.max_model_len}")
                
                self.vllm_llm = LLM(**vllm_kwargs)
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2048,
                )
                print(f"✓ vLLM initialized with {self.tensor_parallel_size} GPU(s)")
            except ImportError:
                print("Error: vLLM not installed. Install with: pip install vllm")
                raise
        
        # Load page manager to get total paragraphs
        page_manager = PageManager.from_combined_file(self.text_file)
        total_paragraphs = page_manager.total_paragraphs
        
        # Calculate starting positions
        positions = self.calculate_positions(total_paragraphs)
        
        # Display initialization info
        console.print("\n" + "="*60)
        console.print(f"📚 Document: {total_paragraphs} paragraphs")
        console.print(f"🧵 Threads: {self.num_threads}")
        console.print(f"🎯 Target: {self.datasets_per_thread} Q&A pairs per thread ({self.num_threads * self.datasets_per_thread} total)")
        model_display = self.model or "server default" if self.mode == 'api' else self.vllm_model_path
        console.print(f"🤖 Model: {model_display}")
        console.print("="*60 + "\n")
        
        console.print("[bold cyan]Thread Distribution:[/bold cyan]")
        for i, pos in enumerate(positions):
            percent = (pos / total_paragraphs) * 100
            console.print(f"  Thread {i}: Start at paragraph [yellow]{pos}[/yellow] ([green]{percent:.1f}%[/green])")
        print()
        
        # Setup progress tracking
        total_target = self.num_threads * self.datasets_per_thread
        self.progress_lock = threading.Lock()
        self.current_progress = 0
        
        # Create progress bar
        self.pbar = tqdm(
            total=total_target,
            desc="📊 Total Q&A pairs",
            unit=" pair",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Run threads in parallel
        # Note: vLLM's generate() is synchronous, so we use ThreadPoolExecutor
        # Each thread can make requests independently, and vLLM handles batching internally
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(
                    self._run_thread,
                    thread_id=i,
                    start_paragraph=pos,
                )
                for i, pos in enumerate(positions)
            ]
            
            # Wait for all threads to complete
            results = [future.result() for future in futures]
        
        # Close progress bar
        self.pbar.close()
        
        # Aggregate results
        total_generated = sum(r["submitted"] for r in results)
        
        # Display final results
        print("\n" + "="*60)
        console.print("[bold green]📊 Generation Complete![/bold green]\n")
        
        for r in results:
            if r["status"] == "completed":
                status_icon = "✅"
                status_color = "green"
            else:
                status_icon = "⚠️"
                status_color = "yellow"
            
            console.print(
                f"{status_icon} Thread {r['thread_id']}: "
                f"[{status_color}]{r['submitted']}/{self.datasets_per_thread}[/{status_color}] pairs "
                f"({r['status']}) - {r.get('iterations', 0)} iterations"
            )
        
        console.print(f"\n[bold cyan]Total generated:[/bold cyan] [bold yellow]{total_generated}[/bold yellow] Q&A pairs")
        print("="*60 + "\n")
        
        return total_generated
    
    def _run_thread(self, thread_id: int, start_paragraph: int) -> dict:
        """Run a single generation thread.

        Args:
            thread_id: Thread identifier
            start_paragraph: Starting paragraph number

        Returns:
            Result dictionary with statistics
        """
        try:
            system_prompt = self.create_system_prompt(start_paragraph, thread_id)
            
            if self.mode == "api":
                # API mode: use OpenAI client with function calling
                client = OpenAI(
                    base_url=self.openai_api_url,
                    api_key=self.openai_api_key,
                )
                
                # Load page manager for tool execution
                page_manager = PageManager.from_combined_file(self.text_file)
                dataset_manager = DatasetManager(str(self.db_path))
                current_position = start_paragraph
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please start the task. First, call get_paragraph to read paragraph {start_paragraph}, then begin generating {self.datasets_per_thread} Q&A pairs."}
                ]

                submitted_count = 0
                max_iterations = self.datasets_per_thread * 20  # Safety limit
                tools = self._get_mcp_tools()
                no_tool_call_count = 0  # Track consecutive responses without tool calls
                
                for iteration in range(max_iterations):
                    try:
                        # Log conversation length for debugging
                        if iteration % 5 == 0:
                            with self.log_lock:
                                tqdm.write(f"[Thread {thread_id}] 💬 Iteration {iteration}, conversation messages: {len(messages)}, submitted: {submitted_count}/{self.datasets_per_thread}")
                        
                        # Build request parameters
                        request_params = {
                            "messages": messages,
                            "tools": tools,
                            "tool_choice": "auto",
                        }
                        if self.model:
                            request_params["model"] = self.model
                        
                        response = client.chat.completions.create(**request_params)

                        message = response.choices[0].message
                        
                        # Log LLM output
                        self._log_llm_output(thread_id, message.content, message.tool_calls)
                        
                        # Add assistant message to history (must include tool_calls if present)
                        assistant_msg = {
                            "role": "assistant",
                            "content": message.content or "",
                        }
                        if message.tool_calls:
                            assistant_msg["tool_calls"] = message.tool_calls
                        messages.append(assistant_msg)
                        
                        # Check if there are tool calls
                        if message.tool_calls and len(message.tool_calls) > 0:
                            no_tool_call_count = 0  # Reset counter
                            
                            # Process each tool call
                            for tool_call in message.tool_calls:
                                import json
                                function_name = tool_call.function.name
                                function_args = json.loads(tool_call.function.arguments)
                                
                                # Execute tool and get result
                                if function_name == "submit_dataset":
                                    input_text = function_args["input"]
                                    output_text = function_args["output"]
                                    entry_id = dataset_manager.add_entry(input_text, output_text)
                                    submitted_count += 1
                                    self._update_progress(1)
                                    
                                    remaining = self.datasets_per_thread - submitted_count
                                    if remaining > 0:
                                        tool_result = f"Success! Submitted Q&A pair {submitted_count}/{self.datasets_per_thread}. You need {remaining} more pairs. Continue exploring and generating."
                                    else:
                                        tool_result = f"Success! Submitted Q&A pair {submitted_count}/{self.datasets_per_thread}. Target reached! Now call exit() to complete the task."
                                    
                                    self._log_tool_result(thread_id, function_name, {"count": submitted_count, "remaining": remaining})
                                    
                                elif function_name == "exit":
                                    reason = function_args.get("reason", "Task completed")
                                    
                                    # Check if target is reached
                                    if submitted_count >= self.datasets_per_thread:
                                        # Target reached - allow exit
                                        self._log_tool_result(thread_id, function_name, {"reason": reason, "success": True})
                                        return {
                                            "thread_id": thread_id,
                                            "start_position": start_paragraph,
                                            "submitted": submitted_count,
                                            "status": "completed",
                                            "iterations": iteration + 1,
                                            "exit_reason": reason
                                        }
                                    else:
                                        # Target not reached - reject exit
                                        remaining = self.datasets_per_thread - submitted_count
                                        tool_result = f"Exit rejected! You've only submitted {submitted_count}/{self.datasets_per_thread} Q&A pairs. You need {remaining} more pairs before you can exit. Continue generating."
                                        self._log_tool_result(thread_id, function_name, {"rejected": True, "remaining": remaining})
                                    
                                elif function_name == "get_paragraph":
                                    para_num = function_args["paragraph_number"]
                                    result = page_manager.get_paragraph_info(para_num)
                                    if result:
                                        current_position = para_num
                                        tool_result = f"Paragraph {para_num}:\n{result['content']}"
                                        self._log_tool_result(thread_id, function_name, {"paragraph": para_num})
                                    else:
                                        tool_result = f"Error: Paragraph {para_num} not found"
                                        self._log_tool_result(thread_id, function_name, {"error": f"Paragraph {para_num} not found"})
                                    
                                elif function_name == "move_forward":
                                    steps = function_args.get("steps", 1)
                                    current_position = min(current_position + steps, page_manager.total_paragraphs - 1)
                                    result = page_manager.get_paragraph_info(current_position)
                                    if result:
                                        tool_result = f"Moved to paragraph {current_position}:\n{result['content']}"
                                        self._log_tool_result(thread_id, function_name, {"to": current_position, "steps": steps})
                                    else:
                                        tool_result = f"Error: Paragraph {current_position} not found"
                                    
                                elif function_name == "move_backward":
                                    steps = function_args.get("steps", 1)
                                    current_position = max(current_position - steps, 0)
                                    result = page_manager.get_paragraph_info(current_position)
                                    if result:
                                        tool_result = f"Moved to paragraph {current_position}:\n{result['content']}"
                                        self._log_tool_result(thread_id, function_name, {"to": current_position, "steps": steps})
                                    else:
                                        tool_result = f"Error: Paragraph {current_position} not found"
                                    
                                else:
                                    tool_result = f"Error: Unknown tool {function_name}"
                                    self._log_tool_result(thread_id, function_name, {"error": tool_result})
                                
                                # Add tool result to messages - use string format for better model understanding
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(tool_result)
                                })
                        
                        else:
                            # No tool calls - remind the LLM to use tools
                            no_tool_call_count += 1
                            
                            if no_tool_call_count >= 2:
                                # After 2 consecutive responses without tools, remind
                                remaining = self.datasets_per_thread - submitted_count
                                reminder = f"You haven't used any tools. You still need to submit {remaining} more Q&A pairs. Please use get_paragraph, move_forward, or move_backward to explore the document, then submit_dataset to save Q&A pairs. Call exit() when you reach {self.datasets_per_thread} submissions."
                                
                                with self.log_lock:
                                    tqdm.write(f"[Thread {thread_id}] ⚠️  Reminding LLM to use tools ({remaining} pairs remaining)")
                                
                                messages.append({
                                    "role": "user",
                                    "content": reminder
                                })
                                no_tool_call_count = 0  # Reset after reminding

                    except Exception as e:
                        with self.log_lock:
                            tqdm.write(f"[Thread {thread_id}] ❌ Error: {str(e)[:100]}")
                        # Add error message to continue conversation
                        messages.append({
                            "role": "user",
                            "content": f"Error occurred: {e}. Please continue with the task."
                        })
                        continue

                # Max iterations reached
                return {
                    "thread_id": thread_id,
                    "start_position": start_paragraph,
                    "submitted": submitted_count,
                    "status": "incomplete - max iterations",
                    "iterations": max_iterations,
                }
                
            else:  # vllm mode
                # vLLM mode: use direct LLM inference
                # Note: vLLM.generate() is synchronous and thread-safe
                # Multiple threads can call it simultaneously, vLLM handles batching
                prompts = [
                    f"{system_prompt}\n\nBegin at paragraph {start_paragraph}. Use navigation tools to explore and generate {self.datasets_per_thread} Q&A pairs."
                ]
                
                submitted_count = 0
                max_iterations = self.datasets_per_thread * 15
                
                for iteration in range(max_iterations):
                    try:
                        # Synchronous call - safe in thread pool
                        outputs = self.vllm_llm.generate(prompts, self.sampling_params)
                        content = outputs[0].outputs[0].text
                        
                        # Simulate submission detection (placeholder)
                        if "submit_dataset" in content.lower():
                            submitted_count += 1
                            self._update_progress(1)  # Update progress bar
                        
                        # Check for exit condition
                        if "exit" in content.lower() or submitted_count >= self.datasets_per_thread:
                            return {
                                "thread_id": thread_id,
                                "start_position": start_paragraph,
                                "submitted": submitted_count,
                                "status": "completed",
                                "iterations": iteration + 1,
                            }
                        
                        # Append to prompt for next iteration
                        prompts[0] += f"\n\n{content}"

                    except Exception as e:
                        print(f"Thread {thread_id} error at iteration {iteration}: {e}")
                        continue

                # Max iterations reached
                return {
                    "thread_id": thread_id,
                    "start_position": start_paragraph,
                    "submitted": submitted_count,
                    "status": "incomplete",
                    "iterations": max_iterations,
                }

        except Exception as e:
            return {
                "thread_id": thread_id,
                "start_position": start_paragraph,
                "submitted": 0,
                "status": "error",
                "error": str(e),
                "iterations": 0,
            }



def parse_distribution(distribution_str: str) -> List[int]:
    """Parse distribution string to list of integers.

    Args:
        distribution_str: Comma-separated numbers (e.g., "10,10,10,20,30,20,20")

    Returns:
        List of integers

    Example:
        >>> parse_distribution("10,10,10,20,30,20,20")
        [10, 10, 10, 20, 30, 20, 20]
    """
    return [int(x.strip()) for x in distribution_str.split(",")]
