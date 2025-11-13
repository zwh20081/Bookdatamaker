"""DeepSeek OCR text extraction module."""

import asyncio
import base64
from pathlib import Path
from typing import List, Optional, Literal
import os

import httpx
from PIL import Image
import torch


class OCRExtractor:
    """Extract text from images using DeepSeek OCR API or local transformers model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.deepseek.com/v1",
        mode: Literal["api", "local"] = "api",
        local_model_path: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        """Initialize OCR extractor.

        Args:
            api_key: DeepSeek API key (required for API mode)
            api_url: DeepSeek API base URL
            mode: "api" for API calls, "local" for self-hosted transformers model
            local_model_path: Path to local model (for local mode)
            batch_size: Batch size for local transformers processing
        """
        self.mode = mode
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.local_model_path = local_model_path or "deepseek-ai/DeepSeek-OCR"
        self.batch_size = batch_size
        self.llm = None

        if mode == "api":
            if not api_key:
                raise ValueError("API key is required for API mode")
            self.client = httpx.AsyncClient(
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        else:
            # Local mode - initialize transformers model
            self._init_local_model()

    async def __aenter__(self) -> "OCRExtractor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self.mode == "api" and self.client:
            await self.client.aclose()

    def _init_local_model(self) -> None:
        """Initialize local transformers model."""

        from transformers import AutoModel, AutoTokenizer
        
        # Set CUDA device
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_model_path, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.local_model_path,
            trust_remote_code=True,
            use_safetensors=True
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)
        


    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to bytes
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")

    async def extract_text(self, image_path: Path) -> str:
        """Extract text from a single image.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text content

        Raises:
            httpx.HTTPError: If API request fails (API mode)
        """
        if self.mode == "api":
            return await self._extract_text_api(image_path)
        else:
            return await self._extract_text_local(image_path)

    async def _extract_text_api(self, image_path: Path) -> str:
        """Extract text using API.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text content
        """
        image_b64 = self._encode_image(image_path)

        response = await self.client.post(
            f"{self.api_url}/ocr",
            json={
                "image": image_b64,
                "model": "deepseek-ocr",
            },
        )
        response.raise_for_status()

        result = response.json()
        return result.get("text", "")

    async def _extract_text_local(self, image_path: Path) -> str:
        """Extract text using local transformers model.

        Args:
            image_path: Path to image file

        Returns:
            Extracted text content
        """
        from concurrent.futures import ThreadPoolExecutor
        
        image = Image.open(image_path).convert("RGB")
        # Use grounding prompt for markdown conversion
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "

        # Run in thread pool executor (inference is synchronous)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            output_text = await loop.run_in_executor(
                executor, lambda: self._run_inference(image, prompt)
            )

        return output_text
    
    def _run_inference(self, image: Image.Image, prompt: str) -> str:
        """Run model inference (synchronous helper).
        
        Args:
            image: PIL Image
            prompt: Prompt text
            
        Returns:
            Extracted text
        """
        inputs = self.model.prepare_inputs(
            images=[image],
            prompts=[prompt],
            tokenizer=self.tokenizer
        )
        
        # Move inputs to GPU
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=8192)
        
        # Decode output
        output_text = self.tokenizer.decode(
            outputs[0].cpu().tolist(), 
            skip_special_tokens=True
        )
        
        return output_text

    async def extract_from_directory(
        self, directory: Path, extensions: Optional[List[str]] = None
    ) -> List[tuple[Path, str]]:
        """Extract text from all images in a directory.

        Args:
            directory: Directory containing images
            extensions: List of file extensions to process (default: common image formats)

        Returns:
            List of tuples (image_path, extracted_text)
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))

        # Use batch processing for local mode
        if self.mode == "local":
            return await self._extract_batch_local(sorted(image_files))

        # Sequential processing for API mode
        results = []
        for image_path in sorted(image_files):
            try:
                text = await self.extract_text(image_path)
                results.append((image_path, text))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((image_path, ""))

        return results

    async def _extract_batch_local(self, image_paths: List[Path]) -> List[tuple[Path, str]]:
        """Batch extract text using local transformers model.

        Args:
            image_paths: List of image paths

        Returns:
            List of tuples (image_path, extracted_text)
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        if not image_paths:
            return []

        all_results = []
        total_images = len(image_paths)
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        # Process in batches with progress bar
        with tqdm(total=total_images, desc="OCR Processing", unit="image") as pbar:
            for i in range(0, total_images, self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                
                # Prepare batch input
                batch_images = []
                batch_prompts = []
                valid_paths = []

                for image_path in batch_paths:
                    try:
                        image = Image.open(image_path).convert("RGB")
                        batch_images.append(image)
                        batch_prompts.append(prompt)
                        valid_paths.append(image_path)
                    except Exception as e:
                        print(f"\nError loading {image_path}: {e}")

                if not batch_images:
                    pbar.update(len(batch_paths))
                    continue

                # Run batch inference in thread pool (inference is synchronous)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    output_texts = await loop.run_in_executor(
                        executor, lambda: self._run_batch_inference(batch_images, batch_prompts)
                    )

                # Collect batch results
                for j, text in enumerate(output_texts):
                    if j < len(valid_paths):
                        all_results.append((valid_paths[j], text))
                
                # Update progress bar
                pbar.update(len(batch_paths))

        return all_results
    
    def _run_batch_inference(self, images: List[Image.Image], prompts: List[str]) -> List[str]:
        """Run batch model inference (synchronous helper).
        
        Args:
            images: List of PIL Images
            prompts: List of prompt texts
            
        Returns:
            List of extracted texts
        """
        inputs = self.model.prepare_inputs(
            images=images,
            prompts=prompts,
            tokenizer=self.tokenizer
        )
        
        # Move inputs to GPU
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=8192)
        
        # Decode outputs
        output_texts = []
        for output in outputs:
            text = self.tokenizer.decode(
                output.cpu().tolist(), 
                skip_special_tokens=True
            )
            output_texts.append(text)
        
        return output_texts

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        # Split by double newlines or more
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # If no double newlines, split by single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        return paragraphs

    async def extract_from_document(
        self, document_path: Path, prefer_text: bool = False
    ) -> List[tuple[int, str]]:
        """Extract text from PDF or EPUB document.

        Args:
            document_path: Path to PDF or EPUB file
            prefer_text: Try to extract text directly before OCR

        Returns:
            List of tuples (page_number, extracted_text)
        """
        from .document_parser import extract_document_pages

        pages = extract_document_pages(document_path, prefer_text=prefer_text)
        
        # Separate text pages and image pages
        text_results = []
        image_pages = []
        
        for page_num, content in pages:
            if isinstance(content, str):
                # Text already extracted
                text_results.append((page_num, content))
            else:
                # Image - need OCR
                image_pages.append((page_num, content))
        
        # Process images
        if image_pages:
            if self.mode == "local":
                # Batch process all images with transformers
                image_results = await self._extract_batch_from_images(image_pages)
            else:
                # API mode - process sequentially
                image_results = []
                import tempfile
                
                for page_num, image in image_pages:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        image.save(tmp.name)
                        tmp_path = Path(tmp.name)
                    
                    try:
                        text = await self.extract_text(tmp_path)
                        image_results.append((page_num, text))
                    finally:
                        tmp_path.unlink()
            
            # Combine and sort results
            all_results = text_results + image_results
            all_results.sort(key=lambda x: x[0])  # Sort by page number
            return all_results
        
        return text_results
    
    async def _extract_batch_from_images(
        self, image_pages: List[tuple[int, Image.Image]]
    ) -> List[tuple[int, str]]:
        """Batch extract text from images using local transformers model.

        Args:
            image_pages: List of tuples (page_number, image)

        Returns:
            List of tuples (page_number, extracted_text)
        """
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        if not image_pages:
            return []
        
        all_results = []
        total_pages = len(image_pages)
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        
        # Process in batches with progress bar
        with tqdm(total=total_pages, desc="OCR Processing", unit="page") as pbar:
            for i in range(0, total_pages, self.batch_size):
                batch = image_pages[i:i + self.batch_size]
                
                # Prepare batch input
                batch_images = []
                batch_prompts = []
                page_numbers = []
                
                for page_num, image in batch:
                    batch_images.append(image.convert("RGB"))
                    batch_prompts.append(prompt)
                    page_numbers.append(page_num)
                
                # Run batch inference in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    output_texts = await loop.run_in_executor(
                        executor, lambda: self._run_batch_inference(batch_images, batch_prompts)
                    )
                
                # Collect batch results
                for j, text in enumerate(output_texts):
                    all_results.append((page_numbers[j], text))
                
                # Update progress bar
                pbar.update(len(batch))
        
        return all_results
