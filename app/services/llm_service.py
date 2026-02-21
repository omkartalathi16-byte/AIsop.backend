import logging
import os
from typing import List, Dict, Any, Optional
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"):
        self.model_path = model_path
        self.llm: Optional[Llama] = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            # Handle MinGW DLL dependencies on Windows
            if os.name == 'nt':
                mingw_bin = "C:\\mingw\\bin"
                if os.path.exists(mingw_bin) and mingw_bin not in os.environ["PATH"]:
                    os.environ["PATH"] = mingw_bin + os.pathsep + os.environ["PATH"]
                    logger.info(f"Added {mingw_bin} to PATH for LLM dependencies")

            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found at {self.model_path}")
                return

            logger.info(f"Initializing Llama model from {self.model_path}")
            # n_ctx should be large enough for RAG chunks
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,       # Matched to i5-4200U (4 logical processors)
                n_batch=512,
                verbose=False
            )
            logger.info("Llama model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Llama model: {e}")

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        if not self.llm:
            return "LLM Service is not initialized."

        try:
            # Simple prompt formatting for Qwen
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            response = self.llm(
                formatted_prompt,
                max_tokens=512,
                stop=["<|im_end|>", "<|im_start|>", "user:", "assistant:"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {e}"

    def synthesize_rag_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a response based on the query and retrieved context chunks.
        """
        context_text = "\n\n".join([
            f"Source: {chunk.get('metadata', {}).get('title', 'Unknown')}\n{chunk.get('content', '')}"
            for chunk in context_chunks
        ])

        system_prompt = (
            "You are a helpful SOP assistant. Answer using the context below. "
            "If not in context, say you don't know based on provided material. "
            "Be very brief and direct."
        )

        prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        return self.generate_response(prompt, system_prompt)
