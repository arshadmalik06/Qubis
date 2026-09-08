import json
import httpx
from typing import AsyncGenerator

class BISLLMAgent:
    """
    Handles all communication with the local Ollama LLM, enforcing strict 
    system prompts and guardrails for BIS document retrieval.
    """
    def __init__(self, model_name: str = "qwen2.5:7b-instruct", ollama_url: str = "http://127.0.0.1:11434/api/generate"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # The master instruction set for the AI
        self.system_prompt = (
            "You are an elite, enterprise-grade AI assistant specializing exclusively in Bureau of Indian Standards (BIS) documents. "
            "Your sole purpose is to answer user queries using ONLY the provided context.\n\n"
            "STRICT RULES:\n"
            "1. If the answer is not contained in the provided context, you MUST output exactly: 'The available BIS documentation does not contain this specific rule.'\n"
            "2. Never hallucinate, guess, or bring in outside knowledge.\n"
            "3. You must cite the exact Standard ID and Clause ID provided in the context at the end of your response.\n"
            "4. Maintain a professional, technical, and objective tone."
        )

    def _build_prompt(self, query: str, context_chunks: list) -> str:
        """Assembles the system prompt, retrieved chunks, and user query into one string."""
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            content = chunk.get("content", "")
            meta = chunk.get("metadata", {})
            std_id = meta.get("standard_id", "Unknown Standard")
            clause_id = meta.get("clause_id", "Unknown Clause")
            
            context_text += f"\n--- Context Chunk {i+1} ---\nSource: {std_id}, Clause {clause_id}\nContent:\n{content}\n"

        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
            f"USER QUERY: {query}\n\n"
            "ANSWER: "
        )
        return full_prompt

    async def generate_stream(self, query: str, context_chunks: list) -> AsyncGenerator[dict, None]:
        """Streams the LLM response chunk by chunk."""
        # Yield the sources as a custom event first so frontend can display citations
        yield {"data": json.dumps({"sources": context_chunks})}
        
        prompt = self._build_prompt(query, context_chunks)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1  # Low temperature forces the model to be strict and factual
            }
        }

        try:
            # Using httpx for non-blocking asynchronous HTTP requests
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.ollama_url, json=payload, timeout=60.0) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            # Yield in the exact format required by SSE
                            yield {"data": json.dumps({"token": data.get("response", "")})}
                            
        except httpx.ConnectError:
            # Graceful failure if Ollama isn't running
            error_msg = "CRITICAL ERROR: LLM Engine (Ollama) is offline or unreachable. Please verify the service is running on port 11434."
            yield {"data": json.dumps({"token": error_msg})}
        except Exception as e:
            # Catch-all for any other backend failures
            error_msg = f"SYSTEM ERROR: An unexpected failure occurred during generation - {str(e)}"
            yield {"data": json.dumps({"token": error_msg})}

    async def generate_completion(self, prompt: str, system_prompt: str = None) -> str:
        """
        Non-streaming generation for structured outputs (e.g., JSON checklists).
        Returns the complete response as a single string — avoids SSE parsing entirely.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or self.system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 4096,
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.ollama_url, json=payload, timeout=180.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
        except httpx.ConnectError:
            raise ConnectionError(
                "CRITICAL: LLM Engine (Ollama) is offline or unreachable on port 11434."
            )
        except Exception as e:
            raise RuntimeError(f"LLM completion failed: {str(e)}")