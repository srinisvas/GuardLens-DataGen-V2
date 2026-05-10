"""
inference_backend.py

Pluggable inference backends for the dataset generation pipeline.
Replaces the hardcoded Ollama HTTP calls with a backend abstraction
that supports:

  1. OllamaBackend    - Original localhost:11434 (local dev)
  2. VLLMBackend      - vLLM OpenAI-compatible server (HPC production)
  3. HFLocalBackend   - Direct HuggingFace transformers (single-GPU fallback)

Usage:
    backend = VLLMBackend(model="Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000")
    generator = LLMTurnGenerator(backend=backend)
    validator = CausalValidator(backend=backend)
"""

import os
import re
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import requests


class InferenceBackend(ABC):
    """Abstract base. All backends expose generate() and chat()."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "",
                 temperature: float = 0.8, max_tokens: int = 120) -> str:
        """Single-turn completion. Used by LLMTurnGenerator and LocalParaphraser."""
        ...

    @abstractmethod
    def chat(self, messages: List[Dict], temperature: float = 0.3,
             max_tokens: int = 200) -> str:
        """Multi-turn chat. Used by CausalValidator."""
        ...

    def wait_until_ready(self, timeout: int = 300, interval: int = 5):
        """Poll the backend until it responds. Used in SLURM scripts."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                if self.health_check():
                    return True
            except Exception:
                pass
            time.sleep(interval)
        raise TimeoutError(
            f"Backend not ready after {timeout}s. "
            f"Check that the server is running and the URL is correct."
        )

    def health_check(self) -> bool:
        return True


# ---------------------------------------------------------
# Ollama (local development)
# ---------------------------------------------------------

class OllamaBackend(InferenceBackend):
    def __init__(self, model: str = "qwen2.5:3b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, system: str = "",
                 temperature: float = 0.8, max_tokens: int = 120) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "top_p": 0.92,
                        "num_predict": max_tokens},
        }
        resp = requests.post(f"{self.base_url}/api/generate",
                             json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def chat(self, messages: List[Dict], temperature: float = 0.3,
             max_tokens: int = 200) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False,
                  "options": {"temperature": temperature,
                              "num_predict": max_tokens}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def health_check(self) -> bool:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
        return resp.status_code == 200


# ---------------------------------------------------------
# vLLM OpenAI-compatible server (HPC production)
# ---------------------------------------------------------

class VLLMBackend(InferenceBackend):
    """
    Talks to a vLLM server running with --api-key and
    --served-model-name via the OpenAI-compatible API.

    Launch the server with:
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-7B-Instruct \
            --tensor-parallel-size 1 \
            --port 8000
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct",
                 base_url: str = "http://localhost:8000",
                 api_key: str = "EMPTY"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def generate(self, prompt: str, system: str = "",
                 temperature: float = 0.8, max_tokens: int = 120) -> str:
        # Use the chat completions endpoint with a system + user message
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(self, messages: List[Dict], temperature: float = 0.3,
             max_tokens: int = 200) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.92,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload, headers=self.headers, timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            # Some vLLM versions use /v1/models instead
            try:
                resp = requests.get(f"{self.base_url}/v1/models",
                                    headers=self.headers, timeout=5)
                return resp.status_code == 200
            except Exception:
                return False


# ---------------------------------------------------------
# Direct HuggingFace transformers (single-GPU fallback)
# ---------------------------------------------------------

class HFLocalBackend(InferenceBackend):
    """
    Loads the model directly into GPU memory via transformers.
    No server needed. Useful for single-GPU jobs or debugging.

    Requires: pip install transformers torch accelerate
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct",
                 device: str = "auto", torch_dtype: str = "bfloat16"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
            device_map=device, trust_remote_code=True,
        )
        self.model_obj.eval()
        self.model_name = model

    def _run(self, messages: List[Dict], temperature: float,
             max_tokens: int) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(
            self.model_obj.device
        )
        with torch.no_grad():
            outputs = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=0.92,
                do_sample=True,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, prompt: str, system: str = "",
                 temperature: float = 0.8, max_tokens: int = 120) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._run(messages, temperature, max_tokens)

    def chat(self, messages: List[Dict], temperature: float = 0.3,
             max_tokens: int = 200) -> str:
        return self._run(messages, temperature, max_tokens)


# ---------------------------------------------------------
# Factory
# ---------------------------------------------------------

def create_backend(backend_type: str = "vllm", **kwargs) -> InferenceBackend:
    """
    Factory function. Environment variables override kwargs.

    Env vars:
        DATASET_BACKEND       = ollama | vllm | hf
        DATASET_MODEL         = model name or path
        DATASET_BASE_URL      = server URL (ollama/vllm)
        DATASET_DEVICE        = auto | cuda:0 | ... (hf only)
    """
    backend_type = os.environ.get("DATASET_BACKEND", backend_type).lower()
    model = os.environ.get("DATASET_MODEL", kwargs.get("model", ""))
    base_url = os.environ.get("DATASET_BASE_URL", kwargs.get("base_url", ""))

    if backend_type == "ollama":
        return OllamaBackend(
            model=model or "qwen2.5:3b",
            base_url=base_url or "http://localhost:11434",
        )
    elif backend_type == "vllm":
        return VLLMBackend(
            model=model or "Qwen/Qwen2.5-7B-Instruct",
            base_url=base_url or "http://localhost:8000",
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
    elif backend_type == "hf":
        return HFLocalBackend(
            model=model or "Qwen/Qwen2.5-7B-Instruct",
            device=os.environ.get("DATASET_DEVICE", kwargs.get("device", "auto")),
            torch_dtype=kwargs.get("torch_dtype", "bfloat16"),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
