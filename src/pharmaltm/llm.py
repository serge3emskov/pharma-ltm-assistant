"""LLM-бэкенды с одним интерфейсом `chat()`.

* QwenTransformersLLM — локально в Colab (T4), 4-bit через bitsandbytes.
* OpenAICompatLLM — любой OpenAI-совместимый сервер: vLLM, Ollama, LM Studio.
* CachedLLM — кэш ответов на диск: повторный прогон оценки не тратит GPU.
* ScriptedLLM — ответы из функции, для тестов без модели.

Параметр `meta` несёт служебные сведения о вызове (задача, раздел).
Настоящие модели его игнорируют, скриптованные — используют для маршрутизации.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

Messages = list[dict[str, str]]

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


class LLM(Protocol):
    name: str

    def chat(
        self,
        messages: Messages,
        *,
        max_new_tokens: int = 512,
        json_mode: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> str: ...


# --------------------------------------------------------------------------- разбор JSON

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)


def extract_json_object(text: str) -> dict:
    """Первый JSON-объект в ответе модели.

    Модель может обернуть JSON в ```json, добавить пояснение до или после
    или оставить блок <think>. Ищем первый `{`, с которого начинается
    корректный объект. Ничего не нашли — ValueError.
    """
    cleaned = _FENCE_RE.sub("", _THINK_RE.sub("", text))
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("в ответе нет JSON-объекта")


# --------------------------------------------------------------------------- Qwen в Colab


class QwenTransformersLLM:
    """Qwen через transformers. По умолчанию 4-bit NF4 с вычислениями в fp16 (T4 не умеет bf16)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_id + (":4bit" if load_in_4bit else ":fp16")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def chat(self, messages: Messages, *, max_new_tokens: int = 512, json_mode: bool = False,
             meta: dict[str, Any] | None = None) -> str:
        # enable_thinking=False отключает режим рассуждений у гибридных Qwen3; Instruct-2507 его просто игнорирует
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # извлечение должно быть воспроизводимым
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- OpenAI-совместимый сервер


class OpenAICompatLLM:
    """vLLM (`vllm serve ...`), Ollama (`http://localhost:11434/v1`), LM Studio и т. п."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0,
                 temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"{model}@{self.base_url}"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.timeout = timeout
        self.temperature = temperature

    def chat(self, messages: Messages, *, max_new_tokens: int = 512, json_mode: bool = False,
             meta: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"] or ""


# --------------------------------------------------------------------------- кэш


class CachedLLM:
    """Кэш ответов в JSONL. Ключ — модель, сообщения и лимит токенов."""

    def __init__(self, inner: LLM, path: str):
        self.inner = inner
        self.name = inner.name
        self.path = path
        self.hits = 0
        self.misses = 0
        self._cache: dict[str, str] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["output"]

    def _key(self, messages: Messages, max_new_tokens: int, json_mode: bool) -> str:
        payload = json.dumps([self.inner.name, messages, max_new_tokens, json_mode], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chat(self, messages: Messages, *, max_new_tokens: int = 512, json_mode: bool = False,
             meta: dict[str, Any] | None = None) -> str:
        key = self._key(messages, max_new_tokens, json_mode)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        output = self.inner.chat(messages, max_new_tokens=max_new_tokens, json_mode=json_mode, meta=meta)
        self._cache[key] = output
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "meta": meta, "output": output}, ensure_ascii=False) + "\n")
        return output


# --------------------------------------------------------------------------- для тестов


@dataclass
class ScriptedLLM:
    """Ответ вычисляет функция `responder(messages, meta)`. Все вызовы записываются."""

    responder: Callable[[Messages, dict[str, Any]], str]
    name: str = "scripted"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, messages: Messages, *, max_new_tokens: int = 512, json_mode: bool = False,
             meta: dict[str, Any] | None = None) -> str:
        meta = dict(meta or {})
        out = self.responder(messages, meta)
        self.calls.append({"messages": messages, "meta": meta, "output": out})
        return out


def timed(fn: Callable[[], str]) -> tuple[str, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0
