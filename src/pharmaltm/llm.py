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

    def prompt(self, messages: Messages) -> str:
        # enable_thinking=False отключает режим рассуждений у гибридных Qwen3 (Qwen3-8B); Instruct-2507 его игнорирует
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Жадная генерация для списка промптов. Для пакета нужен left padding: у всех строк ответ
        начинается с одной позиции, и новые токены срезаются по длине входа."""
        tok = self.tokenizer
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            inputs = tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        finally:
            tok.padding_side = prev_side
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # извлечение должно быть воспроизводимым
                temperature=None, top_p=None, top_k=None,
                pad_token_id=pad_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return [tok.decode(row, skip_special_tokens=True).strip() for row in new_tokens]

    def chat(self, messages: Messages, *, max_new_tokens: int = 512, json_mode: bool = False,
             meta: dict[str, Any] | None = None) -> str:
        return self._generate([self.prompt(messages)], max_new_tokens)[0]

    def _is_oom(self, exc: BaseException) -> bool:
        oom = getattr(self.torch.cuda, "OutOfMemoryError", None)
        return (oom is not None and isinstance(exc, oom)) or "out of memory" in str(exc).lower()

    def chat_batch(self, messages_list: list[Messages], *, max_new_tokens: int = 512, json_mode: bool = False,
                   metas: list[dict[str, Any] | None] | None = None, batch_size: int = 8) -> list[str]:
        """Пакетная генерация. Промпты сортируются по длине, чтобы меньше места уходило на padding.
        При нехватке памяти пакет уменьшается вдвое, вплоть до одного промпта."""
        prompts = [self.prompt(m) for m in messages_list]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        outputs: list[str | None] = [None] * len(prompts)
        bs, i = max(1, batch_size), 0
        while i < len(order):
            idx = order[i:i + bs]
            try:
                texts = self._generate([prompts[j] for j in idx], max_new_tokens)
            except Exception as exc:  # noqa: BLE001 — нужен именно перехват OOM от CUDA
                if self._is_oom(exc) and bs > 1:
                    self.torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    continue
                raise
            for j, text in zip(idx, texts):
                outputs[j] = text
            i += len(idx)
        return [o or "" for o in outputs]

    def unload(self) -> None:
        """Освобождает видеопамять перед загрузкой следующей модели."""
        import gc

        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


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

    def _store(self, records: list[tuple[str, dict[str, Any] | None, str]]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for key, meta, output in records:
                self._cache[key] = output
                fh.write(json.dumps({"key": key, "meta": meta, "output": output}, ensure_ascii=False) + "\n")

    def chat(self, messages: Messages, *, max_new_tokens: int = 512, json_mode: bool = False,
             meta: dict[str, Any] | None = None) -> str:
        key = self._key(messages, max_new_tokens, json_mode)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        output = self.inner.chat(messages, max_new_tokens=max_new_tokens, json_mode=json_mode, meta=meta)
        self._store([(key, meta, output)])
        return output

    def chat_batch(self, messages_list: list[Messages], *, max_new_tokens: int = 512, json_mode: bool = False,
                   metas: list[dict[str, Any] | None] | None = None, batch_size: int = 8,
                   on_progress: Callable[[int, int], None] | None = None) -> list[str]:
        """Генерирует только промахи кэша, пакетами; каждый пакет сразу пишется на диск,
        поэтому оборванный прогон продолжается с того же места."""
        metas = metas or [None] * len(messages_list)
        keys = [self._key(m, max_new_tokens, json_mode) for m in messages_list]
        todo: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k not in self._cache and k not in todo:
                todo[k] = i
        self.hits += len(keys) - len(todo)
        self.misses += len(todo)
        pending = sorted(todo.items(), key=lambda kv: -sum(len(m["content"]) for m in messages_list[kv[1]]))
        done = 0
        for start in range(0, len(pending), max(1, batch_size)):
            chunk = pending[start:start + max(1, batch_size)]
            outs = chat_many(self.inner, [messages_list[i] for _, i in chunk], max_new_tokens=max_new_tokens,
                             json_mode=json_mode, metas=[metas[i] for _, i in chunk], batch_size=batch_size)
            self._store([(k, metas[i], out) for (k, i), out in zip(chunk, outs)])
            done += len(chunk)
            if on_progress:
                on_progress(done, len(pending))
        return [self._cache[k] for k in keys]


def chat_many(llm: Any, messages_list: list[Messages], *, max_new_tokens: int = 512, json_mode: bool = False,
              metas: list[dict[str, Any] | None] | None = None, batch_size: int = 8,
              on_progress: Callable[[int, int], None] | None = None) -> list[str]:
    """Пакетный вызов, если бэкенд умеет; иначе по одному."""
    if not messages_list:
        return []
    metas = metas or [None] * len(messages_list)
    if hasattr(llm, "chat_batch"):
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "json_mode": json_mode, "metas": metas,
                                  "batch_size": batch_size}
        if on_progress is not None and isinstance(llm, CachedLLM):
            kwargs["on_progress"] = on_progress
        return llm.chat_batch(messages_list, **kwargs)
    outs = []
    for i, (m, meta) in enumerate(zip(messages_list, metas), 1):
        outs.append(llm.chat(m, max_new_tokens=max_new_tokens, json_mode=json_mode, meta=meta))
        if on_progress:
            on_progress(i, len(messages_list))
    return outs


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
