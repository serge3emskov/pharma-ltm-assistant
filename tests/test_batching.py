"""Пакетная генерация: логика QwenTransformersLLM на моках (numpy вместо тензоров) и кэш."""
import contextlib
import json

import numpy as np
import pytest

from pharmaltm.llm import CachedLLM, QwenTransformersLLM, ScriptedLLM, chat_many

PAD = 0


class FakeOOM(RuntimeError):
    pass


class FakeTorch:
    class cuda:  # noqa: N801 — повторяет интерфейс torch.cuda
        OutOfMemoryError = FakeOOM
        emptied = 0

        @classmethod
        def empty_cache(cls):
            cls.emptied += 1

        @staticmethod
        def is_available():
            return True

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class FakeTokenizer:
    """Промпт «pN» кодируется токенами [N, N, ...] длиной len(prompt); ответ — токены 1000+N."""

    pad_token_id = PAD
    eos_token_id = 99

    def __init__(self):
        self.padding_side = "right"
        self.sides_seen = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False and add_generation_prompt and not tokenize
        return messages[-1]["content"]

    def __call__(self, prompts, return_tensors, padding):
        self.sides_seen.append(self.padding_side)
        ids = [[int(p.split("#")[0][1:])] * len(p) for p in prompts]
        width = max(map(len, ids))
        rows = [[PAD] * (width - len(r)) + r if self.padding_side == "left" else r + [PAD] * (width - len(r))
                for r in ids]
        arr = np.array(rows)

        class Batch(dict):
            def to(self, device):
                return self

        return Batch(input_ids=arr, attention_mask=(arr != PAD).astype(int))

    def decode(self, row, skip_special_tokens):
        return " ".join(f"ответ-{t - 1000}" for t in row if t >= 1000) + "  "


class FakeModel:
    device = "cpu"

    def __init__(self, oom_above=None):
        self.oom_above = oom_above
        self.batch_sizes = []

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, temperature, top_p, top_k,
                 pad_token_id):
        assert do_sample is False and pad_token_id == PAD
        if self.oom_above and len(input_ids) > self.oom_above:
            raise FakeOOM("CUDA out of memory")
        self.batch_sizes.append(len(input_ids))
        first = [row[row != PAD][0] for row in input_ids]
        new = np.array([[1000 + int(f)] for f in first])
        return np.concatenate([input_ids, new], axis=1)


def make_llm(oom_above=None):
    llm = object.__new__(QwenTransformersLLM)  # без загрузки весов
    llm.torch, llm.tokenizer, llm.model, llm.name = FakeTorch, FakeTokenizer(), FakeModel(oom_above), "fake"
    return llm


def msgs(text):
    return [{"role": "user", "content": text}]


def test_batch_keeps_order_and_uses_left_padding():
    llm = make_llm()
    prompts = ["p1", "p2" + "#" * 30, "p3" + "#" * 5, "p4"]
    out = llm.chat_batch([msgs(p) for p in prompts], max_new_tokens=8, batch_size=8)
    assert out == ["ответ-1", "ответ-2", "ответ-3", "ответ-4"]  # порядок входа, ответы обрезаны
    assert llm.tokenizer.sides_seen == ["left"] and llm.tokenizer.padding_side == "right"  # сторона восстановлена


def test_single_chat_matches_batch():
    llm = make_llm()
    assert llm.chat(msgs("p7")) == llm.chat_batch([msgs("p7")])[0] == "ответ-7"


def test_batches_split_by_size():
    llm = make_llm()
    llm.chat_batch([msgs(f"p{i}") for i in range(1, 6)], batch_size=2)
    assert llm.model.batch_sizes == [2, 2, 1]


def test_oom_halves_batch_and_finishes():
    llm = make_llm(oom_above=2)
    out = llm.chat_batch([msgs(f"p{i}") for i in range(1, 6)], batch_size=8)
    assert out == [f"ответ-{i}" for i in range(1, 6)]
    assert max(llm.model.batch_sizes) <= 2 and FakeTorch.cuda.emptied >= 1


def test_non_oom_errors_are_raised():
    llm = make_llm()

    def boom(*a, **k):
        raise ValueError("другая ошибка")

    llm.model.generate = boom
    with pytest.raises(ValueError):
        llm.chat_batch([msgs("p1"), msgs("p2")])


# ---------------------------------------------------------------- кэш и chat_many


class BatchScripted(ScriptedLLM):
    def __init__(self, responder):
        super().__init__(responder)
        self.batches = []

    def chat_batch(self, messages_list, *, max_new_tokens=512, json_mode=False, metas=None, batch_size=8):
        self.batches.append(len(messages_list))
        return [self.chat(m, max_new_tokens=max_new_tokens, json_mode=json_mode, meta=meta)
                for m, meta in zip(messages_list, metas or [None] * len(messages_list))]


def test_cached_batch_generates_only_misses_and_persists(tmp_path):
    inner = BatchScripted(lambda m, meta: m[-1]["content"].upper())
    path = tmp_path / "c.jsonl"
    llm = CachedLLM(inner, str(path))
    llm.chat(msgs("a"))
    progress = []
    out = llm.chat_batch([msgs("a"), msgs("b"), msgs("c"), msgs("b")], batch_size=2,
                         on_progress=lambda d, t: progress.append((d, t)))
    assert out == ["A", "B", "C", "B"]
    assert inner.batches == [2]  # «a» из кэша, «b» дублируется — одна генерация на b и c
    assert progress == [(2, 2)]
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    again = CachedLLM(inner, str(path))
    assert again.chat_batch([msgs("c")]) == ["C"] and len(inner.calls) == 3


def test_chat_many_falls_back_to_sequential():
    llm = ScriptedLLM(lambda m, meta: m[-1]["content"] + "!")
    progress = []
    assert chat_many(llm, [msgs("x"), msgs("y")], on_progress=lambda d, t: progress.append(d)) == ["x!", "y!"]
    assert progress == [1, 2] and chat_many(llm, []) == []
