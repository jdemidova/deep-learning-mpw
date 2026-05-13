from __future__ import annotations

from collections.abc import Iterable
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Protocol, Sequence


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalize_text(text: str) -> str:
    return text.casefold().strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


class CaptionTokenizerProtocol(Protocol):
    @property
    def pad_idx(self) -> int:
        ...

    def build(self, captions: Iterable[str]) -> "CaptionTokenizerProtocol":
        ...

    def encode(self, caption: str, max_len: int | None = None) -> list[int]:
        ...


class CaptionTokenizer:
    def __init__(
        self,
        min_freq: int = 5,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1.")

        special_tokens = tuple(special_tokens)
        required_tokens = {"<pad>", "<bos>", "<eos>", "<unk>"}
        missing_tokens = required_tokens.difference(special_tokens)
        if missing_tokens:
            raise ValueError(
                f"special_tokens must include {sorted(required_tokens)}; "
                f"missing {sorted(missing_tokens)}."
            )

        self.min_freq = min_freq
        self.special_tokens = special_tokens
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}

    def __len__(self) -> int:
        return len(self.stoi)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    @property
    def vocab_size(self) -> int:
        return len(self)

    @classmethod
    def from_captions(
        cls,
        captions: Iterable[str],
        min_freq: int = 5,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> "CaptionTokenizer":
        return cls(min_freq=min_freq, special_tokens=special_tokens).build(captions)

    def build(self, captions: Iterable[str]) -> "CaptionTokenizer":
        counter = Counter()
        for caption in captions:
            counter.update(tokenize(caption))

        words = [
            word
            for word, count in counter.items()
            if count >= self.min_freq and word not in self.special_tokens
        ]
        vocab = list(self.special_tokens) + sorted(words)

        self.stoi = {token: index for index, token in enumerate(vocab)}
        self.itos = {index: token for token, index in self.stoi.items()}
        return self

    def _require_vocab(self) -> None:
        if not self.stoi or not self.itos:
            raise RuntimeError("Tokenizer vocabulary is empty. Call build(...) first.")

    @property
    def pad_idx(self) -> int:
        self._require_vocab()
        return self.stoi["<pad>"]

    @property
    def bos_idx(self) -> int:
        self._require_vocab()
        return self.stoi["<bos>"]

    @property
    def eos_idx(self) -> int:
        self._require_vocab()
        return self.stoi["<eos>"]

    @property
    def unk_idx(self) -> int:
        self._require_vocab()
        return self.stoi["<unk>"]

    def encode_tokens(
        self,
        tokens: Sequence[str],
        max_len: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        self._require_vocab()

        if max_len is not None and max_len < 1:
            raise ValueError("max_len must be at least 1 when provided.")

        if add_special_tokens:
            tokens = ["<bos>", *tokens, "<eos>"]

        token_ids = [self.stoi.get(token, self.unk_idx) for token in tokens]

        if max_len is not None:
            token_ids = token_ids[:max_len]
            if add_special_tokens and token_ids and token_ids[-1] != self.eos_idx:
                token_ids[-1] = self.eos_idx

        return token_ids

    def encode(
        self,
        caption: str,
        max_len: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self.encode_tokens(
            tokenize(caption),
            max_len=max_len,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: Iterable[int],
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        self._require_vocab()

        tokens: list[str] = []
        for token_id in token_ids:
            token = self.itos.get(int(token_id), "<unk>")

            if not skip_special_tokens or token not in self.special_tokens:
                tokens.append(token)

            if stop_at_eos and token == "<eos>":
                break

        return " ".join(tokens)

    def to_dict(self) -> dict[str, object]:
        self._require_vocab()
        return {
            "min_freq": self.min_freq,
            "special_tokens": list(self.special_tokens),
            "stoi": dict(self.stoi),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, state: dict[str, object]) -> "CaptionTokenizer":
        tokenizer = cls(
            min_freq=int(state["min_freq"]),
            special_tokens=tuple(state["special_tokens"]),
        )
        tokenizer.stoi = {str(token): int(index) for token, index in state["stoi"].items()}
        tokenizer.itos = {index: token for token, index in tokenizer.stoi.items()}
        return tokenizer

    @classmethod
    def load(cls, path: str | Path) -> "CaptionTokenizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(state)


Vocabulary = CaptionTokenizer


__all__ = [
    "CaptionTokenizer",
    "SPECIAL_TOKENS",
    "TOKEN_PATTERN",
    "Vocabulary",
    "normalize_text",
    "tokenize",
]


