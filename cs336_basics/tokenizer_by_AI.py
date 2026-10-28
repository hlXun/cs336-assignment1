import pickle
import regex as re
from collections import Counter
from typing import Iterable, Iterator


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = dict(vocab)
        special_tokens = list(special_tokens or [])
        special_tokens.sort(key=len, reverse=True)
        for special_token in special_tokens:
            if special_token.encode('utf-8') not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token.encode('utf-8')
        self.merges = list(merges)
        self.special_tokens = special_tokens
        self.special_tokens_set = set(special_tokens)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges_rank = {k: v for v, k in enumerate(self.merges)}

        # ✅ 预编译正则，避免每次 encode 重建
        self._pat = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        if special_tokens:
            self._special_pat = re.compile(
                '(' + '|'.join(re.escape(s) for s in special_tokens) + ')'
            )
        else:
            self._special_pat = None
        self._max_special_len = max((len(s) for s in special_tokens), default=0)

        # ✅ pre-token -> token ids 缓存
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def merge_pair(
        self,
        bytes_text: list[bytes],
    ) -> tuple[list[bytes], bool]:
        # 原样保留
        max_rank = len(self.merges)
        target_pair = []
        for pair in zip(bytes_text[:-1], bytes_text[1:]):
            rank = self.merges_rank.get(pair, len(self.merges))
            if max_rank > rank:
                max_rank = rank
                target_pair = pair
        if max_rank == len(self.merges):
            return bytes_text, False
        merged_text: list[bytes] = []
        i = 0
        while i < len(bytes_text):
            if i < len(bytes_text) - 1 and (bytes_text[i], bytes_text[i + 1]) == target_pair:
                merged_text.append(bytes_text[i] + bytes_text[i + 1])
                i += 1
            else:
                merged_text.append(bytes_text[i])
            i += 1
        return merged_text, True

    # ✅ 新增：单个 pre-token 的 BPE + 缓存
    def _encode_pretoken(self, pre_token: str) -> list[int]:
        cached = self._cache.get(pre_token)
        if cached is not None:
            return cached
        pb = [bytes([n]) for n in pre_token.encode('utf-8')]
        while True:
            pb, merged = self.merge_pair(pb)
            if not merged:
                break
        ids = [self.inv_vocab[b] for b in pb]
        self._cache[pre_token] = ids
        return ids

    def encode(
        self,
        text: str,
    ) -> list[int]:
        result: list[int] = []
        texts = self._special_pat.split(text) if self._special_pat else [text]
        for t in texts:
            if t in self.special_tokens_set:
                result.append(self.inv_vocab[t.encode('utf-8')])
                continue
            for m in self._pat.finditer(t):
                result.extend(self._encode_pretoken(m.group()))
        return result

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        # special token 的严格前缀集合（用于跨 chunk 保留）
        special_token_prefix: set[str] = set()
        for special_token in self.special_tokens:
            for k in range(len(special_token)):
                special_token_prefix.add(special_token[:k])

        buffer = ''
        max_st_len = self._max_special_len

        for stream in iterable:
            stream = buffer + stream
            buffer = ''

            # 只需检查末尾 max_st_len-1 个字符的后缀
            check_start = (
                max(0, len(stream) - (max_st_len - 1)) if max_st_len > 0 else len(stream)
            )
            k = len(stream)
            for k in range(check_start, len(stream) + 1):
                if stream[k:] in special_token_prefix:
                    buffer = stream[k:]
                    break
            stream = stream[:k]

            texts = self._special_pat.split(stream) if self._special_pat else [stream]
            n = len(texts)
            for i, text in enumerate(texts):
                if text in self.special_tokens_set:
                    yield self.inv_vocab[text.encode('utf-8')]
                    continue
                pre_tokens = list(self._pat.finditer(text))
                if not pre_tokens:
                    continue
                # 除最后一个 pre-token 外全部处理
                for pre in pre_tokens[:-1]:
                    yield from self._encode_pretoken(pre.group())
                # 最后一个 pre-token：若是最后一个 text 则进 buffer 等下一轮拼接
                last = pre_tokens[-1]
                if i == n - 1:
                    buffer = last.group() + buffer
                else:
                    yield from self._encode_pretoken(last.group())

        # ✅ 收尾：处理剩下的 buffer
        texts = self._special_pat.split(buffer) if self._special_pat else [buffer]
        for text in texts:
            if text in self.special_tokens_set:
                yield self.inv_vocab[text.encode('utf-8')]
                continue
            for m in self._pat.finditer(text):
                yield from self._encode_pretoken(m.group())

    def decode(
        self,
        ids: list[int],
    ) -> str:
        bytes_text = b''.join(self.vocab[i] for i in ids)
        return bytes_text.decode('utf-8', errors='replace')