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
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges_rank = {k: v for v, k in enumerate(self.merges)}
    
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
    
    def encode(
        self,
        text: str,
    ) -> list[int]:
        result: list[int] = [] 
        pattern = '(' + '|'.join(re.escape(special_token) for special_token in self.special_tokens) + ')'
        if len(self.special_tokens):
            texts = re.split(pattern, text)
        else:
            texts = [text]
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        inv_vocab = self.inv_vocab
        for text in texts:
            if text in self.special_tokens:
                token_id = inv_vocab.get(text.encode('utf-8'))
                result.append(token_id)
                continue
            pre_tokens = re.finditer(PAT, text)
            for match in pre_tokens:
                pre_token = match.group()
                pre_token_bytes = [bytes([n]) for n in pre_token.encode('utf-8')]
                while True:
                    pre_token_bytes, is_merged= self.merge_pair(pre_token_bytes)
                    if is_merged == False:
                        break
                for token_bytes in pre_token_bytes:
                    token_id = inv_vocab.get(token_bytes, -1)
                    result.append(token_id)
        return result
    
    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        special_token_prefix: set[str] = set()
        for special_token in self.special_tokens:
            for k in range(len(special_token)):
                special_token_prefix.add(special_token[:k])
        inv_vocab = self.inv_vocab
        buffer: str = ''
        pattern = '(' + '|'.join([re.escape(special_token) for special_token in self.special_tokens]) + ')'
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for stream in iterable:
            stream = buffer + stream
            buffer = ''
            for k in range(len(stream) + 1):
                if stream[k:] in special_token_prefix:
                    buffer = stream[k:]
                    break
            stream = stream[:k]
            if len(self.special_tokens):
                texts = re.split(pattern, stream)
            else:
                texts = [stream]
            for i in range(len(texts)):
                text = texts[i]
                if text in self.special_tokens:
                    yield inv_vocab[text.encode('utf-8')]
                    continue
                pre_tokens = re.finditer(PAT, text)
                pre = next(pre_tokens, None)
                if pre != None:
                    for cur in pre_tokens:
                        token_ids = self.encode(pre.group())
                        pre_token = pre.group()
                        pre_token_bytes = [bytes([n]) for n in pre_token.encode('utf-8')]
                        while True:
                            pre_token_bytes, is_merged = self.merge_pair(pre_token_bytes)
                            if is_merged == False:
                                break
                        for token_bytes in pre_token_bytes:
                            token_id = inv_vocab.get(token_bytes, -1)
                            yield token_id
                        pre = cur
                else:
                    continue
                if i == len(texts) - 1:
                    buffer = pre.group() + buffer
                else:
                    pre_token = pre.group()
                    pre_token_bytes = [bytes([n]) for n in pre_token.encode('utf-8')]
                    while True:
                        pre_token_bytes, is_merged = self.merge_pair(pre_token_bytes)
                        if is_merged == False:
                            break
                    for token_bytes in pre_token_bytes:
                        token_id = inv_vocab.get(token_bytes, -1)
                        yield token_id
           
        if len(self.special_tokens):
            texts = re.split(pattern, buffer)
        else:
            texts = [buffer]
        for text in texts:
            if text in self.special_tokens:
                yield inv_vocab[text.encode('utf-8')]
                continue
            pre_tokens = re.finditer(PAT, text)
            for match in pre_tokens:
                pre_token = match.group()
                token_ids = self.encode(pre_token)
                for token_id in token_ids:
                    yield token_id
             
    def decode(
        self,
        ids: list[int],
    ) -> str:
        bytes_list: list[bytes] = []
        for id in ids:
            bytes_list.append(self.vocab[id])
        bytes_text: bytes = b''
        for b in bytes_list:
            bytes_text += b  
        text = bytes_text.decode('utf-8', errors='replace')
        return text