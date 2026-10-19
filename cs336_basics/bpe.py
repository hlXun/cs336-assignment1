import os
import regex as re
from collections import Counter

def vocab_init(
    special_tokens: list[str],
) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    assert len(vocab) == 256 + len(special_tokens)
    return vocab

def pre_tokenization(
    raw_text: str,
    special_tokens: list[str],
) -> dict[str, int]:
    pretoken_freq = Counter()
    special_tokens.sort(key=len, reverse=True)
    pattern = '|'.join(re.escape(special_token) for special_token in special_tokens)
    if len(special_tokens):
        texts = re.split(pattern, raw_text)
    else:
        texts = []
        texts.append(raw_text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for text in texts:
        result = re.finditer(PAT, text)
        for match in result:
            pretoken_freq[match.group()] += 1
    return pretoken_freq

def pretoken_freq_init(
    str_pretoken_freq: dict[str, int],
) -> dict[tuple[bytes, ...], int]:
    bytes_pretoken_freq = Counter()
    for k, v in str_pretoken_freq.items():
        new_key = tuple(bytes([n]) for n in k.encode('utf-8'))
        bytes_pretoken_freq[new_key] = v
    return bytes_pretoken_freq

def get_pair_frequency(
    pretoken_freq: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    pair_frequency = Counter()
    for k, v in pretoken_freq.items():
        for pair in zip(k[:-1], k[1:]):
            pair_frequency[pair] += v
    return pair_frequency
    
def get_merge_pair(
    pretoken_freq: dict[tuple[bytes, ...], int],
) -> tuple[bytes, bytes] | None:
    pair_frequency = get_pair_frequency(pretoken_freq)
    if not pair_frequency:
        return None
    max_v = max(pair_frequency.values())
    candidates = [k for k, v in pair_frequency.items() if v == max_v]
    merge_pair = candidates[0]
    for candidate in candidates:
        if candidate > merge_pair:
            merge_pair = candidate
    return merge_pair

def merge_update(
    pretoken_freq: dict[tuple[bytes, ...], int],
    merge_pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    new_pretoken_freq = Counter()
    for k, v in pretoken_freq.items():
        if len(k) == 1:
            new_pretoken_freq[k] += v
            continue
        new_k = []
        i = 0
        while i < len(k):
            if i < len(k) - 1 and k[i] == merge_pair[0] and k[i + 1] == merge_pair[1]:
                new_k.append(k[i] + k[i + 1])
                i += 1
            else:
                new_k.append(k[i])
            i += 1
        new_pretoken_freq[tuple(new_k)] += v
    return new_pretoken_freq      

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = vocab_init(special_tokens)
    merge: list[tuple[bytes, bytes]] = []
    with open(input_path, "r", encoding='utf-8') as f:
        text = f.read()
        str_pretoken_freq = pre_tokenization(text, special_tokens)
        pretoken_freq = pretoken_freq_init(str_pretoken_freq)
    while len(vocab) < vocab_size:
        merge_pair = get_merge_pair(pretoken_freq)
        if merge_pair == None:
            break
        merge.append(merge_pair)
        vocab[len(vocab)] = merge_pair[0] + merge_pair[1]
        pretoken_freq = merge_update(pretoken_freq, merge_pair)
    return vocab, merge