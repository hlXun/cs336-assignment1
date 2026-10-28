import os
import regex as re
from collections import Counter, defaultdict
import heapq
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

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

def worker(
    input_path: str,
    boundary: tuple[int, int],
    special_tokens: list[str],
) -> dict[str, int]:
    start, end = boundary
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        str_pretoken_freq = pre_tokenization(chunk, special_tokens)
    return str_pretoken_freq

def pre_tokenization(
    raw_text: str,
    special_tokens: list[str],
) -> dict[str, int]:
    pretoken_freq = Counter()
    special_tokens = sorted(special_tokens, key=len, reverse=True)
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

def pretoken_states_init(
    str_pretoken_freq: dict[str, int],  
) -> dict[int, tuple[tuple[bytes, ...], int]]:
    pre_token_states = dict()
    for i, (k, v) in enumerate(str_pretoken_freq.items()):
        new_key = tuple(bytes([n]) for n in k.encode('utf-8'))
        pre_token_states[i] = (new_key, v)
    return pre_token_states

def get_pair_frequency_init(
    pretoken_states: dict[int, tuple[tuple[bytes, ...], int]],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[int]]]:
    pair_frequency = Counter()
    inv_pair_index = defaultdict(set)
    for i, (k, v) in pretoken_states.items():
        for pair in zip(k[:-1], k[1:]):
            pair_frequency[pair] += v
            inv_pair_index[pair].add(i)
    return pair_frequency, inv_pair_index

class Warpper_pair_freq[T, U]:
    def __init__(self, pair: T, freq: U):
        self.pair = pair
        self.freq = freq
    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.pair > other.pair

def get_merge_pair(
    pair_frequency_pq: list[Warpper_pair_freq[tuple[bytes, bytes], int]],
    pair_frequency: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes] | None:
    while True and len(pair_frequency_pq):
        top_pair_frequency = heapq.heappop(pair_frequency_pq)
        merge_pair, freq = top_pair_frequency.pair, top_pair_frequency.freq
        now_freq = pair_frequency.get(merge_pair, 0)
        if freq == now_freq:
            return merge_pair
        if now_freq > 0:
            heapq.heappush(pair_frequency_pq, Warpper_pair_freq(merge_pair, now_freq))
    return None

def merge_update(
    pretoken_states: dict[int, tuple[tuple[bytes, ...], int]],
    merge_pair: tuple[bytes, bytes],
    pair_frequency: dict[tuple[bytes, bytes], int],
    inv_pair_index: dict[tuple[bytes, bytes], set[int]],
    pair_frequency_pq: list[Warpper_pair_freq[tuple[bytes, bytes], int]],
) -> None:
    fixed_set = inv_pair_index[merge_pair]
    new_pair = set()
    for i in list(fixed_set):
        pretoken, freq = pretoken_states[i]
        for pair in zip(pretoken[:-1], pretoken[1:]):
            pair_frequency[pair] -= freq
            inv_pair_index[pair].discard(i)
            if pair_frequency[pair] <= 0:
                del pair_frequency[pair]
                inv_pair_index.pop(pair, None)
        j = 0
        new_pretoken = []
        while j < len(pretoken):
            if j + 1 < len(pretoken) and (pretoken[j], pretoken[j + 1]) == merge_pair:
                new_pretoken.append(pretoken[j] + pretoken[j + 1])
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1
        merged_token = merge_pair[0] + merge_pair[1]
        for j in range(len(new_pretoken)):
            if new_pretoken[j] == merged_token:
                if j > 0:
                    new_pair.add((new_pretoken[j - 1], new_pretoken[j]))
                if j + 1 < len(new_pretoken):
                    new_pair.add((new_pretoken[j], new_pretoken[j + 1]))
        pretoken_states[i] = (tuple(new_pretoken), freq)
        for pair in zip(new_pretoken[:-1], new_pretoken[1:]):
            pair_frequency[pair] += freq
            inv_pair_index[pair].add(i)
    for pair in new_pair:
        heapq.heappush(pair_frequency_pq, Warpper_pair_freq(pair, pair_frequency[pair]))
        
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = vocab_init(special_tokens)
    merge: list[tuple[bytes, bytes]] = []
    # with open(input_path, "r", encoding='utf-8') as f:
    #     text = f.read()
    #     str_pretoken_freq = pre_tokenization(text, special_tokens)
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')
        args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args.append([input_path, (start, end), special_tokens])
        with Pool(num_processes) as pool:
            chunks_pretoken_freq = pool.starmap(worker, args)
        str_pretoken_freq = Counter()
        for chunk_pretoken_freq in chunks_pretoken_freq:
            str_pretoken_freq.update(chunk_pretoken_freq)
        del chunks_pretoken_freq
    pretoken_states = pretoken_states_init(str_pretoken_freq)
    pair_frequency, inv_pair_index = get_pair_frequency_init(pretoken_states)
    pair_frequency_pq = []
    for k, v in pair_frequency.items():
        heapq.heappush(pair_frequency_pq, Warpper_pair_freq(k, v))
    while len(vocab) < vocab_size:
        merge_pair = get_merge_pair(pair_frequency_pq, pair_frequency)
        if merge_pair == None:
            break
        merge.append(merge_pair)
        vocab[len(vocab)] = merge_pair[0] + merge_pair[1]
        merge_update(pretoken_states, merge_pair, pair_frequency, inv_pair_index, pair_frequency_pq)
    return vocab, merge