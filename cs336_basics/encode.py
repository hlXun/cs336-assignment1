import pickle
from cs336_basics.tokenizer import Tokenizer
import numpy as np

if __name__ == '__main__':
    vm_path = './data/bpe_tinystories_train.pkl'
    with open(vm_path, "rb") as f:
        vm = pickle.load(f)
        vocab = vm['vocab']
        merges = vm['merges']
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])
    input_path = './data/TinyStoriesV2-GPT4-train.txt'
    out_path = './data/tinystories_train.npy'
    with open(input_path, "r", encoding='utf-8', errors='ignore') as f:
        text = f.read()
        dataset = tokenizer.encode(text)
    np.save(out_path, np.array(dataset, dtype=np.int64))
    print('encode 完成！')