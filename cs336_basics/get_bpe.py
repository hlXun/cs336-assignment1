from cs336_basics.bpe_improved import train_bpe
import pickle
import time
import resource

if __name__ == '__main__':
    input_path = '/home/hlx/study/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    start = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    elapsed_time_m = (time.perf_counter() - start) / 60
    peak_memory_main_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    peak_memory_children_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 / 1024
    print(f'耗时:{elapsed_time_m:.2f}min'
          f'主线程峰值内存:{peak_memory_main_gb:.2f}GB'
          f'子线程峰值内存:{peak_memory_children_gb:.2f}GB')
    result = {'vocab': vocab, 'merges': merges}
    out = '/home/hlx/study/CS336/assignment1-basics/data/bpe_tinystories_train.pkl'
    with open(out, 'wb') as f:
        pickle.dump(result, f)