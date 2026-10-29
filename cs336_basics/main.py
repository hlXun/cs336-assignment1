from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import Transformer_LM
from cs336_basics.decode import decode
import pickle
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vm_path = './data/bpe_tinystories_train.pkl'
    with open(vm_path, "rb") as f:
        vm = pickle.load(f)
        vocab = vm['vocab']
        merges = vm['merges']
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=['<|endoftext|>'])
    prompt = input('请输入文本：')
    prompt = tokenizer.encode(prompt)
    prompt = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0) # [1, context_length]
    
    vocab_size = 10000
    d_model = 512
    d_ff = 1344
    num_heads = 16
    context_length = 256
    theta = 10000
    num_layers = 4
    model = Transformer_LM(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        context_length=1000,
        theta=theta,
        num_layers=num_layers,
        device=device,
        dtype=torch.float,
    )
    
    checkpoint = './cs336_basics/check_points/trained.pt'
    if checkpoint is not None:
        check_point = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(check_point['model_state_dict'])
    
    result = decode(model, prompt, tokenizer, max_num_tokens=1000)
    print(result)