import os
from tqdm import tqdm
import torch
from collections import defaultdict



class LMDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, data_dir, vocab, batch_size, split ):
        self.data_dir = data_dir
        self.vocab = vocab
        
        self.data = self.load_data(split, batch_size) # bsz x (len(data)/bsz)
        self.start = 0
        self.end = self.data.size(1)
        self.split = split

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:                   # in a worker process split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(self.data.transpose(1,0)[iter_start:iter_end])

    


    def load_data(self, split, bsz):
        with open(os.path.join(self.data_dir, f"{split}.txt"), "r") as fn:
            data = [line.strip() for line in fn.readlines()]

        print('binarizing data ...')
        doc = []
        for line in tqdm(data):
            if line != '':
                doc.append(torch.tensor(self.vocab.encode_line(line)))
        data = torch.cat(doc)

        nstep = data.size(0) // bsz
        return data[ : nstep * bsz].view(bsz, -1)


class Vocab:

    def __init__(self, 
            data_dir, 
            eos_token='<eos>', 
            pad_token='<pad>', 
            mask_token=None):

        self.data_dir = data_dir
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mask_token = mask_token

        self.build_vocab()

    def build_vocab(self, min_freq=0, max_num=1000000):
        
        self.vocab = defaultdict(int)
        self.tok2id = {}
        self.id2tok = []


        # counting vocab
        print("building vocab...")
        with open(os.path.join(self.data_dir, "train.txt"), "r") as f:
            data = f.readlines()
            for line in tqdm(data):
                line = line.strip().split()
                for tok in line:
                    self.vocab[tok] += 1

        
        self.vocab = {a: self.vocab[a] for a in self.vocab if self.vocab[a] >= min_freq}
        
        # sorting by freq
        self.vocab = list(sorted(self.vocab.items(), key=lambda x:x[1], reverse=True))
        print(self.vocab[:10])

        # self.vocab = self.vocab[:max_num]
        
        # add unk token 
        # self.vocab.append(('<unk>', 0))

        self.id2tok = [self.mask_token, self.pad_token, self.eos_token] + [a[0] for a in self.vocab]
        self.id2tok = [tok for tok in self.id2tok if tok is not None]
        self.tok2id = {a: i for i, a in enumerate(self.id2tok)}
        
        print('end building vocab ...')
        # print('vocab size', len(self.tok2id),  len(self.vocab), len(self.id2tok))
        print(self.id2tok[:10])

    def encode_line(self, line, add_eos=True):
        line = [self.tok2id[tok] for tok in line.split()]

        return line + [self.eos_idx] if add_eos and self.eos_token else line

    def decode_tokids(self, tensor):
        tokens = []
        for tokid in tensor:
            tokens.append(self.id2tok[tokid])
        
        # tokens = [t if t != self.eos_token else '\n' for t in tokens]
        return ' '.join(tokens)

    @property
    def eos_idx(self):
        return self.tok2id[self.eos_token]

    @property
    def mask_idx(self):
        return self.tok2id[self.mask_token]


    @property
    def padding_idx(self):
        return self.tok2id[self.pad_token]


    @property
    def size(self):
        return len(self.tok2id)



if __name__ == '__main__':
    data_dir = "ptb"
    batch_size = 5
    batch_length = 20
    vocab = Vocab(data_dir)
    trainset = LMDataset(data_dir, vocab, batch_size, 'train')
    trainloader=  torch.utils.data.DataLoader(trainset, batch_size=batch_length)
    trainloader = iter(trainloader)

    while True:
        samples = next(trainloader).transpose(1,0)
        print(samples.shape, vocab.decode_tokids(samples[0]) )
