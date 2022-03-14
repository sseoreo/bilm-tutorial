import os
import math
import numpy as np

from tqdm import tqdm
import torch
from vocab import Vocab


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




class BiLMDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, vocab, seq_len: int=512, split='train'):
        
        super(BiLMDataset, self).__init__()
        self.vocab = vocab
        with open(os.path.join(data_dir, f"{split}.txt"), "r") as f:
            self.data = [line.strip() for line in f.readlines() if line != '']
            
        
        self.seq_len = seq_len
        self.eos_id = vocab.eos_idx
        self.pad_id = vocab.padding_idx
        self.mask_id = vocab.mask_idx


    def __getitem__(self, i):
        seq = self.vocab.encode_line(self.data[i], add_eos=True)[: self.seq_len]
        
        
        encoded_input = torch.tensor( seq+ [self.pad_id] * (self.seq_len - len(seq))).long().contiguous()
        # masked_labels = torch.tensor( seq+ [self.pad_id] * (self.seq_len  - len(seq))).long().contiguous()
        
        # assert len(masked_input) == len(masked_labels) == self.seq_len
        return encoded_input

    def __len__(self):
        return len(self.data)

    def get_vocab(self):
        return self.vocab


if __name__ == '__main__':
    data_dir = "ptb"
    batch_size = 5
    batch_length = 20

    if False:
        vocab = Vocab(data_dir)

        trainset = LMDataset(data_dir, vocab, batch_size, 'train')
        trainloader=  torch.utils.data.DataLoader(trainset, batch_size=batch_length)
        trainloader = iter(trainloader)

        while True:
            samples = next(trainloader).transpose(1,0)
            print(samples.shape, vocab.decode_tokids(samples[0]) )

    else:
        vocab = Vocab(data_dir, mask_token='<mask>')

        dataset = BERTLanguageModelingDataset(data_dir, vocab, seq_len=16)

        # 작동 테스트
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch, (mlm_train, mlm_target) in enumerate(dataloader):
            print(mlm_train.size(), vocab.decode_tokids(mlm_train[0]))
            print(mlm_target.size(), vocab.decode_tokids(mlm_target[0]))
            # break
