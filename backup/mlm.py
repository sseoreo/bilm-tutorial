import os
import random
import torch
from typing import List

from vocab import Vocab


class BERTLanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, vocab, eos_id: str='<eos>', 
                mask_id: str='<mask>', pad_id: str="<pad>", seq_len: int=512, mask_frac: float=0.15, p: float=0.5):
        """Initiate language modeling dataset.
        Arguments:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
            p (float): probability for NSP. defaut 0.5
        """
        super(BERTLanguageModelingDataset, self).__init__()
        self.vocab = vocab
        with open(os.path.join(data_dir, "train.txt"), "r") as f:
            self.data = [line.strip() for line in f.readlines() if line != '']
            
        
        self.seq_len = seq_len
        self.eos_id = vocab.eos_idx
        self.pad_id = vocab.padding_idx
        self.mask_id = vocab.mask_idx

        self.p = p
        self.mask_frac = mask_frac

    def __getitem__(self, i):
        seq = self.vocab.encode_line(self.data[i].strip(), add_eos=False)
        seq = seq[:self.seq_len-1]    

        # sentence embedding: 0 for A, 1 for B
        mlm_target = torch.tensor( seq + [self.eos_id] + [self.pad_id] * (self.seq_len - 1 - len(seq))).long().contiguous()
        
        def masking(data):
            data = torch.tensor(data).long().contiguous()
            data_len = data.size(0)
            ones_num = int(data_len * self.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]
            data = data.masked_fill(lm_mask.bool(), self.mask_id)

            return data

        mlm_train = torch.cat([masking(seq), torch.tensor([self.eos_id])]).long().contiguous()
        mlm_train = torch.cat([mlm_train, torch.tensor([self.pad_id] * (self.seq_len - mlm_train.size(0)))]).long().contiguous()

        # mlm_train, mlm_target, sentence embedding, NSP target
        return mlm_train, mlm_target
        # return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab



if __name__ == '__main__':
    data_dir = 'ptb'
    

    vocab = Vocab(data_dir, mask_token='<mask>')

    dataset = BERTLanguageModelingDataset(data_dir, vocab, seq_len=16)

    # 작동 테스트
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    for batch, (mlm_train, mlm_target) in enumerate(dataloader):
        pass
        print(mlm_train.size(), vocab.decode_tokids(mlm_train[0]))
        print(mlm_target.size(), vocab.decode_tokids(mlm_target[0]))
        # break