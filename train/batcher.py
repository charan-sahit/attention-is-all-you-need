import random
import torch
from torch.utils.data import Sampler

class TokenBucketSampler(Sampler):
    def __init__(self, dataset, tokens_per_batch=25_000, chunk_size=10_000, shuffle=True) -> None:
        self.dataset = dataset
        self.tokens_per_batch = tokens_per_batch
        self.chunk_size = chunk_size
        self.shuffle = shuffle

        self.lengths = [
            (i, len(src), len(tgt) + 1)
            for i, (src, tgt) in enumerate(dataset.pairs)
        ]  
    
    def _batches(self):
        sorted_lens = sorted(self.lengths, key=lambda x: max(x[1], x[2]))
        chunks = [sorted_lens[i : i + self.chunk_size]
                    for i in range(0, len(sorted_lens), self.chunk_size)]
        if self.shuffle:
            random.shuffle(chunks)

        batches = []
        for chunk in chunks:
            cur, max_s, max_t = [], 0, 0
            for idx, s_len, t_len in chunk:
                new_s = max(max_s, s_len)
                new_t = max(max_t, t_len)
                if cur and ((len(cur)+1)*new_s > self.tokens_per_batch or (len(cur)+1)*new_t>self.tokens_per_batch):
                    batches.append(cur)
                    cur, max_s, max_t = [idx], s_len, t_len
                else:
                    cur.append(idx)
                    max_s, max_t = new_s, new_t
            if cur:
                batches.append(cur)

        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self._batches():
            yield batch

    def __len__(self):
        total_tokens = sum(max(s, t) for _, s, t in self.lengths)
        return max(1, total_tokens // self.tokens_per_batch)