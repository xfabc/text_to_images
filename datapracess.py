import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class TextToImageDataset(Dataset):
    def __init__(self, data_dir, max_length=32, image_size=64):
        self.data_dir = data_dir
        self.max_length = max_length
        self.image_size = image_size

        self.captions = []
        with open(os.path.join(data_dir, 'captions.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                print(line)
                image_name, caption = line.strip().split(',')
                self.captions.append((image_name, caption))

                # 构建词汇表
            self.vocab = self.build_vocab()
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
            self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)

    def build_vocab(self):
        words = set()
        for _, caption in self.captions:
            for word in caption.split():
                words.add(word)
        words = sorted(list(words))
        words = ['<PAD>', '<UNK>'] + words  # 添加填充和未知词标记
        return words

    def text_to_sequence(self, text):
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>'])
                    for word in text.split()]
        # 截断或填充到固定长度
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence += [self.word2idx['<PAD>']] * (self.max_length - len(sequence))
        return torch.tensor(sequence, dtype=torch.long)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name, caption = self.captions[idx]
        image = Image.open(os.path.join(self.data_dir, 'images', image_name)).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0  # 归一化到[0,1]
        image = torch.tensor(image).permute(2, 0, 1).float()  # 转为C×H×W

        text_seq = self.text_to_sequence(caption)

        return {
            "image": image,
            "text": text_seq,
            "caption": caption
        }
