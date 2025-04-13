import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, text):
        embedded = self.embedding(text)
        encoded = self.transformer(embedded)
        return encoded.mean(dim=1)  # 取平均作为文本特征