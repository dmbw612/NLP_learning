import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_layers = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(self.transformer_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

# 사용 예시
vocab_size = 10000  # 적절한 어휘 크기로 설정
d_model = 512
nhead = 8
num_layers = 6

model = TransformerDecoder(d_model, nhead, num_layers)

# 모델을 통한 예측
input_sequence = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])  # 입력 시퀀스 예시
output_sequence = model(input_sequence)
