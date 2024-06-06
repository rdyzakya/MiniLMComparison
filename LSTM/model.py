import torch
import sys
sys.path.append("..")
from MiniCharGPT.model import EmbeddingMatrix

class LSTMLM(torch.nn.Module):
    def __init__(self, h_dim=512, n_layer=4, n_token=28, bidirectional=False):
        super().__init__()
        self.embedding = EmbeddingMatrix(n_token=n_token, dim=h_dim)
        self.lstm = torch.nn.LSTM(h_dim, h_dim, n_layer, batch_first=True, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(h_dim * 2 if bidirectional else h_dim, n_token)

        self.h_dim = h_dim
        self.n_layer = n_layer
        self.bidirectional = bidirectional
    
    def forward(self, input_ids, hidden):
        x = self.embedding(input_ids)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

if __name__ == "__main__":
    m = LSTMLM()
    x = torch.randint(26, (5,12))

    num_layers = 4
    batch_size = 5
    hidden_dim = 512
    hidden = (torch.zeros(num_layers, batch_size, hidden_dim),
                  torch.zeros(num_layers, batch_size, hidden_dim))
    
    out, h = m(x, hidden)

    print(out.shape)