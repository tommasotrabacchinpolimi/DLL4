import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
from torch.optim import AdamW



# Set the seed
seed = 42
torch.manual_seed(seed)
# Probably, this below must be changed if you work with a M1/M2/M3 Mac
torch.cuda.manual_seed(seed)  # for CUDA
torch.backends.cudnn.deterministic = True  # for CUDNN
torch.backends.benchmark = False  # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
n = 20

class TSPDataset(Dataset):
    def __init__(self, dataset, device):
      tours = []
      graphs = []
      for s in dataset:
        tours.append(s[1][:-1])
        graph = []
        for i in range(len(s[0].nodes)):
          coords = s[0].nodes[i]["pos"]
          graph.append([coords[0], coords[1]])
        graphs.append(graph)

      self.tours_tensor = torch.tensor(tours, dtype=torch.long).to(device)
      self.graphs_tensor = torch.tensor(graphs, dtype=torch.float32).to(device)

      del tours, graphs

    def __len__(self):
        return len(self.graphs_tensor)

    def __getitem__(self, idx):
        return self.graphs_tensor[idx], self.tours_tensor[idx]

def entropy_penalty(vector):
    # Normalize the vector into probabilities
    probabilities = torch.softmax(vector, dim=-1)
    # Compute the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
    return -entropy


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        

    def forward(self, input, target):
      #first_element_loss = self.gamma * (input[:,n-1] ** 2).sum()
      log_input = torch.log(input + 1e-9)
      return F.nll_loss(log_input, target)
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):# max_len is probably n
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding) :
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(1), :])
    
class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward):
      super().__init__(d_model, nhead, dim_feedforward, dropout = 0.0, batch_first = True)


    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_is_causal, memory_is_causal):
        x = tgt
        x = self.norm1(
            x + self._sa_block(x, tgt_mask, None)
        )
        x = self.norm2(
            x
            + self._mha_block(
                x, memory, memory_mask, None
            )
        )
        x = self.norm3(x + self._ff_block(x))

        return x
    

class LastCustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, dim_feedforward):
      super().__init__(d_model, 1, dim_feedforward, dropout = 0.0, batch_first = True)


    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_is_causal, memory_is_causal):
        x = tgt

        x = self.multihead_attn(
            x,
            memory,
            memory,
            attn_mask=memory_mask[:x.size(0),:,:],
            key_padding_mask=None,
            is_causal=False,
            need_weights=True,
            )[1]

        x = x.squeeze(1)

        return x

      
      

class TSPTransformer(nn.Module):
  def __init__ (self, d_d = n, d_e = n, N_d = 2, N_e = 2, num_heads = 4, dim_feedforward = 0, dropout_p = 0.0):
    super(TSPTransformer, self).__init__()
    self.d_d = d_d
    self.d_e = d_e
    self.N_e = N_e
    self.N_d = N_d
    self.num_heads = num_heads
    self.dim_feedforward = dim_feedforward
    self.dropout_p = dropout_p

    self.pos = PositionalEncoding(self.d_d, self.dropout_p, n)
    self.linear1 = nn.Linear(2, d_e)
    self.linear2 = nn.Linear(d_e, d_d)
    self.linear3 = nn.Linear(d_d, n)
    self.embedding = nn.Embedding(n, d_d)
    self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dropout= 0.0, d_model=self.d_e, nhead=self.num_heads, dim_feedforward = self.dim_feedforward), num_layers=self.N_e)
    self.decoder = nn.TransformerDecoder(CustomDecoderLayer(d_model=self.d_d, nhead=self.num_heads, dim_feedforward = self.dim_feedforward), num_layers=self.N_d)
    self.last_decoder_layer = nn.TransformerDecoder(LastCustomDecoderLayer(d_model=self.d_d, dim_feedforward = self.dim_feedforward), num_layers=1)

  def get_tgt_causal_mask(self, tgt, batch_size, size, current_index, device):
        mask = torch.tril(torch.ones(size, size, device = device) == 1)
        mask = mask.float()

        mask = mask.masked_fill(mask == 0, float('-1e9'))
        mask = mask.masked_fill(mask == 1, 0)

        return mask
  
  def get_memory_cities_mask(self, tgt, batch_size, size, current_index, device):
    mask = torch.zeros(size = (batch_size, size, size), device = device)
    mask.float()
    for b in range(batch_size):
        mask[b, tgt[b, :current_index], current_index] = float('-1e9')
    return mask


  def forward(self, src, batch_size, device):
    torch.autograd.set_detect_anomaly(True)
    probs = torch.zeros(size = (batch_size, n, n), device = device)
    tgt = torch.zeros(size = (batch_size, n), dtype=torch.long, device = device)
    for index in range(1, n):
      causal_mask = self.get_tgt_causal_mask(tgt, batch_size, n, index, device)
      cities_mask = torch.repeat_interleave(self.get_memory_cities_mask(tgt, batch_size, n, index, device), self.num_heads, dim=0)
      y = self.linear1(src) * math.sqrt(self.d_e)
      x = self.embedding(tgt) * math.sqrt(self.d_d)
      x = self.pos(x)
      y = self.encoder(y)
      y = self.linear2(y)
      x = self.decoder(x, y, tgt_mask = causal_mask, memory_mask = cities_mask)
      x = self.last_decoder_layer(tgt = x, memory = y, tgt_mask = causal_mask, memory_mask = cities_mask)
      probs[:, index, :] = x[:, :, index]
      next_token = torch.argmax(probs[:, index, :], dim = 1)
      tgt = tgt.clone()
      tgt[:, index] = next_token
    return probs, tgt

def train(model, device, train_loader, validation_dataloader, optimizer, epochs_number):
    model.to(device)
    model.train()
    loss_fn = CustomLoss()
    for epoch in range(epochs_number):
      model.train()
      train_loss = 0.0
      for idx, (coords, tours) in enumerate(train_loader):
          optimizer.zero_grad()
          output, tgt = model(coords, coords.size(0), device)
          loss = loss_fn(output, tours)
          train_loss += loss.item()
          loss.backward()
          optimizer.step()
          
      print(f"Epoch {epoch+1}: Train loss: {train_loss/len(train_dataloader)}")
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
          for coords, tours in validation_dataloader:
              coords, tours = coords.to(device), tours.to(device)
              output, tgt = model(coords, coords.size(0), device)
              loss = loss_fn(output, tours)
              val_loss += loss.item()

      avg_val_loss = val_loss / len(validation_dataloader)
      print(f"Epoch {epoch+1}: Validation loss: {avg_val_loss:.4f}")
    


file_path_train = "/content/drive/MyDrive/train_20_DLL_ass4.pkl"
file_path_val = "/content/drive/MyDrive/valid_20_DLL_ass4.pkl"
device = torch.device("cuda")


# Load the pickle file
with open(file_path_train, "rb") as file_train:
    dataset_train = pickle.load(file_train)

with open(file_path_val, "rb") as file_val:
    dataset_val = pickle.load(file_val)
train_dataset = TSPDataset(dataset_train, device)
validation_dataset = TSPDataset(dataset_val, device)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
model = TSPTransformer()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train(model, device, train_dataloader,validation_dataloader,  optimizer, 100)
# for idx, (coords, tours) in enumerate(dataloader):
#   input = torch.zeros(size = (coords.size(0),n), dtype=torch.long)
#   for i in range(1, n):
#     output = model(coords, input.long(), i, coords.size(0))
#     #softmaxed_tensor = F.softmax(output[:, :, i], dim=-1)
#     #print(output[:, :, i])
#     for b in range(coords.size(0)):
#       s = torch.multinomial(output[b, :, i], num_samples=1).item()
#       input[b, i] = s
#   print(input)
#   break




