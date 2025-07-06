# Boiling-Point-Prediction-Using-GCN-and-GAT
Graph neural networks have been applied in scientific domains to represent data as graphs, consisting of nodes and edges. GCN (Graph Convolutional Network) and GAT (Graph Attention Network) are basic architectures that receive graphs as inputs. These two methods are different from the convolution of CNN and attention of Transformer. This project is the implementation of GCN&GAT from scratch in PyTorch, based on mathematical equations of two methods.

# Overview
You only need train_predict.py to get started after download all .py files and required libraries.
- data.py - create nodes using one-hot encoding and manually define edges
- gcn_gat - a GCN_GAT model combining GCNConv and GATConv
- train_predict.py - training loop and model evaluation
- run_train_predict.py - prediction of carbon dioxide boiling point

# data.py
For simplicity, there are only 6 molecules, and carbon dioxide is used for prediction later. They are represented as molecular formulas, and the bond types are not specified. Because each molecule has a different number of nodes, the model cannot be trained all at once. The solution is to train each molecule separately to get individual losses, and then sum all the losses later.
```python
import torch

def create_element_vector(atomic_number):
    vector = torch.zeros(118)
    vector[atomic_number - 1] = 1
    return vector

H = create_element_vector(1)
C = create_element_vector(6)
N = create_element_vector(7)
O = create_element_vector(8)
S = create_element_vector(16)

water = torch.stack([H, H, O])
ethane = torch.stack([C, C, H, H, H, H, H, H])
methane = torch.stack([C, H, H, H, H])
ammonia = torch.stack([N, H, H, H])
hydrogen_sulfide = torch.stack([H, H, S])
carbon_dioxide = torch.stack([C, O, O])

water_edge_index = torch.tensor([
    [0, 1, 2, 2],
    [2, 2, 0, 1],
], dtype=torch.long)

ethane_edge_index = torch.tensor([
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7],
    [2, 1, 6, 7, 3, 4, 5, 0, 0, 1, 1, 1, 0, 0],
], dtype=torch.long)

methane_edge_index = torch.tensor([
    [0, 0, 0, 0, 1, 2, 3, 4],
    [1, 2, 3, 4, 0, 0, 0, 0],
], dtype=torch.long)

ammonia_edge_index = torch.tensor([
    [0, 0, 0, 1, 2, 3],
    [1, 2, 3, 0, 0, 0],
], dtype=torch.long)

hydrogen_sulfide_edge_index = torch.tensor([
    [0, 1, 2, 2],
    [2, 2, 0, 1],
], dtype=torch.long)

carbon_dioxide_edge_index = torch.tensor([
    [0, 0, 1, 2],
    [1, 2, 0, 0],
], dtype=torch.long)

boiling_point = torch.tensor([100, 
                             78.37,
                             -161.5,
                             -33.34,
                             -60], dtype=torch.float)   # Celsius

data_list = [
    (water, water_edge_index, boiling_point[0]),
    (ethane, ethane_edge_index, boiling_point[1]),
    (methane, methane_edge_index, boiling_point[2]),
    (ammonia, ammonia_edge_index, boiling_point[3]),
    (hydrogen_sulfide, hydrogen_sulfide_edge_index, boiling_point[4]),
]
```

# run_train_predict.py
Although the loss decreases sufficiently, model cannot predict the boiling point of carbon dioxide well, which is -78.46. It can be considered that increasing the dataset and representing more properties will improve the prediction.
```python
from train_predict import train_predict

train_predict(hidden_dim=128, train_num_epochs=4000, print_interval=1000)
```
```text
Epoch 1000, Loss: 13853.6348
Epoch 2000, Loss: 430.0951
Epoch 3000, Loss: 1.2525
Epoch 4000, Loss: 0.1958
tensor([359.0571])
```
