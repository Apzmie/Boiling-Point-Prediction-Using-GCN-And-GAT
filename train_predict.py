from data import data_list, carbon_dioxide, carbon_dioxide_edge_index
from gcn_gat import GCN_GAT
import torch
import torch.nn as nn
import torch.optim as optim

def train_predict(hidden_dim, train_num_epochs, print_interval, node_dim=118):
    model = GCN_GAT(node_dim, hidden_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())

    best_loss = float('inf')
    for epoch in range(train_num_epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0
        for node, edge_index, target in data_list:
            out = model(node, edge_index)
            loss = loss_fn(out, target.unsqueeze(0))
            total_loss += loss

        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        predicted_bp = model(carbon_dioxide, carbon_dioxide_edge_index)
        print(predicted_bp)
