# Boiling-Point-Prediction-Using-GCN-and-GAT
Graph neural networks have been applied in scientific domains to represent data as graphs, consisting of nodes and edges. GCN (Graph Convolutional Network) and GAT (Graph Attention Network) are basic architectures that receive graphs as inputs. These two methods are different from the convolution of CNN and attention of Transformer. This project is the implementation of GCN&GAT from scratch in PyTorch, based on mathematical equations of two methods.

# Overview
You only need train_predict.py to get started after download all .py files and required libraries.
- data.py - create nodes using one-hot encoding and manually define edges
- gcn_gat - a GCN_GAT model combining GCNConv and GATConv
- train_predict.py - training loop and model evaluation
- run_train_predict.py - prediction of carbon dioxide boiling point
