import torch

def subgraph_edges(edges, num_layer):
  A = dense_matrix = edges_to_dense(edges)
  sparse_matrices = []
  for _ in range(num_layer):
    subgraph_matrix = A.T * dense_matrix
    if subgraph_matrix.size(0) != 0:
      norm = torch.max(subgraph_matrix)
      if norm != 0.0:
        subgraph_matrix = subgraph_matrix / norm
    sparse_matrices.append(subgraph_matrix.to_sparse())
    dense_matrix = dense_matrix @ dense_matrix
  return sparse_matrices

def edges_to_dense(edges):
  length = len(edges[0])
  if length == 0:
    num_node = 0
  else:
    num_node = max(max(edges[0]), max(edges[1])) + 1
  sparse_matrix = torch.sparse_coo_tensor(indices = edges, values = torch.ones(length), size = (num_node, num_node))
  dense_matrix = sparse_matrix.to_dense()
  return dense_matrix