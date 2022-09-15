import itertools
from collections import defaultdict

def Process2LineGraph(edges, text, intnode):
  edges = zip(*edges)
  new_edges = [[],[]]
  def Append(u, v):
    new_edges[0].append(u)
    new_edges[1].append(v)
  nodes = text.tolist()
  from_node = defaultdict(list)
  for u, v in edges:
    if nodes[u] == nodes[v] == intnode:
        Append(u, v)
        Append(v, u)
    from_node[u].append(v)
  # opposite direction
  for key, community in from_node.items():
    n = len(community)
    for i in range(n):
      u = community[i]
      for j in range(i+1, n):
        v = community[j]
        Append(u, v)
        Append(v, u)
  # same direction
  for key, community in from_node.items():
    for u in community:
      if u in from_node:
        for v in from_node[u]:
          Append(key, v)
  return new_edges

import itertools
def dense_graph(text):
  num_node = len(text)
  new_edges = [[],[]]
  for u, v in itertools.combinations(range(num_node),2):
    new_edges[0].append(u)
    new_edges[1].append(v)
  return new_edges