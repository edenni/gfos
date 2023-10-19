import torch
from torch_geometric.nn import GraphConv, SAGEConv

x = torch.randn(100, 32)
edge_index = torch.randint(0, 3, (2, 100))

sage = SAGEConv(32, 32, aggr="mean")
graph = GraphConv(32, 32, aggr="mean")

print("Parameters of SAGEConv:")
for name, param in sage.named_parameters():
    print(name, param.size())
print("Parameters of GraphConv:")
for name, param in graph.named_parameters():
    print(name, param.size())

sage.lin_l.weight.data.copy_(graph.lin_rel.weight.data)
sage.lin_l.bias.data.copy_(graph.lin_rel.bias.data)
sage.lin_r.weight.data.copy_(graph.lin_root.weight.data)

out_sage = sage(x, edge_index)
out_graph = graph(x, edge_index)

print("Output of SAGEConv:")
print(out_sage)
print("Output of GraphConv:")
print(out_graph)

torch.testing.assert_close(out_sage, out_graph)
