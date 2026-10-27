
import torch

print("Loading graph files...")
edges = torch.load(
    "edge_index.pt", map_location="cpu", weights_only=True
)
types = torch.load(
    "edge_type.pt", map_location="cpu", weights_only=True
)
weights = torch.load(
    "edge_weight.pt", map_location="cpu", weights_only=True
)

features = torch.load(
    "features.pt", map_location="cpu", weights_only=True
)
labels = torch.load(
    "labels_bot.pt", map_location="cpu", weights_only=True
)

src, dst = edges[0], edges[1]
num_nodes = features.shape[0]
num_edges = edges.shape[1]

print("\n=== GRAPH STRUCTURE ===")
print("Nodes defined by features:", num_nodes)
print("Total edge records:", num_edges)
print("Distinct nodes appearing in edges:",
      torch.unique(edges).numel())
print("Nodes absent from all edges:",
      num_nodes - torch.unique(edges).numel())
print("Self-loop records:", int((src == dst).sum()))
print("Distinct source nodes:", torch.unique(src).numel())
print("Distinct destination nodes:", torch.unique(dst).numel())

print("\n=== PER-RELATION AUDIT ===")
for t in torch.unique(types).tolist():
    mask = types == t
    s, d, w = src[mask], dst[mask], weights[mask]

    # Count unique (source, destination) pairs within this type.
    pairs = torch.stack((s, d), dim=1)
    unique_pairs = torch.unique(pairs, dim=0)

    print(f"\nRelation type {t}")
    print("  Edge records:", s.numel())
    print("  Unique source-destination pairs:",
          unique_pairs.shape[0])
    print("  Repeated pair records:",
          s.numel() - unique_pairs.shape[0])
    print("  Self-loops:", int((s == d).sum()))
    print("  Distinct source nodes:", torch.unique(s).numel())
    print("  Distinct destination nodes:", torch.unique(d).numel())
    print("  Weight min/median/max:",
          w.min().item(),
          w.median().item(),
          w.max().item())

print("\n=== DUPLICATE EDGE AUDIT ===")
triples = torch.stack((src, dst, types), dim=1)
unique_triples = torch.unique(triples, dim=0)

print("Unique (source, destination, type) triples:",
      unique_triples.shape[0])
print("Repeated triple records:",
      num_edges - unique_triples.shape[0])

print("\n=== BOT LABEL DISTRIBUTION ===")
for label, count in zip(*torch.unique(
        labels, return_counts=True)):
    print(f"Label {label.item()}: {count.item()}")

print("\n=== AUDIT COMPLETE ===")

import torch

print("Loading graph files...")
edges = torch.load(
    "edge_index.pt", map_location="cpu", weights_only=True
)
types = torch.load(
    "edge_type.pt", map_location="cpu", weights_only=True
)
weights = torch.load(
    "edge_weight.pt", map_location="cpu", weights_only=True
)

features = torch.load(
    "features.pt", map_location="cpu", weights_only=True
)
labels = torch.load(
    "labels_bot.pt", map_location="cpu", weights_only=True
)

src, dst = edges[0], edges[1]
num_nodes = features.shape[0]
num_edges = edges.shape[1]

print("\n=== GRAPH STRUCTURE ===")
print("Nodes defined by features:", num_nodes)
print("Total edge records:", num_edges)
print("Distinct nodes appearing in edges:",
      torch.unique(edges).numel())
print("Nodes absent from all edges:",
      num_nodes - torch.unique(edges).numel())
print("Self-loop records:", int((src == dst).sum()))
print("Distinct source nodes:", torch.unique(src).numel())
print("Distinct destination nodes:", torch.unique(dst).numel())

print("\n=== PER-RELATION AUDIT ===")
for t in torch.unique(types).tolist():
    mask = types == t
    s, d, w = src[mask], dst[mask], weights[mask]

    # Count unique (source, destination) pairs within this type.
    pairs = torch.stack((s, d), dim=1)
    unique_pairs = torch.unique(pairs, dim=0)

    print(f"\nRelation type {t}")
    print("  Edge records:", s.numel())
    print("  Unique source-destination pairs:",
          unique_pairs.shape[0])
    print("  Repeated pair records:",
          s.numel() - unique_pairs.shape[0])
    print("  Self-loops:", int((s == d).sum()))
    print("  Distinct source nodes:", torch.unique(s).numel())
    print("  Distinct destination nodes:", torch.unique(d).numel())
    print("  Weight min/median/max:",
          w.min().item(),
          w.median().item(),
          w.max().item())

print("\n=== DUPLICATE EDGE AUDIT ===")
triples = torch.stack((src, dst, types), dim=1)
unique_triples = torch.unique(triples, dim=0)

print("Unique (source, destination, type) triples:",
      unique_triples.shape[0])
print("Repeated triple records:",
      num_edges - unique_triples.shape[0])

print("\n=== BOT LABEL DISTRIBUTION ===")
for label, count in zip(*torch.unique(
        labels, return_counts=True)):
    print(f"Label {label.item()}: {count.item()}")

print("\n=== AUDIT COMPLETE ===")
