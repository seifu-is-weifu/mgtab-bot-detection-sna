
import os
import torch

FILES = [
    "edge_index.pt",
    "edge_type.pt",
    "edge_weight.pt",
    "features.pt",
    "labels_bot.pt",
    "labels_stance.pt",
]

data = {}

print("=== MGTAB INITIAL DATA AUDIT ===")

for filename in FILES:
    if not os.path.exists(filename):
        continue

    obj = torch.load(filename, map_location="cpu", weights_only=True)
    data[filename] = obj

    print(f"\nFILE: {filename}")
    print("Type:", type(obj).__name__)

    if not torch.is_tensor(obj):
        print("Not a tensor; inspect separately.")
        continue

    print("Shape:", tuple(obj.shape))
    print("Data type:", obj.dtype)

    if obj.numel() == 0:
        print("WARNING: Empty tensor")
        continue

    if obj.is_floating_point():
        print("NaN values:", int(torch.isnan(obj).sum()))
        print("Infinite values:", int(torch.isinf(obj).sum()))

    print("Minimum:", obj.min().item())
    print("Maximum:", obj.max().item())

    if obj.ndim == 1 and "label" in filename:
        values, counts = torch.unique(obj, return_counts=True)
        print("Label distribution:")
        for value, count in zip(values.tolist(),
                                counts.tolist()):
            print(f"  {value}: {count}")

print("\n=== CROSS-FILE VALIDATION ===")

if "edge_index.pt" in data and "edge_type.pt" in data:
    edges = data["edge_index.pt"]
    types = data["edge_type.pt"]

    print("Edge count:", edges.shape[1])
    print("Edge types:", torch.unique(types).tolist())
    print("Edge/type lengths match:",
          edges.shape[1] == types.numel())

    if "edge_weight.pt" in data:
        weights = data["edge_weight.pt"]
        print("Edge/weight lengths match:",
              edges.shape[1] == weights.numel())
        print("Weight dtype:", weights.dtype)
        print("Weight minimum:", weights.min().item())
        print("Weight maximum:", weights.max().item())

if "features.pt" in data:
    features = data["features.pt"]
    print("Feature rows:", features.shape[0])

    if "edge_index.pt" in data:
        print("Maximum edge node ID:",
              data["edge_index.pt"].max().item())
        print("All edge IDs within feature rows:",
              data["edge_index.pt"].min().item() >= 0
              and data["edge_index.pt"].max().item()
              < features.shape[0])

for filename in ("labels_bot.pt", "labels_stance.pt",
                 "label_stance.pt"):
    if filename in data and "features.pt" in data:
        print(filename, "matches feature rows:",
              data[filename].shape[0]
              == data["features.pt"].shape[0])

print("\n=== AUDIT COMPLETE ===")
