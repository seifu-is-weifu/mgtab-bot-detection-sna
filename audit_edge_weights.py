
import torch

edges = torch.load("edge_index.pt", map_location="cpu",
                   weights_only=True)
types = torch.load("edge_type.pt", map_location="cpu",
                   weights_only=True)
weights = torch.load("edge_weight.pt", map_location="cpu",
                     weights_only=True)

src, dst = edges
print("=== EDGE WEIGHT INVESTIGATION ===")

for t in range(7):
    mask = types == t
    s, d, w = src[mask], dst[mask], weights[mask]

    pairs = torch.stack((s, d), dim=1)
    unique_pairs, inverse, counts = torch.unique(
        pairs, dim=0, return_inverse=True, return_counts=True
    )

    repeated = counts > 1
    pair_weight_sum = torch.zeros(len(unique_pairs))
    pair_weight_sum.scatter_add_(0, inverse, w)

    pair_weight_mean = pair_weight_sum / counts

    print(f"\nRelationship type {t}")
    print("Records:", len(w))
    print("Unique pairs:", len(unique_pairs))
    print("Pairs occurring more than once:", int(repeated.sum()))
    print("Maximum occurrences of one pair:", int(counts.max()))
    print("Self-loop records:", int((s == d).sum()))

    print("Weight quantiles (min, 25%, 50%, 75%, max):")
    print(torch.quantile(
        w.float(),
        torch.tensor([0., .25, .5, .75, 1.])
    ).tolist())

    if repeated.any():
        repeated_means = pair_weight_mean[repeated]
        print("Mean aggregated weight of repeated pairs:")
        print(torch.quantile(
            repeated_means,
            torch.tensor([0., .5, 1.])
        ).tolist())

    loop_weights = w[s == d]
    if len(loop_weights):
        print("Self-loop weight min/median/max:",
              loop_weights.min().item(),
              loop_weights.median().item(),
              loop_weights.max().item())

print("\n=== INVESTIGATION COMPLETE ===")
