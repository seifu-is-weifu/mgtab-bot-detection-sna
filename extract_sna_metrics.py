
import csv
import gc
import torch
import networkx as nx

NUM_NODES = 10199

RELATIONS = {
    0: "followers",
    1: "friends",
    2: "mentions",
    3: "replies",
    4: "quoted_posts",
    5: "URLs",
    6: "hashtags",
}

edges = torch.load(
    "edge_index.pt", map_location="cpu", weights_only=True
)
types = torch.load(
    "edge_type.pt", map_location="cpu", weights_only=True
)
weights = torch.load(
    "edge_weight.pt", map_location="cpu", weights_only=True
)
labels = torch.load(
    "labels_bot.pt", map_location="cpu", weights_only=True
)

metric_rows = []
class_rows = []

for relation_id, relation_name in RELATIONS.items():
    mask = types == relation_id

    src = edges[0, mask].tolist()
    dst = edges[1, mask].tolist()
    edge_weights = weights[mask]

    raw_count = len(src)
    self_loops = sum(s == d for s, d in zip(src, dst))

    # Preserve unique directed pairs for topology metrics.
    unique_pairs = set(zip(src, dst))
    non_loop_pairs = {
        (s, d) for s, d in unique_pairs if s != d
    }

    # Keep all nodes, including isolated nodes.
    G = nx.DiGraph()
    G.add_nodes_from(range(NUM_NODES))
    G.add_edges_from(non_loop_pairs)

    U = G.to_undirected()

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    degree_rows = []
    for node in range(NUM_NODES):
        degree_rows.append({
            "relation": relation_name,
            "node_id": node,
            "bot_label": int(labels[node]),
            "in_degree": in_degrees[node],
            "out_degree": out_degrees[node],
            "total_degree": in_degrees[node] + out_degrees[node],
        })

    for label in (0, 1):
        subset = [
            row for row in degree_rows
            if row["bot_label"] == label
        ]

        for metric in ("in_degree", "out_degree", "total_degree"):
            values = sorted(row[metric] for row in subset)
            n = len(values)
            median = (
                values[n // 2] if n % 2
                else (values[n // 2 - 1] + values[n // 2]) / 2
            )

            class_rows.append({
                "relation": relation_name,
                "bot_label": label,
                "class_size": n,
                "degree_metric": metric,
                "mean": sum(values) / n,
                "median": median,
            })

    weak_components = list(nx.weakly_connected_components(G))
    largest_wcc = max(
        (len(c) for c in weak_components), default=0
    )

    # Clustering is calculated on the simple undirected projection.
    clustering = nx.average_clustering(U)

    metric_rows.append({
        "relation_id": relation_id,
        "relation": relation_name,
        "raw_edge_records": raw_count,
        "unique_directed_pairs_including_loops": len(unique_pairs),
        "self_loop_records": self_loops,
        "unique_non_loop_edges": len(non_loop_pairs),
        "repeated_records": raw_count - len(unique_pairs),
        "nodes": NUM_NODES,
        "nodes_with_edges": G.number_of_nodes() - len(
            list(nx.isolates(G))
        ),
        "directed_density": nx.density(G),
        "mean_in_degree": sum(in_degrees.values()) / NUM_NODES,
        "mean_out_degree": sum(out_degrees.values()) / NUM_NODES,
        "weakly_connected_components": len(weak_components),
        "largest_weak_component_nodes": largest_wcc,
        "largest_weak_component_fraction": largest_wcc / NUM_NODES,
        "strongly_connected_components": nx.number_strongly_connected_components(G),
        "average_undirected_clustering": clustering,
        "edge_weight_min": edge_weights.min().item(),
        "edge_weight_median": edge_weights.median().item(),
        "edge_weight_max": edge_weights.max().item(),
    })

    print(
        f"Completed {relation_name}: "
        f"{len(non_loop_pairs)} unique non-loop edges"
    )

    del G, U, mask, src, dst, unique_pairs, non_loop_pairs
    del degree_rows, in_degrees, out_degrees
    gc.collect()

with open("sna_metrics_by_relation.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metric_rows[0].keys())
    writer.writeheader()
    writer.writerows(metric_rows)

with open("degree_comparison_by_bot_label.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=class_rows[0].keys())
    writer.writeheader()
    writer.writerows(class_rows)

print("\n=== BASELINE METRICS COMPLETE ===")
print("Saved: sna_metrics_by_relation.csv")
print("Saved: degree_comparison_by_bot_label.csv")
print("\nSummary:")
for row in metric_rows:
    print(
        f"{row['relation']}: "
        f"edges={row['unique_non_loop_edges']:,}, "
        f"density={row['directed_density']:.6f}, "
        f"components={row['weakly_connected_components']:,}, "
        f"clustering={row['average_undirected_clustering']:.6f}"
    )
