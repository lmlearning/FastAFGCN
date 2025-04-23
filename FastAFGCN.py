import argparse
import json
import os
import numpy as np
import networkx as nx
import onnxruntime as ort
import torch

BLANK = 0
IN = 1
OUT = 2

# ------------------------- grounded extension -------------------------
def solve_grounded(adj_matrix):
    labelling = np.zeros((adj_matrix.shape[0]), dtype=np.int8)
    unattacked = np.where(np.sum(adj_matrix, axis=0) == 0)[0]
    labelling[unattacked] = IN
    cascade = True
    while cascade:
        new_out = np.unique(np.nonzero(adj_matrix[unattacked, :])[1])
        new_out = [i for i in new_out if labelling[i] != OUT]
        if new_out:
            labelling[new_out] = OUT
            affected = np.unique(np.nonzero(adj_matrix[new_out, :])[1])
        else:
            affected = []
        new_in = []
        for i in affected:
            attackers = np.nonzero(adj_matrix[:, i])[0]
            if np.all(labelling[attackers] == OUT):
                new_in.append(i)
        if new_in:
            labelling[new_in] = IN
            unattacked = np.array(new_in)
        else:
            cascade = False
    return np.where(labelling == IN)[0]

# ------------------------- parser and graph loader -------------------------
def read_af_input(path):
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if not l.startswith('#')]
    args, atts = [], []
    for line in lines:
        parts = line.split()
        if parts[0] == 'p' and parts[1] == 'af':
            args = [str(i) for i in range(1, int(parts[2]) + 1)]
        elif len(parts) == 2:
            atts.append((parts[0], parts[1]))
    return args, atts

def reindex_graph(nxg):
    mapping = {node: i for i, node in enumerate(nxg.nodes())}
    return nx.relabel_nodes(nxg, mapping), mapping

# ------------------------- threshold loader -------------------------
def load_thresholds(json_path):
    with open(json_path) as f:
        return json.load(f)

# ------------------------- ONNX runtime solver -------------------------
def predict_with_onnx(onnx_path, edge_index, num_nodes, threshold=0.5):
    # Xavier random node features
    x = torch.empty((num_nodes, 128), dtype=torch.float32)
    torch.nn.init.xavier_uniform_(x)

    # Prepare inputs
    inputs = {
        "x": x.numpy(),
        "edge_index": edge_index.astype(np.int64),
    }

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], inputs)[0]
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > threshold).astype(bool)
    return preds

# ------------------------- main logic -------------------------
def main(args):
    file_path = args.filepath
    argument = args.argument
    task = args.task
    in_features = args.in_features
    onnx_model = task + "_int8.onnx"
    thresholds = load_thresholds(args.thresholds_file)
    threshold = thresholds.get(task, 0.5)

    af_args, af_atts = read_af_input(file_path)
    nxg = nx.DiGraph()
    nxg.add_nodes_from(af_args)
    nxg.add_edges_from(af_atts)
    nxg, mapping = reindex_graph(nxg)
    inv_mapping = {v: k for k, v in mapping.items()}

    adj = nx.to_numpy_array(nxg)
    grounded_in = solve_grounded(adj)

    arg_idx = mapping.get(argument)
    if arg_idx in grounded_in:
        print("YES")
        return

    if threshold == 1.0:
        print("NO")
        return

    # Create edge_index (PyG format)
    edge_index = np.array(list(nxg.edges())).T  # shape (2, E)

    preds = predict_with_onnx(onnx_model, edge_index, nxg.number_of_nodes(), threshold)

    if preds[arg_idx]:
        print("YES")
    else:
        print("NO")

# ------------------------- entry -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_features", type=int, default=128)
    parser.add_argument("--filepath", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--argument", required=True)
    parser.add_argument("--thresholds_file", default="thresholds.json")
    args = parser.parse_args()
    main(args)
