import torch
import ast
from collections import defaultdict
import argparse
import os
def load_neuron(file_path):
    with open(file_path, 'r') as f:
        return [ast.literal_eval(line.strip()) for line in f if line.strip()]

def group_neurons_by_layer(neuron_list):
    layer_map = defaultdict(list)
    for layer, neuron_id in neuron_list:
        layer_map[layer].append(neuron_id)
    return layer_map

def extract_submatrix_by_layer(layer_map, activation_freq):
    layer_matrices = {}
    for layer, neuron_ids in layer_map.items():
        rows = [activation_freq[layer, nid, :] for nid in neuron_ids]
        layer_matrices[layer] = torch.stack(rows)
    return layer_matrices


def compute_hsic(mat1, mat2, sigma=None):
    mat1 = (mat1 - mat1.mean(dim=1, keepdim=True)) / (mat1.std(dim=1, keepdim=True) + 1e-6)
    mat2 = (mat2 - mat2.mean(dim=1, keepdim=True)) / (mat2.std(dim=1, keepdim=True) + 1e-6)

    n1, n2 = mat1.size(0), mat2.size(0)
    n_samples = mat1.size(1)

    if sigma is None:
        x = mat1.view(-1, n_samples)
        y = mat2.view(-1, n_samples)
        dist_x = torch.cdist(x, x).view(-1)
        dist_y = torch.cdist(y, y).view(-1)
        sigma = torch.median(torch.cat([dist_x, dist_y])).item() + 1e-6

    # [n_samples, n_samples]
    H = torch.eye(n_samples, device=mat1.device) - (1.0 / n_samples) * torch.ones(n_samples, n_samples,
                                                                                  device=mat1.device)

    # Compute Gaussian kernels for mat1 and mat2
    x = mat1.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist_x = (x - x_t).pow(2).squeeze(2)
    K = torch.exp(-dist_x / (2 * sigma ** 2))

    y = mat2.unsqueeze(2)
    y_t = y.transpose(1, 2)
    dist_y = (y - y_t).pow(2).squeeze(2)
    L = torch.exp(-dist_y / (2 * sigma ** 2))

    # Centered kernels
    KH = torch.matmul(K, H)
    HL = torch.matmul(H, L)

    # Compute HSIC scores
    hsic_scores = torch.zeros((n1, n2), device=mat1.device)
    for i in range(n1):
        for j in range(n2):
            hsic_scores[i, j] = torch.trace(torch.matmul(KH[i], HL[j]))

    max_sim1 = hsic_scores.max(dim=1).values.mean().item()
    max_sim2 = hsic_scores.max(dim=0).values.mean().item()
    return (max_sim1 + max_sim2) / 2




def compute_layerwise_sim(specific_mats, overlap_mats, sim_func, device):
    sims = []
    for layer in sorted(set(specific_mats) & set(overlap_mats)):
        mat1 = specific_mats[layer].to(device)
        mat2 = overlap_mats[layer].to(device)
        sim = sim_func(mat1, mat2)
        sims.append(sim)
        print(f"Layer {layer}: {sim}")
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    return sims, mean_sim


def save_result(layer_sims_dict, total_layers, bridge_langs, filename):
    with open(filename, 'w') as f:

        header = "\t".join(["Layer"] + bridge_langs + ["Ranking"])
        f.write(header + "\n")

        # Save per-layer similarities and rankings
        for layer in range(total_layers):
            sims = [layer_sims_dict.get((layer, lang), 0.0) for lang in bridge_langs]
            sims_str = "\t".join([f"{sim:.6f}" for sim in sims])

            layer_sims = {lang: layer_sims_dict.get((layer, lang), 0.0) for lang in bridge_langs}
            sorted_langs = sorted(layer_sims.items(), key=lambda x: x[1], reverse=True)
            layer_ranking = ">".join([lang for lang, _ in sorted_langs])
            f.write(f"{layer}\t{sims_str}\t{layer_ranking}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute HSIC")
    parser.add_argument("--source", default='zh', help="Source language code")
    parser.add_argument("--target", default='he', help="Target language code")
    parser.add_argument("--langs", nargs='+', default=['en','de', 'fr', 'it', 'pt', 'es'],  #'en',
                        help="Bridge languages list")
    parser.add_argument("--gpu", type=int, default=0, help="Specify the GPU to use (default: 0)")

    args = parser.parse_args()

    source_lang = args.source
    target_lang = args.target
    bridge_langs = args.langs

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer_sims_dict = {}


    for bridge_lang in bridge_langs:
        activation_file = f"./bli_activation/activation_value_{source_lang}_{target_lang}_llama.pt"  #_mis
        specific_file = f"./llama_neuron/top_neurons_{source_lang}_{target_lang}.txt"
        overlap_file = f"./llama_neuron/top_neurons_{bridge_lang}_specific_bridge_{source_lang}_{target_lang}_k.txt"


        activation_freq = torch.load(activation_file)  # [num_layers, intermediate_size, num_samples]
        specific_neurons = load_neuron(specific_file)
        overlap_neurons = load_neuron(overlap_file)

        specific_dict = group_neurons_by_layer(specific_neurons)
        overlap_dict = group_neurons_by_layer(overlap_neurons)

        specific_layer_mats = extract_submatrix_by_layer(specific_dict, activation_freq)
        overlap_layer_mats = extract_submatrix_by_layer(overlap_dict, activation_freq)

        total_layers = activation_freq.shape[0]

        layerwise_sims, mean_sim = compute_layerwise_sim(
            specific_layer_mats, overlap_layer_mats, compute_hsic, device)

        for layer, sim in enumerate(layerwise_sims):
            layer_sims_dict[(layer, bridge_lang)] = sim

    save_result(layer_sims_dict, total_layers, bridge_langs, f"./llama_result/{source_lang}_{target_lang}_bridge.txt")

if __name__ == "__main__":
    main()
