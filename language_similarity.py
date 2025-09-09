import torch
import torch.nn.functional as F
from collections import defaultdict

langs = ['zh', 'ja', 'ar', 'he', 'id', 'tl', 'fi', 'hu', 'sw', 'en', 'de', 'fr', 'it', 'pt', 'es']

# load data
n, over_zero = [], []
for lang in langs:
    data = torch.load(f'activation/activation.{lang}.flores.llama', weights_only=True)
    n.append(int(data['n']))
    over_zero.append(data['over_zero'])

n = torch.tensor(n)
over_zero = torch.stack(over_zero, dim=-1)

num_layers, intermediate_size, lang_num = over_zero.size()
activation_probs = over_zero / n

# select top neurons
top_rate = 0.01
flattened_probs = activation_probs.view(-1, lang_num)
top_prob_values = torch.kthvalue(flattened_probs, k=round(flattened_probs.size(0) * (1 - top_rate)), dim=0).values
selected_positions = activation_probs >= top_prob_values.unsqueeze(0).unsqueeze(0)
print(top_prob_values)

# save neuron ID
selected_ids = defaultdict(list)  
for lang_idx in range(lang_num):
    selected_neurons = torch.where(selected_positions[:, :, lang_idx] == 1)
    selected_ids[lang_idx] = list(zip(selected_neurons[0].tolist(), selected_neurons[1].tolist()))

for lang_idx, neuron_ids in selected_ids.items():
    with open(f'top_neuron/top_neurons_{langs[lang_idx]}.txt', 'w') as f:
        f.writelines(f"({l}, {n})\n" for l, n in neuron_ids)

def save_neuron_ids_with_probs(selected_ids, activation_probs):
    for lang_idx, neuron_ids in selected_ids.items():
        with open(f'top_neuron/top_neurons_with_probs_{langs[lang_idx]}.txt', 'w') as f:
            for l, n in neuron_ids:
                f.write(f"({l},{n}), {activation_probs[l, n, lang_idx].item()}\n")

save_neuron_ids_with_probs(selected_ids, activation_probs)


lang_indices = torch.arange(lang_num)
combined = torch.stack(torch.where(selected_positions), dim=-1)
grouped = defaultdict(list)

for position in combined.tolist():
    layer_index, neuron_index, lang_index = position
    grouped[lang_index].append((layer_index, neuron_index))

# compute language similarity
def compute_similarity(matrix1, matrix2):
    cos_sim = F.cosine_similarity(matrix1.unsqueeze(0), matrix2.unsqueeze(0), dim=1)
    return cos_sim.item()

def compute_activation_similarity():
    similarity_matrix = torch.zeros((lang_num, lang_num), dtype=torch.float)
    for i in lang_indices:
        for j in lang_indices:
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue

            lang1_indices = set(grouped[i.item()])
            lang2_indices = set(grouped[j.item()])
            intersection_indices = lang1_indices.intersection(lang2_indices)

            if len(intersection_indices) == 0:
                similarity_matrix[i, j] = 0.0
                continue

            lang1_activation = activation_probs[:, list(intersection_indices), i.item()]
            lang2_activation = activation_probs[:, list(intersection_indices), j.item()]

            similarity = compute_similarity(lang1_activation.flatten(), lang2_activation.flatten())
            similarity_matrix[i, j] = similarity
    print(similarity_matrix)


# Overlap Matrix
def compute_overlap_matrix():
    unique_langs = torch.unique(torch.tensor([i for i in range(lang_num)]))

    overlap_matrix = torch.zeros((len(unique_langs), len(unique_langs)), dtype=torch.int)

    for i, lang1 in enumerate(unique_langs):
        for j, lang2 in enumerate(unique_langs):

            indices1 = set(grouped[lang1.item()])
            indices2 = set(grouped[lang2.item()])

            overlap = len(indices1.intersection(indices2))
            overlap_matrix[i, j] = overlap

    print(overlap_matrix)

compute_activation_similarity()
compute_overlap_matrix()
