import argparse

def read_similarity_file(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return {}, []

    header = lines[0].split('\t')
    bridge_langs = header[1:-1]

    sim_dict = {}
    for line in lines[1:]:
        parts = line.split('\t')
        if parts[0].lower().startswith('average'):
            continue
        layer = int(parts[0])
        sims = [float(s) for s in parts[1:-1]]
        sim_dict.update({(layer, lang): sim for lang, sim in zip(bridge_langs, sims)})

    return sim_dict, bridge_langs

def seclect_bridge(sim_dict, bridge_langs, target_layers):
    avg_sims = {lang: sum(sim_dict.get((layer, lang), 0.0) for layer in target_layers) / len(target_layers)
                for lang in bridge_langs}
    top_lang = max(avg_sims, key=avg_sims.get) if avg_sims else None
    return top_lang, avg_sims

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default='id')
    parser.add_argument("--target", default='he')
    parser.add_argument("--start_layer", type=int, default=10)
    parser.add_argument("--end_layer", type=int, default=21)
    args = parser.parse_args()

    input_file = f"./llama_result/{args.source}_{args.target}_bridge.txt"
    target_layers = range(args.start_layer, args.end_layer + 1)

    sim_dict, bridge_langs = read_similarity_file(input_file)
    top_lang, avg_sims = seclect_bridge(sim_dict, bridge_langs, target_layers)

    print(f"language pair: {args.source}_{args.target}")
    print("best bridge:", top_lang)
    print(", ".join([f"{lang}:{sim:.6f}" for lang, sim in avg_sims.items()]))

if __name__ == "__main__":
    main()
