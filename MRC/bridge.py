import os
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--question_lang', default='ind_Latn')
parser.add_argument('--passage_lang', default='swh_Latn')
parser.add_argument('--bridge_lang', default='eng_Latn')
parser.add_argument('--model_dir', default='/data/xkx/mllms/LLM-Research/Meta-Llama-3-8B')
args = parser.parse_args()


llm = LLM(model=args.model_dir, dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)



def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data



def build_samples(questions, passages):
    p_dict = {p['link']: p["flores_passage"] for p in passages}
    samples = []
    for q in questions:
        if q['link'] in p_dict:
            samples.append({
                "link": q["link"],
                "ds": q["ds"],
                "question_number": q["question_number"],
                "passage": p_dict[q['link']],
                "question": q["question"],
                "mc_answers": [q[f"mc_answer{i}"] for i in range(1, 5)],
                "correct_answer_num": q["correct_answer_num"]
            })
    return samples


# translate
def translate_passages(samples, source_lang, target_lang):
    translated = []
    prompts = [f"Translate the following text from {source_lang} to {target_lang}:\n{sample['passage']}\nTranslation: "
               for sample in samples]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512, stop=["\n"]))
    for s, o in zip(samples, outputs):
        s["translated_passage"] = o.outputs[0].text.strip()
        translated.append(s)
    
    return translated


# prompt
def format_prompt(passage, question, answers, question_lang):
    choice_str = "\n".join(f"{chr(64 + i)}: {a}" for i, a in enumerate(answers, 1))
    return (
        f"Answer the following question based on the passage. Respond with A, B, C, or D.\n"
        f"Passage: {passage}\nQuestion: {question}\nChoices:\n{choice_str}\nAnswer:"
    )


# extract answer
def extract_answer(text):
    match = re.search(r'\b[A-D]\b', text.strip().upper())
    return match.group(0) if match else None


def evaluate(samples, use_translated=False, max_samples=500):
    prompts, keys = [], []
    for s in samples[:max_samples]:
        passage = s["translated_passage"] if use_translated else s["passage"]
        prompt = format_prompt(passage, s["question"], s["mc_answers"], question_lang=args.question_lang)
        prompts.append(prompt)
        keys.append(f"{s['link']}_{s['ds']}_{s['question_number']}")

    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10, stop=["\n"]))
    correct = total = 0
    for s, o, key in zip(samples, outputs, keys):
        pred = extract_answer(o.outputs[0].text)
        gt = chr(64 + int(s["correct_answer_num"]))
        if pred == gt:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def main():
    print(f"Q_lang={args.question_lang} | P_lang={args.passage_lang} | B_lang={args.bridge_lang}")

    questions = load_data(f'/data/xkx/Improve_LowResource_Language/belebele/data/{args.question_lang}.jsonl')
    passages = load_data(f'/data/xkx/Improve_LowResource_Language/belebele/data/{args.passage_lang}.jsonl')
    samples = build_samples(questions, passages)

    print("[step1]")
    acc_base = evaluate(samples, use_translated=False)
    print(f"[zero-shot] {acc_base:.4f}")

    print("[step2]")
    samples_translated = translate_passages(samples, source_lang=args.passage_lang, target_lang=args.bridge_lang)
    acc_aux = evaluate(samples_translated, use_translated=True)
    print(f"[bridge] {acc_aux:.4f}")


if __name__ == "__main__":
    main()
