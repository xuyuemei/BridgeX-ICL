import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import os
import re
import json
import csv
import itertools
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='llama_translation with vllm')
parser.add_argument('--src_lang', default='en', help='source language')
parser.add_argument('--trg_lang', default='sw', help='target language')
parser.add_argument('--aid_lang', default='de', help='aid language')
args = parser.parse_args()

lang_dic = {"en": "English", "de": "Deutsch", "sw": "kiswahili", 'zh': '中文', 'ja': '日本語',
            'tl': 'Tagalog', 'he': 'עִברִית', 'ar': 'عربي', 'id': 'Indonesia', 'pt': 'Português', 'fr': 'Français',
            'it': 'Italiano', 'es': 'Español','fi':'suomi','hu':'magyar'}

def extract_en(pairs_sentence):
    translation_dict = {}
    num = 0
    for i, src_word in enumerate(pairs_sentence.keys()):
        sentences = pairs_sentence[src_word]

        sentence = sentences.split('\n')[0]  # 仅分割一次
        sentence = sentence.replace('.', '')
        if '→' in sentence:
            parts = sentence.split('→')[1]
            if ": " in parts:
                part = parts.split(': ')[1]
            else:
                part = parts.split(':')[1]
            part = part.split(',')[0]

        if part is None:
            print(src_word, "No translations found.")
            num += 1
        else:
            translation = part.replace("'", '').replace('"', '').replace(' "', '').replace('「', '').replace('」', '')
            translation_dict.update({src_word: translation})
    print("*****invalid:", num)
    return translation_dict


def extract(content):
    translation_dict = {}
    num = 0

    if isinstance(content, dict):
        json_objects = [content]
    else:
        json_objects = []
        json_objects_str = re.findall(r'\{[^{}]*\}', content)
        for json_str in json_objects_str:
            try:
                json_object = json.loads(json_str)
                json_objects.append(json_object)
            except json.JSONDecodeError as e:
                print(f"JSON: {json_str} - error: {e}")

    for json_object in json_objects:
        for src_word, sentences in json_object.items():
            sentence = sentences.split('\n')[0]
            sentence = sentence.replace('.', '')
            if '→' in sentence:
                parts = sentence.split('→')[2]
                if ": " in parts:
                    part = parts.split(': ')[1]
                else:
                    part = parts.split(':')[1]
                part = part.split(',')[0]
                part = part.split('(')[0]
                part = part.split(' ')[0]
                part = part.split('?')[0]

                if part is None:
                    print(src_word, "No translations found.")
                    num += 1
                else:
                    translation = part.replace("'", '').replace('"', '').replace(' "', '').replace('「', '').replace('」',
                                                                                                                    '')
                    print(translation)
                    translation_dict.update({src_word: translation})

    print("*****invalid:", num)
    return translation_dict

def wordsim_evaluate(src_lang, trg_lang, data_dict, aid_dict):
    src_gold_words = []
    trg_gold_words = []
    binary_lexicon = {}

    file_lexicon = open(f'../bli_datasets/{args.src_lang}-{args.trg_lang}.txt', 'r', encoding='utf-8')
    for line in file_lexicon.readlines():
        line = line.rstrip("\n")
        line = line.replace("\t", " ")
        line = line.replace("  ", " ")
        src_word, trg_word = line.split(' ')
        if src_word not in binary_lexicon:
            binary_lexicon.update({src_word: [trg_word]})
        else:
            binary_lexicon[src_word].append(trg_word)
        src_gold_words.append(src_word)
        trg_gold_words.append(trg_word)
    file_lexicon.close()

    count = 0
    hit_count = 0

    for key, value in binary_lexicon.items():
        a = 0
        src_gold_word = key
        trg_gold_words = value
        if src_gold_word in data_dict.keys():
            hit_count = hit_count + 1
            aid_word = aid_dict[src_gold_word]
            candidate_words = [data_dict[src_gold_word].translate(str.maketrans("", "", "ְֱֲֳִֵֶַָׇֹֻּֽֿׁׂׅׄ")).strip().lower()]
            candidate_words = list(set(candidate_words))
            candidate_words = candidate_words + [candidate_word.lower() for candidate_word in candidate_words]
            if candidate_words is not None:
                for candidate_word in candidate_words:
                    candidate_word = candidate_word.strip()
                    trg_gold_words = [word.strip() for word in trg_gold_words]
                    if any(candidate_word == trg_word for trg_word in trg_gold_words):
                        count = count + 1
                        a = 1
                        break
                    else:
                        print("failed:", src_gold_word, candidate_word, trg_gold_words)
        # with open(f"./bli/llama3_{src_lang}_{aid_lang}_{trg_lang}.txt", "a", encoding="utf-8") as f:
        #     f.write(f"{src_gold_word} {aid_word} {candidate_word} {','.join(trg_gold_words)} {a}\n")
    acc1 = count / len(binary_lexicon)
    print('Acc: {0:.4f}'.format(acc1))
    print(f"There are {hit_count} words in the bilingual dictionary that also appear in the constructed dictionary.")
    print(f"Among these {hit_count} words, {count} were found in both dictionaries.")
    return acc1

model_dir = '/data/xkx/mllms/LLM-Research/Meta-Llama-3-8B'
llm = LLM(model=model_dir, dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

src_lang = lang_dic[args.src_lang]
trg_lang = lang_dic[args.trg_lang]
aid_lang = lang_dic[args.aid_lang]

file_lexicon = open(f'../bli_datasets/{args.src_lang}-{args.trg_lang}.txt', 'r', encoding='utf-8')
src_gold_words = []
for line in file_lexicon.readlines():
    line = line.rstrip("\n")
    line = line.replace("\t", " ")
    line = line.replace("  ", " ")
    src_word, trg_word = line.split(' ')
    src_gold_words.append(src_word)
file_lexicon.close()

# First translation: src_lang -> aid_lang
json_path_aid = f"./bli/llama3_{args.src_lang}_{args.aid_lang}_{args.trg_lang}_aid_vllm.json"
if os.path.exists(json_path_aid):
    print("****file already exists")
    with open(json_path_aid, "r", encoding="utf-8") as f:
        pairs_s2t = json.load(f)
        data_dict = extract_en(pairs_s2t)
else:
    pairs_s2t = {}
    prompts = []
    batch_words = []

    for src_word in src_gold_words:
        if src_word in pairs_s2t:
            continue
        prompt = f'{src_lang}:"{src_word}"→{aid_lang}:'
        prompts.append(prompt)
        batch_words.append(src_word)

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=30, stop=["\n"])
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    for src_word, output in zip(batch_words, outputs):
        out = output.outputs[0].text.strip()
        full_line = f"{src_lang}:\"{src_word}\"→{aid_lang}:\"{out}\""
        pairs_s2t[src_word] = full_line

    with open(json_path_aid, "w", encoding="utf-8") as f:
        json.dump(pairs_s2t, f, indent=4, ensure_ascii=False)
    data_dict = extract_en(pairs_s2t)

# Second translation: src_lang -> aid_lang -> trg_lang
json_path_zero = f"./bli/llama3_{args.src_lang}_{args.aid_lang}_{args.trg_lang}_zero_vllm.json"
if os.path.exists(json_path_zero):
    print("****file already exists")
    with open(json_path_zero, "r", encoding="utf-8") as f:
        pairs_s2t = json.load(f)  
else:
    pairs_s2t = {}
    prompts = []
    batch_words = []

    for src_word in src_gold_words:
        if src_word in pairs_s2t:
            continue
        if src_word in data_dict.keys():
            aid_word = data_dict[src_word]
            prompt = f'{src_lang}:"{src_word}"→{aid_lang}:"{aid_word}"→{trg_lang}:'
            prompts.append(prompt)
            batch_words.append(src_word)

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=30, stop=["\n"])
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    for src_word, output in zip(batch_words, outputs):
        out = output.outputs[0].text.strip()
        full_line = f"{src_lang}:\"{src_word}\"→{aid_lang}:\"{data_dict[src_word]}\"→{trg_lang}:\"{out}\""
        pairs_s2t[src_word] = full_line

    with open(json_path_zero, "w", encoding="utf-8") as f:
        json.dump(pairs_s2t, f, indent=4, ensure_ascii=False)
        
translation_dict = extract(pairs_s2t)

# evaluate
acc = wordsim_evaluate(args.src_lang, trg_lang, translation_dict, data_dict)
