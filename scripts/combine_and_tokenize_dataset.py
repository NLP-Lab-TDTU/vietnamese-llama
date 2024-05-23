import os
from os import path
import glob
from itertools import chain
from argparse import ArgumentParser

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, DatasetDict

DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    args = ArgumentParser()
    args.add_argument('--tokenizer_name', type=str, default="init_model/Llama-3-mini")
    args.add_argument('--output_dir', type=str, default='processed_data')
    args.add_argument('--block_size', type=int, default=8192)
    args.add_argument('--overwrite_cache', action='store_true')
    args.add_argument('--num_proc', type=int, default=os.cpu_count() * 3 // 4)
    args.add_argument('--seed', type=int, default=42)
    return args.parse_args()

args = parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

def apply_chat_template(sample):
    try:
        return {'text': tokenizer.apply_chat_template(sample['messages'], tokenize=False)}
    except:
        return {'text': ''}

datanames = [
    "CC-MAIN-2023-50",
    "CC-MAIN-2023-40",
    "CC-MAIN-2023-23",
    "CC-MAIN-2023-14",
    "CC-MAIN-2023-06",
    "CC-MAIN-2022-49",
    "CC-MAIN-2022-40",
    "CC-MAIN-2022-33",
    "CC-MAIN-2022-27",
    "CC-MAIN-2022-21",
    "CC-MAIN-2022-05",
    "CC-MAIN-2021-49",
    "CC-MAIN-2021-43",
    "CC-MAIN-2021-39",
    "CC-MAIN-2021-31",
    "CC-MAIN-2021-25",
    "CC-MAIN-2021-21",
    "CC-MAIN-2021-17",
    "CC-MAIN-2021-10",
    "CC-MAIN-2021-04",
]

datasets = []
for dataname in datanames:
    print("Loading", dataname)
    dataset = load_dataset(
        "parquet",
        data_files=glob.glob(path.join(DIR, '../data', dataname, "*.parquet")),
        split="train",
        num_proc=args.num_proc
    )
    if 'messages' in dataset.column_names:
        dataset = dataset.map(apply_chat_template, num_proc=args.num_proc)
    if 'text' in dataset.column_names:
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])

    dataset = dataset.filter(lambda x: x['text'] != '' and x['text'] != None, num_proc=args.num_proc)

    datasets.append(dataset)

datanames = [
    "vietgpt/openwebtext_en",
    "vietgpt/the_pile_openwebtext2",
    "vietgpt/SlimOrca-Dedup",
    "vietgpt/no_robots",
]

for dataname in datanames:
    print("Loading", dataname)
    dataset = load_dataset(
        "parquet",
        data_files=glob.glob(path.join(DIR, '../data', dataname, "**/*.parquet")),
        split="train",
        num_proc=args.num_proc
    )
    if 'messages' in dataset.column_names:
        dataset = dataset.map(apply_chat_template, num_proc=args.num_proc)
    if 'text' in dataset.column_names:
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])

    dataset = dataset.filter(lambda x: x['text'] != '' and x['text'] != None, num_proc=args.num_proc)

    datasets.append(dataset)

raw_dataset = concatenate_datasets(datasets)

raw_dataset = raw_dataset.shuffle(seed=args.seed)

block_size = args.block_size

def tokenize_function_and_group_texts(examples):
    texts = [text+'<|end_of_text|>' for text in examples["text"]]
    examples = tokenizer(texts)

    examples['attention_mask'] = [[idx + 1] * len(value) for idx, value in enumerate(examples['attention_mask'])]
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = raw_dataset.map(
    tokenize_function_and_group_texts,
    batched=True,
    batch_size=1000,
    num_proc=args.num_proc,
    remove_columns=["text"],
    load_from_cache_file=not args.overwrite_cache,
    desc=f"Running tokenizer on dataset & Grouping texts in chunks of {block_size}",
)

if os.path.exists(args.output_dir) and args.overwrite_cache:
    os.remove(args.output_dir)

dataset_dict = DatasetDict({
    "train": lm_datasets
})

dataset_dict.save_to_disk(args.output_dir)
