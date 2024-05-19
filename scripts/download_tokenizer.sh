#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1

files=(
    "special_tokens_map.json"
    "tokenizer.json"
    "tokenizer_config.json"
)

for file in "${files[@]}"
do
    echo "Downloading $file"
    huggingface-cli download \
    --repo-type model \
    --local-dir ./init_model/Llama-3-mini \
    --local-dir-use-symlinks False \
    vietgpt/Llama-3-mini $file
done
