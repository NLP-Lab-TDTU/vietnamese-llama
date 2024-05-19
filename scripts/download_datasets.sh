#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1

snapshots=(
    "CC-MAIN-2023-50"
    "CC-MAIN-2023-40"
    "CC-MAIN-2023-23"
    "CC-MAIN-2023-14"
    "CC-MAIN-2023-06"
    "CC-MAIN-2022-49"
    "CC-MAIN-2022-40"
    "CC-MAIN-2022-33"
    "CC-MAIN-2022-27"
    "CC-MAIN-2022-21"
    "CC-MAIN-2022-05"
    "CC-MAIN-2021-49"
    "CC-MAIN-2021-43"
    "CC-MAIN-2021-39"
    "CC-MAIN-2021-31"
    "CC-MAIN-2021-25"
    "CC-MAIN-2021-21"
    "CC-MAIN-2021-17"
    "CC-MAIN-2021-10"
    "CC-MAIN-2021-04"
)

for snapshot in "${snapshots[@]}"
do
    echo "Downloading $snapshot"
    huggingface-cli download \
    --repo-type dataset \
    vietgpt/commoncrawl \
    --include $snapshot/*.parquet \
    --local-dir-use-symlinks False \
    --local-dir ./data
done

en_datasets=(
    "vietgpt/openwebtext_en"
    "vietgpt/the_pile_openwebtext2"
)

for dataset in "${en_datasets[@]}"
do
    echo "Downloading $dataset"
    huggingface-cli download \
    --repo-type dataset \
    $dataset \
    --include data/*.parquet \
    --local-dir-use-symlinks False \
    --local-dir ./data/$dataset
done

sft_datasets=(
    "vietgpt/SlimOrca-Dedup"
    "vietgpt/no_robots"
)

for dataset in "${sft_datasets[@]}"
do
    echo "Downloading $dataset"
    huggingface-cli download \
    --repo-type dataset \
    $dataset \
    --include data/*.parquet \
    --local-dir-use-symlinks False \
    --local-dir ./data/$dataset
done
