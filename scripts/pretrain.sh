#!/bin/bash

INPUT_FILE="data/sbu_captions-hindi-english-pretrain/conversations.jsonl"
TRAIN_FILE="data/sbu_captions-hindi-english-pretrain/train.jsonl"
TEST_FILE="data/sbu_captions-hindi-english-pretrain/test.jsonl"
TRAIN_SIZE=0.9

# Calculate split point
total_lines=$(wc -l < "$INPUT_FILE")
train_lines=$(awk "BEGIN {print int($total_lines * $TRAIN_SIZE)}")

# Shuffle and split
shuf "$INPUT_FILE" | head -n "$train_lines" > "$TRAIN_FILE"
shuf "$INPUT_FILE" | tail -n "+$((train_lines + 1))" > "$TEST_FILE"
sed -i 's|processed_data|data/sbu_captions-hindi-english-pretrain|g' "$TRAIN_FILE"
sed -i 's|processed_data|data/sbu_captions-hindi-english-pretrain|g' "$TEST_FILE"
echo "Train: $train_lines lines"
echo "Test: $((total_lines - train_lines)) lines"

echo "Data split completed: "

python -m src.pretrain \
    --train_file "$TRAIN_FILE" \
    --test_file "$TEST_FILE" \
    --language_model "models/gemma-3-1b-it" \
    --vision_model "models/siglip-base-patch16-256-multilingual" \
    --num_image_tokens 64 \
    