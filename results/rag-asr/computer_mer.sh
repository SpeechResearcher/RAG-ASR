#!/bin/bash

# 设置环境变量或配置文件路径
export PYTHON_SCRIPT="/home/yhl522/wenet/tools/compute-wer.py"

# 启用错误检测
set -e

languages=("Dutch" "German" "Czech" "Polish" "French" "Hungarian" "Finnish" "Romanian" "Slovak" "Spanish" "Italian" "Estonian" "Lithuanian" "Croatian" "Slovene" "English" "Scottish" "Irish" "NorthernIrish" "Indian" "Vietnamese" "Canadian" "American")

BASE_DIR="/home/yhl522/English_Accent_DataSet/results/rag-asr"

# 检查目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "Directory $BASE_DIR does not exist, skipping..."
    continue
fi

for lang in "${languages[@]}"; do
    echo "Processing $lang"
    
    reference_file="$BASE_DIR/${lang}_reference.txt"
    hypothesis_file="$BASE_DIR/${lang}_hypothesis.txt"
    output_file="$BASE_DIR/wer-${lang}.txt"
    
    # 检查文件是否存在
    if [ ! -f "$reference_file" ] || [ ! -f "$hypothesis_file" ]; then
        echo "Files $reference_file or $hypothesis_file do not exist, skipping..."
        continue
    fi
    
    python "$PYTHON_SCRIPT" \
        --char 0 \
        --v 1 \
        "$reference_file" \
        "$hypothesis_file" > "$output_file"
    
    echo "WER for $lang: $(tail -n 1 "$output_file")"
    tail -10 "$output_file" | grep -oP 'Overall -> \K[0-9.]+'
done


