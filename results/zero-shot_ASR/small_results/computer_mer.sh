#!/bin/bash

# 设置环境变量或配置文件路径
export BASE_DIR="/s5r2/yhl522/English_Accent_DataSet/data/small_results"
export PYTHON_SCRIPT="/home/yhl522/wenet/tools/compute-wer.py"

# 启用错误检测
set -e
languages=("Dutch" "German" "Czech" "Polish" "French" "Hungarian"  "Finnish" "Romanian" "Slovak" "Spanish" "Italian" "Estonian" "Lithuanian" "Croatian" "Slovene" "English" "Scottish" "Irish" "NorthernIrish" "Indian" "Vietnamese" "Canadian" "American")

for lang in "${languages[@]}";
do
    echo "Processing $lang"
	python "$PYTHON_SCRIPT" \
            --char 0 \
            --v 1 \
            "$BASE_DIR/${lang}_reference.txt" \
            "$BASE_DIR/${lang}_hypothesis.txt" > "$BASE_DIR/wer-${lang}.txt"
	tail -6 "$BASE_DIR/wer-${lang}.txt"
done

