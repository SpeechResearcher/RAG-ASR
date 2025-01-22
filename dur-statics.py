from datasets import load_dataset, DatasetDict
import torchaudio

accent_encoding = {
    0: "Dutch",
    1: "German",
    2: "Czech",
    3: "Polish",
    4: "French",
    5: "Hungarian",
    6: "Finnish",
    7: "Romanian",
    8: "Slovak",
    9: "Spanish",
    10: "Italian",
    11: "Estonian",
    12: "Lithuanian",
    13: "Croatian",
    14: "Slovene",
    15: "English",
    16: "Scottish",
    17: "Irish",
    18: "NorthernIrish",
    19: "Indian",
    20: "Vietnamese",
    21: "Canadian",
    22: "American"
}

def calculate_dataset_duration(dataset):
    total_duration = 0
    for item in dataset:
        # 假设音频在 'audio' 字段
        # waveform, sample_rate = torchaudio.load(item['audio']['array'])
        # duration = waveform.shape[1] / sample_rate  # 秒
        duration = len(item['audio']['array']) / 16000.0
        total_duration += duration
    
    # 转换为小时
    return total_duration / 3600

# 加载数据集
datasets = load_dataset("/s5r2/yhl522/English_Accent_DataSet", split="test")
unique_accents = datasets.unique("accent")
accent_datasets = DatasetDict()

for unique_accent in unique_accents:
    if unique_accent not in accent_encoding:
        print(f"Warning: Accent {unique_accent} not found in encoding.")
        continue
        
    lan = accent_encoding[unique_accent]
    accent_datasets[lan] = datasets.filter(lambda x: x["accent"] == unique_accent)
    # 计算并打印每个语言的数据时长
    duration_hours = calculate_dataset_duration(accent_datasets[lan])
    print(f"Language {lan}: {duration_hours:.2f} hours")
