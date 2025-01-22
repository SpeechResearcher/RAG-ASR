import sys
sys.path.append('/home/yhl522/CosyVoice-main/third_party/Matcha-TTS')
sys.path.append('/home/yhl522/CosyVoice-main')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
import json
import shutil
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from pathlib import Path
from whisper.normalizers import EnglishTextNormalizer
import numpy as np
import torch
import torchaudio.functional as F
import torchaudio
import shutil

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

# cuda添加os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os, sys
sys.path.insert(0, os.path.abspath('third_party/Matcha-TTS'))
def save_to_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


# cosyvoice = CosyVoice('/s5r2/yhl522/ckpts/pretrained_models/CosyVoice-300M')
cosyvoice = CosyVoice2('/s5r2/yhl522/ckpts/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
datasets = load_dataset("/s5r2/yhl522/English_Accent_DataSet", split="test")
unique_accents = datasets.unique("accent")


infer_audio_dir = "/s5r2/yhl522/English_Accent_DataSet/infer_audio"
if not os.path.exists(f"{infer_audio_dir}"):
    os.makedirs(f"{infer_audio_dir}")
normal = EnglishTextNormalizer()

audio_id_to_meta = {}

accent_datasets = DatasetDict()
for unique_accent in unique_accents:    
    lan = accent_encoding[unique_accent]
    accent_datasets[lan] = datasets.filter(lambda x: x["accent"] == unique_accent)
    accent_datasets[lan] = accent_datasets[lan].map(lambda x: {"normal_text": normal(x["raw_text"])})

    for dataset in accent_datasets[lan]:
        try:
            audio_id = dataset['audio_id']
            ref_text = dataset['normal_text']
            audio_array = dataset['audio']['array'].astype(np.float32)
            
            if len(audio_array.shape) == 1:
                audio_array = audio_array[np.newaxis, :]
            audio = torch.from_numpy(audio_array)
            
            spk = dataset['speaker_id']
            if spk not in audio_id_to_meta:
                audio_id_to_meta[spk]={}
            
            infer_text = "Please transcribe the speech to text."
            
            for i, j in enumerate(cosyvoice.inference_zero_shot(infer_text, ref_text, audio, stream=False)):
                if not os.path.exists(f"{infer_audio_dir}/{lan}/"):
                    os.makedirs(f"{infer_audio_dir}/{lan}/")
                infer_audio_path = f'{infer_audio_dir}/{lan}/zero_shot_{audio_id}.wav'
                torchaudio.save(infer_audio_path, j['tts_speech'], cosyvoice.sample_rate)
            
            # ori_infer_audio_path = f'{infer_audio_dir}/lan/zero_shot_{audio_id}.wav'
            # infer_audio_path = f'{infer_audio_dir}/{lan}/zero_shot_{audio_id}.wav'
            # shutil.move(f'{ori_infer_audio_path}', infer_audio_path)
            
            audio_id_to_meta[spk][audio_id] = {
                "text": ref_text,
                "speaker_id": spk,
                "duration": dataset['duration'],
                "accent_id": unique_accent,
                "accent": lan,
                "tts_audio": infer_audio_path,
                "tts_sr": cosyvoice.sample_rate
            }

        except Exception as e:
            print(f"Error processing {lan} data: {e}")
    print(f"{lan} done")

audio_id_to_meta_json_path = "/s5r2/yhl522/English_Accent_DataSet/audio_id_to_meta.json"
with open(audio_id_to_meta_json_path, 'w') as f:
    json.dump(audio_id_to_meta, f, indent=4)
print(f"Audio ID to metadata mapping saved to {audio_id_to_meta_json_path}")
                
                
