import json
import torch
import os
from jiwer import wer, cer
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
import wespeaker
from datasets import Dataset, DatasetDict, load_dataset
from whisper.normalizers import EnglishTextNormalizer
import whisper
import numpy as np
import librosa
from pathlib import Path
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

wespeaker_model = wespeaker.load_model('english')
wespeaker_model.set_gpu(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = whisper.load_model("/home/yhl522/whisper_model/large-v3.pt")
normal = EnglishTextNormalizer()

ori_data = {}
with open("/home/yhl522/English_Accent_DataSet/audio_id_to_meta.json") as f:
    audio_id_to_meta = json.load(f)
    for speaker_id, meta in audio_id_to_meta.items():
        for audio_id, meta in meta.items():
            ori_data[audio_id] = meta

ori_audio_dir = "/s5r2/yhl522/English_Accent_DataSet/audio"
if not os.path.exists(f"{ori_audio_dir}"):
    os.makedirs(f"{ori_audio_dir}")

meta = {}
lan2spk = {}
datasets = load_dataset("/s5r2/yhl522/English_Accent_DataSet", split="test")
unique_accents = datasets.unique("accent")
accent_datasets = DatasetDict()
for unique_accent in unique_accents:
    if unique_accent not in accent_encoding:
        print(f"Warning: Accent {unique_accent} not found in encoding.")
        continue
    lan = accent_encoding[unique_accent]

    spk2utt = {}
    accent_datasets[lan] = datasets.filter(lambda x: x["accent"] == unique_accent)
    accent_datasets[lan] = accent_datasets[lan].map(lambda x: {"normal_text": normal(x["raw_text"])})
    
    reference_lines = []
    hypothesis_lines = []
    
    meta[lan] = accent_datasets[lan]
    for dict1 in tqdm(meta[lan]):
        if lan not in lan2spk:
            lan2spk[lan] = {}
        audio_id = dict1['audio_id']
        speaker_id = dict1['speaker_id']
        if speaker_id not in spk2utt:
            spk2utt[speaker_id]= {}
        if audio_id not in ori_data:
            continue
        dict1["tts_audio"] = ori_data[audio_id]["tts_audio"]
        dict1["tts_sr"] = ori_data[audio_id]["tts_sr"]
        if not os.path.exists(dict1["tts_audio"]):
            continue
        # tts_ref_text = normal("Please transcribe the speech to text.")
        # tts_audio_result = normal(model.transcribe(dict1["tts_audio"], language='en', without_timestamps=True)['text'])
        # tts_wer = wer(tts_ref_text, tts_audio_result)
        # if tts_wer > 0.05:
        #     continue
        spk2utt[speaker_id][audio_id] = dict1
    lan2spk[lan][speaker_id] = spk2utt[speaker_id]
    
    if not os.path.exists(f"{ori_audio_dir}/{lan}/"):
        os.makedirs(f"{ori_audio_dir}/{lan}/")

    for speaker,dict1 in tqdm(lan2spk[lan].items()):
        current_speaker_len = len(dict1)
        current_speaker_maxtrix = {}
        for key2, dict2 in dict1.items():
            audio_id = dict2['audio_id']
            infer_audio_prompt = dict2['audio']['array'].astype(np.float32)
            infer_audio_prompt = torch.tensor(infer_audio_prompt, dtype=torch.float32).unsqueeze(0)
            ori_audio_path = f'{ori_audio_dir}/{lan}/{audio_id}.wav'
            torchaudio.save(ori_audio_path, infer_audio_prompt, 16000)
            print(f"{lan}/{speaker}/{audio_id}")
            query_audio = ori_audio_path
            best_score = 0
            best_similarity = 0
            best_utt_path = ""
            best_wer = 0
            for key3,dict3 in dict1.items():
                generate_audio = dict3['tts_audio']
                target_audio = generate_audio
                
                tts_ref_text = normal("Please transcribe the speech to text.")
                tts_audio_result = normal(model.transcribe(generate_audio, language='en', without_timestamps=True)['text'])
                tts_wer = wer(tts_ref_text, tts_audio_result)
                
                try:
                    similarity = wespeaker_model.compute_similarity(query_audio, target_audio)
                    
                    current_score = similarity + 1 - tts_wer
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_similarity = similarity
                        best_wer = tts_wer
                        best_utt_path = generate_audio
                except Exception as e:
                    print(e)
                    continue
            lan2spk[lan][speaker_id][audio_id][best_utt_path]=best_utt_path
            
            tts_sr = ori_data[audio_id]["tts_sr"]
            tts_audio, tts_sr = librosa.load(best_utt_path, sr=tts_sr) 
            tts_audio_16k = librosa.resample(tts_audio, orig_sr=tts_sr, target_sr=16000) 
            concat_audio = np.concatenate([tts_audio_16k,dict2['audio']['array'].astype(np.float32)], axis=0)
            dur = len(concat_audio) / 16000.0
            if dur > 30:
                continue
            result = model.transcribe(concat_audio, language='en', without_timestamps=True, prompt="Please transcribe the speech to text.",prefix="Please transcribe the speech to text.")['text']
            ref_text = dict2['normal_text']
            hyp_text = normal(result)
            reference_lines.append(f"{audio_id}\t{ref_text}\n")
            hypothesis_lines.append(f"{audio_id}\t{hyp_text}\n")
            
    with open(f"./results/rag-asr/{lan}_reference.txt", "a") as f1:
        f1.writelines(reference_lines)
        
    with open(f"./results/rag-asr/{lan}_hypothesis.txt", "a") as f2:
        f2.writelines(hypothesis_lines)


with open("/home/yhl522/English_Accent_DataSet/new_audio_id_to_meta.json", "w") as f:
    json.dump(lan2spk, f, indent=4)
