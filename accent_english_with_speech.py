import os
from datasets import Dataset, DatasetDict, load_dataset
from whisper.normalizers import EnglishTextNormalizer
import whisper
import numpy as np
import json
import librosa
from pathlib import Path
from jiwer import wer

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

data = {}
with open("/home/yhl522/English_Accent_DataSet/audio_id_to_meta.json") as f:
    audio_id_to_meta = json.load(f)
    for speaker_id, meta in audio_id_to_meta.items():
        for audio_id, meta in meta.items():
            data[audio_id] = meta

models = ["tiny", "small", "medium", "large-v3"]

for model in models:
    
    print(f"Starting {model}")
    model_path = f"/home/yhl522/whisper_model/{model}.pt"
    whisper_dir = "/home/yhl522/whisper_model"
    model = whisper.load_model(model_path, device="cuda")
    model_name = Path(model_path).stem
    normal = EnglishTextNormalizer()

    datasets = load_dataset("/s5r2/yhl522/English_Accent_DataSet", split="test")
    unique_accents = datasets.unique("accent")

    # Ensure the results directory exists
    os.makedirs(f"./{model_name}_results", exist_ok=True)

    accent_datasets = DatasetDict()
    for unique_accent in unique_accents:
        if unique_accent not in accent_encoding:
            print(f"Warning: Accent {unique_accent} not found in encoding.")
            continue
        
        lan = accent_encoding[unique_accent]
        accent_datasets[lan] = datasets.filter(lambda x: x["accent"] == unique_accent)
        accent_datasets[lan] = accent_datasets[lan].map(lambda x: {"normal_text": normal(x["raw_text"])})
        
        reference_lines = []
        hypothesis_lines = []

        for dataset in accent_datasets[lan]:
            try:
                audio_id = dataset['audio_id']
                ref_text = dataset['normal_text']
                audio = dataset['audio']['array'].astype(np.float32)
                
                tts_audio_path = data[audio_id]["tts_audio"]
                if not os.path.exists(tts_audio_path):
                    continue
                tts_sr = data[audio_id]["tts_sr"]
                tts_audio, tts_sr = librosa.load(tts_audio_path, sr=tts_sr) 
                tts_audio_16k = librosa.resample(tts_audio, orig_sr=tts_sr, target_sr=16000)
                
                concat_audio = np.concatenate([tts_audio_16k,audio], axis=0)
                
                tts_ref_text = normal("Please transcribe the speech to text.")
                tts_audio_result = normal(model.transcribe(tts_audio_path, language='en', without_timestamps=True)['text'])
                tts_wer = wer(tts_ref_text, tts_audio_result)
                
                
                # computer dur
                dur = len(concat_audio) / 16000.0
                if dur > 30 or tts_wer > 0.05:
                    continue
                
                # a1
                # result = model.transcribe(audio, language='en', without_timestamps=True)['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, initial_prompt="please transcribe the speech to text")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, initial_prompt="Please transcribe the speech to text")['text']
                
                # a2
                # result = model.transcribe(audio, language='en', without_timestamps=True, initial_prompt="Please transcribe the speech to text.")['text']
                
                
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="please transcribe the speech to text")['text']
                
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="please transcribe the speech to text.")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="Please transcribe the speech to text")['text']
                
                # a3
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="Please transcribe the speech to text.")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="please transcribe the speech to text")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="please transcribe the speech to text.")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="Please transcribe the speech to text")['text']
                
                # a4
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="Please transcribe the speech to text.")['text']
                
                # a7
                # result = model.transcribe(concat_audio, language='en', without_timestamps=True, initial_prompt="Please transcribe the speech to text.")['text']
                
                # a8
                result = model.transcribe(concat_audio, language='en', without_timestamps=True, prefix="Please transcribe the speech to text.")['text']
                
                # a5
                # result = model.transcribe(concat_audio, language='en', without_timestamps=True, prompt="Please transcribe the speech to text.",prefix="Please transcribe the speech to text.")['text']
                # a6
                # result = model.transcribe(concat_audio, language='en', without_timestamps=True, prompt="Please transcribe the speech to text.", initial_prompt="Please transcribe the speech to text.")['text']
                
                hyp_text = normal(result)

                reference_lines.append(f"{audio_id}\t{ref_text}\n")
                hypothesis_lines.append(f"{audio_id}\t{hyp_text}\n")
            except Exception as e:
                print(f"Error processing {lan} data: {e}")

        with open(f"./{model_name}_results/{lan}_reference.txt", "a") as f1:
            f1.writelines(reference_lines)
        
        with open(f"./{model_name}_results/{lan}_hypothesis.txt", "a") as f2:
            f2.writelines(hypothesis_lines)
        
        print(f"{lan} done")
    print("All done")