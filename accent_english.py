import os
from datasets import Dataset, DatasetDict, load_dataset
from whisper.normalizers import EnglishTextNormalizer
import whisper
import numpy as np
from pathlib import Path

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
                # result = model.transcribe(audio, language='en', without_timestamps=True)['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, initial_prompt="please transcribe the speech to text")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, initial_prompt="Please transcribe the speech to text")['text']
                result = model.transcribe(audio, language='en', without_timestamps=True, initial_prompt="Please transcribe the speech to text.")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="please transcribe the speech to text")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="please transcribe the speech to text.")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="Please transcribe the speech to text")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prompt="Please transcribe the speech to text.")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="please transcribe the speech to text")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="please transcribe the speech to text.")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="Please transcribe the speech to text")['text']
                # result = model.transcribe(audio, language='en', without_timestamps=True, prefix="Please transcribe the speech to text.")['text']
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