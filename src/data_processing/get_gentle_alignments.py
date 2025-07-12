import os
import json
import time
import re
import tempfile
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

# === Konfiguration ===
dataset_base = Path(r"C:/Users/sebas/Desktop/commonVoiceDataset/datasets_original/en")
clips_folder = dataset_base / "clips"
tsv_path = dataset_base / "validated.tsv"
wav_folder = dataset_base / "wav"  
alignment_output_folder = dataset_base / "gentle_alignments"
alignment_output_folder.mkdir(parents=True, exist_ok=True)
error_log_path = dataset_base / "alignment_errors.log"

# === Gentle-Server-Ports ===
gentle_ports = [8765, 8766, 8767, 8768, 8769, 8770]
gentle_servers = [f"http://localhost:{port}/transcriptions?async=false" for port in gentle_ports]
num_workers = len(gentle_servers)

# === Hilfsfunktionen ===
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
    sentence = re.sub(r'[^a-zA-Z ]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

def align_with_gentle(wav_path, transcript, gentle_url):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(transcript)
        temp_file_path = temp_file.name

    try:
        with open(wav_path, 'rb') as audio_file, open(temp_file_path, 'rb') as transcript_file:
            files = {'audio': audio_file, 'transcript': transcript_file}
            response = requests.post(gentle_url, files=files, timeout=30)
            response.raise_for_status()

            if not response.text.strip():
                return None

            return response.json()
    except Exception as e:
        return str(e)
    finally:
        os.remove(temp_file_path)

# === Worker-Funktion mit zugewiesenem Server ===
def process_row(row_and_url):
    row, gentle_url = row_and_url
    sentence = row['sentence']
 
    filename = Path(row['path']).with_suffix('.wav')  # ändert .mp3 zu .wav – egal was davor steht

    wav_file = wav_folder / filename

    json_output_path = alignment_output_folder / Path(row['path']).with_suffix('.json')

 

    if not wav_file.exists() or json_output_path.exists():
        return None  # Überspringen

    transcript = preprocess_sentence(sentence)
    result = align_with_gentle(str(wav_file), transcript, gentle_url)

    if isinstance(result, dict):
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return "ok"
    else:
        with open(error_log_path, "a", encoding="utf-8") as log:
            log.write(f"{wav_file} failed: {result}\n")
        return "fail"

# === Hauptprogramm ===
if __name__ == "__main__":

    print("Starte Gentle-Alignment-Prozess...")
    print(f"TSV-Datei: {tsv_path}")
    print("Lade TSV-Datei...")
    data = np.genfromtxt(tsv_path, delimiter='\t', names=True, dtype=None, encoding='utf-8')
    print(f"Lese {len(data)} Zeilen ein...")

    # === Aufgaben zuteilen ===
    tasks = []
    for i, row in enumerate(data):
        gentle_url = gentle_servers[i % num_workers]
        tasks.append((row, gentle_url))

    print(f"Starte Pool mit {num_workers} Prozessen...\n")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_row, tasks), total=len(tasks), desc="Aligning", unit="samples"))

    total = len(data)
    aligned = results.count("ok")
    failed = results.count("fail")

    print(f"\nFertig. Gesamt: {total}, Erfolgreich: {aligned}, Fehlerhaft: {failed}")
    print(f"Fehler: {error_log_path}")
