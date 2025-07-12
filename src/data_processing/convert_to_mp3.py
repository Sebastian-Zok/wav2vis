import os
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === Pfade anpassen ===
mp3_ordner = r"C:\Users\sebas\Desktop\commonVoiceDataset\data\en\clips"
wav_ordner = os.path.join(mp3_ordner, "converted_wav")
fehler_log = os.path.join(mp3_ordner, "fehler_log.txt")

# === Zielordner vorbereiten ===
os.makedirs(wav_ordner, exist_ok=True)

# === Alle MP3-Dateien auflisten ===
alle_dateien = [f for f in os.listdir(mp3_ordner) if f.lower().endswith(".mp3")]

# === Konvertierungsfunktion für einen Prozess ===
def konvertiere(datei):
    mp3_pfad = os.path.join(mp3_ordner, datei)
    wav_datei = os.path.splitext(datei)[0] + ".wav"
    wav_pfad = os.path.join(wav_ordner, wav_datei)

    # Überspringen, wenn bereits konvertiert
    if os.path.exists(wav_pfad):
        return None

    try:
        audio = AudioSegment.from_mp3(mp3_pfad)
        audio.export(wav_pfad, format="wav")
    except Exception as e:
        return f"{datei}: {str(e)}"
    return None

# === Hauptprogramm mit Multiprocessing ===
if __name__ == "__main__":
    # Anzahl Prozesse: 1 Kern Reserve für das System
    num_prozesse = max(1, cpu_count() - 1)

    with Pool(processes=num_prozesse) as pool:
        for fehler in tqdm(pool.imap_unordered(konvertiere, alle_dateien), total=len(alle_dateien), desc="Konvertiere MP3s", unit="Dateien"):
            if fehler:
                with open(fehler_log, "a", encoding="utf-8") as log:
                    log.write(fehler + "\n")
