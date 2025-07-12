import json
from pathlib import Path
from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map
from collections import Counter

SAMPLE_RATE = 16000
INPUT_FOLDER = Path("data/en/gentle_alignments")  # Pfad zu JSON-Dateien
OUTPUT_FOLDER = Path("data/en/phn")                # Zielpfad für .phn-Dateien
LOG_FILE = Path("conversion_errors.log")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def convert_file(json_path: Path) -> str:
    try:
        output_path = OUTPUT_FOLDER / json_path.with_suffix(".phn").name
        if output_path.exists():
            return "exists"

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        words = [w for w in data.get("words", []) if w.get("case") == "success"]
        if not words:
            return "no_words"

        lines = []

        # Anfangs-SIL (vor erstem Wort)
        if words[0]["start"] > 0.01:
            lines.append(f"0 {int(words[0]['start'] * SAMPLE_RATE)} SIL")

        # Durch alle Wörter iterieren
        for i, word in enumerate(words):
            word_start = word["start"]
            word_end = word["end"]

            # SIL zwischen zwei Wörtern
            if i > 0:
                prev_end = words[i - 1]["end"]
                if word_start - prev_end > 0.01:
                    sil_start = int(prev_end * SAMPLE_RATE)
                    sil_end = int(word_start * SAMPLE_RATE)
                    lines.append(f"{sil_start} {sil_end} SIL")

            # Phoneme pro Wort
            phone_start = word_start
            for phone in word.get("phones", []):
                duration = phone["duration"]
                phone_label = phone["phone"]
                start_sample = int(phone_start * SAMPLE_RATE)
                end_sample = int((phone_start + duration) * SAMPLE_RATE)
                lines.append(f"{start_sample} {end_sample} {phone_label}")
                phone_start += duration

        # End-SIL
        last_end = words[-1]["end"]
        max_dur = data.get("duration", last_end)
        if max_dur - last_end > 0.01:
            start_sample = int(last_end * SAMPLE_RATE)
            end_sample = int(max_dur * SAMPLE_RATE)
            lines.append(f"{start_sample} {end_sample} SIL")

        # Speichern
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        return "ok"

    except Exception as e:
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"{json_path.name} failed: {str(e)}\n")
        return "fail"

# === Hauptlauf ===
if __name__ == "__main__":
    all_json_files = list(INPUT_FOLDER.glob("*.json"))
    print(f"{len(all_json_files)} JSON-Dateien gefunden. CPUs verfügbar: {cpu_count()}")

    # Mit Fortschrittsanzeige und Multiprocessing
    results = process_map(
        convert_file,
        all_json_files,
        max_workers=cpu_count(),
        chunksize=10,
        desc="Konvertiere .json → .phn"
    )

    # Zusammenfassung
    stats = Counter(results)
    print("\nFertig.")
    print(f"  ✅ Erfolgreich: {stats['ok']}")
    print(f"  ⏭️  Übersprungen (bereits vorhanden): {stats['exists']}")
    print(f"  ❌ Fehler: {stats['fail']}")
    print(f"  ⚠️  Kein gültiges Wort: {stats['no_words']}")
    if stats['fail'] > 0:
        print(f"  ➜ Siehe Logdatei: {LOG_FILE}")
