# Quran vs Tafsir Audio Splitter

A Python CLI tool that uses machine learning to automatically separate audio recordings containing both **Quran recitation** (Arabic, melodic) and **Tafsir explanation** (Kurdish, conversational speech) into individual segment files.

Built for processing ~240 audio tracks from *Tafsira Sanahi*, where each track contains alternating Quran verses and their Kurdish explanations.

## How It Works

1. **Acoustic Feature Extraction** — Each audio file is analyzed using 3-second sliding windows (1-second hop). ~75 features are extracted per window including MFCCs, delta MFCCs, spectral centroid, rolloff, bandwidth, contrast, flatness, chroma, zero-crossing rate, RMS energy, and tonnetz.

2. **ML Classification** — A RandomForest classifier (200 trees, balanced classes) with StandardScaler is trained on manually labeled timestamps to classify each window as either *quran* or *tafsir*.

3. **Post-Processing** — Raw predictions are refined with median filtering (kernel size 7), minimum segment duration enforcement (3 seconds), and silence-boundary snapping for clean cuts.

4. **Parallel Export** — Segments are exported as MP3 files using all available CPU cores (ProcessPoolExecutor for files, ThreadPoolExecutor for I/O).

## Project Structure

```
├── main.py                  # Main CLI tool (~930 lines)
├── requirements.txt         # Python dependencies
├── labels/                  # Manual labels for training (JSON)
│   └── 001.json             # Labeled timestamps for 001.mp3
├── tefisra_sanahi_truck/    # Source audio files (001.mp3 – 240.mp3)
├── model.joblib             # Trained model (generated)
└── output/                  # Generated output (generated)
    ├── segments/            # Prediction JSONs per file
    ├── quran/               # Exported Quran segments
    └── tafsir/              # Exported Tafsir segments
```

## Setup

**Requirements:** Python 3.10+, ffmpeg

```bash
# Install ffmpeg (macOS)
brew install ffmpeg

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Label Training Data

Place your audio files in `tefisra_sanahi_truck/`. Use the `listen` command to play audio with a live timestamp display, then create label files in `labels/`.

```bash
python main.py listen 001.mp3
```

Label files are JSON arrays with start/end times in seconds and a type field:

```json
[
  {"start": 0.00, "end": 3.45, "type": "quran"},
  {"start": 3.45, "end": 81.30, "type": "tafsir"},
  ...
]
```

### 2. Train the Model

```bash
python main.py train
```

Trains a RandomForest classifier on all labeled files in `labels/` and saves the model to `model.joblib`. Reports cross-validation accuracy.

### 3. Preview Predictions

Check predictions before committing to export:

```bash
# Preview a single file
python main.py preview --file 001.mp3

# Preview all files
python main.py preview --all

# Resume from a specific file
python main.py preview --all --start-after 100
```

### 4. Split Audio Files

Export Quran and Tafsir segments as separate MP3 files:

```bash
# Split a single file
python main.py split --file 001.mp3

# Split all files
python main.py split --all

# Resume from a specific file
python main.py split --all --start-after 100
```

#### Output Naming Strategies

**Folder mode** (default) — each track gets its own subfolder:

```bash
python main.py split --all --naming folder
# output/quran/001/001.mp3, 002.mp3, 003.mp3, ...
# output/tafsir/001/001.mp3, 002.mp3, 003.mp3, ...
```

**Flat mode** — all segments in one folder with track prefix:

```bash
python main.py split --all --naming flat
# output/quran/001_001.mp3, 001_002.mp3, ...
# output/tafsir/001_001.mp3, 001_002.mp3, ...
```

## Performance

- Trained on 1 labeled file (001.mp3, 64 segments) → **98.3% cross-validation accuracy**
- Average prediction confidence: **97.6%**
- Uses all available CPU cores for parallel processing
- MP3 export at 128kbps

## Dependencies

| Package | Purpose |
|---------|---------|
| librosa | Audio feature extraction |
| pydub | MP3 slicing and export |
| scikit-learn | RandomForest classifier |
| numpy | Numerical computation |
| soundfile | Audio I/O |
| joblib | Model persistence & parallelism |
| tqdm | Progress bars |
