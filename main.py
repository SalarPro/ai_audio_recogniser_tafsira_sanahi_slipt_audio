#!/usr/bin/env python3
"""
Quran vs Tafsir Audio Splitter
==============================
CLI tool to separate Quran recitation (Arabic, melodic) from Tafsir explanation
(Kurdish, conversational speech) in audio recordings.

Usage:
    python main.py listen <file>          - Play audio with live timestamp display
    python main.py train                  - Train classifier from labeled data
    python main.py preview <file>         - Preview predicted segments for one file
    python main.py preview --all          - Preview predicted segments for all files
    python main.py split --file <file>    - Split one file into quran/tafsir segments
    python main.py split --all            - Split all files into quran/tafsir segments
"""

import os
import sys
import json
import glob
import argparse
import warnings
import subprocess
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "tefisra_sanahi_truck"
LABELS_DIR = BASE_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "output"
QURAN_OUTPUT = OUTPUT_DIR / "quran"
TAFSIR_OUTPUT = OUTPUT_DIR / "tafsir"
SEGMENTS_DIR = OUTPUT_DIR / "segments"  # JSON prediction outputs
MODEL_PATH = BASE_DIR / "model.joblib"

# Feature extraction parameters
SAMPLE_RATE = 22050
WINDOW_SEC = 3.0        # Analysis window in seconds
HOP_SEC = 1.0           # Hop between windows in seconds
N_MFCC = 13             # Number of MFCC coefficients

# Post-processing parameters
MEDIAN_FILTER_SIZE = 7   # Median filter kernel size (in windows)
MIN_SEGMENT_SEC = 3.0    # Minimum segment duration in seconds

# Export parameters
EXPORT_BITRATE = "128k"  # MP3 export bitrate

# Parallelism
N_CORES = os.cpu_count() or 4  # Use all available CPU cores
N_WORKERS_FILES = N_CORES  # Workers for parallel file processing (use all cores)


# ─── Feature Extraction ─────────────────────────────────────────────────────

def extract_features_for_window(y_window, sr):
    """Extract acoustic features from a single audio window."""
    features = []

    # MFCCs (13 coefficients × mean + std = 26 features)
    mfccs = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # Delta MFCCs (13 features) - captures temporal dynamics
    delta_mfccs = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta_mfccs, axis=1))

    # Spectral centroid (mean + std = 2 features)
    cent = librosa.feature.spectral_centroid(y=y_window, sr=sr)
    features.append(np.mean(cent))
    features.append(np.std(cent))

    # Spectral rolloff (mean + std = 2 features)
    rolloff = librosa.feature.spectral_rolloff(y=y_window, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # Spectral bandwidth (mean + std = 2 features)
    bandwidth = librosa.feature.spectral_bandwidth(y=y_window, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # Zero-crossing rate (mean + std = 2 features)
    zcr = librosa.feature.zero_crossing_rate(y=y_window)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # Chroma features (12 bins × mean = 12 features)
    chroma = librosa.feature.chroma_stft(y=y_window, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # RMS energy (mean + std = 2 features)
    rms = librosa.feature.rms(y=y_window)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Spectral contrast (7 bands × mean = 7 features)
    try:
        contrast = librosa.feature.spectral_contrast(y=y_window, sr=sr)
        features.extend(np.mean(contrast, axis=1))
    except Exception:
        features.extend([0.0] * 7)

    # Spectral flatness (mean = 1 feature) - helps distinguish tonal (melodic) from noise-like (speech)
    flatness = librosa.feature.spectral_flatness(y=y_window)
    features.append(np.mean(flatness))

    # Tonnetz (6 features) - tonal centroid features, good for melodic content
    try:
        harmonic = librosa.effects.harmonic(y_window)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        features.extend(np.mean(tonnetz, axis=1))
    except Exception:
        features.extend([0.0] * 6)

    return np.array(features, dtype=np.float64)


def _extract_single_window(args):
    """Extract features for a single window (top-level function for pickling)."""
    y_window, sr, is_silent, n_features = args
    if is_silent:
        return np.zeros(n_features)
    return extract_features_for_window(y_window, sr)


def extract_features_from_audio(y, sr, show_progress=False, parallel=True):
    """Extract features from entire audio using sliding windows.
    Uses parallel processing across all CPU cores when parallel=True.
    Set parallel=False when called from a subprocess worker to avoid nested parallelism.
    
    Returns:
        features: np.array of shape (n_windows, n_features)
        times: np.array of window center times in seconds
    """
    window_samples = int(WINDOW_SEC * sr)
    hop_samples = int(HOP_SEC * sr)
    n_windows = max(1, (len(y) - window_samples) // hop_samples + 1)

    # Pre-compute all windows and their metadata
    windows = []
    times = []
    
    # Get n_features from a dummy extraction
    n_features = len(extract_features_for_window(np.random.randn(window_samples) * 0.01, sr))

    for i in range(n_windows):
        start_sample = i * hop_samples
        end_sample = start_sample + window_samples
        if end_sample > len(y):
            end_sample = len(y)
            start_sample = max(0, end_sample - window_samples)

        y_window = y[start_sample:end_sample]
        is_silent = np.max(np.abs(y_window)) < 1e-6
        windows.append((y_window, sr, is_silent, n_features))

        center_time = (start_sample + end_sample) / 2 / sr
        times.append(center_time)

    if parallel:
        # Parallel feature extraction using all cores (for single-file mode)
        if show_progress:
            print(f"  Extracting features from {n_windows} windows using {N_CORES} cores...")

        features_list = Parallel(n_jobs=N_CORES, backend="loky", verbose=0)(
            delayed(_extract_single_window)(w) for w in tqdm(
                windows, desc="  Extracting features", unit="win", leave=False,
                disable=not show_progress
            )
        )
    else:
        # Sequential extraction (for use inside subprocess workers)
        features_list = [_extract_single_window(w) for w in windows]

    return np.array(features_list), np.array(times)


def load_audio(filepath):
    """Load audio file and return (samples, sample_rate)."""
    print(f"  Loading audio: {filepath}")
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    print(f"  Duration: {format_time(duration)} ({duration:.1f}s)")
    return y, sr


# ─── Label Loading ───────────────────────────────────────────────────────────

def load_labels(label_path):
    """Load labels from a JSON file."""
    with open(label_path, 'r') as f:
        segments = json.load(f)
    return segments


def get_labels_for_windows(segments, times):
    """Assign a label to each window based on its center time."""
    labels = []
    for t in times:
        label = None
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                label = seg["type"]
                break
        if label is None:
            # Window falls in a gap — assign nearest segment's label
            min_dist = float("inf")
            for seg in segments:
                mid = (seg["start"] + seg["end"]) / 2
                dist = abs(t - mid)
                if dist < min_dist:
                    min_dist = dist
                    label = seg["type"]
        labels.append(label)
    return np.array(labels)


# ─── Training ────────────────────────────────────────────────────────────────

def cmd_train(args):
    """Train classifier from labeled data."""
    label_files = sorted(glob.glob(str(LABELS_DIR / "*.json")))
    if not label_files:
        print("ERROR: No label files found in labels/ directory.")
        print("Please create at least one label file (e.g., labels/001.json).")
        sys.exit(1)

    print(f"Found {len(label_files)} label file(s):")
    for lf in label_files:
        print(f"  - {Path(lf).name}")

    all_features = []
    all_labels = []
    all_groups = []  # For cross-validation: which file each sample came from

    for group_id, label_path in enumerate(label_files):
        stem = Path(label_path).stem  # e.g., "001"
        audio_path = AUDIO_DIR / f"{stem}.mp3"

        if not audio_path.exists():
            print(f"WARNING: Audio file not found: {audio_path}, skipping.")
            continue

        print(f"\nProcessing {stem}.mp3...")
        segments = load_labels(label_path)
        y, sr = load_audio(str(audio_path))
        features, times = extract_features_from_audio(y, sr, show_progress=True)
        labels = get_labels_for_windows(segments, times)

        all_features.append(features)
        all_labels.append(labels)
        all_groups.append(np.full(len(labels), group_id))

        quran_count = np.sum(labels == "quran")
        tafsir_count = np.sum(labels == "tafsir")
        print(f"  Windows: {len(labels)} total ({quran_count} quran, {tafsir_count} tafsir)")

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)

    # Replace any NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nTotal training samples: {len(y)}")
    print(f"  Quran windows:  {np.sum(y == 'quran')}")
    print(f"  Tafsir windows: {np.sum(y == 'tafsir')}")

    # Build pipeline with scaler + classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    # Cross-validation if multiple files
    if len(label_files) > 1:
        print("\nCross-validation (leave-one-file-out)...")
        logo = LeaveOneGroupOut()
        scores = cross_val_score(pipeline, X, y, groups=groups, cv=logo, scoring="accuracy")
        print(f"  CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        for i, score in enumerate(scores):
            print(f"    Fold {i+1} (test on {Path(label_files[i]).stem}.mp3): {score:.4f}")
    else:
        # Single file — do k-fold CV
        print("\nCross-validation (5-fold on single file)...")
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        print(f"  CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Train final model on all data
    print("\nTraining final model on all labeled data...")
    pipeline.fit(X, y)

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print("\nDone! You can now use:")
    print("  python main.py preview <file>    - to check predictions")
    print("  python main.py split --all       - to split all files")


# ─── Prediction & Post-processing ───────────────────────────────────────────

def predict_segments(y, sr, model, show_progress=False, parallel=True):
    """Predict quran/tafsir segments for an audio signal.
    
    Returns:
        segments: list of {"start": float, "end": float, "type": str, "confidence": float}
    """
    features, times = extract_features_from_audio(y, sr, show_progress=show_progress, parallel=parallel)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Get predictions and probabilities
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    confidence = np.max(probabilities, axis=1)

    # Apply median filter for smoothing
    predictions = median_filter_labels(predictions, kernel_size=MEDIAN_FILTER_SIZE)

    # Convert window predictions to segments
    segments = windows_to_segments(predictions, times, confidence)

    # Enforce minimum segment duration
    segments = merge_short_segments(segments, min_duration=MIN_SEGMENT_SEC)

    # Snap boundaries to silence points
    segments = snap_to_silence(segments, y, sr)

    return segments


def median_filter_labels(labels, kernel_size=7):
    """Apply median filter to label sequence to remove jitter."""
    if len(labels) < kernel_size:
        return labels

    # Convert labels to numeric
    label_map = {"quran": 0, "tafsir": 1}
    reverse_map = {0: "quran", 1: "tafsir"}
    numeric = np.array([label_map.get(l, 0) for l in labels])

    # Apply median filter
    from scipy.ndimage import median_filter
    filtered = median_filter(numeric, size=kernel_size)

    return np.array([reverse_map[v] for v in filtered])


def windows_to_segments(predictions, times, confidence):
    """Convert per-window predictions to contiguous segments."""
    segments = []
    current_type = predictions[0]
    current_start = times[0] - WINDOW_SEC / 2  # Start of first window
    current_start = max(0, current_start)
    conf_scores = [confidence[0]]

    for i in range(1, len(predictions)):
        if predictions[i] != current_type:
            # End current segment
            seg_end = (times[i-1] + times[i]) / 2  # Midpoint between windows
            segments.append({
                "start": round(current_start, 2),
                "end": round(seg_end, 2),
                "type": current_type,
                "confidence": round(float(np.mean(conf_scores)), 3)
            })
            current_type = predictions[i]
            current_start = seg_end
            conf_scores = [confidence[i]]
        else:
            conf_scores.append(confidence[i])

    # Last segment
    last_end = times[-1] + WINDOW_SEC / 2
    segments.append({
        "start": round(current_start, 2),
        "end": round(last_end, 2),
        "type": current_type,
        "confidence": round(float(np.mean(conf_scores)), 3)
    })

    return segments


def merge_short_segments(segments, min_duration=3.0):
    """Merge segments shorter than min_duration into their neighbors."""
    if len(segments) <= 1:
        return segments

    merged = True
    while merged:
        merged = False
        new_segments = []
        i = 0
        while i < len(segments):
            seg = segments[i]
            duration = seg["end"] - seg["start"]

            if duration < min_duration and len(segments) > 1:
                merged = True
                # Merge with the neighbor that has the same type, or the longer one
                if i == 0:
                    # Merge with next
                    segments[i+1]["start"] = seg["start"]
                elif i == len(segments) - 1:
                    # Merge with previous
                    new_segments[-1]["end"] = seg["end"]
                else:
                    # Merge with whichever neighbor has the same type, or the longer one
                    prev_same = new_segments[-1]["type"] == seg["type"] if new_segments else False
                    next_same = segments[i+1]["type"] == seg["type"]

                    if prev_same:
                        new_segments[-1]["end"] = seg["end"]
                    elif next_same:
                        segments[i+1]["start"] = seg["start"]
                    else:
                        # Neither neighbor matches — merge with longer neighbor
                        prev_dur = new_segments[-1]["end"] - new_segments[-1]["start"] if new_segments else 0
                        next_dur = segments[i+1]["end"] - segments[i+1]["start"]
                        if prev_dur >= next_dur and new_segments:
                            new_segments[-1]["end"] = seg["end"]
                        else:
                            segments[i+1]["start"] = seg["start"]
                i += 1
            else:
                new_segments.append(seg)
                i += 1

        segments = new_segments

    return segments


def snap_to_silence(segments, y, sr, search_window=0.5):
    """Snap segment boundaries to nearest silence/low-energy point."""
    for seg in segments:
        for key in ["start", "end"]:
            time_sec = seg[key]
            # Search in a small window around the boundary
            search_start = max(0, time_sec - search_window)
            search_end = min(len(y) / sr, time_sec + search_window)

            start_sample = int(search_start * sr)
            end_sample = int(search_end * sr)

            if end_sample <= start_sample:
                continue

            chunk = y[start_sample:end_sample]

            # Compute short-time energy
            frame_length = int(0.02 * sr)  # 20ms frames
            hop_length = int(0.01 * sr)    # 10ms hop

            if len(chunk) < frame_length:
                continue

            energies = []
            for j in range(0, len(chunk) - frame_length, hop_length):
                frame = chunk[j:j + frame_length]
                energies.append(np.sum(frame ** 2))

            if not energies:
                continue

            # Find the minimum energy point
            min_idx = np.argmin(energies)
            min_time = search_start + min_idx * hop_length / sr
            seg[key] = round(min_time, 3)

    # Ensure no overlap and segments are properly ordered
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i-1]["end"]:
            midpoint = (segments[i]["start"] + segments[i-1]["end"]) / 2
            segments[i-1]["end"] = round(midpoint, 3)
            segments[i]["start"] = round(midpoint, 3)

    return segments


# ─── Preview ─────────────────────────────────────────────────────────────────

def _process_single_file_preview(audio_path_str, model_path_str, base_dir_str, segments_dir_str):
    """Process a single file for preview (top-level function for multiprocessing)."""
    audio_path = Path(audio_path_str)
    base_dir = Path(base_dir_str)
    segments_dir = Path(segments_dir_str)
    model = joblib.load(model_path_str)

    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    total_duration = len(y) / sr
    segments = predict_segments(y, sr, model, show_progress=False, parallel=False)

    # Save segments JSON
    segments_dir.mkdir(parents=True, exist_ok=True)
    seg_path = segments_dir / f"{audio_path.stem}.json"
    with open(seg_path, 'w') as f:
        json.dump(segments, f, indent=2)

    # Compute stats
    quran_time = sum(s["end"] - s["start"] for s in segments if s["type"] == "quran")
    tafsir_time = sum(s["end"] - s["start"] for s in segments if s["type"] == "tafsir")
    avg_conf = float(np.mean([s["confidence"] for s in segments]))
    n_quran = len([s for s in segments if s["type"] == "quran"])
    n_tafsir = len([s for s in segments if s["type"] == "tafsir"])
    low_conf = [s for s in segments if s["confidence"] < 0.7]

    return {
        "name": audio_path.relative_to(base_dir),
        "total_duration": total_duration,
        "segments": segments,
        "quran_time": quran_time,
        "tafsir_time": tafsir_time,
        "avg_conf": avg_conf,
        "n_quran": n_quran,
        "n_tafsir": n_tafsir,
        "low_conf": low_conf,
        "seg_path": str(seg_path),
    }


def cmd_preview(args):
    """Preview predicted segments for a file."""
    model = load_model()

    if args.all:
        files = sorted(AUDIO_DIR.glob("*.mp3"))
        # Apply --start-after filter
        if hasattr(args, 'start_after') and args.start_after:
            start_after = args.start_after.replace('.json', '').replace('.mp3', '')
            files = [f for f in files if f.stem > start_after]
            if not files:
                print(f"No files found after {start_after}")
                return
            print(f"Skipping files up to {start_after}, starting from {files[0].stem}.mp3")
    elif args.file:
        files = [AUDIO_DIR / args.file]
    else:
        print("ERROR: Specify --file <name> or --all")
        sys.exit(1)

    if len(files) == 1:
        # Single file — process with progress display
        audio_path = files[0]
        if not audio_path.exists():
            print(f"ERROR: File not found: {audio_path}")
            return

        print(f"\n{'='*60}")
        y, sr = load_audio(str(audio_path))
        total_duration = len(y) / sr
        segments = predict_segments(y, sr, model, show_progress=True)

        print(f"\n{audio_path.relative_to(BASE_DIR)} (total: {format_time(total_duration)})")
        print_segments(segments)

        SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
        seg_path = SEGMENTS_DIR / f"{audio_path.stem}.json"
        with open(seg_path, 'w') as f:
            json.dump(segments, f, indent=2)
        print(f"\n  Segments saved to: {seg_path}")

        quran_time = sum(s["end"] - s["start"] for s in segments if s["type"] == "quran")
        tafsir_time = sum(s["end"] - s["start"] for s in segments if s["type"] == "tafsir")
        avg_conf = np.mean([s["confidence"] for s in segments])
        low_conf = [s for s in segments if s["confidence"] < 0.7]

        print(f"\n  Stats:")
        print(f"    Quran:  {format_time(quran_time)} ({len([s for s in segments if s['type']=='quran'])} segments)")
        print(f"    Tafsir: {format_time(tafsir_time)} ({len([s for s in segments if s['type']=='tafsir'])} segments)")
        print(f"    Avg confidence: {avg_conf:.3f}")
        if low_conf:
            print(f"    ⚠ {len(low_conf)} segment(s) with low confidence (<0.7):")
            for s in low_conf:
                print(f"      [{s['type'].upper()}] {format_time(s['start'])} - {format_time(s['end'])} (conf: {s['confidence']:.3f})")
    else:
        # Multiple files — parallel processing
        print(f"\nProcessing {len(files)} files using {N_WORKERS_FILES} parallel workers...")
        print(f"(Each worker uses {N_CORES} cores for feature extraction)")

        with ProcessPoolExecutor(max_workers=N_WORKERS_FILES) as executor:
            futures = {
                executor.submit(
                    _process_single_file_preview,
                    str(audio_path), str(MODEL_PATH), str(BASE_DIR), str(SEGMENTS_DIR)
                ): audio_path
                for audio_path in files if audio_path.exists()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
                try:
                    result = future.result()
                    tqdm.write(f"\n{'='*60}")
                    tqdm.write(f"{result['name']} (total: {format_time(result['total_duration'])})")
                    for seg in result['segments']:
                        label = seg['type'].upper()
                        start = format_time(seg['start'])
                        end = format_time(seg['end'])
                        tqdm.write(f"  [{label:6s}] {start} - {end}")
                    tqdm.write(f"  Stats: {result['n_quran']} quran, {result['n_tafsir']} tafsir, conf: {result['avg_conf']:.3f}")
                    tqdm.write(f"  Saved: {result['seg_path']}")
                except Exception as e:
                    audio_path = futures[future]
                    tqdm.write(f"  ERROR processing {audio_path.name}: {e}")


def print_segments(segments):
    """Pretty-print segments list."""
    prev_type = None
    for seg in segments:
        label = seg["type"].upper()
        start = format_time(seg["start"])
        end = format_time(seg["end"])
        conf = seg.get("confidence", 1.0)

        # Add blank line between quran→tafsir pairs
        if prev_type == "tafsir" and seg["type"] == "quran":
            print()

        conf_marker = "" if conf >= 0.7 else f" (conf: {conf:.2f}) !"
        print(f"  [{label:6s}] {start} - {end}{conf_marker}")
        prev_type = seg["type"]


# ─── Split & Export ──────────────────────────────────────────────────────────

def _split_single_file(audio_path_str, model_path_str, segments_dir_str, quran_dir_str, tafsir_dir_str, export_bitrate, naming="folder"):
    """Split a single audio file into quran/tafsir segments (top-level for multiprocessing)."""
    audio_path = Path(audio_path_str)
    segments_dir = Path(segments_dir_str)
    quran_dir = Path(quran_dir_str)
    tafsir_dir = Path(tafsir_dir_str)
    stem = audio_path.stem

    # Check if segments JSON already exists
    seg_path = segments_dir / f"{stem}.json"
    if seg_path.exists():
        with open(seg_path, 'r') as f:
            segments = json.load(f)
    else:
        # Predict segments
        model = joblib.load(model_path_str)
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        segments = predict_segments(y, sr, model, show_progress=False, parallel=False)
        segments_dir.mkdir(parents=True, exist_ok=True)
        with open(seg_path, 'w') as f:
            json.dump(segments, f, indent=2)

    # Load MP3 with pydub for slicing
    audio = AudioSegment.from_mp3(str(audio_path))

    quran_idx = 0
    tafsir_idx = 0

    # Export all segments (use threads for parallel I/O within this file)
    export_tasks = []
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        chunk = audio[start_ms:end_ms]

        if seg["type"] == "quran":
            quran_idx += 1
            if naming == "folder":
                track_dir = quran_dir / stem
                track_dir.mkdir(parents=True, exist_ok=True)
                out_path = track_dir / f"{quran_idx:03d}.mp3"
            else:
                out_path = quran_dir / f"{stem}_{quran_idx:03d}.mp3"
        else:
            tafsir_idx += 1
            if naming == "folder":
                track_dir = tafsir_dir / stem
                track_dir.mkdir(parents=True, exist_ok=True)
                out_path = track_dir / f"{tafsir_idx:03d}.mp3"
            else:
                out_path = tafsir_dir / f"{stem}_{tafsir_idx:03d}.mp3"

        export_tasks.append((chunk, str(out_path), export_bitrate))

    # Parallel MP3 export using threads (I/O-bound)
    def _export_chunk(task):
        chunk, path, bitrate = task
        chunk.export(path, format="mp3", bitrate=bitrate)

    with ThreadPoolExecutor(max_workers=4) as tex:
        list(tex.map(_export_chunk, export_tasks))

    return stem, quran_idx, tafsir_idx


def cmd_split(args):
    """Split audio files into quran/tafsir segments."""
    model = load_model()

    QURAN_OUTPUT.mkdir(parents=True, exist_ok=True)
    TAFSIR_OUTPUT.mkdir(parents=True, exist_ok=True)

    if args.all:
        files = sorted(AUDIO_DIR.glob("*.mp3"))
        # Apply --start-after filter
        if hasattr(args, 'start_after') and args.start_after:
            start_after = args.start_after.replace('.json', '').replace('.mp3', '')
            files = [f for f in files if f.stem > start_after]
            if not files:
                print(f"No files found after {start_after}")
                return
            print(f"Skipping files up to {start_after}, starting from {files[0].stem}.mp3")
    elif args.file:
        files = [AUDIO_DIR / args.file]
    else:
        print("ERROR: Specify --file <name> or --all")
        sys.exit(1)

    print(f"Processing {len(files)} file(s) using {N_WORKERS_FILES} parallel workers...")
    print(f"Hardware: {N_CORES} CPU cores available")

    naming = getattr(args, 'naming', 'folder')
    print(f"Naming strategy: {naming}" + (" (output/<type>/<track>/NNN.mp3)" if naming == "folder" else " (output/<type>/<track>_NNN.mp3)"))

    if len(files) == 1:
        # Single file — just run directly
        audio_path = files[0]
        if not audio_path.exists():
            print(f"WARNING: File not found: {audio_path}")
            return
        stem, qn, tn = _split_single_file(
            str(audio_path), str(MODEL_PATH), str(SEGMENTS_DIR),
            str(QURAN_OUTPUT), str(TAFSIR_OUTPUT), EXPORT_BITRATE, naming
        )
        print(f"  {stem}.mp3: {qn} quran + {tn} tafsir segments exported")
    else:
        # Multiple files — parallel processing
        with ProcessPoolExecutor(max_workers=N_WORKERS_FILES) as executor:
            futures = {}
            for audio_path in files:
                if not audio_path.exists():
                    continue
                future = executor.submit(
                    _split_single_file,
                    str(audio_path), str(MODEL_PATH), str(SEGMENTS_DIR),
                    str(QURAN_OUTPUT), str(TAFSIR_OUTPUT), EXPORT_BITRATE, naming
                )
                futures[future] = audio_path

            for future in tqdm(as_completed(futures), total=len(futures), desc="Splitting files", unit="file"):
                try:
                    stem, qn, tn = future.result()
                    tqdm.write(f"  {stem}.mp3: {qn} quran + {tn} tafsir segments exported")
                except Exception as e:
                    audio_path = futures[future]
                    tqdm.write(f"  ERROR splitting {audio_path.name}: {e}")

    print(f"\nDone! Output saved to:")
    print(f"  Quran:  {QURAN_OUTPUT}")
    print(f"  Tafsir: {TAFSIR_OUTPUT}")


# ─── Listen Helper ───────────────────────────────────────────────────────────

def cmd_listen(args):
    """Play audio with timestamp display to help with manual labeling."""
    audio_path = AUDIO_DIR / args.file
    if not audio_path.exists():
        print(f"ERROR: File not found: {audio_path}")
        sys.exit(1)

    y, sr = load_audio(str(audio_path))
    total_duration = len(y) / sr
    print(f"\nPlaying: {audio_path.name} (total: {format_time(total_duration)})")
    print("Press Ctrl+C to stop. Note timestamps for labeling.\n")
    print("Tip: Create labels/{stem}.json with format:")
    print('  [{"start": 0.0, "end": 30.5, "type": "quran"}, ...]')
    print()

    # Try to play with afplay (macOS) or ffplay
    try:
        import time
        import threading

        def show_time():
            start = time.time()
            try:
                while True:
                    elapsed = time.time() - start
                    if elapsed > total_duration:
                        break
                    sys.stdout.write(f"\r  ⏱ {format_time(elapsed)} / {format_time(total_duration)}  ")
                    sys.stdout.flush()
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            print()

        timer_thread = threading.Thread(target=show_time, daemon=True)
        timer_thread.start()

        # Try afplay (macOS), then ffplay, then aplay
        for player in ["afplay", "ffplay -nodisp -autoexit", "aplay"]:
            cmd_parts = player.split() + [str(audio_path)]
            try:
                subprocess.run(cmd_parts, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        else:
            print("\nERROR: No audio player found (tried afplay, ffplay, aplay).")
            print("You can still label files by listening in any audio player.")

    except KeyboardInterrupt:
        print(f"\n\nStopped. Current time was shown above for labeling reference.")


# ─── Utilities ───────────────────────────────────────────────────────────────

def format_time(seconds):
    """Format seconds as mm:ss.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0
    minutes = int(seconds // 60)
    secs = seconds % 60
    whole_secs = int(secs)
    centiseconds = int((secs - whole_secs) * 100)
    return f"{minutes:02d}:{whole_secs:02d}.{centiseconds:02d}"


def load_model():
    """Load trained model or exit with error."""
    if not MODEL_PATH.exists():
        print("ERROR: No trained model found.")
        print("Run 'python main.py train' first.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quran vs Tafsir Audio Splitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py listen 001.mp3          Play audio with live timestamps
  python main.py train                   Train model from labels/ directory
  python main.py preview --file 001.mp3  Preview predictions for one file
  python main.py preview --all           Preview predictions for all files
  python main.py split --file 001.mp3    Split one file into segments
  python main.py split --all             Split all 240 files
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # listen
    p_listen = subparsers.add_parser("listen", help="Play audio with timestamp display")
    p_listen.add_argument("file", help="Audio filename (e.g., 001.mp3)")

    # train
    p_train = subparsers.add_parser("train", help="Train classifier from labeled data")

    # preview
    p_preview = subparsers.add_parser("preview", help="Preview predicted segments")
    p_preview_group = p_preview.add_mutually_exclusive_group(required=True)
    p_preview_group.add_argument("--file", help="Single file to preview (e.g., 001.mp3)")
    p_preview_group.add_argument("--all", action="store_true", help="Preview all files")
    p_preview.add_argument("--start-after", help="Skip files up to and including this one (e.g., 166.mp3 or 166)")

    # split
    p_split = subparsers.add_parser("split", help="Split audio into quran/tafsir segments")
    p_split_group = p_split.add_mutually_exclusive_group(required=True)
    p_split_group.add_argument("--file", help="Single file to split (e.g., 001.mp3)")
    p_split_group.add_argument("--all", action="store_true", help="Split all files")
    p_split.add_argument("--start-after", help="Skip files up to and including this one (e.g., 166.mp3 or 166)")
    p_split.add_argument("--naming", choices=["folder", "flat"], default="folder",
                         help="Output naming strategy: 'folder' = output/quran/001/001_001.mp3 (default), 'flat' = output/quran/001_001.mp3")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Ensure output directories exist
    QURAN_OUTPUT.mkdir(parents=True, exist_ok=True)
    TAFSIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    commands = {
        "listen": cmd_listen,
        "train": cmd_train,
        "preview": cmd_preview,
        "split": cmd_split,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
