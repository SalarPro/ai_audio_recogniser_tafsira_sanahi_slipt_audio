## Plan: Audio Segmenter — Quran vs Tafsir Splitter

**TL;DR**: Build a Python CLI tool that uses acoustic feature extraction (MFCCs, spectral features) + a trained classifier to detect Quran recitation vs Kurdish Tafsir speech in each of the 240 MP3 files. The user manually labels 3–5 files with timestamps; the tool trains on those, then processes all files — splitting them into `quran/` and `tafsir/` output folders. The melodic Quran + different speaker makes this highly separable with simple ML — no need for heavy deep learning.

**Steps**

1. **Create project structure** at the workspace root:
   - `main.py` — single entry-point script
   - `labels/` — folder for manual timestamp labels (JSON files)
   - `output/quran/` — Quran segments output
   - `output/tafsir/` — Tafsir segments output
   - `requirements.txt` — dependencies

2. **Dependencies** (`requirements.txt`):
   - `librosa` — audio feature extraction (MFCCs, spectral centroid, chroma, zero-crossing rate)
   - `pydub` — audio slicing and MP3 export
   - `scikit-learn` — SVM/Random Forest classifier
   - `numpy` — numerical operations
   - `soundfile` — audio I/O backend for librosa
   - `ffmpeg` — required system dependency for pydub MP3 handling

3. **Manual labeling format** — create a simple JSON schema for the user to label 3–5 files. Example `labels/001.json`:
   ```json
   [
     {"start": 0.0, "end": 150.5, "type": "quran"},
     {"start": 150.5, "end": 480.0, "type": "tafsir"},
     {"start": 480.0, "end": 620.3, "type": "quran"},
     ...
   ]
   ```
   Add a **helper command** (`python main.py listen 001.mp3`) that plays the audio and prints the current timestamp so labeling is easier — the user can note down switch points.

4. **Feature extraction pipeline** — for each audio file, process in overlapping windows (e.g., 3-second windows, 1-second hop):
   - 13 MFCCs (mean + std = 26 features) — captures timbre/vocal quality
   - Spectral centroid (mean + std) — melodic recitation has different spectral center than speech
   - Spectral rolloff — energy distribution differences
   - Zero-crossing rate — speech vs. chanting distinction
   - Chroma features (12 bins, mean) — musical/melodic content
   - RMS energy — volume patterns
   - Total: ~45 features per window

5. **Train classifier** (`python main.py train`):
   - Load labeled JSON files from `labels/`
   - Extract features for each labeled segment from the corresponding MP3
   - Train a `RandomForestClassifier` (robust, fast, handles feature variety well)
   - Cross-validate with leave-one-file-out to report accuracy
   - Save trained model with `joblib`
   - Print accuracy report so user can judge if more labels are needed

6. **Predict & segment** (`python main.py process [--file 001.mp3 | --all]`):
   - Load trained model
   - For each MP3: extract features in sliding windows → predict each window → get raw label sequence
   - **Post-processing / smoothing**:
     - Apply median filter to remove jitter (isolated 1–2 second misclassifications)
     - Merge consecutive segments of same type
     - Enforce minimum segment duration (e.g., 5 seconds) — a real Quran or Tafsir segment won't be just 2 seconds
     - Snap boundaries to silence/low-energy points (using `librosa.effects.split`) for clean cuts
   - Output a JSON with detected segments per file for review

7. **Preview mode** (`python main.py preview 001.mp3`):
   - Show predicted segments with timestamps before cutting
   - User can visually verify before committing to split all 240 files
   - Prints something like:
     ```
     /tefisra_sanahi_truck/001.mp3 (total: 23:56:17 (mm:ss:ms))
       [QURAN]  00:00:00 - 00:03:45
       [TAFSIR] 00:03:72 - 01:21:37

       [QURAN]  01:21:37 - 01:26:57
       [TAFSIR] 01:26:57 - 02:06:02

       [QURAN] 02:06:02 - 02:11:42
       [TAFSIR] 02:11:42 - 02:37:65
       
       [QURAN] 02:37:65 - 02:41:85
       [TAFSIR] 02:41:85 - 02:54:49
       
       [QURAN] 02:54:49 - 02:58:41
       [TAFSIR] 02:58:41 - 03:18:33
       
       [QURAN] 03:18:33 - 03:23:24
       [TAFSIR] 03:23:24 - 04:02:87
       
       [QURAN] 04:02:87 - 04:24:49
       [TAFSIR] 04:24:49 - 05:32:57
       
       [QURAN] 05:32:57 - 05:46:42
       [TAFSIR] 05:46:42 - 06:25:22
       
       [QURAN] 06:25:22 - 06:32:13
       [TAFSIR] 06:32:13 - 06:56:93
       
       [QURAN] 06:56:93 - 07:07:80
       [TAFSIR] 07:07:80 - 08:12:79
       
       [QURAN] 08:12:79 - 08:27:45
       [TAFSIR] 08:27:45 - 09:15:51
       
       [QURAN] 09:15:51 - 09:27:05
       [TAFSIR] 09:27:05 - 09:46:32
       
       [QURAN] 09:46:32 - 10:00:10
       [TAFSIR] 10:00:10 - 10:20:00
       
       [QURAN] 10:20:00 - 10:35:55
       [TAFSIR] 10:35:55 - 10:55:83
       
       [QURAN] 10:55:83 - 11:07:50
       [TAFSIR] 11:07:50 - 11:25:12
       
       [QURAN] 11:25:12 - 11:37:43
       [TAFSIR] 11:37:43 - 12:02:55
       
       [QURAN] 12:02:55 - 12:17:83
       [TAFSIR] 12:17:83 - 12:35:21
       
       [QURAN] 12:35:21 - 12:45:86
       [TAFSIR] 12:45:86 - 13:06:82
       
       [QURAN] 13:06:82 - 13:15:59
       [TAFSIR] 13:15:59 - 13:25:39
       
       [QURAN] 13:25:39 - 13:51:31
       [TAFSIR] 13:51:31 - 14:19:08
       
       [QURAN] 14:19:08 - 14:38:28
       [TAFSIR] 14:38:28 - 15:01:99
       
       [QURAN] 15:01:99 - 15:10:18
       [TAFSIR] 15:10:18 - 15:22:07
       
       [QURAN] 15:22:07 - 15:34:89
       [TAFSIR] 15:34:89 - 15:49:72
       
       [QURAN] 15:49:72 - 16:11:89
       [TAFSIR] 16:11:89 - 16:51:56
       
       [QURAN] 16:51:56 - 16:59:17
       [TAFSIR] 16:59:17 - 17:15:83

       [QURAN] 17:15:83 - 17:40:75
       [TAFSIR] 17:40:75 - 18:16:07

       [QURAN] 18:16:07 - 18:47:52
       [TAFSIR] 18:47:52 - 19:20:04

       [QURAN] 19:20:04 - 19:33:96
       [TAFSIR] 19:33:96 - 19:58:67

       [QURAN] 19:58:67 - 20:27:68
       [TAFSIR] 20:27:68 - 20:56:85

       [QURAN] 20:56:85 - 21:22:97
       [TAFSIR] 21:22:97 - 21:49:60

       [QURAN] 21:49:60 - 22:05:12
       [TAFSIR] 22:05:12 - 22:26:00

       [QURAN] 22:26:00 - 23:03:01
       [TAFSIR] 23:03:01 - 23:56:17 
       ---[the end]---
     ```

8. **Split & export** (`python main.py split [--file 001.mp3 | --all]`):
   - Use `pydub` to slice the MP3 at detected boundaries
   - Save Quran segments to `output/quran/001_001.mp3`, `output/quran/001_002.mp3`, ...
   - Save Tafsir segments to `output/tafsir/001_001.mp3`, `output/tafsir/001_002.mp3`, ...
   - Naming: `{originalFileName}_{segmentIndex}.mp3`
   - Preserve audio quality (avoid re-encoding if possible, or use high bitrate)

9. **Progress & logging**:
   - Show progress bar (`tqdm`) for batch processing of 240 files
   - Log any files where confidence is low so user can review those manually

**Verification**
- After labeling 3–5 files, run `python main.py train` — expect >95% accuracy given the acoustic differences
- Run `python main.py preview 001.mp3` on a labeled file to compare predictions vs manual labels
- Run `python main.py preview` on an unlabeled file and manually spot-check
- If accuracy is insufficient, label 1–2 more files and retrain
- Final: `python main.py split --all` to process all 240 files

**Decisions**
- **RandomForest over deep learning**: Melodic Quran vs conversational Kurdish is acoustically very distinct — simple ML with good features will work and is fast to train on 3–5 labeled files. No GPU needed.
- **3-second windows**: Long enough to capture melodic patterns, short enough for decent boundary resolution. Boundaries are then refined by snapping to silence.
- **Single `main.py` script**: Since this is a one-time-use terminal tool, keeping it in one file is simplest.
- **JSON labels**: Easy to hand-write, easy to parse. The listen helper makes labeling faster.
