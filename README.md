# Real-Time Drowsiness Detection Using Deep Learning (CPU-Only)

A drowsiness detector that runs on an ordinary laptop **with no GPU**. It classifies eye crops as **open (awake)** or **closed (sleepy)** with fine-tuned CNNs, compares models on accuracy and speed, and turns the frame-by-frame predictions into a real-time **drowsiness alarm** using MediaPipe face landmarks and a PERCLOS-style rule.

> Built and trained entirely on an HP 15s laptop (Intel Core i3-1115G4, 8 GB RAM, integrated graphics). Training each model took 4 to 8 minutes.

<!-- Add your demo GIF or video link here, e.g. ![demo](assets/demo.gif) -->

---

## Highlights

- **86.8% accuracy, 0.977 AUC (88.9% balanced accuracy) on 6 people the model never saw** (subject-wise test, fine-tuned ResNet18).
- **Found and fixed a leaky split.** The dataset's own train/val/test split is random per image, so the same 37 people appear in all three. On that split ResNet18 scores 96.9%, but that number is optimistic. I re-split **by person** and report that result as the headline.
- **Personalised to my own webcam:** fine-tuning on 379 of my own webcam crops raised closed-eye accuracy on a held-out part of the same recording from 76.2% to 96.8%. This is a weak test (one person, one session, 127 test crops), explained below.
- **Model comparison** of ResNet18 vs MobileNetV3-Small under both protocols, with precision, recall, F1, AUC, CPU latency, model size and training time.
- **End-to-end real-time pipeline:** webcam or video, then MediaPipe Face Mesh, eye crops, CNN, temporal logic, then alarm. EAR is shown live as a classical reference.
- Works on **live webcam or a video file**, with a guided recorder for your own test video.
- Everything runs locally; no images leave your machine.

---

## Results

### Headline: subject-wise evaluation (unseen people)

All images from the 37 subjects were pooled and re-split **by person**: 25 subjects for training, 6 for validation, 6 for testing (the subject lists are in `results/subject_wise/subject_split.json`). Models were fine-tuned on 10,000 images (5,000 per class) for 3 epochs at 64x64 grayscale, and the checkpoint with the best validation accuracy was tested.

Test set: **7,190 images from 6 people** (2,591 awake, 4,599 sleepy).

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | AUC | Latency / crop | Size | Train time |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet18** | **86.84%** | **88.92%** | 0.976 | 0.815 | **0.888** | **0.977** | ~19-22 ms | 44.8 MB | 7.9 min |
| MobileNetV3-Small | 65.10% | 63.55% | 0.745 | 0.691 | 0.717 | 0.707 | ~6-16 ms | 6.2 MB | 4.6 min |
| *Always predict "sleepy"* | *63.96%* | *50.00%* | | | | | | | |

![Subject-wise comparison](assets/comparison_subject_wise.png)

| Confusion matrix (test) | ResNet18 | MobileNetV3-Small |
|---|---|---|
| Awake predicted awake (TN) | 2,497 | 1,502 |
| Awake predicted sleepy (FP, false alarm) | 94 | 1,089 |
| Sleepy predicted awake (FN, missed) | 852 | 1,420 |
| Sleepy predicted sleepy (TP) | 3,747 | 3,179 |

![Subject-wise confusion matrices](assets/confusion_matrices_subject_wise.png)

**What this shows**

- **ResNet18 generalises to new people; MobileNetV3-Small does not (yet).** MobileNetV3-Small's accuracy (65.1%) is barely above the 64.0% you would get by always answering "sleepy", and its AUC is 0.707. Its training loss kept falling (0.30 to 0.11) while validation accuracy stayed near 79%, a sign it fits the training people rather than learning a general rule.
- **ResNet18 is cautious on new people:** only 3.6% of awake images raise a false alarm (94 of 2,591), but it misses 18.5% of sleepy images (852 of 4,599). The high AUC (0.977) means its ranking is good, so a decision threshold chosen on the validation people could improve recall. I have not done this yet.
- **Accuracy alone can mislead here** because the test people are 64% sleepy, so balanced accuracy and AUC are reported too.
- Both models peaked on validation at epoch 2 (see `results/subject_wise/history.json`).

### For comparison: the dataset's own random split (optimistic)

Same training recipe, but using the dataset's provided train/val/test folders. Test set: 16,981 images (8,591 awake, 8,390 sleepy).

| Model | Accuracy | Precision | Recall | F1 | AUC | Latency / crop | Train time |
|---|---|---|---|---|---|---|---|
| ResNet18 | 96.93% | 0.956 | 0.983 | 0.969 | 0.997 | 22.5 ms | 6.7 min |
| MobileNetV3-Small | 83.52% | 0.874 | 0.778 | 0.824 | 0.926 | 6.2 ms | 4.0 min |

Plots: `assets/comparison_random_split.png`, `assets/confusion_matrices_random_split.png`.

**Why this number is optimistic.** The dataset fork splits images randomly (`train_test_split`, `random_state=42`, 60/20/20) without looking at who is in each picture. Each of the 37 people contributes about 2,300 near-identical frames, so the test set contains people the model has already seen. The drop from 96.9% to 86.8% for ResNet18 is consistent with this. The two experiments use different test sets, so the comparison is indicative rather than exact.

### Fine-tuning on my own webcam data (Experiment C)

The dataset's images are close-up infrared eye crops, while my laptop webcam gives visible-light crops, so I recorded my own and fine-tuned the subject-wise ResNet18 on them.

- **Data:** 255 open-eye and 251 closed-eye crops (64x64 grayscale, same cropping as `realtime.py`), saved with the `o` / `c` keys in `my_data/s0/open` and `my_data/s0/closed`. Open-eye frames included times I was looking down (eyes open, so labelled open).
- **Split:** the first 75% of each class (by time) for training (379 crops), the last 25% for testing (127 crops: 64 open, 63 closed).
- **Fine-tuning:** from `models/subject_wise/resnet18.pt`, AdamW (lr 1e-4), batch 32, 8 epochs, same augmentation as before, 1.9 minutes on the laptop. The final-epoch weights were saved; the test crops were not used to choose epochs.

| Held-out webcam crops (127) | Before fine-tuning | After fine-tuning |
|---|---|---|
| Accuracy | 88.2% | **98.4%** |
| Open eyes correct (64 crops) | 100.0% | 100.0% |
| Closed eyes correct (63 crops) | 76.2% (48/63) | **96.8% (61/63)** |
| F1 (closed) | 0.865 | **0.984** |

**Read this carefully:** the improvement is in recognising **closed** eyes on my webcam. Open-eye accuracy was already 100% on this held-out part, so these numbers do **not** by themselves prove that the looking-down false alarm is fixed. The test crops come from the same recording as the training crops (neighbouring frames look alike), there are only 127 of them, test accuracy swung between epochs (74.8% to 100%), and the result applies to one person on one laptop camera. A fair check needs a second recording on another day or in other lighting, plus live testing of looking down with open eyes.

### Latency note

CPU latency varies with background load (MobileNetV3-Small measured 6.2 ms in one run and 16.2 ms in another on the same laptop). For a clean number, close other programs and run `python benchmark_latency.py --size 64`, which reports the median of 300 runs.

---

## How it works

```
Webcam / video frame
        |
        v
MediaPipe Face Mesh  --> 6 landmarks per eye  --> EAR (shown as a reference)
        |
        v
Eye crops (grayscale, 64x64)  --> CNN (ResNet18)  --> p(closed) per eye, averaged
        |
        v
Temporal logic over time
   - eyes closed continuously for >= 1.5 s            -> ALARM
   - PERCLOS (closed fraction in last 4 s) >= 0.5      -> ALARM
        |
        v
On-screen warning + beep
```

- **Classifier input:** grayscale eye image, resized to 64x64, scaled to [0, 1], repeated to 3 channels and normalised with ImageNet statistics inside the model wrapper.
- **Fine-tuning:** full fine-tuning of ImageNet-pretrained weights with a new 2-class head. AdamW (lr 3e-4, weight decay 1e-4), cosine learning-rate schedule, batch size 64, cross-entropy loss.
- **Augmentation:** random horizontal flip, brightness and contrast jitter, light Gaussian noise.
- **Alarm decision uses the CNN only.** EAR (threshold 0.20) is displayed for comparison but does not trigger the alarm.
- For video files, timing uses the **video's own timestamps**, so the 1.5 s and 4 s thresholds mean the same thing whether the video is processed fast or slowly.
- **"Sleepy" means eyes closed.** The classifier detects closed vs open eyes; drowsiness is inferred from how long and how often the eyes stay closed.

---

## Dataset

**MRL Eye Dataset** (Media Research Lab, VSB - Technical University of Ostrava): 84,898 near-infrared eye images from **37 subjects**, captured with 3 sensors under different lighting, with and without glasses and reflections. This project uses the Kaggle fork with `awake` / `sleepy` folders (sleepy = closed eyes).

- Original dataset: http://mrl.cs.vsb.cz/eyedataset
- Kaggle fork used: https://www.kaggle.com/datasets/imadeddinedjerarda/mrl-eye-dataset

```
data/
  train/  awake/  sleepy/
  val/    awake/  sleepy/
  test/   awake/  sleepy/
```

| Fork's split | Images |
|---|---|
| Train | 50,937 |
| Val | 16,980 |
| Test | 16,981 |

Subject-wise split used for the headline results (all splits pooled, then split by person):

| Split | People | Images in pool | Images used |
|---|---|---|---|
| Train | 25 | 66,449 | 10,000 (5,000 per class) |
| Val | 6 | 11,259 | 3,445 (2,000 awake + 1,445 sleepy) |
| Test | 6 | 7,190 | all 7,190 |

The dataset is not included here; please download it from the sources above and respect its license. Credit goes to the MRL dataset authors.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── train_subjectwise.py   # headline experiment: re-split by person, train, compare
├── train_folders.py       # experiment on the dataset's own train/val/test folders
├── train_compare.py       # shared model / training / evaluation code
├── realtime.py            # real-time detector (webcam or video file)
├── record_video.py        # guided 70-second test-video recorder
├── benchmark_latency.py   # clean CPU latency benchmark
├── finetune_mine.py       # fine-tune on my own webcam crops (Experiment C)
├── collect_data.py        # optional: guided, auto-labelled webcam recorder (not used for reported results)
├── finetune_sessions.py   # optional: fine-tuning with separate recording sessions (not used for reported results)
├── check_leakage.py       # optional: look for shared people / near-duplicates across splits
├── models/
│   ├── subject_wise/      # resnet18.pt, mobilenet_v3_small.pt (headline experiment)
│   ├── random_split/      # weights from the dataset's own split
│   └── webcam_finetuned/  # resnet18_mine.pt (optional; personalised to one person)
├── results/
│   ├── subject_wise/      # results.csv, history.json, subject_split.json
│   └── random_split/      # results.csv, history.json
└── assets/                # figures used in this README
```

`train_subjectwise.py` and `train_folders.py` import from `train_compare.py`, so keep them in the same folder.

---

## Setup

Requires Python 3.10 to 3.12 (Windows tested). No GPU needed.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` pins `mediapipe==0.10.14`. Newer MediaPipe releases removed the `mp.solutions` API used here.

---

## Usage

### 1. Run the demo with the included weights

```bash
# live webcam
python realtime.py --weights models/subject_wise/resnet18.pt --model resnet18 --size 64 --source 0

# or a video file
python realtime.py --weights models/subject_wise/resnet18.pt --model resnet18 --size 64 --source path/to/video.mp4
```

Press `q` to quit. The overlay shows `p(closed)`, EAR and PERCLOS; a red warning and a beep appear when the alarm triggers. Options: `--max_closed` (seconds of continuous closure, default 1.5), `--perclos` (default 0.5), `--window` (seconds, default 4). For a faster alarm try `--max_closed 1.0 --window 3`.

### 2. Record your own test video

```bash
python record_video.py
```

Follow the on-screen prompts. It saves `videos/my_test.mp4`; then run `realtime.py` with `--source videos/my_test.mp4`.

### 3. Fine-tune on your own webcam data

```bash
# record crops while running the detector: press o (eyes open) or c (eyes closed); hold the key to save many frames
python realtime.py --weights models/subject_wise/resnet18.pt --model resnet18 --size 64 --source 0
# put them in my_data/s0/open and my_data/s0/closed, delete wrong labels, then:
python finetune_mine.py --weights models/subject_wise/resnet18.pt --model resnet18 --train my_data/s0 --out mine
python realtime.py --weights mine/resnet18_mine.pt --model resnet18 --size 64 --source 0
```

Optionally add `--test my_data/s1` with a second recording for a fairer before/after. Your crops are pictures of your face region, so keep `my_data/` out of public repositories (it is in `.gitignore`).

### 4. Reproduce the results

```bash
# headline: subject-wise (unseen people)
python train_subjectwise.py --data path/to/data --out results_subjectwise --models mobilenet_v3_small resnet18 --size 64 --epochs 3 --max_per_class 5000

# dataset's own split (optimistic)
python train_folders.py --data path/to/data --out results_random --models mobilenet_v3_small resnet18 --size 64 --epochs 3 --max_per_class 5000
```

Each run writes `results.csv`, `history.json`, comparison and confusion-matrix plots, and the weights. Use `--seed 1` and `--seed 2` with `train_subjectwise.py` to try different choices of test people.

---

## Real-world test and known issues

I recorded a guided ~70-second video of myself with the laptop camera and ran the detector on it (qualitative, one person, one video):

- Eyes open: `p(closed)` stayed low and no alarm fired.
- Eyes closed for about 8 seconds: `p(closed)` rose towards 1.0 and the alarm triggered after about 1.5 seconds.
- Normal blinks did not trigger the alarm.

Further use on the live webcam showed two issues:

- **The alarm can be late.** This follows from the rule that requires about 1.5 s of continuous closure, plus per-frame processing time on a CPU. `--max_closed` and `--window` can shorten it.
- **Looking down with open eyes can trigger a false "drowsy" alarm.** Likely cause (not yet verified): the training images are close-up near-infrared eye crops, while webcam crops are different in look and angle, and looking down makes the eyelids cover part of the eye. I fine-tuned on my own webcam crops (Experiment C above); whether this removes the looking-down false alarms still needs a live check and a second recording.

None of these observations is a measured accuracy.

---

## Limitations

- **Only 6 test people and one random seed** for the headline result, so the numbers could change with a different choice of people. No confidence intervals yet.
- **MobileNetV3-Small did not generalise to new people** in this setup; it was not tuned further.
- **Domain gap to webcams.** Training data is near-infrared close-ups; the demo uses visible-light webcam crops. This is the likely cause of the looking-down false alarms.
- **Real-world check is qualitative:** one person, no frame-by-frame ground truth.
- **Limited training:** a 10,000-image subset and 3 epochs.
- **Eye state is only one drowsiness cue.** Yawning, head nodding and longer behaviour patterns are not used.
- **Alarm thresholds are heuristic:** 1.5 s, PERCLOS 0.5 and a 4 s window were not tuned on labelled data.
- **Webcam fine-tuning is personal and weakly tested:** one person, one laptop camera, 506 crops, and a test set taken from the same recording (127 crops). The fine-tuned model may not work for other people.
- **Not evaluated:** glasses, extreme lighting, other cameras.
- **Not a safety system.** This is a learning and portfolio project; do not rely on it for driving or any safety-critical use.

---

## Future work

- Record a second webcam session (another day, other lighting) and re-run the before/after comparison; check looking-down frames separately.
- Choose the decision threshold on the validation people to trade precision for recall.
- Several random subject splits (mean and standard deviation); EfficientNet-B0; higher input resolution.
- Label a test video frame by frame to measure alarm delay and false-alarm rate.
- Add yawning and head-pose cues; ONNX / int8 for speed; a small web UI.

---

## Tech stack

Python, PyTorch (CPU), torchvision, OpenCV, MediaPipe Face Mesh, scikit-learn, pandas, NumPy, Matplotlib.

## References

- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- A. Howard et al., *Searching for MobileNetV3*, ICCV 2019.
- T. Soukupova and J. Cech, *Real-Time Eye Blink Detection using Facial Landmarks*, CVWW 2016 (Eye Aspect Ratio).
- Y. Kartynnik et al., *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs*, 2019 (MediaPipe Face Mesh).
- D. Dinges and R. Grace, *PERCLOS: A Valid Psychophysiological Measure of Alertness*, FHWA, 1998.
- MRL Eye Dataset, Media Research Lab, VSB - Technical University of Ostrava.

## Author

**[Your Name]** | [LinkedIn link] | [Email]

## License

Add a license of your choice (for example MIT) before publishing. The dataset has its own license from its original authors.
