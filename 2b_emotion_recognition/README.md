# Assignment 2b: Multimodal emotion recognition on TESS

Three systems recognize 7 emotions from **speech only**, **text only** and **speech + text**. Each is built from the blocks in the assignment figure: preprocessing → feature extraction → temporal / contextual modelling → (fusion) → classifier.

Every system has a main version, built on frozen pretrained encoders, and a from-scratch baseline. All of them are evaluated on three test conditions with 3 seeds each.

**Report:** part B of [`Report/Report.pdf`](../Report/Report.pdf) at the repository root (source: `Report/Report_2b.md`)
**All numbers:** [`Results/RESULTS.md`](Results/RESULTS.md)

## Architecture

| Block | Speech pipeline | Text pipeline |
|---|---|---|
| Preprocessing | resample to 16 kHz mono, trim leading/trailing silence (30 dB), zero-mean / unit-variance | lowercase, strip punctuation, tokenize (BERT WordPiece, or words for the baseline) |
| Feature extraction | **WavLM-base-plus** (frozen), learned weighted sum of its 13 layers → T × 768 (50 frames/s). Baseline: 40 MFCC + Δ + ΔΔ → T × 120 (100 frames/s) | **BERT-base-uncased** (frozen), learned weighted sum of its 13 layers → tokens × 768. Baseline: word embeddings learned from scratch |
| Temporal / contextual modelling | projection to 256 → 2-layer **BiLSTM** → attention pooling → 256-d | projection to 256 → 2-layer **Transformer encoder** (4 heads) → attention pooling → 256-d |
| Fusion | **Gated multimodal unit** (GMU): a per-dimension gate z mixes the speech and text vectors. Ablation: concatenation | |
| Classifier | MLP 256 → 128 → 7 | same |

Training: AdamW (lr 1e-3, weight decay 0.01), one-cycle LR schedule over a fixed budget of 15 epochs, batch 32, label smoothing 0.05; the final weights are used. There is no early stopping: in the unseen-speaker splits the validation set is the training speaker, which reaches 100 % within a few epochs and so can't choose an epoch that transfers. Validation is only logged.

## Test conditions

| Split | Train / val / test | What it measures |
|---|---|---|
| `words` | 140 / 30 / 30 of the 200 words; both speakers in each | emotion on **words never heard in training** |
| `spk_OAF` | train + val = YAF only; test = all 1,400 OAF clips | emotion on an **unseen speaker** |
| `spk_YAF` | the reverse | same |

## Folder layout (as required)

This folder is the brief's `project/` folder (renamed so the repository can hold both parts of the assignment).

```
2b_emotion_recognition/
├── models/
│   ├── common.py              shared: data, splits, preprocessing, feature caches, blocks, training loop
│   ├── speech_pipeline/       train.py  test.py
│   ├── text_pipeline/         train.py  test.py
│   └── fusion_pipeline/       train.py  test.py
├── Results/
│   ├── RESULTS.md             all tables (accuracy, macro-F1, per class, separability, fusion, failure cases)
│   ├── accuracy_table.csv     the 3 model families x 3 test conditions, mean ± std over 3 seeds
│   ├── *.csv                  the other tables
│   ├── plots/                 confusion matrices, t-SNE of the 3 blocks of the fusion model, training curves
│   └── runs/                  per run: metrics.json, per-clip predictions.csv; embeddings .npz where used (speech unseen-speaker runs: fewshot.py; GMU blocks: §5.4)
├── checkpoints/               trained weights, seed 0 only (seeds 1–2 are not in git: 270 MB; retrain with --seed 1 / 2)
├── Report/                    Report_2b.md (the PDF is part B of ../Report/Report.pdf)
├── run_all.py                 trains + tests the whole grid (63 runs)
├── text_ablation.py           fusion models re-tested with shuffled / constant text
├── fusion_blocks.py           block outputs inside the fusion model (separability)
├── fewshot.py                 few-shot recalibration on the unseen speaker
├── prosody_baseline.py        pitch / duration features per clip + prosody-only baseline
├── make_results.py            builds Results/ from Results/runs/
├── README.md
└── requirements.txt
```

## Setup

All commands run inside this folder (`cd 2b_emotion_recognition`). The first run downloads WavLM and BERT from Hugging Face (about 1.1 GB).

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # or conda/micromamba with python=3.11
pip install -r requirements.txt

# TESS from Kaggle (pip install kaggle; needs a Kaggle API token in ~/.kaggle/)
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d tess_zip
mkdir -p data && mv "tess_zip/TESS Toronto emotional speech set data" data/TESS
rm -rf "data/TESS/TESS Toronto emotional speech set data"   # the zip has a duplicate copy nested inside
```

The data folder must hold 14 folders (`OAF_angry` … `YAF_sad`) with 2,800 WAV files in total. A different location can be given with `TESS_DIR=/path/to/TESS`.

## Run

One model:

```bash
python models/speech_pipeline/train.py --features wavlm --split words --seed 0
python models/speech_pipeline/test.py  --features wavlm --split words --seed 0

python models/text_pipeline/train.py   --features bert  --split spk_OAF
python models/text_pipeline/test.py    --features bert  --split spk_OAF

python models/fusion_pipeline/train.py --speech wavlm --text bert --fusion gmu --split spk_YAF
python models/fusion_pipeline/test.py  --speech wavlm --text bert --fusion gmu --split spk_YAF
```

Options:

* `--features mfcc` (speech) / `--features scratch` (text): the from-scratch baselines.
* `--speech mfcc --text scratch --fusion gmu`: the from-scratch fusion baseline.
* `--fusion concat`: the concatenation ablation.
* `--epochs N`: training budget (default 15).

The first run of each feature type extracts the features and caches them in `cache/` (WavLM: 5.3 GB of disk, about 4–10 min on CPU or GPU). The cache is not in the repository, so the first `test.py` of a WavLM model on a fresh clone extracts it first. WavLM features are kept on the GPU during training: allow about 6 GB of GPU memory.

Checkpoints are saved on the CPU, so `test.py` works on any device (`DEVICE=cpu`, `cuda` or `mps`).

Everything: the 63 runs (7 variants × 3 splits × 3 seeds), the extra analyses, then all tables and plots:

```bash
python run_all.py --quick    # optional smoke test: 1 split, 1 seed, 2 epochs
python run_all.py            # 63 runs; about 20 min per split on a Kaggle T4 GPU, several hours on an Apple M4; DEVICE=cpu or cuda to override
python text_ablation.py      # fusion models re-tested with the text shuffled / held constant (needs all 3 seeds' checkpoints)
python fusion_blocks.py      # speech / text / fusion block outputs inside the GMU model (separability, text-term variance)
python fewshot.py            # unseen speaker: classifier refit on k labelled clips per emotion
python prosody_baseline.py   # pitch / duration features per clip (used in report §5.1) + a prosody-only baseline
python make_results.py       # Results/RESULTS.md, csvs and plots
```

Note: `test.py` (over)writes `Results/runs/<run>_*`. With only the committed seed-0 checkpoints, `test.py` reproduces every seed-0 run (checked on a fresh clone: identical metrics); the seed-1/2 numbers, `text_ablation.py` and `fusion_blocks.py` need those seeds retrained first.

`run_all.py --splits words` / `--seeds 1 2` run only some splits or seeds, so they can run in parallel processes or on different machines.

## Notes

* **The text carries no emotion information.** Every TESS transcript is "Say the word ___", and the same 200 words are recorded in all 7 emotions. So the text-only model is at chance by construction. This is the expected result, not a bug, and the report discusses what it means for fusion.
* **One file is misnamed.** `OAF_neutral/OA_bite_neutral.wav` is missing the "F", so the speaker is taken from the folder name, not the file name.
* **The sample rates differ.** One file is 96 kHz (all others 24.4 kHz); everything is resampled to 16 kHz.
