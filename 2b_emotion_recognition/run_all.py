"""Train + test every model variant on every split and seed, in one process (features load once).

python run_all.py            # full grid: 3 splits x 7 model variants x 3 seeds = 63 runs
python run_all.py --quick    # probe: words split, seed 0, 2 epochs
python run_all.py --splits words spk_OAF    # only some splits (to run splits in parallel processes)
python run_all.py --seeds 1 2               # only some seeds
Each run is the same as calling the pipeline's train.py then test.py with these arguments.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = [
    ("speech_pipeline", ["--features", "wavlm"]),
    ("speech_pipeline", ["--features", "mfcc"]),
    ("text_pipeline", ["--features", "bert"]),
    ("text_pipeline", ["--features", "scratch"]),
    ("fusion_pipeline", ["--speech", "wavlm", "--text", "bert", "--fusion", "gmu"]),
    ("fusion_pipeline", ["--speech", "wavlm", "--text", "bert", "--fusion", "concat"]),
    ("fusion_pipeline", ["--speech", "mfcc", "--text", "scratch", "--fusion", "gmu"]),
]


def run(pipeline, script, args):
    sys.modules.pop("train", None)  # each pipeline has its own train.py; don't reuse another's
    sys.path.insert(0, str(ROOT / "models" / pipeline))
    sys.argv = [script] + args
    try:
        runpy.run_path(str(ROOT / "models" / pipeline / script), run_name="__main__")
    finally:
        sys.path.pop(0)


def arg_list(flag):
    rest = sys.argv[sys.argv.index(flag) + 1:]
    return rest[:next((i for i, a in enumerate(rest) if a.startswith("--")), len(rest))]


def main():
    quick = "--quick" in sys.argv
    splits, seeds = (["words"], [0]) if quick else (["words", "spk_OAF", "spk_YAF"], [0, 1, 2])
    if "--splits" in sys.argv:
        splits = arg_list("--splits")
    if "--seeds" in sys.argv:
        seeds = [int(a) for a in arg_list("--seeds")]
    for split in splits:
        for seed in seeds:
            for pipeline, args in VARIANTS:
                a = args + ["--split", split, "--seed", str(seed)]
                print(f"\n### {pipeline} {' '.join(a)}", flush=True)
                run(pipeline, "train.py", a + (["--epochs", "2"] if quick else []))
                run(pipeline, "test.py", a)


if __name__ == "__main__":
    main()
