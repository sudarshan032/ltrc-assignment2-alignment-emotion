#!/usr/bin/env bash
# Full forced-alignment pipeline: data prep -> dictionaries -> 6 alignment runs (+ scratch with a wide beam) -> analysis -> Praat figures.
# Needs: an env with Montreal Forced Aligner 3.x (see README), and Praat on PATH for the figures.
# Usage: bash run_pipeline.sh            (MFA env active)   or
#        MFA="micromamba run -n mfa" bash run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
MFA=${MFA:-}            # optional prefix to run commands inside the MFA env
J=2                     # only 2 speakers, so more jobs don't help
DICTS=$(${MFA} python -c "from montreal_forced_aligner.config import get_temporary_directory as g; print(g())")/pretrained_models/dictionary

echo "== 1. pretrained models"
for m in acoustic:english_us_arpa dictionary:english_us_arpa g2p:english_us_arpa \
         acoustic:english_mfa dictionary:english_us_mfa g2p:english_us_mfa; do
  ${MFA} mfa model download "${m%%:*}" "${m#*:}"
done

echo "== 2. corpus (SPHERE -> wav, transcript normalization)"
rm -rf corpus && ${MFA} python scripts/prepare_corpus.py

echo "== 3. dictionaries: lexicon words + G2P for out-of-vocabulary words"
mkdir -p dictionaries models
# own G2P model, trained on the english_us_arpa lexicon (~55 min; skipped if models/own_g2p_arpa.zip exists)
[ -f models/own_g2p_arpa.zip ] || ${MFA} mfa train_g2p --clean -j 8 "$DICTS/english_us_arpa.dict" models/own_g2p_arpa.zip
${MFA} mfa g2p corpus models/own_g2p_arpa.zip dictionaries/oov_arpa_g2p.dict --dictionary_path english_us_arpa --num_pronunciations 1 --clean -j $J
# for comparison: MFA's pretrained G2P on the same words
${MFA} mfa g2p corpus english_us_arpa dictionaries/oov_arpa_pretrained_g2p.dict --dictionary_path english_us_arpa --num_pronunciations 1 --clean -j $J
${MFA} mfa g2p corpus english_us_mfa  dictionaries/oov_mfa_g2p.dict  --dictionary_path english_us_mfa  --num_pronunciations 1 --clean -j $J
${MFA} python scripts/build_dictionaries.py "$DICTS/english_us_arpa.dict" dictionaries/oov_arpa_g2p.dict dictionaries/custom_arpa.dict
${MFA} python scripts/build_dictionaries.py "$DICTS/english_us_mfa.dict"  dictionaries/oov_mfa_g2p.dict  dictionaries/custom_mfa.dict

echo "== 4. alignment runs"
rm -rf outputs
# a) pretrained ARPAbet model (main run)
${MFA} mfa align corpus dictionaries/custom_arpa.dict english_us_arpa outputs/arpa --clean -j $J
# b) same + MFA boundary fine-tuning
${MFA} mfa align corpus dictionaries/custom_arpa.dict english_us_arpa outputs/arpa_finetune --fine_tune --clean -j $J
# c) pretrained model adapted (MAP) to these 6 files, then used to align
${MFA} mfa adapt corpus dictionaries/custom_arpa.dict english_us_arpa models/english_us_arpa_adapted.zip --clean -j $J
${MFA} mfa align corpus dictionaries/custom_arpa.dict models/english_us_arpa_adapted.zip outputs/arpa_adapted --clean -j $J
# d) pretrained IPA model (english_mfa) with the US IPA dictionary
${MFA} mfa align corpus dictionaries/custom_mfa.dict english_mfa outputs/mfa_ipa --clean -j $J
# e) acoustic model trained from scratch on only these 6 files (~1.6 min of speech)
${MFA} mfa train corpus dictionaries/custom_arpa.dict models/scratch_arpa.zip --output_directory outputs/scratch --clean -j $J

# e2) the scratch model again with a 10x wider beam: is its failure the model or the search?
${MFA} mfa align corpus dictionaries/custom_arpa.dict models/scratch_arpa.zip outputs/scratch_widebeam --config_path config/widebeam.yaml --clean -j $J
# f) main model + reduced pronunciations (no /t/ after /n/): can the aligner skip unreleased /t/?
${MFA} python scripts/add_reduced_variants.py dictionaries/custom_arpa.dict dictionaries/custom_arpa_reduced.dict
${MFA} mfa align corpus dictionaries/custom_arpa_reduced.dict english_us_arpa outputs/arpa_reduced --clean -j $J

echo "== 5. analysis tables"
${MFA} python scripts/analyze.py
${MFA} python scripts/compare_reduced.py
[ -n "$(ls manual_labels/*.TextGrid 2>/dev/null)" ] && ${MFA} python scripts/evaluate_manual.py

echo "== 6. Praat figures + vowel formants"
PRAAT=${PRAAT:-$(command -v praat || ls /Applications/Praat.app/Contents/MacOS/Praat 2>/dev/null || true)}
if [ -n "$PRAAT" ]; then
  P=$PWD
  draw() {
    local tg="$P/outputs/$3/$1/$2.TextGrid"
    if [ -f "$tg" ]; then "$PRAAT" --run scripts/draw_alignment.praat "$P/corpus/$1/$2.wav" "$tg" "$4" "$5" "$P/figures/praat_$6.png" "$7"
    else echo "no TextGrid for $2 in run $3 (MFA could not align it): skipping figure"; fi
  }
  draw isle_sess0131 ISLE_SESS0131_BLOCKD02_01_sprt1 arpa    0.3 2.4 isle01_arpa    "ISLE 01 - I SAID WHITE NOT BAIT (english_us_arpa)"
  draw f2b F2BJRLP2 arpa 6.4  8.2  f2b2_appointments_arpa "F2BJRLP2 'judicial appointments' (english_us_arpa)"
  { printf "word\tphone\tstart\tend\tF1_Hz\tF2_Hz\n"
    for u in 01 02 03; do
      "$PRAAT" --run scripts/vowel_formants.praat "$P/corpus/isle_sess0131/ISLE_SESS0131_BLOCKD02_${u}_sprt1.wav" \
                                               "$P/outputs/arpa/isle_sess0131/ISLE_SESS0131_BLOCKD02_${u}_sprt1.TextGrid"
    done; } > results/isle_vowel_formants.tsv
else
  echo "Praat not found (set PRAAT=/path/to/praat): skipping Praat figures"
fi
echo "done: TextGrids in outputs/, tables in results/, figures in figures/"
