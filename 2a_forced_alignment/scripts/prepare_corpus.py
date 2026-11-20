"""Turn the provided wav/ + transcripts/ folders into an MFA corpus.

corpus/<speaker>/<utt>.wav + <utt>.lab   (MFA's required layout)

* The three BU Radio files (F2BJRLP*.wav) are NIST SPHERE files with a .wav name.
  Their header is read and the raw 16-bit PCM is rewritten as a normal RIFF wav.
* Transcripts are joined onto one line and lightly normalized (see normalize()).
"""
import re
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_WAV, SRC_TXT = ROOT / "data/wav", ROOT / "data/transcripts"
OUT = ROOT / "corpus"


def read_audio(path):
    """Return (pcm_bytes, sample_rate). Handles RIFF and NIST SPHERE."""
    raw = path.read_bytes()
    if raw[:4] == b"RIFF":
        with wave.open(str(path)) as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            return w.readframes(w.getnframes()), w.getframerate()
    assert raw[:7] == b"NIST_1A", f"unknown audio format: {path}"
    head_len = int(raw[8:16])
    head = raw[:head_len].decode("ascii", "ignore")
    field = lambda k: re.search(rf"^{k} -\w+ (\S+)", head, re.M).group(1)
    assert field("sample_n_bytes") == "2" and field("channel_count") == "1"
    assert field("sample_coding") == "pcm" and field("sample_byte_format") == "01"  # little-endian
    pcm = raw[head_len:]
    assert len(pcm) == 2 * int(field("sample_count"))
    return pcm, int(field("sample_rate"))


def normalize(text):
    """Keep the words as spoken; only fix what the tokenizer would get wrong."""
    text = " ".join(text.split())                      # join lines, collapse spaces
    text = re.sub(r"(\w)- (\w)", r"\1 \2", text)       # "seventy- six" -> "seventy six"
    # acronyms read letter by letter -> separate letters (in the dictionary)
    text = text.replace("S.J.C.'s", "S J C's").replace("WBUR's", "W B U R's")
    return text


def speaker_of(utt):
    return "f2b" if utt.startswith("F2B") else "isle_sess0131"


def main():
    OUT.mkdir(exist_ok=True)
    for wav in sorted(SRC_WAV.glob("*.wav")):
        utt = wav.stem
        txt = next(SRC_TXT.glob(f"{utt}.[tT][xX][tT]"))
        pcm, sr = read_audio(wav)
        d = OUT / speaker_of(utt)
        d.mkdir(exist_ok=True)
        with wave.open(str(d / f"{utt}.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(pcm)
        lab = normalize(txt.read_text(encoding="utf-8", errors="replace"))
        (d / f"{utt}.lab").write_text(lab + "\n")
        print(f"{utt:34s} {len(pcm) / 2 / sr:6.2f}s  {lab[:70]}")


if __name__ == "__main__":
    sys.exit(main())
