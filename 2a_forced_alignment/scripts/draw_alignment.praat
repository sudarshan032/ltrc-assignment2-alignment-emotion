# Draw waveform + spectrogram + TextGrid (words, phones) for a time window, save as PNG.
# Usage: praat --run draw_alignment.praat <wav> <TextGrid> <tmin> <tmax> <out.png> <title>
form Draw alignment
    sentence wav
    sentence textgrid
    real tmin 0
    real tmax 0
    sentence out
    sentence title
endform

sound = Read from file: wav$
tg = Read from file: textgrid$
selectObject: sound
spec = To Spectrogram: 0.005, 5000, 0.002, 20, "Gaussian"

Erase all
Font size: 10
Line width: 1

# waveform
Select outer viewport: 0, 10, 0, 1.6
selectObject: sound
Draw: tmin, tmax, 0, 0, "no", "Curve"
Draw inner box
Text top: "no", replace$(title$, "_", "\_ ", 0)

# spectrogram
Select outer viewport: 0, 10, 1.4, 4.86
selectObject: spec
Paint: tmin, tmax, 0, 5000, 100, "yes", 50, 6, 0, "no"
Draw inner box
Marks left every: 1, 1000, "yes", "yes", "no"
Text left: "yes", "Hz"

# TextGrid with boundaries extended over the spectrogram
Select outer viewport: 0, 10, 1.4, 6.2
selectObject: tg
Draw: tmin, tmax, "yes", "yes", "yes"

Select outer viewport: 0, 10, 0, 6.2
Save as 300-dpi PNG file: out$
