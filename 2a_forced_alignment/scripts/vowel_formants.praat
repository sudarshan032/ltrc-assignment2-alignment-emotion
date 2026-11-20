# Print F1/F2 at the midpoint of every stressed vowel (label ending in 1) in a TextGrid.
# Usage: praat --run vowel_formants.praat <wav> <TextGrid>
form Vowel formants
    sentence wav
    sentence textgrid
endform
sound = Read from file: wav$
tg = Read from file: textgrid$
selectObject: sound
formant = To Formant (burg): 0, 5, 5500, 0.025, 50
selectObject: tg
nPh = Get number of intervals: 2
for i to nPh
    selectObject: tg
    lab$ = Get label of interval: 2, i
    if right$(lab$, 1) = "1"
        t0 = Get start time of interval: 2, i
        t1 = Get end time of interval: 2, i
        w = Get interval at time: 1, (t0 + t1) / 2
        word$ = Get label of interval: 1, w
        selectObject: formant
        f1 = Get value at time: 1, (t0 + t1) / 2, "hertz", "linear"
        f2 = Get value at time: 2, (t0 + t1) / 2, "hertz", "linear"
        appendInfoLine: word$, tab$, lab$, tab$, fixed$(t0, 3), tab$, fixed$(t1, 3), tab$, round(f1), tab$, round(f2)
    endif
endfor
