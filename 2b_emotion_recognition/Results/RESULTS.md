# Results

All numbers: test set, mean ± std over 3 seeds (0, 1, 2). Chance = 14.3 %.

Names: `words` = unseen words (both speakers); `spk_OAF` = unseen speaker, tested on OAF (trained on YAF); `spk_YAF` = tested on YAF (trained on OAF). Run files: `runs/<model>_<split>_s<seed>_*`.

Plots: [confusion](plots/confusion_unseen_speaker.png), [t-SNE unseen words](plots/tsne_unseen_words.png), [t-SNE test YAF](plots/tsne_unseen_speaker_YAF.png), [t-SNE test OAF](plots/tsne_unseen_speaker_OAF.png), [training curves](plots/training_curves.png).

## Accuracy (%)

| model                                          | Unseen words (both speakers)   | Unseen speaker: test OAF   | Unseen speaker: test YAF   |   Unseen speaker (mean) |
|:-----------------------------------------------|:-------------------------------|:---------------------------|:---------------------------|------------------------:|
| Speech: WavLM + BiLSTM                         | 100.0 ± 0.0                    | 61.8 ± 3.5                 | 66.8 ± 7.8                 |                    64.3 |
| Speech: MFCC + BiLSTM (from scratch)           | 99.9 ± 0.1                     | 39.3 ± 9.8                 | 47.0 ± 3.7                 |                    43.2 |
| Text: BERT + Transformer                       | 14.3 ± 0.0                     | 14.3 ± 0.0                 | 14.3 ± 0.0                 |                    14.3 |
| Text: word emb. + Transformer (from scratch)   | 14.3 ± 0.0                     | 14.3 ± 0.0                 | 14.3 ± 0.0                 |                    14.3 |
| Fusion: WavLM + BERT, gated (GMU)              | 100.0 ± 0.0                    | 55.7 ± 7.0                 | 66.1 ± 1.8                 |                    60.9 |
| Fusion: WavLM + BERT, concat                   | 100.0 ± 0.0                    | 56.0 ± 3.9                 | 57.2 ± 9.3                 |                    56.6 |
| Fusion: MFCC + word emb., gated (from scratch) | 99.6 ± 0.1                     | 36.1 ± 10.7                | 48.8 ± 6.8                 |                    42.5 |

## Macro-F1 (%)

| model                                          | Unseen words (both speakers)   | Unseen speaker: test OAF   | Unseen speaker: test YAF   |
|:-----------------------------------------------|:-------------------------------|:---------------------------|:---------------------------|
| Speech: WavLM + BiLSTM                         | 100.0 ± 0.0                    | 56.5 ± 3.2                 | 63.2 ± 8.3                 |
| Speech: MFCC + BiLSTM (from scratch)           | 99.9 ± 0.1                     | 32.7 ± 9.1                 | 42.8 ± 3.5                 |
| Text: BERT + Transformer                       | 3.8 ± 0.4                      | 3.6 ± 0.0                  | 3.6 ± 0.1                  |
| Text: word emb. + Transformer (from scratch)   | 3.6 ± 0.0                      | 5.4 ± 1.4                  | 4.6 ± 0.9                  |
| Fusion: WavLM + BERT, gated (GMU)              | 100.0 ± 0.0                    | 49.5 ± 9.7                 | 63.6 ± 0.2                 |
| Fusion: WavLM + BERT, concat                   | 100.0 ± 0.0                    | 50.4 ± 3.6                 | 53.6 ± 12.9                |
| Fusion: MFCC + word emb., gated (from scratch) | 99.6 ± 0.1                     | 27.2 ± 10.2                | 46.1 ± 6.6                 |

## Per-class accuracy (%) — recall of each emotion, 3 seeds pooled

| model                                | test           |   angry |   disgust |   fear |   happy |   neutral |   pleasant surprise |   sad |
|:-------------------------------------|:---------------|--------:|----------:|-------:|--------:|----------:|--------------------:|------:|
| Speech: WavLM + BiLSTM               | unseen words   |   100   |     100   |  100   |   100   |     100   |               100   | 100   |
| Speech: WavLM + BiLSTM               | unseen speaker |    34.5 |      85.8 |   99   |     3.8 |      99.2 |                39.7 |  87.9 |
| Speech: MFCC + BiLSTM (from scratch) | unseen words   |   100   |     100   |  100   |   100   |     100   |                99.4 | 100   |
| Speech: MFCC + BiLSTM (from scratch) | unseen speaker |    54.9 |      54.2 |   65.2 |     0   |      77.6 |                 3.9 |  46.2 |
| Fusion: WavLM + BERT, gated (GMU)    | unseen words   |   100   |     100   |  100   |   100   |     100   |               100   | 100   |
| Fusion: WavLM + BERT, gated (GMU)    | unseen speaker |    27.7 |      78.3 |   94.5 |     3.2 |      92.8 |                50.7 |  79   |

## Separability of the blocks inside the multimodal (GMU) model (silhouette on 256-d test vectors, 3 seeds)

| block                   | test set                     | silhouette (mean ± std, 3 seeds)   |
|:------------------------|:-----------------------------|:-----------------------------------|
| Temporal block (speech) | Unseen words (both speakers) | 0.87 ± 0.01                        |
| Contextual block (text) | Unseen words (both speakers) | -0.02 ± 0.00                       |
| Fusion block (GMU)      | Unseen words (both speakers) | 0.96 ± 0.00                        |
| Temporal block (speech) | Unseen speaker: test YAF     | 0.48 ± 0.02                        |
| Contextual block (text) | Unseen speaker: test YAF     | 0.00 ± 0.00                        |
| Fusion block (GMU)      | Unseen speaker: test YAF     | 0.49 ± 0.03                        |
| Temporal block (speech) | Unseen speaker: test OAF     | 0.35 ± 0.06                        |
| Contextual block (text) | Unseen speaker: test OAF     | 0.00 ± 0.00                        |
| Fusion block (GMU)      | Unseen speaker: test OAF     | 0.36 ± 0.09                        |

## Failure cases (GMU fusion, unseen speaker, seed 0, largest margin; stats = clip / her average for the true emotion)

| confusion                    |   errors of this type (3 seeds) | clip           | seeds wrong   | speech-only same error   | mean F0 Hz (clip / her avg)   | F0 range Hz   | duration s   | nearest training emotion (prosody)   |
|:-----------------------------|--------------------------------:|:---------------|:--------------|:-------------------------|:------------------------------|:--------------|:-------------|:-------------------------------------|
| happy -> angry               |                             556 | YAF_love_happy | 3/3           | 3/3                      | 282 / 281                     | 192 / 190     | 1.41 / 1.67  | angry                                |
| angry -> fear                |                             443 | OAF_hash_angry | 3/3           | 2/3                      | 270 / 257                     | 53 / 118      | 1.63 / 1.48  | happy                                |
| pleasant surprise -> disgust |                             348 | OAF_time_ps    | 3/3           | 3/3                      | 233 / 224                     | 225 / 209     | 2.14 / 2.00  | disgust                              |
| happy -> disgust             |                             345 | OAF_dime_happy | 3/3           | 3/3                      | 269 / 263                     | 277 / 274     | 1.89 / 1.97  | happy                                |
| angry -> neutral             |                             311 | YAF_pick_angry | 3/3           | 2/3                      | 242 / 242                     | 60 / 146      | 2.02 / 2.04  | neutral                              |

## Acoustic profile per speaker and emotion (all 200 clips per cell)

| emotion           | mean F0 Hz (OAF / YAF)   | F0 range Hz (OAF / YAF)   | duration s (OAF / YAF)   |
|:------------------|:-------------------------|:--------------------------|:-------------------------|
| angry             | 257 / 242                | 118 / 146                 | 1.48 / 2.04              |
| disgust           | 189 / 220                | 116 / 192                 | 2.31 / 2.39              |
| fear              | 297 / 322                | 89 / 172                  | 1.54 / 1.62              |
| happy             | 263 / 281                | 274 / 190                 | 1.97 / 1.67              |
| neutral           | 188 / 194                | 35 / 19                   | 2.01 / 2.03              |
| pleasant surprise | 224 / 323                | 209 / 329                 | 2.00 / 1.88              |
| sad               | 202 / 217                | 153 / 115                 | 2.52 / 2.17              |

## Does prosody explain the unseen-speaker confusions? (nearest training emotion by mean F0, F0 range, duration vs the speech WavLM model's top prediction)

| test speaker   | true emotion      | nearest training emotion (prosody)   | model's top prediction   | agree   |
|:---------------|:------------------|:-------------------------------------|:-------------------------|:--------|
| YAF            | angry             | pleasant surprise                    | neutral (44 %)           | False   |
| YAF            | disgust           | sad                                  | disgust (84 %)           | False   |
| YAF            | fear              | fear                                 | fear (99 %)              | True    |
| YAF            | happy             | angry                                | angry (81 %)             | True    |
| YAF            | neutral           | neutral                              | neutral (99 %)           | True    |
| YAF            | pleasant surprise | happy                                | pleasant surprise (66 %) | False   |
| YAF            | sad               | disgust                              | sad (94 %)               | False   |
| OAF            | angry             | happy                                | angry (50 %)             | False   |
| OAF            | disgust           | sad                                  | disgust (88 %)           | False   |
| OAF            | fear              | fear                                 | fear (99 %)              | True    |
| OAF            | happy             | happy                                | disgust (82 %)           | False   |
| OAF            | neutral           | neutral                              | neutral (99 %)           | True    |
| OAF            | pleasant surprise | angry                                | disgust (82 %)           | False   |
| OAF            | sad               | disgust                              | sad (82 %)               | False   |

## Text ablation: fusion accuracy with the text input shuffled or held constant (mean over seeds)

| split   | fusion   |   normal |   text_shuffled |   text_constant |
|:--------|:---------|---------:|----------------:|----------------:|
| words   | gmu      |    100   |           100   |           100   |
| words   | concat   |    100   |           100   |           100   |
| spk_OAF | gmu      |     55.7 |            55.7 |            55.7 |
| spk_OAF | concat   |     56   |            56   |            55.9 |
| spk_YAF | gmu      |     66.1 |            66.1 |            66.1 |
| spk_YAF | concat   |     57.2 |            57.2 |            56.7 |

## Prosody baseline (16 hand-crafted features + logistic regression; accuracy %)

| split   |   prosody_16_features |   duration_only |
|:--------|----------------------:|----------------:|
| words   |                  88.8 |            36.9 |
| spk_OAF |                  51.1 |            29.8 |
| spk_YAF |                  52.4 |            25.6 |

## Few-shot recalibration on the unseen speaker (classifier refit on k labelled clips per emotion; accuracy % on her other clips, mean of both directions, 3 seeds, 20 draws)

| model        |   k |   zero_shot |   recalibrated |
|:-------------|----:|------------:|---------------:|
| speech_mfcc  |   1 |        43.2 |           62.8 |
| speech_mfcc  |   2 |        43.2 |           75.1 |
| speech_mfcc  |   5 |        43.2 |           85.4 |
| speech_mfcc  |  10 |        43.2 |           90.1 |
| speech_mfcc  |  20 |        43.2 |           93.1 |
| speech_wavlm |   1 |        64.3 |           81.9 |
| speech_wavlm |   2 |        64.3 |           86.8 |
| speech_wavlm |   5 |        64.3 |           92.4 |
| speech_wavlm |  10 |        64.2 |           95.2 |
| speech_wavlm |  20 |        64.3 |           96.8 |
