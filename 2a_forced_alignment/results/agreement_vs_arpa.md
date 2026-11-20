| run vs arpa   | tier   |   boundaries |   mean |diff| (ms) |   median (ms) |   within 20 ms (%) |   > 100 ms |
|:--------------|:-------|-------------:|-------------------:|--------------:|-------------------:|-----------:|
| arpa_finetune | words  |          488 |                0.1 |             0 |              100   |          0 |
| arpa_finetune | phones |         2126 |                0.1 |             0 |               99.9 |          0 |
| arpa_adapted  | words  |          488 |                0.2 |             0 |              100   |          0 |
| arpa_adapted  | phones |         2126 |                0.1 |             0 |              100   |          0 |
| arpa_reduced  | words  |          488 |                0.1 |             0 |              100   |          0 |
| arpa_reduced  | phones |         2124 |                0.1 |             0 |               99.9 |          0 |
| mfa_ipa       | words  |          488 |               18.4 |            10 |               94.1 |          8 |
| scratch       | words  |          478 |             1010   |           580 |                5   |        419 |
| scratch       | phones |         2004 |              998.3 |           580 |                6.5 |       1731 |
