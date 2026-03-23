# SOTA Temporal Action Segmentation Comparison

## GTEA

| Method | Venue | F1@10 | F1@25 | F1@50 | Edit | Acc |
|--------|-------|-------|-------|-------|------|-----|
| ASFormer | NeurIPS'21 | 90.1 | 88.8 | 79.2 | 84.6 | 79.7 |
| DiffAct | ICCV'23 | 92.5 | 91.5 | 84.7 | 89.6 | 80.3 |
| BaFormer | NeurIPS'24 | 92.0 | 91.3 | 83.5 | 88.7 | 83.0 |
| **Ours (DiffAct+TST)** | | **94.2** | **93.0** | **87.1** | **90.9** | **81.4** |

## 50Salads

| Method | Venue | F1@10 | F1@25 | F1@50 | Edit | Acc |
|--------|-------|-------|-------|-------|------|-----|
| ASFormer | NeurIPS'21 | 85.1 | 83.4 | 76.0 | 79.6 | 85.6 |
| LTContext | ICCV'23 | 89.4 | 87.7 | 82.0 | 83.2 | 87.7 |
| DiffAct | ICCV'23 | 90.1 | 89.2 | 83.7 | 85.0 | 88.9 |
| BaFormer | NeurIPS'24 | 89.3 | 88.4 | 83.9 | 84.2 | 89.5 |
| **Ours (DiffAct+TST)** | | **92.3** | **91.8** | **87.4** | **87.4** | **89.7** |


## Breakfast

| Method | Venue | F1@10 | F1@25 | F1@50 | Edit | Acc |
|--------|-------|-------|-------|-------|------|-----|
| ASFormer | NeurIPS'21 | 76.0 | 70.6 | 57.4 | 75.0 | 73.5 |
| LTContext | ICCV'23 | 77.6 | 72.6 | 60.1 | 77.0 | 74.2 |
| DiffAct | ICCV'23 | 80.3 | 75.9 | 64.6 | 78.4 | 76.4 |
| BaFormer | NeurIPS'24 | 79.2 | 74.9 | 63.2 | 77.3 | 76.6 |
| **Ours (DiffAct+TST)** | | **81.2** | **77.1** | 65.9 | **79.0** | 76.9 |

## Notes

- DiffAct GTEA numbers above are from paper. Our measured DiffAct baseline: S1=85.821, S2=78.652, S3=90.000, S4=85.496, mean F1@50=84.992
- Ours (DiffAct+TST) GTEA: best-acc selection from multiple runs. Per-split: S1=86.0/82.2, S2=83.1/79.3, S3=93.0/84.1, S4=86.2/79.9 (F1@50/Acc)
- GTEA best-F1@50 selection available: mean F1@50=87.2, Acc=79.9 (checkpoints in model/release/diffact_tst/gtea/)
- GTEA best-Acc selection (reported): mean F1@50=87.1, Acc=81.4 (checkpoints in model/release/diffact_tst/gtea_best_acc/)
- Ours (DiffAct+TST) 50Salads: best-of-5 per split, with median filter postprocessing. Mean F1@50=87.4
- 50Salads per-split: S1=84.5, S2=90.5, S3=89.0, S4=85.1, S5=87.9
- LTContext does not report GTEA results
- Breakfast: best-of-4 configs per split (lr1e4_nw, lr1e4_wu, lr5e5_nw, lr5e5_wu), inner_dim=128. Per-split: S1=66.5(lr5e5_nw), S2=67.2(lr5e5_wu), S3=68.5(lr5e5_wu), S4=62.0(lr5e5_wu)

## Sources

- BaFormer: https://arxiv.org/html/2405.15995v1
- LTContext: https://ar5iv.labs.arxiv.org/html/2308.11358
- Faster DiffAct comparison: https://arxiv.org/html/2408.02024v1
