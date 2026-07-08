# Attention-based Regression Model

## Project Overview

希望能夠將attention機制套入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-07-08)

### Completed

- 目前已經完成scenerioA的模擬資料，以及embedding問題。
- 針對A，設計可以訓練的attention機制，詳見模擬資料/A。

### Current Findings

- 碰到attention matrix訓練完後collapse的問題。
- 所有的變數X都必須依賴某一個變數，與模擬資料不符。

---

## Next Steps

- [ ] 確認sofmax過程中，運算是否出現問題
- [ ] 檢查score是否有數值膨脹的問題
- [x] Visualize the attention matrix using heatmaps.
- [ ] Add LayerNorm and residual connections.
- [ ] Implement Multi-Head Attention.
- [ ] Compare the attention model with a baseline MLP.

---
當研究生好累