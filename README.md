# Attention-based Regression Model

## Project Overview

希望能夠將attention機制套入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-07-17)

### Completed

- 目前已經可以得到訓練完成的attention weight matrix
- 了解attention matrix套入GLM的地方

### Current Findings

- attention weight matrix的效果非常不穩定，可能是因為模型需要學習太多參數
- embedding的解釋要仔細，到底是可以訓練還是依照原始資料的觀測值決定
- 重複訓練多次之後，效果仍然不穩定，甚至有些loss不收斂

---

## Next Steps

- [x] 拓展attention到整筆資料
- [ ] 測試attention matrix的穩定性
- [ ] 完成GLM的beta估計
- [ ] 完成attention_adjust這個branch

---
當研究生好累🥲