# Attention-based Regression Model

## Project Overview

希望能夠將attention機制套入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-07-14)

### Completed

- 目前已經可以得到訓練完成的attention weight matrix
- attention weight matrix 只針對某一筆資料中的一個row

### Current Findings

- attention weight matrix的效果非常不穩定，可能是因為模型需要學習太多參數

---

## Next Steps

- [ ] 了解在產生attention之前，是否會先設計loss
- [x] 視覺化attention weight matrix
- [ ] 與老師討論模型的可行性
- [ ] 完成attention_adjust這個branch

---
當研究生好累🥲