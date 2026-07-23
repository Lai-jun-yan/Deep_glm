# Attention-based Regression Model

## Project Overview

希望能夠將attention機制套入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-07-23)

### Completed

- 完成部分scenarioA的模擬資料
- 重新設計attention matrix套入GLM的地方

### Current Findings

- attention weight matrix的效果非常不穩定，可能是因為模型需要學習太多參數
- 重新預測多次之後，要估計的β符合預期

---

## Next Steps

- [x] 拓展attention到整筆資料
- [x] 測試估計β的穩定性
- [ ] 完成scenario A的資料模擬
- [ ] 與老師討論
- [x] 探討是否要減少模型學習的自由度
- [x] 完成attention_adjust這個branch

---
當研究生好累🥲