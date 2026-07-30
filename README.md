# Attention-based Regression Model

## Project Overview

希望能夠將attention機制套入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-07-30)

### Completed

- 重新設計attention matrix套入GLM的地方
- 在scenarioB中，設計OLS會overfitting的情況
- 與Ridge共同比較

### Current Findings

- 在scenarioB的情況下，DeepGLM表現有提升，可是幅度不大
- 在資料樣本數小且存在變數高度共線性的情況下，Deep GLM相較OLS的beta bias大幅下降
- DeepGLM與Ridge的表現非常相似，無明顯差別
- Attention weight matrix的變異很大，無法與模擬資料吻合

---

## Next Steps

- [x] 拓展attention到整筆資料
- [x] 測試估計β的穩定性
- [x] 測試Ridge的效果
- [ ] 與老師討論
- [x] 探討是否要減少模型學習的自由度
- [x] 試試看原始資料不做標準化

---
當研究生好累🥲