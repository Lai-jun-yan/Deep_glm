# Attention-based Regression Model

## Project Overview

希望能夠將attention機制嵌入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-08-07)

### Completed

- 在scenarioB中，設計OLS會overfitting的情況
- 在scenarioC中，測試老師的想法

### Current Findings

- 在scenarioB的情況下，DeepGLM表現有提升，可是幅度不大
- 在資料樣本數小且存在變數高度共線性的情況下，Deep GLM相較OLS的beta bias大幅下降
- DeepGLM與Ridge的表現非常相似，無明顯差別
- Adaptive Regularixation Matrix在解釋上能與scenarioB對上
- 在scenarioC中，OLS、Ridge以及DeepGLM三者表現指標無明顯差別
- 單次訓練的v3以及v4小幅改善DeepGLM，贏過OLS、Ridge

---

## Next Steps

- [ ] 驗證整筆資料跑attention_v3(v4).ipynb
- [ ] 與老師討論後續方案
