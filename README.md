# Attention-based Regression Model

## Project Overview

希望能夠將attention機制嵌入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-08-12)

### Completed

- 在scenarioB中，設計OLS會overfitting的情況
- 在scenarioC中，測試老師的想法
- 使用colab跑模擬，設定checkpoint並將結果保存在google drive

### Current Findings

- 在scenarioB的情況下，DeepGLM表現有提升，可是幅度不大
- Adaptive Regularixation Matrix在解釋上能與scenarioB對上
- 在scenarioC中，DeepGLM相較OLS和Ridge，表現指標(係數偏差、MSE、R^2)皆為最佳

---

## Next Steps

- [ ] 使用scenarioC的資料跑其他機器學習模型
- [ ] 嘗試其他的資料結構
- [ ] 與老師討論後續方案
