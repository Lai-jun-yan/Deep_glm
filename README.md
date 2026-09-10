# Attention-based Regression Model

## Project Overview

希望能夠將attention機制嵌入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-09-10)

### Completed

- 完成Simulation的500次模擬
- 完成Simulation_with_different_variale_cor的500次模擬
- 建立好XGBoost的模擬架構

### Current Findings

- 在Simulation的情況下，DeepGLM的平均表現最佳，但與OLS以及Ridge相比，並未達到統計顯著
- 在Simulation_with_different_variale_cor的情況下，DeepGLM的平均表現最佳，與OLS相比並未達到統計顯著，但Ridge(顯著)且lasso(邊界)
- 從Simulation_with_different_variale_cor的模擬中發現，模型似乎會動態懲罰較小效果的變數以提升預測表現

---

## Next Steps

- [ ] 使用Simulation的資料跑其他機器學習模型
- [ ] 了解XGBoost的原理
- [X] 評估Simulation_with_different_variale_cor的performance
- [X] Simulation_with_different_variale_cor的500次模擬

