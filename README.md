# Attention-based Regression Model

## Project Overview

希望能夠將attention機制套入GLM模型中，兼顧模型預測的準確性以及可解釋性。

---

## Current Progress (2026-07-12)

### Completed

- 目前已經完成scenerioA的模擬資料，以及embedding問題。
- 針對A，設計可以訓練的attention機制，詳見模擬資料/A。
- 新增一個branch，針對attention機制重新手刻。

### Current Findings

- 用attention擬合殘差，可能會有main effect的效果在裡面。
- 解釋z的部分，因為是變數之間的非線性關係函數，所以會是最重要的部分

---

## Next Steps

- [x] 確認sofmax過程中，運算是否出現問題
- [x] 檢查score是否有數值膨脹的問題
- [x] 視覺化attention matrix
- [ ] 與老師討論模型的可行性
- [ ] 完成attention_adjust這個branch

---
當研究生好累🥲