# Attention-based Regression Model

## Project Goal
Implement a self-attention model from scratch and investigate whether the learned attention matrix can reveal feature interactions.

---

## Progress Log

### 2026-07-08

#### Completed
- Implemented a self-attention layer from scratch using PyTorch.
- Implemented Query, Key and Value projections.
- Implemented scaled dot-product attention.
- Added a regression head for continuous target prediction.
- Successfully trained the model using MSE loss.

#### Verified
- Attention tensor shape:
  - Input: `(500, 50, 1)`
  - Q/K/V: `(500, 50, 8)`
  - Attention matrix: `(500, 50, 50)`
- Verified that each row of the attention matrix sums to 1.

#### Current Issue
- The trained attention matrix collapses to nearly one-hot distributions.
- Most query features attend almost exclusively to a single feature.
- Need to investigate:
  - input normalization
  - initialization
  - learning rate
  - residual connection
  - layer normalization
  - multi-head attention

#### Next Steps
- Standardize input features.
- Inspect score matrix before softmax.
- Visualize attention heatmaps.
- Compare attention before and after training.