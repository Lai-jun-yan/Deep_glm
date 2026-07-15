import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# 先讀進模擬資料
data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\embedding_data.csv")

cols = ["X2","X3","X1"] # 針對變數標準化，後面做softmax的時候，數值才不會爆掉

data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()

# 為了測試模型架構，先使用第一筆資料
test = data.iloc[0,:3]

shape = (8,1) # 投影成8維，為了方便後續的attention計算

### 定義需要學習的地方
random = torch.randn(shape, requires_grad = True) # 需要學習的地方

test = torch.tensor(test, dtype = torch.float32) # tensor毛好多，統一設定dtype = float32

test = test.reshape(1,3)

wq_shape = (4,8) # KQ space先設定4維，反正比embedding的小

wq = torch.randn(wq_shape, requires_grad = True) # 需要學習的地方

wk_shape = (4,8)

dk = 4

wk = torch.randn(wq_shape, requires_grad = True) # 需要學習的地方

wv_shape = (8,8) # 需要變回原本embedding的維度

wv = torch.randn(wv_shape, requires_grad = True) # 需要學習的地方

# y = 0.2*x1 + 1.5*x2 + 0.8*x3 可以從匯出模擬資料.ipynb得到正確的模型模擬
y_true = data.iloc[0]["Y"] 

y_true = torch.tensor(y_true, dtype = torch.float32)

proj = nn.Linear(8,1)

linear = nn.Linear(3,1)

optimizer = torch.optim.Adam(
    [random,wq,wk,wv]+
    list(proj.parameters())+
    list(linear.parameters()),
    lr=0.001
)

loss_history = []

initial_attn = None


epochs = 1000

for epoch in range(epochs):

    # =====================
    # Forward propagation
    # =====================

    E = random @ test


    Q = wq @ E
    K = wk @ E

    scores = K.T @ Q
    scores = scores / (4 ** 0.5)

    attn = F.softmax(scores, dim=0)


    # 儲存第一次 attention
    if epoch == 0:
        initial_attn = attn.detach().clone()


    V = wv @ E

    delta_E = V @ attn

    New_E = E + delta_E


    # Prediction

    z = proj(New_E.T)

    z = z.reshape(1,-1)

    y_hat = linear(z)


    # Loss

    loss = (y_true - y_hat.squeeze())**2

    loss_history.append(loss.item())


    # Backpropagation

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



# =====================
# 訓練完成後重新 forward
# 取得最後 attention
# =====================

with torch.no_grad():

    E = random @ test

    Q = wq @ E

    K = wk @ E

    scores = K.T @ Q

    scores = scores / (dk ** 0.5)

    final_attn = F.softmax(scores, dim=0)


### 看一下最後得到的attention weight matrix
print("Initial Attention Matrix:")
print(initial_attn)


print("\nFinal Attention Matrix:")
print(final_attn)

# =====================
# 畫 loss curve 看模型學習的趨勢
# =====================

plt.figure(figsize=(6,4))

plt.plot(loss_history)

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.title("Training Loss")

plt.show()

### 可視化attention matrix
# 設定變數名稱
labels = cols

fig, axes = plt.subplots(1,2,figsize=(10,4))

sns.heatmap(
    initial_attn.detach().numpy(),
    annot=True,
    fmt=".3f",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[0]
)

axes[0].set_title("Initial Attention")


sns.heatmap(
    final_attn.detach().numpy(),
    annot=True,
    fmt=".3f",
    xticklabels=labels,
    yticklabels=labels,
    ax=axes[1]
)

axes[1].set_title("Final Attention")

plt.show()
