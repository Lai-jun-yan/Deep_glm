import pandas as pd
import numpy as np

# 先讀進模擬資料
# data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\embedding_data.csv")

data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\B\raw_data.csv")

cols = data.columns[:-1].to_list() 

whole = data.copy()

data = whole.iloc[0:35,:]

# data[cols] = (data[cols] - data[cols].mean()) / data[cols].std() # 針對變數標準化，後面做softmax的時候，數值才不會爆掉

validation = whole.iloc[35:50,:]

# validation[cols] = (validation[cols] - validation[cols].mean()) / validation[cols].std()

# ### 先用傳統統計模型驗證
X = data[cols]

XTX = X.T @ X

# y = data["Y"]

# # beta
# beta = np.linalg.inv(X.T @ X) @ X.T @ y

# print("手算:")
# print(beta)
# print("---------------------------------------------")

# 用套件驗證
import statsmodels.api as sm

X = data[cols]

model = sm.OLS(
    data["Y"],
    X          # 不加 constant
)

result = model.fit()

# print("套件驗證:")
# print(result.params)
# print("---------------------------------------------")

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# ======================
# Split X and Y
# ======================

X_train = data[cols].values
y_train = data["Y"].values

# X_test = validation[cols].values
# y_test = validation["Y"].values


# ======================
# Ridge model
# ======================

ridge = Ridge(alpha=1.0, fit_intercept=False)

ridge.fit(
    X_train,
    y_train,
)
# print("套件算Ridge的係數:")
# print(ridge.coef_)
# print("---------------------------------------------")

# # 手算Ridge
# lamda = 1
# I = np.eye(
#     len(cols)
# )
# beta_ridge = np.linalg.inv(X_train.T @ X_train + lamda * I) @ X_train.T @ y_train
# print("手算Ridge:")
# print(beta_ridge)


import torch

X = torch.tensor(
    data[cols].values,
    dtype=torch.float32
)

y = torch.tensor(
    data["Y"].values,
    dtype=torch.float32
)

# y_true = y.mean()

# y_true = y_true.reshape(1,1)

N = X.shape[0]
P = X.shape[1]

import torch.nn as nn

embedding_dim = len(data)
dk = 4

wq = torch.randn(
    dk,
    embedding_dim,
    requires_grad=True
)

wk = torch.randn(
    dk,
    embedding_dim,
    requires_grad=True
)

# 先不需要V
# wv = torch.randn(
#     embedding_dim,
#     embedding_dim,
#     requires_grad=True
# )

# proj = nn.Linear(
#     embedding_dim,
#     1
# )

# linear = nn.Linear(
#     P,
#     1
# )

optimizer = torch.optim.Adam(

    [wq,wk], # wv

    # +list(proj.parameters())

    # +list(linear.parameters()),

    lr=0.001

)

import numpy as np
import torch.nn.functional as F

lam = 1

p = X.shape[1]

I = torch.eye(
    p,
    dtype=X.dtype,
    device=X.device
)

loss_history = []

initial_attn = None

epochs = 1000

for epoch in range(epochs):

    E = X

    Q = wq @ E

    K = wk @ E

    scores = K.T @ Q 

    scores = scores / np.sqrt(dk)

    attn = F.softmax(
        scores,
        dim=0
    )

    A = attn @ attn.T + lam * I

    # beta
    beta = torch.linalg.solve(
    X.T @ X + A,
    X.T @ y
    )

    # V = wv @ E

    # delta_E = V @ attn

    # New_E = E + delta_E

    # z = proj(
    # New_E.T
    # )

    # z = z.reshape(1,-1)

    # y_hat = linear(z)

#    mse_loss = F.mse_loss(
#    y_hat,
#    y
#    )

#    lambda_attn=0.1

#    loss = (
#        mse_loss
#        +
#        lambda_attn *
#        torch.mean(attn**2)
#    )

    y_hat = X @ beta

    loss = F.mse_loss(
    y_hat,
    y
    )


    if epoch == 0:
        initial_attn = attn.detach().clone()

    #     mean_initial_attn = initial_attn.mean(dim=0)

    #     print(E.shape)

    #     print(Q.shape)

    #     print(K.shape)

    #     print(scores.shape)

    #     print(attn.shape)

    #     print(V.shape)

    #     print(delta_E.shape)

    #     print(New_E.shape)

    #     print(z.shape)

    #     print(y_hat.shape)
        
    #     print(y.shape)


    loss_history.append(
        loss.item()
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

# =====================
# 訓練完成後重新 forward
# 取得最後 attention
# =====================

with torch.no_grad():

    E = X

    Q = wq @ E

    K = wk @ E

    scores = K.T @ Q 

    scores = scores / np.sqrt(dk)

    attn = F.softmax(
        scores,
        dim=0
    )

    final_attn = attn.clone()

    A = attn @ attn.T + lam * I

    adaptive = A

    # beta
    beta1 = torch.linalg.solve(
    X.T @ X + A,
    X.T @ y
    )

# print("訓練完之後的beta:")
# print(beta1)

# =====================
# Average attention matrix across all samples
# =====================

# mean_attn = final_attn.mean(dim=0)

# import matplotlib.pyplot as plt

# plt.plot(loss_history)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# # 轉成 numpy
attn_matrix = final_attn.detach().numpy()

# # 設定變數名稱
# labels = cols

# plt.figure(figsize=(5,4))

# sns.heatmap(
#     attn_matrix,
#     annot=True,        # 顯示數值
#     fmt=".3f",         # 小數三位
#     xticklabels=labels,
#     yticklabels=labels,
#     cmap="viridis"
# )

# plt.xlabel("Target variable")
# plt.ylabel("Source variable")
# plt.title("Attention Weight Matrix")

# plt.show()

# fig, axes = plt.subplots(1,2,figsize=(10,4))

# sns.heatmap(
#     initial_attn.detach().numpy(),
#     annot=True,
#     fmt=".3f",
#     xticklabels=labels,
#     yticklabels=labels,
#     ax=axes[0]
# )

# axes[0].set_title("Initial Attention")


# sns.heatmap(
#     final_attn.detach().numpy(),
#     annot=True,
#     fmt=".3f",
#     xticklabels=labels,
#     yticklabels=labels,
#     ax=axes[1]
# )

# axes[1].set_title("Final Attention")


# plt.show()

### 做驗證

beta_ols = result.params.values
beta_attn = beta1.detach().numpy()
beta_ridge = ridge.coef_

X_val = validation[cols].values
y_val = validation["Y"].values

y_pred_ols = X_val @ beta_ols

y_pred_attn = X_val @ beta_attn

y_pred_ridge = X_val @ beta_ridge

from sklearn.metrics import mean_squared_error

mse_ols = mean_squared_error(
    y_val,
    y_pred_ols
)

mse_attn = mean_squared_error(
    y_val,
    y_pred_attn
)

mse_ridge = mean_squared_error(
    y_val,
    y_pred_ridge
)

rmse_ols = np.sqrt(mse_ols)
rmse_attn = np.sqrt(mse_attn)
rmse_ridge = np.sqrt(mse_ridge)

from sklearn.metrics import r2_score

ols_r2 = r2_score(y_val, y_pred_ols)
attn_r2 = r2_score(y_val, y_pred_attn)
ridge_r2 = r2_score(y_val, y_pred_ridge)

# 模擬資料生成時的實際係數
true_beta = pd.read_csv(
    r"C:\Users\USER\Desktop\碩論\程式碼\B\true_beta.csv"
)

beta_true = true_beta["True_beta"].values
ols_bias = abs(beta_ols - beta_true)
attn_bias = abs(beta_attn - beta_true)
ridge_bias = abs(beta_ridge - beta_true)

beta_table = pd.DataFrame({
    "Variable": cols,
    "OLS_beta": beta_ols,
    "Ridge_beta" : beta_ridge,
    "Attention_beta": beta_attn,
    "Simulation_beta": beta_true
})

print("\n---------------------------------------------")
print("三種方法估計之 Beta 比較:")
print("-"*45)
print(beta_table.round(6).to_string(index=False))
print(f"OLS的係數偏差:{ols_bias.sum():.6f}")
print(f"Ridge的係數偏差:{ridge_bias.sum():.6f}")
print(f"DeepGLM的係數偏差:{attn_bias.sum():.6f}")
print("-"*45)

print("\n---------------------------------------------")
print("透過驗證集比較三者的表現:")
print("-"*52)
print(f"{'Model':<12}{'MSE':>12}{'RMSE':>12}{'R²':>12}")
print("-"*52)
print(f"{'OLS':<12}{mse_ols:>12.6f}{rmse_ols:>12.6f}{ols_r2:>12.6f}")
print(f"{'Ridge':<12}{mse_ridge:>12.6f}{rmse_ridge:>12.6f}{ridge_r2:>12.6f}")
print(f"{'Attention':<12}{mse_attn:>12.6f}{rmse_attn:>12.6f}{attn_r2:>12.6f}")
print("-"*52)
print("")

print("-"*148)
print("XTX vs Attention matrix:")
print("XTX")
print(XTX)
print("")

# attn_table = pd.DataFrame(attn_matrix,index = cols, columns = cols)
# adaptive = final_attn @ final_attn.T
Adaptive = adaptive.detach().numpy()
Adaptive = pd.DataFrame(Adaptive,index = cols, columns = cols)

print("Attention adaptive matrix")
print(Adaptive)
print("-"*148)

# plt.figure(figsize=(6,6))

# plt.scatter(
#     y_val,
#     y_pred_ols,
#     facecolors="none",
#     edgecolors="blue",
#     s=60,
#     linewidth=1.5,
#     label="OLS"
# )

# plt.scatter(
#     y_val,
#     y_pred_attn,
#     color="red",
#     s=25,
#     alpha=0.7,
#     label="Attention"
# )

# low = min(y_val.min(), y_pred_ols.min(), y_pred_attn.min())
# high = max(y_val.max(), y_pred_ols.max(), y_pred_attn.max())

# plt.plot([low, high], [low, high], "k--")

# plt.xlabel("True Y")
# plt.ylabel("Predicted Y")
# plt.legend()
# plt.show()

# ---------------------------------------------
# 三種方法估計之 Beta 比較:
# ---------------------------------------------
# Variable  OLS_beta  Ridge_beta  Attention_beta  Simulation_beta
#       X1  1.635490    0.958468        0.957475              1.0
#       X2  0.237266    0.914536        0.912719              0.8
#       X3 -1.060083    0.929475        0.859372              1.0
#       X4  2.988039    0.961210        1.027643              0.8
#       X5  1.066174    0.027232       -0.019836              0.0
#       X6 -0.933764    0.056189        0.100773              0.0
#       X7  2.651680    0.149983        0.189106              0.0
#       X8 -2.423137    0.078081        0.035426              0.0
#       X9 -3.665030    0.042250       -0.118525              0.0
#      X10  3.820952    0.142533        0.296251              0.0
# OLS的係數偏差:20.007084
# Ridge的係數偏差:0.884070
# DeepGLM的係數偏差:1.283434
# ---------------------------------------------

# ---------------------------------------------
# 透過驗證集比較三者的表現:
# ----------------------------------------------------
# Model                MSE        RMSE          R²
# ----------------------------------------------------
# OLS             0.844377    0.918900    0.750640
# Ridge           0.804461    0.896918    0.762428
# Attention       0.799357    0.894068    0.763935
# ----------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# XTX vs Attention matrix:
# XTX
#             X1         X2         X3         X4         X5         X6         X7         X8         X9        X10
# X1   31.525156  31.029135  -6.617261  -6.610576   0.653742   0.743589   2.696802   2.814295  -4.341903  -4.046230
# X2   31.029135  30.576924  -6.623684  -6.614032   0.835689   0.922801   2.650712   2.775045  -4.444043  -4.153152
# X3   -6.617261  -6.623684  29.067973  29.110975   2.792613   2.900075  -1.846449  -1.909958   1.898791   1.795308
# X4   -6.610576  -6.614032  29.110975  29.173889   2.566447   2.678153  -1.776221  -1.836486   1.754328   1.653830
# X5    0.653742   0.835689   2.792613   2.566447  27.010466  26.907692  -3.857609  -3.710478  -4.901831  -5.089852
# X6    0.743589   0.922801   2.900075   2.678153  26.907692  26.832070  -3.699140  -3.548345  -4.896430  -5.069314
# X7    2.696802   2.650712  -1.846449  -1.776221  -3.857609  -3.699140  31.240741  31.291875  -3.464660  -3.354636
# X8    2.814295   2.775045  -1.909958  -1.836486  -3.710478  -3.548345  31.291875  31.378845  -3.696105  -3.583933
# X9   -4.341903  -4.444043   1.898791   1.754328  -4.901831  -4.896430  -3.464660  -3.696105  32.578339  32.796507
# X10  -4.046230  -4.153152   1.795308   1.653830  -5.089852  -5.069314  -3.354636  -3.583933  32.796507  33.050216

# Attention adaptive matrix
#                X1            X2            X3            X4            X5            X6            X7            X8            X9           X10
# X1   1.000000e+00  2.032338e-10  9.516559e-17  4.713460e-17  1.334299e-05  8.017427e-06  3.367065e-07  2.729726e-06  1.396610e-13  1.402950e-13
# X2   2.032338e-10  1.000000e+00  5.981310e-17  2.963189e-17  8.426959e-06  5.064635e-06  8.086051e-07  6.555729e-06  8.690796e-14  8.734512e-14
# X3   9.516559e-17  5.981310e-17  1.203407e+00  4.363963e-02  8.790894e-02  1.646811e-02  2.169248e-21  1.766755e-20  3.772797e-01  9.174642e-03
# X4   4.713460e-17  2.963189e-17  4.363963e-02  1.009372e+00  1.635389e-02  3.064267e-03  9.535893e-21  7.768273e-20  8.143907e-02  1.981562e-03
# X5   1.334299e-05  8.426959e-06  8.790894e-02  1.635389e-02  4.236980e+00  1.043717e+00  8.164532e-30  6.565671e-29  1.578288e-03  3.837633e-05
# X6   8.017427e-06  5.064635e-06  1.646811e-02  3.064267e-03  1.043717e+00  1.419364e+00  3.092968e-29  2.504205e-28  3.193991e-04  7.775710e-06
# X7   3.367065e-07  8.086051e-07  2.169248e-21  9.535893e-21  8.164532e-30  3.092968e-29  1.023890e+00  1.946920e-01  6.267456e-33  4.818197e-32
# X8   2.729726e-06  6.555729e-06  1.766755e-20  7.768273e-20  6.565671e-29  2.504205e-28  1.946920e-01  2.586705e+00  5.075649e-32  3.901929e-31
# X9   1.396610e-13  8.690796e-14  3.772797e-01  8.143907e-02  1.578288e-03  3.193991e-04  6.267456e-33  5.075649e-32  1.728931e+00  1.775170e-02
# X10  1.402950e-13  8.734512e-14  9.174642e-03  1.981562e-03  3.837633e-05  7.775710e-06  4.818197e-32  3.901929e-31  1.775170e-02  1.000433e+00
# ----------------------------------------------------------------------------------------------------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
sns.heatmap(
    adaptive,
    xticklabels=cols,
    yticklabels=cols,
    cmap="viridis"
)
plt.show()
