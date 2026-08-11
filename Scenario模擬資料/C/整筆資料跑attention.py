import pandas as pd
import numpy as np

# 先讀進模擬資料
# data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\embedding_data.csv")

data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\C\raw_data.csv")

cols = data.columns[:-1].to_list() 

whole = data.copy()

data = whole.iloc[0:700,:]

# data[cols] = (data[cols] - data[cols].mean()) / data[cols].std() # 針對變數標準化，後面做softmax的時候，數值才不會爆掉

validation = whole.iloc[700:1000,:]

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

y = y.reshape(-1, 1)

X_Y_train = torch.cat(
    (X, y),
    dim=1
)

X_Y_features = X_Y_train.t()

# dk = 32

d_k = 32**2
W_Q = nn.Linear(
    X_Y_features.shape[1],
    d_k,
    bias=False
)

W_K = nn.Linear(
    X_Y_features.shape[1],
    d_k,
    bias=False
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

    list(W_Q.parameters()) +
    list(W_K.parameters()), 
    
    lr=0.001

)

import numpy as np
import torch.nn.functional as F

lam = 1

# Deep GLM 
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

    Q = W_Q(X_Y_features)
    
    K = W_K(X_Y_features)

    scores = torch.matmul(
        Q,
        K.transpose(-2,-1)
    ) 

    # scores = scores / np.sqrt(dk)

    attention_matrix = F.softmax(
        scores / (d_k ** 0.5),
        dim=-1
    )

    # 把Y再從矩陣中拿掉
    A = attention_matrix[:-1,:-1]

    # 矩陣乘上Y的變異數
    var_y = torch.var(y)

    A_var_y = A * var_y


    # beta
    if torch.linalg.det(A)>0:
        beta = torch.linalg.solve(X.T @ X + I + A/torch.trace(A),X.T @ y)
    else:
        beta = torch.linalg.solve(X.T @ X + I - A/torch.trace(A),X.T @ y)

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
        initial_attn = attention_matrix.detach().clone()

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

    Q = W_Q(X_Y_features)
    
    K = W_K(X_Y_features)

    scores = torch.matmul(
        Q,
        K.transpose(-2,-1)
    ) 

    # scores = scores / np.sqrt(dk)

    attention_matrix = F.softmax(
        scores / (d_k ** 0.5),
        dim=-1
    )

    # 把Y再從矩陣中拿掉
    A = attention_matrix[:-1,:-1]

    final_attn = A.clone()

    # 矩陣乘上Y的變異數
    var_y = torch.var(y)

    # A_var_y = A * var_y


    # beta
    if torch.linalg.det(A)>0:
        beta1 = torch.linalg.solve(X.T @ X + I + A/torch.trace(A),X.T @ y)
    else: 
        beta1 = torch.linalg.solve(X.T @ X + I - A/torch.trace(A),X.T @ y)

    # adaptive
    if torch.linalg.det(A)>0:
        adaptive = I + A/torch.trace(A)
    else:
        adaptive = I - A/torch.trace(A)

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
beta_attn = beta_attn.flatten()
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
    r"C:\Users\USER\Desktop\碩論\程式碼\C\true_beta.csv"
)

beta_true = true_beta["True_Coefficient"].values
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
#  Variable  OLS_beta  Ridge_beta  Attention_beta  Simulation_beta
# Intercept  1.932738    1.929794        1.929802              1.0
#        X1  1.984513    1.981674        1.981711              2.0
#        X2 -1.459544   -1.457278       -1.457705             -1.5
#        X3  0.522878    0.522690        0.522681              0.5
#        X4 -0.132968   -0.132743       -0.132724              0.0
#        X5  3.017042    3.012712        3.012280              3.0
#        X6  0.004333    0.004400        0.004416              0.0
#        X7 -0.049727   -0.049651       -0.049636              0.0
#        X8 -0.059095   -0.059340       -0.059348              0.0
#        X9 -0.039714   -0.039510       -0.039482              0.0
#       X10  0.013323    0.013375        0.013366              0.0
# OLS的係數偏差:1.327760
# Ridge的係數偏差:1.325262
# DeepGLM的係數偏差:1.324319
# ---------------------------------------------

# ---------------------------------------------
# 透過驗證集比較三者的表現:
# ----------------------------------------------------
# Model                MSE        RMSE          R²
# ----------------------------------------------------
# OLS             3.850518    1.962274    0.796702
# Ridge           3.850267    1.962210    0.796715
# Attention       3.850074    1.962161    0.796725
# ----------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# XTX vs Attention matrix:
# XTX
#             Intercept          X1          X2          X3          X4          X5          X6          X7          X8          X9         X10
# Intercept  700.000000   -6.770654   26.077024   21.763461   -4.691188  -10.094301   53.093997   30.854726  -20.491921   13.770219   12.456642
# X1          -6.770654  674.920381   25.724427   13.663655   -2.975140   30.739848   -7.770265  -17.670628   21.737582    2.647245   -7.674679
# X2          26.077024   25.724427  694.313341  -53.488038   14.160446   -3.922648   27.700151   37.356076   26.643042   32.962421  -17.266176
# X3          21.763461   13.663655  -53.488038  659.797241    8.603685   40.357047  -42.237083  -16.812509  -11.149045   -0.080183   -4.268836
# X4          -4.691188   -2.975140   14.160446    8.603685  670.059749   16.163796    8.712044  -56.664918   -5.031325   15.691337  -22.114637
# X5         -10.094301   30.739848   -3.922648   40.357047   16.163796  683.556426    2.372052    7.705121  -39.818941   33.165520   -0.965993
# X6          53.093997   -7.770265   27.700151  -42.237083    8.712044    2.372052  691.076370   25.333483  -38.593981   47.212928   -3.047354
# X7          30.854726  -17.670628   37.356076  -16.812509  -56.664918    7.705121   25.333483  680.596092   12.790555   -2.313504  -22.465193
# X8         -20.491921   21.737582   26.643042  -11.149045   -5.031325  -39.818941  -38.593981   12.790555  705.487454    0.816101   19.246314
# X9          13.770219    2.647245   32.962421   -0.080183   15.691337   33.165520   47.212928   -2.313504    0.816101  718.485868   67.629982
# X10         12.456642   -7.674679  -17.266176   -4.268836  -22.114637   -0.965993   -3.047354  -22.465193   19.246314   67.629982  650.504528

# Attention adaptive matrix
#               Intercept            X1            X2            X3            X4            X5            X6            X7            X8            X9           X10
# Intercept  1.000000e+00 -5.657038e-23 -1.455760e-31 -3.276538e-25 -5.683062e-29 -4.593656e-21 -4.325397e-27 -1.681131e-26 -1.692538e-27 -1.970688e-27 -7.176840e-27
# X1        -3.947737e-26  1.000000e+00 -3.100261e-31 -4.750338e-27 -1.390985e-29 -3.379033e-20 -1.294060e-29 -3.389228e-30 -4.268333e-30 -3.480790e-30 -1.378718e-29
# X2        -2.397504e-33 -4.212362e-32  8.000000e-01 -1.573511e-31 -4.846340e-27 -8.434350e-34 -7.839699e-26 -3.988344e-28 -5.915292e-29 -4.112278e-29 -1.346118e-28
# X3        -1.775188e-26 -1.777696e-25 -1.199318e-30  1.000000e+00 -1.857458e-27 -9.803526e-25 -3.512807e-28 -2.804235e-28 -6.292839e-27 -8.272507e-27 -5.156541e-27
# X4        -1.541428e-44 -2.222459e-42  0.000000e+00 -1.401298e-45  1.000000e+00 -4.484155e-44  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
# X5         0.000000e+00  0.000000e+00 -2.000000e-01  0.000000e+00 -3.016309e-40  1.000000e+00 -2.951000e-37 -1.655522e-40 -1.723597e-43 -5.094799e-40 -1.033598e-41
# X6        -5.060113e-16 -5.310965e-18 -6.446970e-17 -5.974650e-18 -5.551999e-17 -9.768359e-18  8.000000e-01 -1.603806e-15 -7.359887e-18 -1.390782e-15 -3.499292e-17
# X7        -1.898019e-10 -7.252057e-11 -5.300928e-13 -1.352086e-09 -8.312610e-10 -9.872843e-10 -1.441211e-08  8.000000e-01 -6.775712e-09 -9.148690e-09 -3.900765e-09
# X8        -5.605194e-45 -1.266774e-42  0.000000e+00 -7.006492e-45  0.000000e+00 -8.547921e-44  0.000000e+00  0.000000e+00  1.000000e+00  0.000000e+00 -5.605194e-45
# X9        -4.955953e-12 -5.987145e-13 -4.025052e-12 -3.326546e-12 -1.122935e-11 -9.242912e-13 -4.084271e-10 -3.195115e-11 -4.236246e-12  8.000000e-01 -1.806208e-09
# X10       -1.016905e-15 -4.444309e-16 -1.548452e-19 -9.191284e-16 -1.183032e-16 -8.233640e-16 -4.173431e-17 -1.315294e-17 -2.333746e-16 -6.550975e-15  8.000000e-01
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(6,5))
# sns.heatmap(
#     adaptive,
#     xticklabels=cols,
#     yticklabels=cols,
#     cmap="viridis"
# )
# plt.show()
