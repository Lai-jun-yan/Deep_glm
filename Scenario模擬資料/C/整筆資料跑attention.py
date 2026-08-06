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

dk = 32

d_k = 32
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

    scores = scores / np.sqrt(dk)

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
    beta = torch.linalg.solve(
        X.T @ X + A + I*torch.trace(A),
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

    scores = scores / np.sqrt(dk)

    attention_matrix = F.softmax(
        scores / (d_k ** 0.5),
        dim=-1
    )

    # 把Y再從矩陣中拿掉
    A = attention_matrix[:-1,:-1]

    final_attn = A.clone()

    # 矩陣乘上Y的變異數
    var_y = torch.var(y)

    A_var_y = A * var_y


    # beta
    beta1 = torch.linalg.solve(
        X.T @ X + A + I*torch.trace(A),
        X.T @ y
    )

    # adaptive
    adaptive = A + I*torch.trace(A)

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
# Intercept  1.932738    1.929794        1.932735              1.0
#        X1  1.984513    1.981674        1.984514              2.0
#        X2 -1.459544   -1.457278       -1.459541             -1.5
#        X3  0.522878    0.522690        0.522878              0.5
#        X4 -0.132968   -0.132743       -0.132968              0.0
#        X5  3.017042    3.012712        3.017042              3.0
#        X6  0.004333    0.004400        0.004333              0.0
#        X7 -0.049727   -0.049651       -0.049727              0.0
#        X8 -0.059095   -0.059340       -0.059095              0.0
#        X9 -0.039714   -0.039510       -0.039714              0.0
#       X10  0.013323    0.013375        0.013323              0.0
# OLS的係數偏差:1.327760
# Ridge的係數偏差:1.325262
# DeepGLM的係數偏差:1.327759
# ---------------------------------------------

# ---------------------------------------------
# 透過驗證集比較三者的表現:
# ----------------------------------------------------
# Model                MSE        RMSE          R²
# ----------------------------------------------------
# OLS             3.850518    1.962274    0.796702
# Ridge           3.850267    1.962210    0.796715
# Attention       3.850520    1.962274    0.796702
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
# Intercept  1.141316e-03  2.929108e-18  1.050843e-20  1.017596e-18  1.061735e-19  1.284657e-16  1.420866e-19  3.348156e-20  3.321961e-20  3.050440e-19  2.282000e-19
# X1         1.032470e-02  1.508340e-03  1.708400e-02  1.879803e-02  3.883033e-02  6.866149e-03  2.084685e-01  1.104890e-01  3.195002e-02  4.607396e-01  9.608199e-02
# X2         2.282071e-14  1.455526e-13  1.141316e-03  1.987396e-14  3.002764e-15  5.232749e-13  1.814327e-15  8.727979e-16  3.385448e-15  1.751717e-15  3.489278e-15
# X3         2.237716e-02  6.439485e-03  3.314753e-02  1.717568e-03  2.801774e-02  5.648539e-03  1.569482e-01  1.643942e-01  9.884112e-02  2.787044e-01  2.049020e-01
# X4         2.664312e-12  2.303501e-12  7.209225e-14  2.953637e-13  1.141316e-03  1.699644e-11  1.016825e-13  6.135712e-13  3.524404e-13  8.708732e-14  1.384582e-13
# X5         2.276471e-02  8.147311e-03  1.300289e-02  2.787817e-02  3.454196e-01  1.339356e-03  9.861804e-02  1.174891e-01  1.125114e-01  9.028731e-02  1.636835e-01
# X6         1.027486e-14  9.953313e-14  2.269753e-16  9.268695e-15  5.213067e-16  1.637347e-13  1.141316e-03  4.342408e-16  2.889823e-15  6.869040e-16  1.255918e-15
# X7         3.270963e-13  2.906106e-12  2.134757e-14  6.769744e-13  3.574754e-13  1.031140e-11  6.739349e-14  1.141316e-03  9.365872e-14  1.034298e-13  3.254211e-13
# X8         2.772078e-13  6.886524e-13  2.605262e-14  6.939582e-13  1.876143e-13  6.099008e-12  4.935094e-13  1.130615e-13  1.141316e-03  1.141804e-13  2.203408e-14
# X9         2.579872e-13  5.975212e-13  3.314653e-15  6.674288e-14  4.848431e-15  4.526685e-13  9.598549e-15  9.769734e-15  1.299139e-14  1.141316e-03  2.086056e-15
# X10        3.431162e-11  3.207988e-11  1.704087e-12  1.635223e-11  1.460323e-12  7.288208e-11  3.429688e-12  9.188829e-12  1.182164e-12  4.051646e-13  1.141316e-03
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
