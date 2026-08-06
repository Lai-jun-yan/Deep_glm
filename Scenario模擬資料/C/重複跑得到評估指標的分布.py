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

# XTX = X.T @ X

y = data["Y"]

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def train_attention_model(X, y, epochs=1000, lr=0.001):

    if isinstance(X, pd.DataFrame): 
        X = torch.tensor(
            X.values,
            dtype=torch.float32
        )

    if isinstance(y, pd.Series):
        y = torch.tensor(
            y.values,
            dtype=torch.float32
        )

    N = X.shape[0]
    P = X.shape[1]

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


    optimizer = torch.optim.Adam(
        list(W_Q.parameters()) +
        list(W_K.parameters()), 

        lr=lr

    )


    lam = 1

    # Deep GLM 
    p = X.shape[1]

    I = torch.eye(
        p,
        dtype=X.dtype,
        device=X.device
    )


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


        y_hat = X@beta


        loss = F.mse_loss(
            y_hat,
            y
        )


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        final_beta = torch.linalg.solve(
            X.T @ X + A + I*torch.trace(A),
            X.T @ y
        )

        # adaptive
        adaptive_matrix = A + I*torch.trace(A)


    return (
        final_beta.detach(),
        final_attn.detach(),
        adaptive_matrix.detach()
    )

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tqdm import tqdm

n_repeat = 100

beta_results = []
mse_results = []
rmse_results = []
r2_results = []
attention_results = []
adaptive_results = []

for seed in tqdm(range(n_repeat), desc="Simulation"):

    torch.manual_seed(seed)
    np.random.seed(seed)


    beta_attn, attn_matrix, adaptive_matrix = train_attention_model(
        X,
        y
    )


    beta_results.append(
        beta_attn.numpy()
    )

    attention_results.append(
        attn_matrix.numpy()
    )

    adaptive_results.append(
    adaptive_matrix.numpy()
    )


    # validation

    X_val = validation[cols].values

    y_pred = (
        torch.tensor(
            X_val,
            dtype=torch.float32
        )
        @ beta_attn
    ).numpy()

    y_val = validation["Y"].values

    mse = mean_squared_error(
        y_val,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_val,
        y_pred
    )


    mse_results.append(mse)
    rmse_results.append(rmse)
    r2_results.append(r2)

beta_results = np.array(beta_results)

beta_mean = beta_results.mean(axis=0)

beta_sd = beta_results.std(axis=0)

# 用套件驗證OLS
import statsmodels.api as sm

X = data[cols]

model = sm.OLS(
    data["Y"],
    X          # 不加 constant
)

result = model.fit()

beta_ols = result.params.values

y_pred_ols = X_val @ beta_ols

mse_ols = mean_squared_error(
    y_val,
    y_pred_ols
)

rmse_ols = np.sqrt(mse_ols)

ols_r2 = r2_score(y_val, y_pred_ols)

# Ridge
from sklearn.linear_model import Ridge
# ======================
# Split X and Y
# ======================

X_train = data[cols].values
y_train = data["Y"].values

# ======================
# Ridge model
# ======================

ridge = Ridge(alpha=1.0, fit_intercept=False)

ridge.fit(
    X_train,
    y_train,
)

beta_ridge = ridge.coef_

y_pred_ridge = X_val @ beta_ridge

mse_ridge = mean_squared_error(
    y_val,
    y_pred_ridge
)

rmse_ridge = np.sqrt(mse_ridge)

ridge_r2 = r2_score(y_val, y_pred_ridge)

# 模擬資料生成時的實際係數
true_beta = pd.read_csv(
    r"C:\Users\USER\Desktop\碩論\程式碼\C\true_beta.csv"
)

beta_true = true_beta["True_Coefficient"].values
ols_bias = abs(beta_ols - beta_true)
attn_bias = abs(beta_mean.flatten() - beta_true) # 平均係數
ridge_bias = abs(beta_ridge - beta_true)

beta_summary = pd.DataFrame({

    "Beta_mean(DeepGLM)":beta_mean.flatten(),

    "Beta_SD(DeepGLM)":beta_sd.flatten(),

    "OLS":result.params,

    "Ridge":beta_ridge,

    "Simulation": beta_true

})

print(f"重複{n_repeat}次，所得到Beta的分布:")
print(beta_summary)
print(f"OLS的係數偏差:{ols_bias.sum():.6f}")
print(f"Ridge的係數偏差:{ridge_bias.sum():.6f}")
print(f"DeepGLM的平均係數偏差:{attn_bias.sum():.6f}")
print("")
print("------------------------------------------")

metric_summary = pd.DataFrame({

    "Metric":[
        "MSE",
        "RMSE",
        "R2"
    ],

    "Mean(DeepGLM)":[
        np.mean(mse_results),
        np.mean(rmse_results),
        np.mean(r2_results)
    ],

    "SD(DeepGLM)":[
        np.std(mse_results),
        np.std(rmse_results),
        np.std(r2_results)
    ],

    "OLS":[
        mse_ols,
        rmse_ols,
        ols_r2
    ],

    "Ridge":[
        mse_ridge,
        rmse_ridge,
        ridge_r2
    ]

})

print("評估指標的分布:")
print(metric_summary)
print("")
print("------------------------------------------")

attention_results = np.array(attention_results)

attention_mean = attention_results.mean(axis=0)

attention_sd = attention_results.std(axis=0)

attention_summary = pd.DataFrame(
    index=cols,
    columns=cols
)

for i in range(len(cols)):
    for j in range(len(cols)):

        attention_summary.iloc[i,j] = (
            f"{attention_mean[i,j]:.4f}"
            " ± "
            f"{attention_sd[i,j]:.4f}"
        )

print("100次attention weight matrix的結果:")
print(attention_summary)
print("")

print("------------------------------------------")
print("100次平均的Adaptive Regularization Matrix:")
adaptive_results = np.array(adaptive_results)

adaptive_mean = adaptive_results.mean(axis=0)

adaptive_sd = adaptive_results.std(axis=0)

adaptive_summary = pd.DataFrame(
    index=cols,
    columns=cols
)

for i in range(len(cols)):
    for j in range(len(cols)):
        adaptive_summary.iloc[i,j] = (
            f"{adaptive_mean[i,j]:.4f}"
            " ± "
            f"{adaptive_sd[i,j]:.4f}"
        )

print(adaptive_summary)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))

sns.heatmap(
    adaptive_mean,
    xticklabels=cols,
    yticklabels=cols,
    annot=True,
    fmt=".2f",
    cmap="viridis"
)

plt.title("Average Adaptive Regularization Matrix")
plt.show()


# 重複100次，所得到Beta的分布:
#            Beta_mean(DeepGLM)  Beta_SD(DeepGLM)       OLS     Ridge  Simulation
# Intercept            1.932738      2.246866e-06  1.932738  1.929794         1.0
# X1                   1.984512      1.283591e-06  1.984513  1.981674         2.0
# X2                  -1.459541      1.672842e-06 -1.459544 -1.457278        -1.5
# X3                   0.522877      4.703105e-07  0.522878  0.522690         0.5
# X4                  -0.132968      1.344329e-07 -0.132968 -0.132743         0.0
# X5                   3.017041      1.177460e-06  3.017042  3.012712         3.0
# X6                   0.004333      1.419939e-07  0.004333  0.004400         0.0
# X7                  -0.049727      8.764889e-08 -0.049727 -0.049651         0.0
# X8                  -0.059095      8.301131e-08 -0.059095 -0.059340         0.0
# X9                  -0.039714      6.799038e-08 -0.039714 -0.039510         0.0
# X10                  0.013323      6.717141e-08  0.013323  0.013375         0.0
# OLS的係數偏差:1.327760
# Ridge的係數偏差:1.325262
# DeepGLM的平均係數偏差:1.327762

# ------------------------------------------
# 評估指標的分布:
#   Metric  Mean(DeepGLM)   SD(DeepGLM)       OLS     Ridge
# 0    MSE       3.850519  9.340248e-07  3.850518  3.850267
# 1   RMSE       1.962274  2.379955e-07  1.962274  1.962210
# 2     R2       0.796702  4.931426e-08  0.796702  0.796715

# ------------------------------------------
# 100次attention weight matrix的結果:
#                  Intercept               X1               X2               X3               X4               X5               X6               X7               X8               X9              X10
# Intercept  0.0001 ± 0.0003  0.0067 ± 0.0109  0.0095 ± 0.0180  0.0194 ± 0.0302  0.0469 ± 0.0733  0.0025 ± 0.0039  0.0319 ± 0.0606  0.0565 ± 0.1311  0.1192 ± 0.1828  0.0841 ± 0.1336  0.0732 ± 0.1214
# X1         0.0115 ± 0.0095  0.0003 ± 0.0003  0.0146 ± 0.0125  0.0251 ± 0.0263  0.0554 ± 0.0559  0.0019 ± 0.0018  0.1669 ± 0.1764  0.1378 ± 0.1549  0.0916 ± 0.1047  0.1292 ± 0.1299  0.1357 ± 0.1362
# X2         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000
# X3         0.0105 ± 0.0126  0.0084 ± 0.0114  0.0218 ± 0.0275  0.0004 ± 0.0006  0.0370 ± 0.0487  0.0032 ± 0.0043  0.1445 ± 0.1758  0.0672 ± 0.1053  0.0805 ± 0.1055  0.0960 ± 0.1161  0.0706 ± 0.0947
# X4         0.0010 ± 0.0045  0.0003 ± 0.0017  0.0023 ± 0.0097  0.0022 ± 0.0114  0.0000 ± 0.0002  0.0003 ± 0.0015  0.0091 ± 0.0386  0.0117 ± 0.0496  0.0082 ± 0.0405  0.0140 ± 0.0667  0.0108 ± 0.0531
# X5         0.0121 ± 0.0083  0.0065 ± 0.0058  0.0135 ± 0.0126  0.0235 ± 0.0214  0.0747 ± 0.0817  0.0001 ± 0.0001  0.1489 ± 0.1465  0.1074 ± 0.1279  0.2524 ± 0.1814  0.1254 ± 0.1226  0.1355 ± 0.1252
# X6         0.0004 ± 0.0019  0.0009 ± 0.0035  0.0018 ± 0.0071  0.0064 ± 0.0234  0.0134 ± 0.0666  0.0003 ± 0.0012  0.0000 ± 0.0002  0.0079 ± 0.0509  0.0248 ± 0.0959  0.0133 ± 0.0596  0.0107 ± 0.0408
# X7         0.0013 ± 0.0053  0.0015 ± 0.0055  0.0039 ± 0.0164  0.0062 ± 0.0229  0.0153 ± 0.0513  0.0003 ± 0.0013  0.0094 ± 0.0396  0.0001 ± 0.0002  0.0165 ± 0.0687  0.0276 ± 0.1014  0.0179 ± 0.0663
# X8         0.0005 ± 0.0030  0.0001 ± 0.0006  0.0009 ± 0.0058  0.0009 ± 0.0061  0.0021 ± 0.0131  0.0001 ± 0.0005  0.0111 ± 0.0798  0.0020 ± 0.0146  0.0000 ± 0.0001  0.0060 ± 0.0423  0.0062 ± 0.0436
# X9         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000
# X10        0.0007 ± 0.0040  0.0010 ± 0.0063  0.0024 ± 0.0143  0.0010 ± 0.0059  0.0044 ± 0.0276  0.0003 ± 0.0016  0.0053 ± 0.0395  0.0055 ± 0.0354  0.0042 ± 0.0317  0.0013 ± 0.0073  0.0000 ± 0.0001

# ------------------------------------------
# 100次平均的Adaptive Regularization Matrix:
#                  Intercept               X1               X2               X3               X4               X5               X6               X7               X8               X9              X10
# Intercept  0.0013 ± 0.0009  0.0067 ± 0.0109  0.0095 ± 0.0180  0.0194 ± 0.0302  0.0469 ± 0.0733  0.0025 ± 0.0039  0.0319 ± 0.0606  0.0565 ± 0.1311  0.1192 ± 0.1828  0.0841 ± 0.1336  0.0732 ± 0.1214
# X1         0.0115 ± 0.0095  0.0014 ± 0.0009  0.0146 ± 0.0125  0.0251 ± 0.0263  0.0554 ± 0.0559  0.0019 ± 0.0018  0.1669 ± 0.1764  0.1378 ± 0.1549  0.0916 ± 0.1047  0.1292 ± 0.1299  0.1357 ± 0.1362
# X2         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0012 ± 0.0008  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000
# X3         0.0105 ± 0.0126  0.0084 ± 0.0114  0.0218 ± 0.0275  0.0016 ± 0.0013  0.0370 ± 0.0487  0.0032 ± 0.0043  0.1445 ± 0.1758  0.0672 ± 0.1053  0.0805 ± 0.1055  0.0960 ± 0.1161  0.0706 ± 0.0947
# X4         0.0010 ± 0.0045  0.0003 ± 0.0017  0.0023 ± 0.0097  0.0022 ± 0.0114  0.0012 ± 0.0009  0.0003 ± 0.0015  0.0091 ± 0.0386  0.0117 ± 0.0496  0.0082 ± 0.0405  0.0140 ± 0.0667  0.0108 ± 0.0531
# X5         0.0121 ± 0.0083  0.0065 ± 0.0058  0.0135 ± 0.0126  0.0235 ± 0.0214  0.0747 ± 0.0817  0.0013 ± 0.0008  0.1489 ± 0.1465  0.1074 ± 0.1279  0.2524 ± 0.1814  0.1254 ± 0.1226  0.1355 ± 0.1252
# X6         0.0004 ± 0.0019  0.0009 ± 0.0035  0.0018 ± 0.0071  0.0064 ± 0.0234  0.0134 ± 0.0666  0.0003 ± 0.0012  0.0012 ± 0.0008  0.0079 ± 0.0509  0.0248 ± 0.0959  0.0133 ± 0.0596  0.0107 ± 0.0408
# X7         0.0013 ± 0.0053  0.0015 ± 0.0055  0.0039 ± 0.0164  0.0062 ± 0.0229  0.0153 ± 0.0513  0.0003 ± 0.0013  0.0094 ± 0.0396  0.0012 ± 0.0008  0.0165 ± 0.0687  0.0276 ± 0.1014  0.0179 ± 0.0663
# X8         0.0005 ± 0.0030  0.0001 ± 0.0006  0.0009 ± 0.0058  0.0009 ± 0.0061  0.0021 ± 0.0131  0.0001 ± 0.0005  0.0111 ± 0.0798  0.0020 ± 0.0146  0.0012 ± 0.0008  0.0060 ± 0.0423  0.0062 ± 0.0436
# X9         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0012 ± 0.0008  0.0000 ± 0.0000
# X10        0.0007 ± 0.0040  0.0010 ± 0.0063  0.0024 ± 0.0143  0.0010 ± 0.0059  0.0044 ± 0.0276  0.0003 ± 0.0016  0.0053 ± 0.0395  0.0055 ± 0.0354  0.0042 ± 0.0317  0.0013 ± 0.0073  0.0012 ± 0.0008
