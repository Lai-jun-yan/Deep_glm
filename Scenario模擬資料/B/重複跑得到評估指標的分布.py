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

    dk = 4
    embedding_dim = len(X)

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


    optimizer = torch.optim.Adam(
        [wq,wk],
        lr=lr
    )


    lam = 1

    I = torch.eye(
        P,
        dtype=X.dtype
    )


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


        A = attn @ attn.T + lam*I


        beta = torch.linalg.solve(
            X.T@X + A,
            X.T@y
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

        E = X

        Q = wq @ E
        K = wk @ E

        scores = K.T @ Q
        scores = scores / np.sqrt(dk)

        final_attn = F.softmax(
            scores,
            dim=0
        )

        A = final_attn @ final_attn.T + lam*I

        adaptive_matrix = A

        final_beta = torch.linalg.solve(
            X.T@X + A,
            X.T@y
        )


    return (
        final_beta.detach(),
        final_attn.detach(),
        adaptive_matrix.detach()
    )

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

n_repeat = 100

beta_results = []
mse_results = []
rmse_results = []
r2_results = []
attention_results = []
adaptive_results = []


for seed in range(n_repeat):

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
    r"C:\Users\USER\Desktop\碩論\程式碼\B\true_beta.csv"
)

beta_true = true_beta["True_beta"].values
ols_bias = abs(beta_ols - beta_true)
attn_bias = abs(beta_mean - beta_true) # 平均係數
ridge_bias = abs(beta_ridge - beta_true)

beta_summary = pd.DataFrame({

    "Variable":cols,

    "Beta_mean(DeepGLM)":beta_mean,

    "Beta_SD(DeepGLM)":beta_sd,

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
#     Variable  Beta_mean(DeepGLM)  Beta_SD(DeepGLM)       OLS     Ridge  Simulation
# X1        X1            0.833386          0.284778  1.635490  0.958468         1.0
# X2        X2            1.032953          0.283662  0.237266  0.914536         0.8
# X3        X3            0.928490          0.260006 -1.060083  0.929475         1.0
# X4        X4            0.955605          0.259868  2.988039  0.961210         0.8
# X5        X5            0.015304          0.018423  1.066174  0.027232         0.0
# X6        X6            0.066156          0.017534 -0.933764  0.056189         0.0
# X7        X7            0.224073          0.062312  2.651680  0.149983         0.0
# X8        X8            0.000099          0.064483 -2.423137  0.078081         0.0
# X9        X9           -0.035817          0.075331 -3.665030  0.042250         0.0
# X10      X10            0.215403          0.072705  3.820952  0.142533         0.0
# OLS的係數偏差:20.007084
# Ridge的係數偏差:0.884070
# DeepGLM的平均係數偏差:1.183534

# ------------------------------------------
# 評估指標的分布:
#   Metric  Mean(DeepGLM)  SD(DeepGLM)       OLS     Ridge
# 0    MSE       0.798352     0.006811  0.844377  0.804461
# 1   RMSE       0.893497     0.003808  0.918900  0.896918
# 2     R2       0.764232     0.002011  0.750640  0.762428

# ------------------------------------------
# 100次attention weight matrix的結果:
#                   X1               X2               X3               X4               X5               X6               X7               X8               X9              X10
# X1   0.0772 ± 0.2540  0.0781 ± 0.2540  0.0709 ± 0.2367  0.0689 ± 0.2364  0.0887 ± 0.2690  0.0887 ± 0.2690  0.0813 ± 0.2690  0.0809 ± 0.2690  0.1235 ± 0.3220  0.1236 ± 0.3222
# X2   0.0273 ± 0.1133  0.0286 ± 0.1158  0.0323 ± 0.1202  0.0319 ± 0.1201  0.0374 ± 0.1490  0.0366 ± 0.1483  0.0291 ± 0.1464  0.0278 ± 0.1452  0.0113 ± 0.0508  0.0116 ± 0.0519
# X3   0.0210 ± 0.0589  0.0205 ± 0.0589  0.0422 ± 0.1737  0.0424 ± 0.1746  0.0588 ± 0.1795  0.0579 ± 0.1796  0.0605 ± 0.2025  0.0619 ± 0.2032  0.0628 ± 0.2204  0.0633 ± 0.2209
# X4   0.0549 ± 0.1956  0.0541 ± 0.1952  0.0465 ± 0.1957  0.0462 ± 0.1956  0.0657 ± 0.2168  0.0644 ± 0.2167  0.0324 ± 0.1426  0.0331 ± 0.1433  0.0351 ± 0.1703  0.0364 ± 0.1709
# X5   0.1448 ± 0.2883  0.1441 ± 0.2868  0.1769 ± 0.2999  0.1772 ± 0.3009  0.1257 ± 0.2560  0.1267 ± 0.2577  0.1525 ± 0.2822  0.1525 ± 0.2807  0.1644 ± 0.2695  0.1646 ± 0.2702
# X6   0.0607 ± 0.1396  0.0613 ± 0.1405  0.0917 ± 0.1685  0.0924 ± 0.1697  0.0731 ± 0.1548  0.0722 ± 0.1529  0.0886 ± 0.1780  0.0887 ± 0.1778  0.1136 ± 0.1919  0.1141 ± 0.1926
# X7   0.0625 ± 0.0898  0.0633 ± 0.0921  0.0557 ± 0.0936  0.0552 ± 0.0937  0.0420 ± 0.0828  0.0417 ± 0.0824  0.0530 ± 0.0979  0.0523 ± 0.0987  0.0444 ± 0.0872  0.0439 ± 0.0877
# X8   0.3096 ± 0.3832  0.3064 ± 0.3790  0.2593 ± 0.3576  0.2592 ± 0.3608  0.2198 ± 0.3457  0.2201 ± 0.3460  0.1943 ± 0.3366  0.1933 ± 0.3366  0.1892 ± 0.3365  0.1876 ± 0.3340
# X9   0.1998 ± 0.3234  0.2002 ± 0.3248  0.1685 ± 0.3002  0.1699 ± 0.3024  0.2340 ± 0.3382  0.2364 ± 0.3412  0.2274 ± 0.3248  0.2280 ± 0.3249  0.1929 ± 0.3195  0.1925 ± 0.3193
# X10  0.0422 ± 0.0829  0.0434 ± 0.0850  0.0560 ± 0.1309  0.0566 ± 0.1317  0.0548 ± 0.0988  0.0553 ± 0.0993  0.0809 ± 0.1698  0.0815 ± 0.1708  0.0628 ± 0.1397  0.0624 ± 0.1393

# ------------------------------------------
# 100次平均的Adaptive Regularization Matrix:
#                   X1               X2               X3               X4               X5               X6               X7               X8               X9              X10
# X1   1.8192 ± 1.4294  0.0138 ± 0.0189  0.0000 ± 0.0003  0.0001 ± 0.0005  0.0014 ± 0.0061  0.0006 ± 0.0023  0.0017 ± 0.0054  0.0280 ± 0.0738  0.0156 ± 0.0587  0.0016 ± 0.0087
# X2   0.0138 ± 0.0189  1.1554 ± 0.6116  0.0003 ± 0.0020  0.0003 ± 0.0023  0.0014 ± 0.0074  0.0006 ± 0.0029  0.0021 ± 0.0050  0.0692 ± 0.1729  0.0284 ± 0.1082  0.0025 ± 0.0134
# X3   0.0000 ± 0.0003  0.0003 ± 0.0020  1.3384 ± 0.8841  0.0212 ± 0.0258  0.0082 ± 0.0378  0.0012 ± 0.0040  0.0011 ± 0.0034  0.0158 ± 0.0513  0.0990 ± 0.1743  0.0062 ± 0.0138
# X4   0.0001 ± 0.0005  0.0003 ± 0.0023  0.0212 ± 0.0258  1.3693 ± 1.0198  0.0025 ± 0.0090  0.0008 ± 0.0027  0.0016 ± 0.0056  0.0191 ± 0.0548  0.0498 ± 0.0917  0.0040 ± 0.0089
# X5   0.0014 ± 0.0061  0.0014 ± 0.0074  0.0082 ± 0.0378  0.0025 ± 0.0090  2.0187 ± 1.0362  0.4926 ± 0.4502  0.0008 ± 0.0066  0.0011 ± 0.0064  0.0021 ± 0.0167  0.0003 ± 0.0015
# X6   0.0006 ± 0.0023  0.0006 ± 0.0029  0.0012 ± 0.0040  0.0008 ± 0.0027  0.4926 ± 0.4502  1.3576 ± 0.3927  0.0004 ± 0.0021  0.0011 ± 0.0051  0.0014 ± 0.0124  0.0002 ± 0.0011
# X7   0.0017 ± 0.0054  0.0021 ± 0.0050  0.0011 ± 0.0034  0.0016 ± 0.0056  0.0008 ± 0.0066  0.0004 ± 0.0021  1.1094 ± 0.1310  0.3893 ± 0.3905  0.0063 ± 0.0255  0.0014 ± 0.0053
# X8   0.0280 ± 0.0738  0.0692 ± 0.1729  0.0158 ± 0.0513  0.0191 ± 0.0548  0.0011 ± 0.0064  0.0011 ± 0.0051  0.3893 ± 0.3905  2.8062 ± 1.3396  0.0076 ± 0.0358  0.0014 ± 0.0057
# X9   0.0156 ± 0.0587  0.0284 ± 0.1082  0.0990 ± 0.1743  0.0498 ± 0.0917  0.0021 ± 0.0167  0.0014 ± 0.0124  0.0063 ± 0.0255  0.0076 ± 0.0358  2.4634 ± 1.1359  0.3761 ± 0.3758
# X10  0.0016 ± 0.0087  0.0025 ± 0.0134  0.0062 ± 0.0138  0.0040 ± 0.0089  0.0003 ± 0.0015  0.0002 ± 0.0011  0.0014 ± 0.0053  0.0014 ± 0.0057  0.3761 ± 0.3758  1.2022 ± 0.4235
