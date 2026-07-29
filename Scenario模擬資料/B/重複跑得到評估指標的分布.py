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



    return beta.detach()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

n_repeat = 100


beta_results = []

mse_results = []
rmse_results = []
r2_results = []


for seed in range(n_repeat):

    torch.manual_seed(seed)
    np.random.seed(seed)


    beta_attn = train_attention_model(
        X,
        y
    )


    beta_results.append(
        beta_attn.numpy()
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

# 模擬資料生成時的實際係數
true_beta = pd.read_csv(
    r"C:\Users\USER\Desktop\碩論\程式碼\B\true_beta.csv"
)

beta_true = true_beta["True_beta"].values
ols_bias = abs(beta_ols - beta_true)
attn_bias = abs(beta_mean - beta_true)

beta_summary = pd.DataFrame({

    "Variable":cols,

    "Beta_mean(DeepGLM)":beta_mean,

    "Beta_SD(DeepGLM)":beta_sd,

    "OLS":result.params,

    "Simulation": beta_true

})

print(f"重複{n_repeat}次，所得到Beta的分布:")
print(beta_summary)
print(f"OLS的係數偏差:{ols_bias.sum():.6f}")
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
    ]

})

print("評估指標的分布:")
print(metric_summary)

# 重複100次，所得到Beta的分布:
#     Variable  Beta_mean(DeepGLM)  Beta_SD(DeepGLM)       OLS  Simulation
# X1        X1            0.833347          0.284783  1.635490         1.0
# X2        X2            1.032992          0.283669  0.237266         0.8
# X3        X3            0.928448          0.260072 -1.060083         1.0
# X4        X4            0.955636          0.259930  2.988039         0.8
# X5        X5            0.015278          0.018446  1.066174         0.0
# X6        X6            0.066180          0.017555 -0.933764         0.0
# X7        X7            0.224159          0.062455  2.651680         0.0
# X8        X8            0.000009          0.064632 -2.423137         0.0
# X9        X9           -0.035845          0.075417 -3.665030         0.0
# X10      X10            0.215428          0.072788  3.820952         0.0
# OLS的係數偏差:20.007084
# DeepGLM的平均係數偏差:1.183731

# ------------------------------------------
# 評估指標的分布:
#   Metric  Mean(DeepGLM)  SD(DeepGLM)       OLS
# 0    MSE       0.798354     0.006810  0.844377
# 1   RMSE       0.893498     0.003808  0.918900
# 2     R2       0.764231     0.002011  0.750640