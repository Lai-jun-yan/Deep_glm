import pandas as pd
import numpy as np

# 先讀進模擬資料
# data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\embedding_data.csv")

data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\B\raw_data.csv")

cols = data.columns[:-1].to_list() 

whole = data.copy()

data = whole.iloc[0:350,:]

# data[cols] = (data[cols] - data[cols].mean()) / data[cols].std() # 針對變數標準化，後面做softmax的時候，數值才不會爆掉

validation = whole.iloc[350:500,:]

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

beta_summary = pd.DataFrame({

    "Variable":cols,

    "Beta_mean(DeepGLM)":beta_mean,

    "Beta_SD(DeepGLM)":beta_sd,

    "OLS":result.params

})

print(f"重複{n_repeat}次，所得到Beta的分布:")
print(beta_summary)
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
#     Variable  Beta_mean   Beta_SD       OLS
# X1        X1   1.276943  0.275274  2.654570
# X2        X2   0.530943  0.275240 -0.838146
# X3        X3   1.384366  0.284619  3.828601
# X4        X4   0.346070  0.284978 -2.101622
# X5        X5   0.194042  0.047124  1.643738
# X6        X6  -0.156759  0.047254 -1.617740
# X7        X7   0.010736  0.005100  0.051040
# X8        X8  -0.021766  0.005279 -0.067780
# X9        X9  -0.024090  0.009782 -0.524653
# X10      X10   0.036443  0.009943  0.547294

# ------------------------------------------
# 評估指標的分布:
#   Metric      Mean        SD       OLS
# 0    MSE  1.255851  0.002510  1.241123
# 1   RMSE  1.120647  0.001119  1.114057
# 2     R2  0.850026  0.000300  0.851785