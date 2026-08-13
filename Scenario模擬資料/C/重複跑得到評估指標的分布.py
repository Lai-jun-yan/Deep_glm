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

    d_k = 49

    # d_k = 32**2
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
        
        beta = torch.linalg.solve(X.T @ X + I + A.T@A/torch.trace(A),X.T @ y)
        


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

        A_var_y = A * var_y


        # beta

        final_beta = torch.linalg.solve(X.T @ X + I + A.T@A/torch.trace(A),X.T @ y)

        # adaptive

        adaptive_matrix = I + A.T@A/torch.trace(A)


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


# Simulation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:03<00:00,  1.23s/it]
# 重複100次，所得到Beta的分布:
#            Beta_mean(DeepGLM)  Beta_SD(DeepGLM)       OLS     Ridge  Simulation
# Intercept            1.929799          0.000003  1.932738  1.929794         1.0
# X1                   1.981673          0.000004  1.984513  1.981674         2.0
# X2                  -1.457292          0.000017 -1.459544 -1.457278        -1.5
# X3                   0.522638          0.000081  0.522878  0.522690         0.5
# X4                  -0.132728          0.000045 -0.132968 -0.132743         0.0
# X5                   3.012720          0.000006  3.017042  3.012712         3.0
# X6                   0.004396          0.000012  0.004333  0.004400         0.0
# X7                  -0.049646          0.000016 -0.049727 -0.049651         0.0
# X8                  -0.059264          0.000073 -0.059095 -0.059340         0.0
# X9                  -0.039524          0.000041 -0.039714 -0.039510         0.0
# X10                  0.013368          0.000012  0.013323  0.013375         0.0
# OLS的係數偏差:1.327760
# Ridge的係數偏差:1.325262
# DeepGLM的平均係數偏差:1.325117

# ------------------------------------------
# 評估指標的分布:
#   Metric  Mean(DeepGLM)  SD(DeepGLM)       OLS     Ridge
# 0    MSE       3.850260     0.000022  3.850518  3.850267
# 1   RMSE       1.962208     0.000006  1.962274  1.962210
# 2     R2       0.796716     0.000001  0.796702  0.796715

# ------------------------------------------
# 100次attention weight matrix的結果:
#                  Intercept               X1               X2               X3               X4               X5               X6               X7               X8               X9              X10
# Intercept  0.0017 ± 0.0075  0.0000 ± 0.0003  0.0139 ± 0.0491  0.0948 ± 0.2459  0.0103 ± 0.0995  0.0002 ± 0.0022  0.0808 ± 0.2646  0.0004 ± 0.0013  0.0999 ± 0.2993  0.1205 ± 0.2988  0.0174 ± 0.1173
# X1         0.0000 ± 0.0000  0.0010 ± 0.0063  0.0153 ± 0.0501  0.0780 ± 0.2270  0.0003 ± 0.0010  0.0000 ± 0.0001  0.0099 ± 0.0969  0.0002 ± 0.0006  0.3481 ± 0.4743  0.0672 ± 0.2285  0.0200 ± 0.1387
# X2         0.0001 ± 0.0008  0.0004 ± 0.0037  0.0021 ± 0.0201  0.0116 ± 0.0828  0.0001 ± 0.0003  0.0002 ± 0.0015  0.0397 ± 0.1931  0.0094 ± 0.0924  0.2003 ± 0.3997  0.0462 ± 0.2006  0.0098 ± 0.0966
# X3         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0466 ± 0.0832  0.2499 ± 0.3799  0.0002 ± 0.0009  0.0000 ± 0.0000  0.0000 ± 0.0001  0.0001 ± 0.0001  0.0100 ± 0.0995  0.0332 ± 0.1625  0.0000 ± 0.0001
# X4         0.0000 ± 0.0001  0.0000 ± 0.0000  0.0000 ± 0.0004  0.0307 ± 0.0798  0.2358 ± 0.4138  0.0002 ± 0.0024  0.0413 ± 0.1904  0.0000 ± 0.0001  0.0900 ± 0.2861  0.1216 ± 0.3037  0.0301 ± 0.1700
# X5         0.0000 ± 0.0000  0.0000 ± 0.0001  0.0278 ± 0.0689  0.1360 ± 0.2902  0.0002 ± 0.0008  0.0008 ± 0.0044  0.0101 ± 0.0986  0.0001 ± 0.0003  0.0499 ± 0.2174  0.1048 ± 0.2854  0.0002 ± 0.0008
# X6         0.0004 ± 0.0033  0.0000 ± 0.0001  0.0060 ± 0.0338  0.0380 ± 0.1417  0.0001 ± 0.0007  0.0005 ± 0.0033  0.5002 ± 0.4900  0.0100 ± 0.0988  0.0313 ± 0.1708  0.0687 ± 0.2337  0.0301 ± 0.1698
# X7         0.0005 ± 0.0024  0.0004 ± 0.0024  0.0020 ± 0.0191  0.0298 ± 0.1324  0.0004 ± 0.0024  0.0002 ± 0.0011  0.0617 ± 0.2298  0.2980 ± 0.4484  0.1685 ± 0.3709  0.0460 ± 0.1948  0.0025 ± 0.0219
# X8         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0011 ± 0.0046  0.0000 ± 0.0000  0.0000 ± 0.0001  0.0000 ± 0.0000  0.0000 ± 0.0001  0.0000 ± 0.0002  0.7987 ± 0.3994  0.0000 ± 0.0001  0.0000 ± 0.0001
# X9         0.0000 ± 0.0001  0.0001 ± 0.0009  0.0000 ± 0.0000  0.0140 ± 0.0579  0.0062 ± 0.0613  0.0003 ± 0.0026  0.0595 ± 0.2353  0.0001 ± 0.0003  0.1000 ± 0.3000  0.3096 ± 0.4527  0.0701 ± 0.2542
# X10        0.0000 ± 0.0001  0.0000 ± 0.0002  0.0045 ± 0.0315  0.0358 ± 0.1166  0.0001 ± 0.0003  0.0000 ± 0.0003  0.0202 ± 0.1388  0.0001 ± 0.0005  0.1107 ± 0.3072  0.1201 ± 0.2974  0.3984 ± 0.4789

# ------------------------------------------
# 100次平均的Adaptive Regularization Matrix:
#                  Intercept               X1               X2               X3               X4               X5               X6               X7               X8               X9              X10
# Intercept  1.0000 ± 0.0001  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0002 ± 0.0007  0.0001 ± 0.0005  0.0000 ± 0.0000  0.0005 ± 0.0025  0.0000 ± 0.0001
# X1         0.0000 ± 0.0000  1.0000 ± 0.0001  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0001 ± 0.0005  0.0001 ± 0.0008  0.0000 ± 0.0000  0.0005 ± 0.0030  0.0000 ± 0.0001
# X2         0.0000 ± 0.0000  0.0000 ± 0.0000  1.0071 ± 0.0139  0.0285 ± 0.0549  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0001  0.0000 ± 0.0002  0.0009 ± 0.0021  0.0001 ± 0.0005  0.0001 ± 0.0004
# X3         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0285 ± 0.0549  1.1476 ± 0.2199  0.0033 ± 0.0147  0.0000 ± 0.0000  0.0046 ± 0.0122  0.0010 ± 0.0064  0.0002 ± 0.0011  0.0396 ± 0.0847  0.0017 ± 0.0058
# X4         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0033 ± 0.0147  1.0809 ± 0.2242  0.0000 ± 0.0000  0.0010 ± 0.0085  0.0001 ± 0.0004  0.0000 ± 0.0001  0.0004 ± 0.0010  0.0001 ± 0.0002
# X5         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  0.0000 ± 0.0000  1.0000 ± 0.0001  0.0000 ± 0.0003  0.0001 ± 0.0006  0.0000 ± 0.0000  0.0008 ± 0.0037  0.0000 ± 0.0001
# X6         0.0002 ± 0.0007  0.0001 ± 0.0005  0.0000 ± 0.0001  0.0046 ± 0.0122  0.0010 ± 0.0085  0.0000 ± 0.0003  1.3010 ± 0.4246  0.0004 ± 0.0016  0.0005 ± 0.0037  0.0016 ± 0.0053  0.0012 ± 0.0071
# X7         0.0001 ± 0.0005  0.0001 ± 0.0008  0.0000 ± 0.0002  0.0010 ± 0.0064  0.0001 ± 0.0004  0.0001 ± 0.0006  0.0004 ± 0.0016  1.1124 ± 0.2063  0.0018 ± 0.0141  0.0002 ± 0.0008  0.0002 ± 0.0011
# X8         0.0000 ± 0.0000  0.0000 ± 0.0000  0.0009 ± 0.0021  0.0002 ± 0.0011  0.0000 ± 0.0001  0.0000 ± 0.0000  0.0005 ± 0.0037  0.0018 ± 0.0141  1.8768 ± 0.8669  0.0001 ± 0.0002  0.0038 ± 0.0300
# X9         0.0005 ± 0.0025  0.0005 ± 0.0030  0.0001 ± 0.0005  0.0396 ± 0.0847  0.0004 ± 0.0010  0.0008 ± 0.0037  0.0016 ± 0.0053  0.0002 ± 0.0008  0.0001 ± 0.0002  1.3667 ± 0.5881  0.0018 ± 0.0075
# X10        0.0000 ± 0.0001  0.0000 ± 0.0001  0.0001 ± 0.0004  0.0017 ± 0.0058  0.0001 ± 0.0002  0.0000 ± 0.0001  0.0012 ± 0.0071  0.0002 ± 0.0011  0.0038 ± 0.0300  0.0018 ± 0.0075  1.2083 ± 0.3750
