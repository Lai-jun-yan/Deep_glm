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
        if torch.linalg.det(A)>0:
            beta = torch.linalg.solve(X.T @ X + I + A/torch.trace(A),X.T @ y)
        else:
            beta = torch.linalg.solve(X.T @ X + I - A/torch.trace(A),X.T @ y)
        


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
        if torch.linalg.det(A)>0:
            final_beta = torch.linalg.solve(X.T @ X + I + A/torch.trace(A),X.T @ y)
        else: 
            final_beta = torch.linalg.solve(X.T @ X + I - A/torch.trace(A),X.T @ y)

        # adaptive
        if torch.linalg.det(A)>0:
            adaptive_matrix = I + A/torch.trace(A)
        else:
            adaptive_matrix = I - A/torch.trace(A)


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


# Simulation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:01<00:00,  1.22s/it]
# 重複100次，所得到Beta的分布:
#            Beta_mean(DeepGLM)  Beta_SD(DeepGLM)       OLS     Ridge  Simulation
# Intercept            1.929979          0.000608  1.932738  1.929794         1.0
# X1                   1.981866          0.000621  1.984513  1.981674         2.0
# X2                  -1.457531          0.000624 -1.459544 -1.457278        -1.5
# X3                   0.522718          0.000105  0.522878  0.522690         0.5
# X4                  -0.132775          0.000075 -0.132968 -0.132743         0.0
# X5                   3.012907          0.000808  3.017042  3.012712         3.0
# X6                   0.004387          0.000031  0.004333  0.004400         0.0
# X7                  -0.049666          0.000033 -0.049727 -0.049651         0.0
# X8                  -0.059309          0.000094 -0.059095 -0.059340         0.0
# X9                  -0.039531          0.000081 -0.039714 -0.039510         0.0
# X10                  0.013363          0.000022  0.013323  0.013375         0.0
# OLS的係數偏差:1.327760
# Ridge的係數偏差:1.325262
# DeepGLM的平均係數偏差:1.325237

# ------------------------------------------
# 評估指標的分布:
#   Metric  Mean(DeepGLM)  SD(DeepGLM)       OLS     Ridge
# 0    MSE       3.850237     0.000366  3.850518  3.850267
# 1   RMSE       1.962202     0.000093  1.962274  1.962210
# 2     R2       0.796717     0.000019  0.796702  0.796715

# ------------------------------------------
# 100次attention weight matrix的結果:
#                  Intercept               X1               X2               X3               X4               X5               X6               X7               X8               X9              X10
# Intercept  0.0519 ± 0.2175  0.0107 ± 0.0994  0.0127 ± 0.0988  0.0430 ± 0.1937  0.0209 ± 0.1004  0.0181 ± 0.1258  0.2673 ± 0.3728  0.0528 ± 0.1222  0.0567 ± 0.1585  0.0912 ± 0.2090  0.1545 ± 0.2785
# X1         0.0206 ± 0.1320  0.0526 ± 0.2174  0.0042 ± 0.0071  0.0295 ± 0.1433  0.0280 ± 0.1014  0.0277 ± 0.1489  0.1332 ± 0.2544  0.0472 ± 0.1181  0.0967 ± 0.2025  0.1096 ± 0.2160  0.1507 ± 0.2635
# X2         0.0031 ± 0.0070  0.0023 ± 0.0044  0.1688 ± 0.3550  0.0279 ± 0.1395  0.0583 ± 0.1707  0.0032 ± 0.0267  0.1393 ± 0.2192  0.1321 ± 0.2425  0.1435 ± 0.2489  0.1251 ± 0.1983  0.1126 ± 0.1835
# X3         0.0168 ± 0.0321  0.0123 ± 0.0248  0.0313 ± 0.0375  0.1337 ± 0.2239  0.0906 ± 0.1420  0.0087 ± 0.0261  0.1058 ± 0.1581  0.0895 ± 0.1413  0.1334 ± 0.1855  0.1029 ± 0.1377  0.1168 ± 0.1474
# X4         0.0085 ± 0.0080  0.0087 ± 0.0083  0.0270 ± 0.0477  0.0298 ± 0.0991  0.4832 ± 0.3592  0.0067 ± 0.0100  0.0629 ± 0.1157  0.0562 ± 0.1628  0.0789 ± 0.1655  0.0579 ± 0.1263  0.0492 ± 0.0791
# X5         0.0005 ± 0.0007  0.0102 ± 0.0984  0.0110 ± 0.0994  0.0220 ± 0.1398  0.0216 ± 0.1131  0.0403 ± 0.1959  0.1027 ± 0.2367  0.0260 ± 0.0897  0.0339 ± 0.1331  0.1300 ± 0.2759  0.1519 ± 0.2998
# X6         0.0045 ± 0.0098  0.0048 ± 0.0137  0.0299 ± 0.0736  0.0226 ± 0.0872  0.0370 ± 0.1342  0.0034 ± 0.0115  0.6703 ± 0.4226  0.0509 ± 0.1672  0.0446 ± 0.1419  0.0384 ± 0.1167  0.0343 ± 0.1207
# X7         0.0053 ± 0.0066  0.0074 ± 0.0183  0.0256 ± 0.0588  0.0171 ± 0.0522  0.0549 ± 0.1561  0.0035 ± 0.0066  0.0740 ± 0.1596  0.6058 ± 0.4063  0.0492 ± 0.1257  0.0332 ± 0.0632  0.0676 ± 0.1803
# X8         0.0046 ± 0.0124  0.0055 ± 0.0141  0.0065 ± 0.0276  0.0276 ± 0.1392  0.0393 ± 0.1692  0.0030 ± 0.0068  0.0345 ± 0.1455  0.0412 ± 0.1648  0.6351 ± 0.4159  0.0125 ± 0.0222  0.0239 ± 0.0443
# X9         0.0027 ± 0.0055  0.0028 ± 0.0073  0.0204 ± 0.0560  0.0158 ± 0.0999  0.0362 ± 0.1298  0.0025 ± 0.0085  0.0458 ± 0.1317  0.0607 ± 0.2017  0.0452 ± 0.1598  0.6124 ± 0.4027  0.0657 ± 0.1325
# X10        0.0039 ± 0.0078  0.0036 ± 0.0096  0.0191 ± 0.0736  0.0142 ± 0.0721  0.0391 ± 0.1461  0.0053 ± 0.0316  0.0639 ± 0.1922  0.0362 ± 0.1311  0.0691 ± 0.1896  0.0618 ± 0.1461  0.6135 ± 0.4140

# ------------------------------------------
# 100次平均的Adaptive Regularization Matrix:
#                   Intercept                X1                X2                X3                X4                X5                X6                X7                X8                X9               X10
# Intercept   0.9724 ± 0.1460  -0.0100 ± 0.0993   0.0034 ± 0.0334  -0.0102 ± 0.0699  -0.0094 ± 0.0833  -0.0117 ± 0.0826  -0.0313 ± 0.1854   0.0058 ± 0.0246   0.0045 ± 0.0341   0.0061 ± 0.0553   0.0020 ± 0.0623
# X1         -0.0113 ± 0.0912   0.9783 ± 0.1168   0.0001 ± 0.0015  -0.0093 ± 0.0704   0.0017 ± 0.0375  -0.0195 ± 0.1060  -0.0033 ± 0.0590   0.0044 ± 0.0233  -0.0005 ± 0.0453   0.0073 ± 0.0612  -0.0009 ± 0.0586
# X2          0.0003 ± 0.0018   0.0002 ± 0.0011   0.8876 ± 0.2968  -0.0040 ± 0.0314  -0.0318 ± 0.1487   0.0007 ± 0.0067  -0.0043 ± 0.1031  -0.0109 ± 0.1301  -0.0056 ± 0.0679   0.0115 ± 0.0574   0.0056 ± 0.0496
# X3         -0.0101 ± 0.0316  -0.0069 ± 0.0275  -0.0038 ± 0.0267   0.9906 ± 0.0843  -0.0049 ± 0.0784  -0.0056 ± 0.0240  -0.0266 ± 0.1029  -0.0196 ± 0.1194  -0.0196 ± 0.0961   0.0045 ± 0.0386  -0.0053 ± 0.0450
# X4         -0.0002 ± 0.0032   0.0000 ± 0.0031  -0.0143 ± 0.0441  -0.0016 ± 0.0179   1.0002 ± 0.1328  -0.0005 ± 0.0078  -0.0170 ± 0.1029  -0.0375 ± 0.1614  -0.0310 ± 0.1300  -0.0049 ± 0.0845  -0.0044 ± 0.0410
# X5          0.0000 ± 0.0002  -0.0149 ± 0.1487   0.0034 ± 0.0335  -0.0100 ± 0.0700   0.0048 ± 0.0358   0.9683 ± 0.1597  -0.0033 ± 0.0511   0.0018 ± 0.0165   0.0003 ± 0.0301   0.0057 ± 0.0606   0.0011 ± 0.0658
# X6         -0.0023 ± 0.0093  -0.0030 ± 0.0123  -0.0223 ± 0.0690  -0.0186 ± 0.0868  -0.0268 ± 0.1251  -0.0022 ± 0.0112   1.0039 ± 0.1847  -0.0226 ± 0.1413  -0.0233 ± 0.0764  -0.0071 ± 0.0408  -0.0194 ± 0.0996
# X7         -0.0009 ± 0.0048  -0.0051 ± 0.0213  -0.0199 ± 0.0607  -0.0104 ± 0.0528  -0.0446 ± 0.1558  -0.0015 ± 0.0063  -0.0267 ± 0.1152   0.9962 ± 0.1578  -0.0174 ± 0.0784  -0.0081 ± 0.0373  -0.0176 ± 0.1188
# X8         -0.0016 ± 0.0088  -0.0015 ± 0.0067  -0.0022 ± 0.0162  -0.0056 ± 0.0319  -0.0112 ± 0.1097  -0.0013 ± 0.0055  -0.0193 ± 0.1120  -0.0171 ± 0.1065   1.0070 ± 0.1694  -0.0013 ± 0.0159  -0.0035 ± 0.0355
# X9         -0.0008 ± 0.0050  -0.0012 ± 0.0071  -0.0149 ± 0.0513  -0.0058 ± 0.0500  -0.0211 ± 0.1218  -0.0014 ± 0.0074  -0.0118 ± 0.0885  -0.0288 ± 0.1563  -0.0239 ± 0.1323   1.0000 ± 0.1598  -0.0004 ± 0.0561
# X10        -0.0015 ± 0.0070  -0.0016 ± 0.0090  -0.0086 ± 0.0404  -0.0050 ± 0.0366  -0.0213 ± 0.1119  -0.0024 ± 0.0162  -0.0345 ± 0.1599  -0.0240 ± 0.1250  -0.0304 ± 0.1198  -0.0030 ± 0.0502   1.0156 ± 0.1492
