import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import statsmodels.api as sm

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
def generate_data(
    num_subjects=1000,
    num_features=10,
    seed=None
):

    # -----------------------------
    # 固定這一次 simulation 的 seed
    # -----------------------------

    if seed is not None:
        np.random.seed(seed)


    # -----------------------------
    # 1. Generate X
    # -----------------------------

    X_raw = np.random.randn(
        num_subjects,
        num_features
    )


    # -----------------------------
    # 2. Add intercept
    # -----------------------------

    ones_column = np.ones(
        (num_subjects, 1)
    )

    X = np.concatenate(
        (ones_column, X_raw),
        axis=1
    )


    # -----------------------------
    # 3. True beta
    # -----------------------------

    beta_true = np.array([
        1.0,
        2.0,
        -1.5,
        0.5,
        0.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]).reshape(-1, 1)


    # -----------------------------
    # 4. Generate noise
    # -----------------------------

    noise = (
        np.random.randn(
            num_subjects,
            1
        ) * 1
    )


    # -----------------------------
    # 5. Generate Y
    # -----------------------------

    Y = (
        X @ beta_true
        + noise
        + X[:, 2:3] * X[:, 3:4]
        + X[:, 4:5] * X[:, 4:5]
    )


    # -----------------------------
    # 6. Convert to DataFrame
    # -----------------------------

    columns = (
        ["Intercept"]
        + [
            f"X{i}"
            for i in range(
                1,
                num_features + 1
            )
        ]
    )

    data = pd.DataFrame(
        X,
        columns=columns
    )

    data["Y"] = Y.flatten()


    # -----------------------------
    # 7. Return
    # -----------------------------

    return data, beta_true

num_features = 10

cols = (
    ["Intercept"]
    + [
        f"X{i}"
        for i in range(
            1,
            num_features + 1
        )
    ]
)

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
            scores / (d_k ), # d_k ** 0.5，不做開根號
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
            scores / (d_k ), # d_k ** 0.5，不做開根號
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

# ==============================
# Beta
# ==============================

beta_deepglm_results = []
beta_ols_results = []
beta_ridge_results = []


# ==============================
# Prediction metrics
# ==============================

mse_deepglm_results = []
mse_ols_results = []
mse_ridge_results = []

rmse_deepglm_results = []
rmse_ols_results = []
rmse_ridge_results = []

r2_deepglm_results = []
r2_ols_results = []
r2_ridge_results = []


# ==============================
# DeepGLM internal results
# ==============================

attention_results = []
adaptive_results = []

for seed in tqdm(
    range(n_repeat),
    desc="Simulation"
):

    # =================================
    # 1. Generate NEW dataset
    # =================================

    data, beta_true = generate_data(
        num_subjects=1000,
        num_features=10,
        seed=seed
    )


    # =================================
    # 2. Split train / validation
    # =================================

    train_data = data.iloc[
        0:700,
        :
    ].copy()

    validation = data.iloc[
        700:1000,
        :
    ].copy()


    # =================================
    # 3. X / Y
    # =================================

    X = train_data[cols]

    y = train_data["Y"]


    # =================================
    # 4. Torch random seed
    # =================================

    torch.manual_seed(seed)


    beta_deepglm, attn_matrix, adaptive_matrix = (
        train_attention_model(
            X,
            y
        )
    )


    beta_deepglm_results.append(
        beta_deepglm.numpy().flatten()
    )

    attention_results.append(
        attn_matrix.numpy()
    )

    adaptive_results.append(
        adaptive_matrix.numpy()
    )

    # =================================
    # 5. OLS
    # =================================

    model_ols = sm.OLS(
        y,
        X
    )

    result_ols = model_ols.fit()

    beta_ols = result_ols.params.values


    # 儲存 beta

    beta_ols_results.append(
        beta_ols
    )

    # =================================
    # 6. Ridge
    # =================================

    ridge = Ridge(
        alpha=1.0,
        fit_intercept=False
    )

    ridge.fit(
        X.values,
        y.values
    )

    beta_ridge = ridge.coef_


    # 儲存 beta

    beta_ridge_results.append(
        beta_ridge
    )

    # validation
    X_val = validation[cols].values
    y_val = validation["Y"].values

    y_pred_deepglm = (
        torch.tensor(
            X_val,
            dtype=torch.float32
        )
        @ beta_deepglm
    ).detach().numpy().flatten()

    y_pred_ols = (
        X_val @ beta_ols
    )

    y_pred_ridge = (
        X_val @ beta_ridge
    )

    mse_deepglm = mean_squared_error(
        y_val,
        y_pred_deepglm
    )

    mse_deepglm_results.append(
        mse_deepglm
    )

    mse_ols = mean_squared_error(
        y_val,
        y_pred_ols
    )

    mse_ols_results.append(
        mse_ols
    )

    mse_ridge = mean_squared_error(
        y_val,
        y_pred_ridge
    )

    mse_ridge_results.append(
        mse_ridge
    )

    rmse_deepglm = np.sqrt(
        mse_deepglm
    )

    rmse_deepglm_results.append(
        rmse_deepglm
    )

    rmse_ols = np.sqrt(
        mse_ols
    )

    rmse_ols_results.append(
        rmse_ols
    )

    rmse_ridge = np.sqrt(
        mse_ridge
    )

    rmse_ridge_results.append(
        rmse_ridge
    )

    r2_deepglm = r2_score(
        y_val,
        y_pred_deepglm
    )

    r2_deepglm_results.append(
        r2_deepglm
    )

    r2_ols = r2_score(
        y_val,
        y_pred_ols
    )

    r2_ols_results.append(
        r2_ols
    )

    r2_ridge = r2_score(
        y_val,
        y_pred_ridge
    )

    r2_ridge_results.append(
        r2_ridge
    )

# ============================================================
# Simulation results → numpy array
# ============================================================

beta_deepglm_results = np.array(beta_deepglm_results)
beta_ols_results = np.array(beta_ols_results)
beta_ridge_results = np.array(beta_ridge_results)

attention_results = np.array(attention_results)
adaptive_results = np.array(adaptive_results)

# ============================================================
# Beta mean / SD
# ============================================================

beta_deepglm_mean = beta_deepglm_results.mean(axis=0)
beta_deepglm_sd = beta_deepglm_results.std(axis=0)

beta_ols_mean = beta_ols_results.mean(axis=0)
beta_ols_sd = beta_ols_results.std(axis=0)

beta_ridge_mean = beta_ridge_results.mean(axis=0)
beta_ridge_sd = beta_ridge_results.std(axis=0)

beta_true = beta_true.flatten()

# ============================================================
# Absolute Bias
# ============================================================

deepglm_abs_bias = np.abs(
    beta_deepglm_results - beta_true
)

ols_abs_bias = np.abs(
    beta_ols_results - beta_true
)

ridge_abs_bias = np.abs(
    beta_ridge_results - beta_true
)

# ============================================================
# Absolute Bias
# ============================================================

deepglm_abs_bias = np.abs(
    beta_deepglm_results - beta_true
)

ols_abs_bias = np.abs(
    beta_ols_results - beta_true
)

ridge_abs_bias = np.abs(
    beta_ridge_results - beta_true
)

deepglm_bias_mean = deepglm_abs_bias.mean(axis=0)

ols_bias_mean = ols_abs_bias.mean(axis=0)

ridge_bias_mean = ridge_abs_bias.mean(axis=0)

deepglm_total_bias = deepglm_abs_bias.sum(axis=1)

ols_total_bias = ols_abs_bias.sum(axis=1)

ridge_total_bias = ridge_abs_bias.sum(axis=1)

print(
    f"DeepGLM Mean Total Absolute Bias: "
    f"{deepglm_total_bias.mean():.6f}"
)

print(
    f"OLS Mean Total Absolute Bias: "
    f"{ols_total_bias.mean():.6f}"
)

print(
    f"Ridge Mean Total Absolute Bias: "
    f"{ridge_total_bias.mean():.6f}"
)

mse_deepglm_results = np.array(
    mse_deepglm_results
)

mse_ols_results = np.array(
    mse_ols_results
)

mse_ridge_results = np.array(
    mse_ridge_results
)

rmse_deepglm_results = np.array(
    rmse_deepglm_results
)

rmse_ols_results = np.array(
    rmse_ols_results
)

rmse_ridge_results = np.array(
    rmse_ridge_results
)

r2_deepglm_results = np.array(
    r2_deepglm_results
)

r2_ols_results = np.array(
    r2_ols_results
)

r2_ridge_results = np.array(
    r2_ridge_results
)

print("MSE")

print(
    f"DeepGLM: "
    f"{mse_deepglm_results.mean():.6f} "
    f"± {mse_deepglm_results.std():.6f}"
)

print(
    f"OLS: "
    f"{mse_ols_results.mean():.6f} "
    f"± {mse_ols_results.std():.6f}"
)

print(
    f"Ridge: "
    f"{mse_ridge_results.mean():.6f} "
    f"± {mse_ridge_results.std():.6f}"
)

print("\nRMSE")

print(
    f"DeepGLM: "
    f"{rmse_deepglm_results.mean():.6f} "
    f"± {rmse_deepglm_results.std():.6f}"
)

print(
    f"OLS: "
    f"{rmse_ols_results.mean():.6f} "
    f"± {rmse_ols_results.std():.6f}"
)

print(
    f"Ridge: "
    f"{rmse_ridge_results.mean():.6f} "
    f"± {rmse_ridge_results.std():.6f}"
)

print("\nR2")

print(
    f"DeepGLM: "
    f"{r2_deepglm_results.mean():.6f} "
    f"± {r2_deepglm_results.std():.6f}"
)

print(
    f"OLS: "
    f"{r2_ols_results.mean():.6f} "
    f"± {r2_ols_results.std():.6f}"
)

print(
    f"Ridge: "
    f"{r2_ridge_results.mean():.6f} "
    f"± {r2_ridge_results.std():.6f}"
)

beta_summary = pd.DataFrame({

    "Variable": cols,

    "True_Beta": beta_true,

    "DeepGLM_Mean": beta_deepglm_mean,

    "DeepGLM_SD": beta_deepglm_sd,

    "OLS_Mean": beta_ols_mean,

    "OLS_SD": beta_ols_sd,

    "Ridge_Mean": beta_ridge_mean,

    "Ridge_SD": beta_ridge_sd,

    "DeepGLM_Bias": deepglm_bias_mean,

    "OLS_Bias": ols_bias_mean,

    "Ridge_Bias": ridge_bias_mean

})

print(beta_summary)

metric_summary = pd.DataFrame({

    "Metric": [
        "MSE",
        "RMSE",
        "R2"
    ],

    "DeepGLM_Mean": [
        mse_deepglm_results.mean(),
        rmse_deepglm_results.mean(),
        r2_deepglm_results.mean()
    ],

    "DeepGLM_SD": [
        mse_deepglm_results.std(),
        rmse_deepglm_results.std(),
        r2_deepglm_results.std()
    ],

    "OLS_Mean": [
        mse_ols_results.mean(),
        rmse_ols_results.mean(),
        r2_ols_results.mean()
    ],

    "OLS_SD": [
        mse_ols_results.std(),
        rmse_ols_results.std(),
        r2_ols_results.std()
    ],

    "Ridge_Mean": [
        mse_ridge_results.mean(),
        rmse_ridge_results.mean(),
        r2_ridge_results.mean()
    ],

    "Ridge_SD": [
        mse_ridge_results.std(),
        rmse_ridge_results.std(),
        r2_ridge_results.std()
    ]

})

print(metric_summary)