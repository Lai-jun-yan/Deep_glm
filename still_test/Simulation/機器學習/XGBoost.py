import pandas as pd
import numpy as np

n_sim = 10

def generate_data(seed):

    num_subjects = 1000
    num_features = 10

    rng = np.random.default_rng(seed)

    X_raw = rng.standard_normal(
        (num_subjects, num_features)
    )

    ones_column = np.ones(
        (num_subjects, 1)
    )

    X = np.concatenate(
        (ones_column, X_raw),
        axis=1
    )

    # 第一個是 intercept
    beta_true = np.array([
        1.0,   # intercept
        2.0,   # X1 effect
        -1.5,  # X2 effect
        0.5,   # X3 effect
        0.0,   # X4 no effect
        3.0,   # X5 effect
        0.1,
        -0.2,
        0.3,
        -0.4,
        0.5,
    ]).reshape(-1, 1)


    noise = rng.standard_normal(
        (num_subjects, 1)
    )

    Y = (
        X @ beta_true
        + noise
        + X[:, 2:3] * X[:, 3:4] # X2 * X3 interaction
        + X[:, 4:5] * X[:, 4:5] # X4^2 quadratic term
    )

    columns = ["Intercept"] + [f"X{i}" for i in range(1, num_features+1)]

    data = pd.DataFrame(X, columns=columns)

    data["Y"] = Y.flatten()

    return data, beta_true

import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# 1. Generate data
# ============================================================

def generate_data(seed):

    num_subjects = 1000
    num_features = 10

    rng = np.random.default_rng(seed)

    # ------------------------------------------
    # Generate X
    # ------------------------------------------

    X_raw = rng.standard_normal(
        (num_subjects, num_features)
    )

    # Intercept
    ones_column = np.ones(
        (num_subjects, 1)
    )

    X = np.concatenate(
        (ones_column, X_raw),
        axis=1
    )

    # ------------------------------------------
    # True beta
    # ------------------------------------------

    beta_true = np.array([
        1.0,    # intercept
        2.0,    # X1 effect
        -1.5,   # X2 effect
        0.5,    # X3 effect
        0.0,    # X4 no effect
        3.0,    # X5 effect
        0.1,    # X6
        -0.2,   # X7
        0.3,    # X8
        -0.4,   # X9
        0.5     # X10
    ]).reshape(-1, 1)

    # ------------------------------------------
    # Noise
    # ------------------------------------------

    noise = rng.standard_normal(
        (num_subjects, 1)
    )

    # ------------------------------------------
    # Generate Y
    #
    # Linear effect
    # + X2 * X3 interaction
    # + X4^2 quadratic effect
    # + noise
    # ------------------------------------------

    Y = (
        X @ beta_true
        + noise
        + X[:, 2:3] * X[:, 3:4]     # X2 * X3
        + X[:, 4:5] * X[:, 4:5]     # X4^2
    )

    # ------------------------------------------
    # DataFrame
    # ------------------------------------------

    columns = (
        ["Intercept"]
        + [f"X{i}" for i in range(1, num_features + 1)]
    )

    data = pd.DataFrame(
        X,
        columns=columns
    )

    data["Y"] = Y.flatten()

    return data, beta_true


# ============================================================
# 2. Simulation settings
# ============================================================

n_simulations = n_sim 


# ============================================================
# 3. XGBoost hyperparameter search space
# ============================================================

param_grid = {

    "n_estimators": [
        100,
        200,
        300,
        500
    ],

    "max_depth": [
        2,
        3,
        4,
        5
    ],

    "learning_rate": [
        0.01,
        0.05,
        0.1
    ],

    "subsample": [
        0.7,
        0.8,
        1.0
    ],

    "colsample_bytree": [
        0.7,
        0.8,
        1.0
    ],

    "min_child_weight": [
        1,
        3,
        5
    ]
}


# ============================================================
# 4. Storage
# ============================================================

xgb_results = []


# ============================================================
# 5. 500 Simulation
# ============================================================

for sim in tqdm(
    range(n_simulations),
    desc="Simulation"
):

    # --------------------------------------------------------
    # Seed
    #
    # sim = 0  -> seed = 1
    # sim = 1  -> seed = 2
    # ...
    # sim = 499 -> seed = 500
    # --------------------------------------------------------

    data_seed = sim + 1


    # ========================================================
    # Generate data
    # ========================================================

    data, beta_true = generate_data(
        seed=data_seed
    )


    # ========================================================
    # X / Y
    #
    # XGBoost 不需要 Intercept
    # ========================================================

    X = data.drop(
        columns=["Y", "Intercept"]
    )

    y = data["Y"]


    # ========================================================
    # Train / Test split
    #
    # 前 700 = Training
    # 後 300 = Testing
    # ========================================================

    X_train = X.iloc[:700].copy()
    X_test = X.iloc[700:].copy()

    y_train = y.iloc[:700].copy()
    y_test = y.iloc[700:].copy()


    # ========================================================
    # XGBoost model
    # ========================================================

    xgb = XGBRegressor(

        objective="reg:squarederror",

        random_state=data_seed,

        n_jobs=-1
    )


    # ========================================================
    # Cross Validation
    #
    # 注意：
    # CV 只使用 X_train / y_train
    #
    # Testing data 完全不參與 CV
    # ========================================================

    search = RandomizedSearchCV(

        estimator=xgb,

        param_distributions=param_grid,

        # 每一次 simulation 隨機抽 10 組參數
        n_iter=10,

        # 10-fold CV
        cv=10,

        # 用 MSE 選最佳模型
        scoring="neg_mean_squared_error",

        # reproducibility
        random_state=data_seed,

        # 使用所有 CPU cores
        n_jobs=-1
    )


    # ========================================================
    # Fit CV
    # ========================================================

    search.fit(
        X_train,
        y_train
    )


    # ========================================================
    # Best XGBoost model
    # ========================================================

    best_xgb = search.best_estimator_


    # ========================================================
    # Prediction on testing data
    # ========================================================

    y_pred = best_xgb.predict(
        X_test
    )


    # ========================================================
    # Testing MSE
    # ========================================================

    mse = mean_squared_error(
        y_test,
        y_pred
    )


    # ========================================================
    # Testing RMSE
    # ========================================================

    rmse = np.sqrt(mse)


    # ========================================================
    # Testing R2
    # ========================================================

    r2 = r2_score(
        y_test,
        y_pred
    )


    # ========================================================
    # Save result
    # ========================================================

    xgb_results.append({

        "Simulation": sim + 1,

        "Seed": data_seed,

        "MSE": mse,

        "RMSE": rmse,

        "R2": r2,

        # 儲存最佳參數，之後可以檢查
        "n_estimators": search.best_params_[
            "n_estimators"
        ],

        "max_depth": search.best_params_[
            "max_depth"
        ],

        "learning_rate": search.best_params_[
            "learning_rate"
        ],

        "subsample": search.best_params_[
            "subsample"
        ],

        "colsample_bytree": search.best_params_[
            "colsample_bytree"
        ],

        "min_child_weight": search.best_params_[
            "min_child_weight"
        ]
    })


# ============================================================
# 6. Convert results to DataFrame
# ============================================================

xgb_results_df = pd.DataFrame(
    xgb_results
)


# # ============================================================
# # 7. Display individual simulation results
# # ============================================================

# print("\n")
# print("==============================================")
# print("XGBoost Simulation Results")
# print("==============================================")

# print(
#     xgb_results_df[
#         [
#             "Simulation",
#             "Seed",
#             "MSE",
#             "RMSE",
#             "R2"
#         ]
#     ].head(10)
# )


# ============================================================
# 8. Mean
# ============================================================

mean_results = xgb_results_df[
    [
        "MSE",
        "RMSE",
        "R2"
    ]
].mean()


print("\n")
print("==============================================")
print("Mean Performance")
print("==============================================")

print(
    mean_results
)


# # ============================================================
# # 9. Standard deviation
# # ============================================================

# sd_results = xgb_results_df[
#     [
#         "MSE",
#         "RMSE",
#         "R2"
#     ]
# ].std()


# print("\n")
# print("==============================================")
# print("Standard Deviation")
# print("==============================================")

# print(
#     sd_results
# )


# # ============================================================
# # 10. Mean ± SD
# # ============================================================

# print("\n")
# print("==============================================")
# print("Mean ± SD")
# print("==============================================")

# print(
#     f"MSE  : {mean_results['MSE']:.6f} "
#     f"± {sd_results['MSE']:.6f}"
# )

# print(
#     f"RMSE : {mean_results['RMSE']:.6f} "
#     f"± {sd_results['RMSE']:.6f}"
# )

# print(
#     f"R2   : {mean_results['R2']:.6f} "
#     f"± {sd_results['R2']:.6f}"
# )


# # ============================================================
# # 11. Best hyperparameter frequency
# # ============================================================

# print("\n")
# print("==============================================")
# print("Best Hyperparameter Frequency")
# print("==============================================")


# for param in [
#     "n_estimators",
#     "max_depth",
#     "learning_rate",
#     "subsample",
#     "colsample_bytree",
#     "min_child_weight"
# ]:

#     print(f"\n{param}:")
#     print(
#         xgb_results_df[param]
#         .value_counts()
#         .sort_index()
#     )


# # ============================================================
# # 12. Save results
# # ============================================================

# xgb_results_df.to_csv(
#     r"C:\Users\cyguo\Downloads\XGBoost_simulation_results.csv",
#     index=False
# )

# print("\n")
# print("Results saved to:")
# print("XGBoost_simulation_results.csv")