import numpy as np

# 1. Generate synthetic X
num_subjects = 1000
num_features = 10

np.random.seed(123)  # 固定隨機種子，方便重現

X_raw = np.random.randn(num_subjects, num_features)


# 2. Add intercept column
ones_column = np.ones((num_subjects, 1))

X = np.concatenate((ones_column, X_raw), axis=1)

# X shape: (1000, 11)


# 3. Define true beta
# 第一個是 intercept
beta_true = np.array([
    1.0,   # intercept
    2.0,   # X1 effect
    -1.5,  # X2 effect
    0.5,   # X3 effect
    0.0,   # X4 no effect
    3.0,   # X5 effect
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]).reshape(-1, 1)


# 4. Generate Y according to linear + nonlinear model
noise = np.random.randn(num_subjects, 1) * 1

Y = (
    X @ beta_true
    + noise
    + X[:, 2:3] * X[:, 3:4]   # X2 * X3 interaction
    + X[:, 4:5] * X[:, 4:5]   # X4^2 quadratic term
)


import pandas as pd

columns = ["Intercept"] + [f"X{i}" for i in range(1, num_features+1)]

data = pd.DataFrame(X, columns=columns)

data["Y"] = Y.flatten()

# data.to_csv("raw_data.csv", index=False)

# beta_df = pd.DataFrame({
#     "Variable": ["Intercept"] + [f"X{i}" for i in range(1, num_features+1)],
#     "True_Coefficient": beta_true.flatten()
# })

# beta_df.to_csv("true_beta.csv", index=False)
