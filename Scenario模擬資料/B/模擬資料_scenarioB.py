import numpy as np
import pandas as pd

np.random.seed(42)

# ==========================
# Simulation 2
# High Collinearity
# ==========================

n = 200
J = 10

# --------------------------
# 建立五個獨立latent variables
# --------------------------

Z = np.random.normal(
    0,
    1,
    size=(n,5)
)

noise = 0.03

X = np.zeros((n,J))

# 第一組
X[:,0] = Z[:,0]
X[:,1] = Z[:,0] + np.random.normal(0,noise,n)

# 第二組
X[:,2] = Z[:,1]
X[:,3] = Z[:,1] + np.random.normal(0,noise,n)

# 第三組
X[:,4] = Z[:,2]
X[:,5] = Z[:,2] + np.random.normal(0,noise,n)

# 第四組
X[:,6] = Z[:,3]
X[:,7] = Z[:,3] + np.random.normal(0,noise,n)

# 第五組
X[:,8] = Z[:,4]
X[:,9] = Z[:,4] + np.random.normal(0,noise,n)

beta = np.array([
    1,
    0.8,
    1,
    0.8,
    0,
    0,
    0,
    0,
    0,
    0
])

# --------------------------
# Linear model
# --------------------------

mu = X @ beta

# --------------------------
# Noise
# --------------------------

SNR = 5

sigma = np.sqrt(
    np.var(mu)/SNR
)

epsilon = np.random.normal(
    0,
    sigma,
    n
)

Y = mu + epsilon

data = pd.DataFrame(
    X,
    columns=[f"X{i+1}" for i in range(J)]
)

data["Y"] = Y

print("模擬資料的結果:")
print(data.head())
print("-----------------------------------------------------------------")

print("確認變數之間的相關性:")
print(data.iloc[:,:10].corr())
print("-----------------------------------------------------------------")

from sklearn.linear_model import LinearRegression

model = LinearRegression(
    fit_intercept=False
)

model.fit(
    X,
    Y
)

# print("模型前五個係數(母體為1、0.8、0.6、0.4、0.2)")
# print(model.coef_[0:5])

# 存檔
data.to_csv(r"C:\Users\USER\Desktop\碩論\程式碼\B\raw_data.csv",index = False)