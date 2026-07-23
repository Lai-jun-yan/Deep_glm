import numpy as np
import pandas as pd

np.random.seed(42)

n = 200      # sample size
J = 10       # number of predictors
rho = 0.5    # correlation structure

Sigma = np.zeros((J,J))

for l in range(J):
    for m in range(J):
        Sigma[l,m] = rho ** abs(l-m)

X = np.random.multivariate_normal(
    mean=np.zeros(J),
    cov=Sigma,
    size=n
)

beta = np.array([
    1.0,
    0.8,
    0.6,
    0.4,
    0.2,
    0,
    0,
    0,
    0,
    0
])

# 產生noise
sigma = 1

epsilon = np.random.normal(
    0,
    sigma,
    size=n
)

# 先不加截距項
# beta0 = 1

mu = X @ beta # + beta0

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

print("模型前五個係數(母體為1、0.8、0.6、0.4、0.2)")
print(model.coef_[0:5])

# 存檔
# data.to_csv(r"C:\Users\USER\Desktop\碩論\程式碼\A\raw_data.csv",index = False)