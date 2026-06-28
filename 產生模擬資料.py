import numpy as np
import pandas as pd

# 固定亂數種子
rng = np.random.default_rng(42)

n = 100

# 三個自變數都服從常態分佈
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
x3 = rng.normal(0, 1, n)

# 誤差項
epsilon = rng.normal(0, 1, n)

# 建立 y
y = 0.2*x1 + 1.5*x2 + 0.8*x3 + epsilon

df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": x3,
    "y": y
})

# min-max scaling 
def gray_embedding(x):
    result = []
    max = x.max()
    min = x.min()
    for i in range(len(x)):
        z = ((x[i]-min)/(max-min))*255
        result.append(z)
    return result

X1 = gray_embedding(df["x1"])
X2 = gray_embedding(df["x2"])
X3 = gray_embedding(df["x3"])

df_gray = pd.DataFrame({
    "X1":X1,
    "X2":X2,
    "X3":X3,
    "Y":y
})

# y從大排到小
df_gray = df_gray.sort_values("Y", ascending=False).reset_index(drop=True)

# 根據變數與Y之間的相關性排序
corr = df_gray.corr(numeric_only=True)["Y"].drop("Y").abs()

order = corr.sort_values(ascending=False).index

df_gray = df_gray[ list(order) + ["Y"] ]

import matplotlib.pyplot as plt

image = df_gray.iloc[:,:-1].values

plt.figure(figsize=(4,10))

plt.imshow(
    image,
    cmap="gray",
    aspect="auto",
    vmin=0,
    vmax=255
)

plt.xlabel("Variables")
plt.ylabel("Subjects")

plt.show()

