import pandas as pd
import numpy as np

data = pd.read_csv("/home/lai/Deep_glm/embedding_data.csv") #調整讀檔路徑

X = data.iloc[:,:-1]
y = data["Y"]

import torch

X_tensor = torch.tensor(
    X.values,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y.values,
    dtype=torch.float32
)

y_tensor = y_tensor.unsqueeze(1)

import torch.nn as nn

class FeatureNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3,16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16,3)

    def forward(self, x):

        delta = self.fc1(x)

        delta = self.relu(delta)

        delta = self.fc2(delta)

        z = x + delta

        return z, delta
    
class GLMLayer(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(3,1)

    def forward(self,z):

        return self.linear(z)


feature_net = FeatureNet()

glm_layer = GLMLayer()

z, delta = feature_net(X_tensor)

y_hat = glm_layer(z)

criterion = nn.MSELoss()

import torch.optim as optim

model_params = list(feature_net.parameters()) + list(glm_layer.parameters())

optimizer = optim.Adam(model_params, lr=1e-5)

epochs = 20000
lambda_reg = 0.01

loss_history = []
mse_history = []
corr_history = []

for epoch in range(epochs):

    # ---------- Forward ----------
    z, delta = feature_net(X_tensor)

    y_hat = glm_layer(z)

    # ---------- Loss ----------
    prediction_loss = criterion(y_hat, y_tensor)

    correction_loss = torch.mean(delta**2)

    loss = prediction_loss + lambda_reg * correction_loss

    # loss = prediction_loss

    # ---------- Backpropagation ----------
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    # ------------存下模型的學習軌跡------------
    if epoch % 1 == 0:  # 建議先全部存，之後再決定畫不畫
        loss_history.append(loss.item())
        mse_history.append(prediction_loss.item())
        corr_history.append(correction_loss.item())

    # ---------- Print ----------
    if epoch % 200 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Total={loss.item():.4f} | "
            f"MSE={prediction_loss.item():.4f} | "
            f"Correction={correction_loss.item():.4f}"
        )

import matplotlib.pyplot as plt

plt.figure()

plt.plot(loss_history, label="Total Loss")
plt.plot(mse_history, label="MSE")
plt.plot(corr_history, label="Correction")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("/home/lai/Deep_glm/interaction_attention.png", dpi=300, bbox_inches="tight")
plt.show()