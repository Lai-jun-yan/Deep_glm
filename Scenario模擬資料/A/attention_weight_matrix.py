import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv(r"C:\Users\USER\Desktop\碩論\程式碼\scenarioA_embed.csv")

X = data.iloc[:,0:50].values
Y = data.iloc[:,50].values
Y = Y.reshape(-1,1)

X = torch.tensor(
    X,
    dtype=torch.float32
)

Y = torch.tensor(
    Y,
    dtype=torch.long
)

X = X[:,:,None] # 變tensor

class SelfAttention(nn.Module):

    def __init__(self, d_k):
        super().__init__()

        self.Wq = nn.Linear(1,d_k)
        self.Wk = nn.Linear(1,d_k)
        self.Wv = nn.Linear(1,d_k)


    def forward(self,x):

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)


        scores = torch.matmul(
            Q,
            K.transpose(1,2)
        )


        scores = scores / np.sqrt(Q.shape[-1])


        attention_weights = torch.softmax(
            scores,
            dim=-1
        )


        output = torch.matmul(
            attention_weights,
            V
        )


        return output, attention_weights

class AttentionRegression(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = SelfAttention(
            d_k=8
        )

        self.fc = nn.Linear(
            50*8,
            1
        )


    def forward(self,x):

        out, att = self.attention(x)

        out = out.reshape(
            x.shape[0],
            -1
        )

        prediction = self.fc(out)

        return prediction, att

model = AttentionRegression()

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(100):

    optimizer.zero_grad()


    pred, attention = model(X)


    loss = criterion(
        pred,
        Y
    )


    loss.backward()


    optimizer.step()


    if epoch%10==0:
        print(
            epoch,
            loss.item()
        )

model.eval()

with torch.no_grad():
    pred, attention = model(X)

att0 = attention[0].cpu().numpy()

plt.figure(figsize=(8,8))

plt.imshow(
    att0,
    cmap="viridis"
)

plt.colorbar()

plt.xlabel("Key Feature")

plt.ylabel("Query Feature")

plt.title("Attention Matrix")

plt.show()


