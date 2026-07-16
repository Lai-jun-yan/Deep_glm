import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 讀資料
# =========================

data = pd.read_csv(
    r"C:\Users\USER\Desktop\碩論\程式碼\embedding_data.csv"
)


cols = ["X2","X3","X1"]


# standardization

data[cols] = (
    data[cols]
    -
    data[cols].mean()
) / data[cols].std()



X = torch.tensor(
    data[cols].values,
    dtype=torch.float32
)


y = torch.tensor(
    data["Y"].values,
    dtype=torch.float32
)


N = X.shape[0]
P = X.shape[1]



# =========================
# Hyper parameters
# =========================

embedding_dim = 2
dk = 2

epochs = 1000

n_seed = 20



# =========================
# 儲存不同seed attention
# =========================

attention_results = []

loss_results = []



# =========================
# 多次訓練
# =========================

for seed in range(n_seed):

    print(
        "Training seed:",
        seed
    )


    torch.manual_seed(seed)
    np.random.seed(seed)



    # -------------------------
    # 初始化參數
    # -------------------------

    random = torch.randn(
        embedding_dim,
        1,
        requires_grad=False
    )


    wq = torch.randn(
        dk,
        embedding_dim,
        requires_grad=True
    )


    wk = torch.randn(
        dk,
        embedding_dim,
        requires_grad=True
    )


    wv = torch.randn(
        embedding_dim,
        embedding_dim,
        requires_grad=True
    )



    proj = nn.Linear(
        embedding_dim,
        1
    )


    linear = nn.Linear(
        P,
        1
    )



    optimizer = torch.optim.Adam(

        [
            wq,
            wk,
            wv
        ]
        +
        list(proj.parameters())
        +
        list(linear.parameters()),

        lr=0.001

    )



    loss_history = []



    # =========================
    # training
    # =========================

    for epoch in range(epochs):


        # embedding

        E = (
            random.unsqueeze(0)
            *
            X.unsqueeze(1)
        )



        # Q

        Q = torch.matmul(
            wq.unsqueeze(0),
            E
        )


        # K

        K = torch.matmul(
            wk.unsqueeze(0),
            E
        )



        # attention score

        scores = torch.matmul(

            K.transpose(1,2),

            Q

        )


        scores = scores / np.sqrt(dk)



        attn = F.softmax(
            scores,
            dim=1
        )



        # V

        V = torch.matmul(
            wv.unsqueeze(0),
            E
        )


        delta_E = torch.matmul(
            V,
            attn
        )


        New_E = E + delta_E



        # projection

        z = proj(
            New_E.transpose(1,2)
        )


        z = z.squeeze(-1)



        # prediction

        y_hat = linear(z)

        y_hat = y_hat.squeeze()



        loss = F.mse_loss(
            y_hat,
            y
        )



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        loss_history.append(
            loss.item()
        )



    loss_results.append(
        loss_history
    )



    # =========================
    # training完成後取得attention
    # =========================

    with torch.no_grad():


        E = (
            random.unsqueeze(0)
            *
            X.unsqueeze(1)
        )



        Q = torch.matmul(
            wq.unsqueeze(0),
            E
        )


        K = torch.matmul(
            wk.unsqueeze(0),
            E
        )



        scores = torch.matmul(

            K.transpose(1,2),

            Q

        )


        scores = scores / np.sqrt(dk)



        attn = F.softmax(
            scores,
            dim=1
        )



        # average over samples

        mean_attn = attn.mean(
            dim=0
        )



        attention_results.append(
            mean_attn.cpu().numpy()
        )



print("Training finished")



# =========================
# Attention stability analysis
# =========================


attention_array = np.array(
    attention_results
)


print(
    "attention shape:",
    attention_array.shape
)



# mean attention

mean_attention = (
    attention_array.mean(axis=0)
)



# standard deviation

sd_attention = (
    attention_array.std(axis=0)
)



labels = cols



# =========================
# Mean attention
# =========================


plt.figure(
    figsize=(5,4)
)


sns.heatmap(

    mean_attention,

    annot=True,

    fmt=".3f",

    xticklabels=labels,

    yticklabels=labels

)


plt.title(
    "Mean Attention across 20 seeds"
)


plt.show()



# =========================
# Attention instability
# =========================


plt.figure(
    figsize=(5,4)
)


sns.heatmap(

    sd_attention,

    annot=True,

    fmt=".3f",

    xticklabels=labels,

    yticklabels=labels

)


plt.title(
    "Attention instability (SD)"
)


plt.show()



# =========================
# Loss curve
# =========================


plt.figure(
    figsize=(6,4)
)


for i in range(n_seed):

    plt.plot(
        loss_results[i],
        alpha=0.3
    )


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Training loss across seeds"
)


plt.show()