import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from tqdm import *
import matplotlib.pyplot as plt
device = "cuda:0"

class MLP(nn.Module):
    def __init__(self,input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), #几分类问题
        )
    
    def forward(self, x):
        x = self.network(x)
        x = F.softmax(x, dim=1)
        return x

def train(dataloader, model, optimizer, loss_func, epochs, device):
    model = model.to(device)
    model.train()

    losses = []
    for epoch in tqdm(range(epochs)):
        loss = 0 
        for idx, data in enumerate(dataloader):
            x = data[0].to(device).float()
            y = data[1].to(device).long()

            optimizer.zero_grad()
            probs = model(x).float()
            l = loss_func(probs, y)
            l.backward()
            loss+=l.item()
            optimizer.step()
        losses.append(loss)
        
    plt.figure()
    plt.plot(losses)

    return losses[-1]

'''指标计算'''
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def eval(dataloader, model, device):
    model.eval()
    probs = torch.tensor([]).to(device)
    true_labels = torch.tensor([]).to(device)
    with torch.no_grad():
        for idx, data in tqdm(enumerate(dataloader)):
            x = data[0].to(device).float()
            true_labels = torch.cat([true_labels, data[1].to(device)], dim=0)
            prob = model(x).squeeze()
            probs = torch.cat([probs, prob], dim=0)

        # 对多标签问题进行二进制分类处理
        predicted_labels = torch.argmax(probs, dim=1)

        accuracy = balanced_accuracy_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
        recall = recall_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        precision = precision_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        f1 = f1_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')
        auc = roc_auc_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='macro')

        print("模型精度为：{:.4f}".format(accuracy))
        print("模型召回率为：{:.4f}".format(recall))
        print("模型精确率为：{:.4f}".format(precision))
        print("模型F1分数为：{:.4f}".format(f1))
        print("模型AUC值为：{:.4f}".format(auc))

    model.train()

    return predicted_labels
            

if __name__ == "__main__":
    train_df = pd.read_csv('../data/train.csv')
    valid_df = pd.read_csv('../data/valid.csv')
    test_df = pd.read_csv('../data/test.csv')

    train_data = torch.tensor(train_df.to_numpy())
    valid_data = torch.tensor(valid_df.to_numpy())
    test_data = torch.tensor(test_df.to_numpy())

    '''获取标签'''
    train_y = train_data[:,0]
    valid_y = valid_data[:,0]
    test_y = test_data[:,0]

    '''获取特征'''
    train_x = train_data[:, 1:]
    valid_x = valid_data[:, 1:]
    test_x = test_data[:, 1:]

    '''准备数据集'''
    train_dataset = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

    valid_dataset = TensorDataset(valid_x, valid_y)
    valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True, drop_last=False)

    test_dataset = TensorDataset(test_x, test_y)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False)

    loss_fuc = nn.CrossEntropyLoss()
    model = MLP(train_x.shape[1], 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(train_dataloader, model, optimizer, loss_fuc, 30, device)
    print("训练集合指标：\n")
    a=eval(train_dataloader, model, "cuda:0")
    print("验证集合指标：\n")
    b=eval(valid_dataloader, model, "cuda:0")
    print("测试集合指标：\n")
    c=eval(test_dataloader, model, "cuda:0")
    print('\n')

