import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
file_path = "Conbined_Dataset.xlsx"
data = pd.read_excel(file_path)
train_raw = data  # 剔除第一个特征后的数据集

# 数据预处理
all_features_raw = train_raw.iloc[:, :-1]  # 剔除目标变量列 per_price

# 进行 one-hot 编码处理（假设 district 和 C_floor 是需要编码的非数值特征）
categorical_cols = ['district', 'C_floor']  # 需要进行one-hot编码的列名列表
all_features_raw = pd.get_dummies(all_features_raw, columns=categorical_cols, dummy_na=True)

# 检查并处理NaN值
all_features_raw = all_features_raw.fillna(0)  # 将NaN值填充为0，或者根据实际情况进行处理

# 数据标准化处理
scaler = StandardScaler()
numeric_features = ['roomnum', 'hall', 'AREA', 'floor_num', 'school', 'subway']  # 数值特征列名列表
all_features_raw[numeric_features] = scaler.fit_transform(all_features_raw[numeric_features])

# 排除类型为 object 的数据
all_features_raw = all_features_raw.select_dtypes(exclude=['object'])

# 转换为 numpy 数组并确保类型一致
all_features_np = all_features_raw.values.astype(np.float32)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features_np, train_raw['per_price'].values.reshape(-1, 1),
                                                    test_size=0.01, random_state=42)

# 转换为张量
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_features = torch.tensor(X_train, dtype=torch.float32).to(device)
train_labels = torch.tensor(y_train, dtype=torch.float32).to(device)
test_features = torch.tensor(X_test, dtype=torch.float32).to(device)
test_labels = torch.tensor(y_test, dtype=torch.float32).to(device)


# 定义简单的 RNN 模型
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.rnn(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        return out


# 训练和评估函数
def train_and_evaluate(model, train_features, train_labels, test_features, test_labels, optimizer, criterion, device,
                       num_epochs=100, batch_size=64):
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.train()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() / len(test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return model, train_losses, test_losses


# 设置模型参数和优化器
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 1
num_layers = 2

model = RNN(input_dim, hidden_dim, output_dim, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练模型
trained_model, train_losses, test_losses = train_and_evaluate(model, train_features, train_labels, test_features, test_labels, optimizer,
                                   criterion, device)

# 测试集上评估模型
trained_model.eval()
with torch.no_grad():
    test_preds = trained_model(test_features).cpu().numpy()
    true_labels = test_labels.cpu().numpy()
    test_loss = criterion(torch.tensor(test_preds, dtype=torch.float32), test_labels).item()
    print(f'Test Loss: {test_loss:.4f}')

# 绘制 Train Loss 和 Test Loss 随着 epoch 的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
plt.plot(test_losses, label='Test Loss', color='red', linewidth=2)
plt.title('Train Loss and Test Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(true_labels, label='True Values', color='blue', linewidth=2)
plt.plot(test_preds, label='Predictions', color='red', linewidth=2)
plt.title('True Values vs Predictions (Test Set)')
plt.xlabel('Samples')
plt.ylabel('Per Price')
plt.legend()
plt.grid(True)
plt.show()
