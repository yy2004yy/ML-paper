import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import KFold
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件，剔除第一个特征
file_path = r"Conbined_Dataset.xlsx"
data = pd.read_excel(file_path)
train_raw = data  # 剔除第一个特征后的数据集

# 合并数据集进行处理（假设没有测试集，仅演示）
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

# 转换为numpy数组并确保类型一致
all_features_np = all_features_raw.values.astype(np.float32)

# 转换为张量
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_train = train_raw.shape[0]

train_features = torch.tensor(all_features_np, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_raw['per_price'].values.reshape(-1, 1), dtype=torch.float32).to(device)


# 定义Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class PyramidTransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=4, dropout=0.1):
        super(PyramidTransformerRegressor, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        layers = []
        for i in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout)
            layers.append(nn.TransformerEncoder(encoder_layer, 1))
            d_model //= 2  # 每层减少一半的特征数

        self.transformer_encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(d_model * 2, 1)  # 注意这里的 d_model 是最后一层的特征数

    def forward(self, src):
        src = self.input_fc(src).unsqueeze(1)  # 增加序列维度
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output.squeeze(1))  # 移除序列维度
        return output


# 对数均方根误差（log RMSE）函数
def log_rmse(y_true, y_pred):
    squared_log_errors = (torch.log(y_true + 1) - torch.log(y_pred + 1)) ** 2
    mean_squared_log_errors = torch.mean(squared_log_errors)
    return torch.sqrt(mean_squared_log_errors).item()


# 训练模型
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader)  # Accumulate average batch loss
    return train_loss


# 测试模型
def Test_model(model, test_loader, criterion, device):
    model.eval()
    test_preds = []
    true_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            test_preds.extend(outputs.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
        test_preds = torch.tensor(test_preds, dtype=torch.float32)
        true_labels = torch.tensor(true_labels, dtype=torch.float32)
        test_loss = log_rmse(true_labels, test_preds)
    return test_loss, true_labels.numpy(), test_preds.numpy()


# K-fold 交叉验证
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    avg_train_log_rmse = 0.0
    avg_valid_log_rmse = 0.0
    for i, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[valid_idx], y_train[valid_idx]

        train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                      torch.tensor(y_val, dtype=torch.float32))
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        model = PyramidTransformerRegressor(input_dim=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, optimizer, criterion, device)
            valid_loss, _, _ = Test_model(model, valid_loader, criterion, device)
            print(
                f'Fold [{i + 1}/{k}], Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid log RMSE: {valid_loss:.4f}')

        avg_train_log_rmse += train_loss
        avg_valid_log_rmse += valid_loss

    avg_train_log_rmse /= (k * num_epochs)
    avg_valid_log_rmse /= (k * num_epochs)
    return avg_train_log_rmse, avg_valid_log_rmse


# 设置参数
k, num_epochs, lr, weight_decay, batch_size = 3, 100, 0.0005, 0.0005, 64

# 进行 K-fold 交叉验证
train_l, valid_l = k_fold(k, train_features.cpu().numpy(), train_labels.cpu().numpy(), num_epochs, lr, weight_decay,
                          batch_size)

print(f'{k}-fold validation: avg train log RMSE: {train_l:.4f}, avg valid log RMSE: {valid_l:.4f}')

# 训练最终模型
final_model = PyramidTransformerRegressor(input_dim=train_features.shape[1]).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_epochs_final = 100  # 根据需要设置最终训练的轮数

for epoch in range(num_epochs_final):
    final_model.train()
    train_loss = train_model(final_model, train_loader, optimizer, criterion, device)
    print(f'Final Epoch [{epoch + 1}/{num_epochs_final}], Train Loss: {train_loss:.4f}')

# 在测试集上进行评估
final_model.eval()
test_loss, true_labels, test_preds = Test_model(final_model, train_loader, criterion, device)
print(f'Final Test log RMSE: {test_loss:.4f}')

# 绘制预测值与真实值对比图
plt.figure(figsize=(10, 6))
plt.plot(true_labels, label='True Values', color='blue', linewidth=2)
plt.plot(test_preds, label='Predictions', color='red', linewidth=2)
plt.title('True Values vs Predictions')
plt.xlabel('Samples')
plt.ylabel('Per Price')
plt.legend()
plt.grid(True)
plt.show()
