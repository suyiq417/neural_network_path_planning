import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os

# --- PyTorch BP 神经网络模型 ---
class BPNeuralNetworkModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 learning_rate: float = 0.001):
        """
        初始化基于 PyTorch 的 BP 神经网络模型。

        Args:
            input_dim (int): 输入层维度 n (2*N + 1)。
            hidden_dim (int): 隐藏层维度 h。
            output_dim (int): 输出层维度 u (N)。
            learning_rate (float): 学习率。
        """
        super(BPNeuralNetworkModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # 定义网络层
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid() # Sigmoid 激活函数

        # 选择设备 (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device) # 将模型参数移动到选定的设备

        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # 定义损失函数
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """定义前向传播"""
        x = self.hidden_layer(x)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        x = self.sigmoid(x) # 输出层也使用 sigmoid
        return x

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2):
        """使用 PyTorch 训练模型，训练固定的 epochs 数量"""
        print(f"开始训练 PyTorch 模型，共 {epochs} 个周期，批大小 {batch_size}...")

        # 1. 将 NumPy 数组转换为 PyTorch 张量
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        # 2. 创建 TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)

        # 3. 划分训练集和验证集
        if validation_split > 0:
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            if train_size == 0 or val_size == 0:
                print("警告：训练集或验证集大小为0，无法进行有效的验证。")
                train_dataset = dataset
                val_dataset = None
                val_loader = None
            else:
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            train_dataset = dataset
            val_dataset = None
            val_loader = None
            print("警告：未划分验证集 (validation_split=0)。")

        # 4. 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

        # 5. 训练循环
        for epoch in range(epochs):
            self.train() # 设置模型为训练模式
            running_loss = 0.0
            running_mae = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_mae += mae.item()

            epoch_loss = running_loss / len(train_loader)
            epoch_mae = running_mae / len(train_loader)
            history['loss'].append(epoch_loss)
            history['mae'].append(epoch_mae)

            # 验证步骤
            epoch_val_loss = float('nan')
            epoch_val_mae = float('nan')
            if val_loader:
                self.eval() # 设置模型为评估模式
                val_loss = 0.0
                val_mae = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self(inputs)
                        loss = self.criterion(outputs, targets)
                        mae = torch.mean(torch.abs(outputs - targets))
                        val_loss += loss.item()
                        val_mae += mae.item()

                epoch_val_loss = val_loss / len(val_loader)
                epoch_val_mae = val_mae / len(val_loader)
                history['val_loss'].append(epoch_val_loss)
                history['val_mae'].append(epoch_val_mae)

                # 打印每个周期的损失和指标
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, MAE: {epoch_mae:.6f}, Val Loss: {epoch_val_loss:.6f}, Val MAE: {epoch_val_mae:.6f}")
            else:
                 # 如果没有验证集，只打印训练损失
                 print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, MAE: {epoch_mae:.6f}")

        print(f"PyTorch 模型训练完成 {epochs} 个周期。") # 简单的结束信息

        return history # 返回包含损失和指标历史的字典

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """使用 PyTorch 模型进行预测"""
        self.eval() # 设置模型为评估模式
        with torch.no_grad(): # 禁用梯度计算
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            predictions_tensor = self(input_tensor)
            # 将结果移回 CPU 并转换为 NumPy 数组
            predictions = predictions_tensor.cpu().numpy()
        return predictions

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, batch_size: int = 32) -> list:
        """
        使用 PyTorch 模型评估。
        返回 [loss, mae] 列表。
        """
        print("评估 PyTorch 模型...")
        self.eval() # 评估模式
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0.0
        total_mae = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets))
                total_loss += loss.item() * inputs.size(0) # 乘以 batch size 以便后续平均
                total_mae += mae.item() * inputs.size(0)

        avg_loss = total_loss / len(dataset)
        avg_mae = total_mae / len(dataset)
        print(f"评估完成 - Loss (MSE): {avg_loss:.6f}, MAE: {avg_mae:.6f}")
        return [avg_loss, avg_mae]

    def save_model(self, file_path: str):
        """使用 PyTorch 保存模型状态字典。"""
        print(f"将 PyTorch 模型状态字典保存到 {file_path}...")
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 保存模型的状态字典、输入/隐藏/输出维度和学习率
        save_dict = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), # 可选：保存优化器状态
        }
        torch.save(save_dict, file_path)
        print("PyTorch 模型状态字典已保存。")

    @classmethod
    def load_model(cls, file_path: str) -> 'BPNeuralNetworkModel':
        """使用 PyTorch 加载模型状态字典。"""
        # 确定加载设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载字典
        checkpoint = torch.load(file_path, map_location=device) # map_location 确保在不同设备上也能加载

        # 从字典中获取参数
        input_dim = checkpoint['input_dim']
        hidden_dim = checkpoint['hidden_dim']
        output_dim = checkpoint['output_dim']
        learning_rate = checkpoint.get('learning_rate', 0.001) # 如果没保存学习率，使用默认值

        # 创建模型实例
        instance = cls(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       output_dim=output_dim,
                       learning_rate=learning_rate)

        # 加载模型状态字典
        instance.load_state_dict(checkpoint['model_state_dict'])
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint and hasattr(instance, 'optimizer'):
            try:
                instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                 print(f"警告：加载优化器状态时出错: {e}")


        instance.to(device) # 确保模型在正确的设备上
        return instance

    def get_weights(self):
        """获取模型的状态字典 (包含权重和偏置)。"""
        return self.state_dict()

    def set_weights(self, state_dict):
        """加载模型的状态字典 (包含权重和偏置)。"""
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"错误：设置模型权重时出错（可能是结构或键不匹配）：{e}")
        except Exception as e:
            print(f"设置模型权重时发生未知错误：{e}")
