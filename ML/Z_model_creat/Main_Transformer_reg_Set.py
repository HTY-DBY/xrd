import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Other.Other_main_1 import Fixed_random_seed_IN_Other_main, pd_2_np_IN_Other_main, sample_data_IN_Other_main


# 定义 Transformer 回归模型
class TransformerRegressor(nn.Module):
	def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
		"""
		初始化 Transformer 模型
		input_dim：输入数据的特征数量
		model_dim：模型的维度（例如，词嵌入的维度）
		num_heads：自注意力机制中的头数
		num_layers：Transformer 层数（包括编码器和解码器）
		dropout：dropout 比例，用于防止过拟合
		"""
		super(TransformerRegressor, self).__init__()
		self.input_dim = input_dim
		self.embedding = nn.Linear(input_dim, model_dim)  # 输入特征通过嵌入层
		self.transformer = nn.Transformer(
			d_model=model_dim,
			nhead=num_heads,
			num_encoder_layers=num_layers,
			num_decoder_layers=num_layers,
			dropout=dropout,
			batch_first=True)  # 创建 Transformer 模型
		self.fc = nn.Linear(model_dim, 1)  # 输出一个预测值

	def forward(self, x):
		"""
		定义模型的前向传播过程
		x：输入数据，形状为 [batch_size, seq_len, input_dim]
		"""
		batch_size, seq_len, _ = x.shape  # 动态获取序列的长度，虽然它在这里没有被单独使用
		x = self.embedding(x)  # 将输入数据映射到模型维度
		tgt = torch.zeros_like(x)  # 使用全零的向量作为解码器目标
		x = self.transformer(x, tgt)  # Transformer 编码和解码
		output = self.fc(x[:, -1, :])  # 获取最后一个时间步的输出（即序列的最后一步）
		return output


# Transformer 模型设置类
class Transformer_reg_mySet:
	def __init__(self, model_set_dict=None):
		"""
		初始化模型配置
		model_set_dict：一个字典，包含模型配置参数
		"""
		Fixed_random_seed_IN_Other_main()
		self.num_heads = model_set_dict.get('num_heads', 4)  # 自注意力的头数
		self.model_dim = model_set_dict.get('model_dim', 32)  # 模型维度
		self.num_layers = model_set_dict.get('num_layers', 2)  # Transformer 层数
		self.batch_size = model_set_dict.get('batch_size', 32)  # 批处理大小
		self.num_epochs = model_set_dict.get('num_epochs', 100)  # 训练轮数
		self.device = model_set_dict.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 使用的设备
		self.print_log = model_set_dict.get('print_log', 1)  # 是否打印日志
		self.print_each_epoch = model_set_dict.get('print_each_epoch', 1)  # 是否在每个 epoch 打印日志
		self.print_each_epoch_test = model_set_dict.get('print_each_epoch', 1)  # 是否打印测试集日志

	def predict(self, X_test):
		"""
		预测函数，用于测试集的预测
		"""
		self.model.eval()  # 进入评估模式
		with torch.no_grad():
			# 根据 X_test 的类型将其转换为 NumPy 数组
			if isinstance(X_test, pd.DataFrame):
				X_array = pd_2_np_IN_Other_main(X_test)
			elif isinstance(X_test, (np.ndarray, torch.Tensor)):
				X_array = X_test if isinstance(X_test, np.ndarray) else X_test.cpu().numpy()
			X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device).view(X_array.shape[0], 1, -1)  # 转换为 Tensor 并调整维度
			y_test_pred = self.model(X_tensor).cpu().numpy()  # 通过模型进行预测，确保结果在 CPU 上
		return y_test_pred

	def fit(self, X_train, y_train, X_test=None, y_test=None):
		"""
		模型训练函数
		"""

		# 将训练集转换为 Tensor
		X_train, y_train = [torch.tensor(pd_2_np_IN_Other_main(data), dtype=torch.float32).to(self.device)
							for data in [X_train, y_train]]

		data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)  # 数据加载器
		input_dim = X_train.shape[1]  # 输入数据的维度
		self.model = TransformerRegressor(input_dim=input_dim, model_dim=self.model_dim,
										  num_heads=self.num_heads, num_layers=self.num_layers).to(self.device)

		criterion = nn.MSELoss()  # 定义损失函数
		optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Adam 优化器

		if self.print_log == 1:
			print(f"Using device: {self.device}")
			print(f"----------------------------------------")

		# 训练过程
		for epoch in range(self.num_epochs):
			self.model.train()  # 进入训练模式
			total_loss = 0.0  # 累计损失
			for x_batch, y_batch in data_loader:
				x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
				x_batch = x_batch.view(x_batch.size(0), 1, -1)  # 调整维度
				optimizer.zero_grad()
				preds = self.model(x_batch).squeeze()  # 获取模型预测结果
				loss = criterion(preds, y_batch)  # 计算损失
				loss.backward()  # 反向传播
				optimizer.step()  # 更新模型参数
				total_loss += loss.item()

			# 每个 epoch 打印训练信息
			if self.print_each_epoch == 1:
				y_train_pred = self.predict(X_train.cpu())  # 预测训练集
				y_train_pred_in_epoch = r2_score(y_train.cpu().numpy(), y_train_pred)  # 计算 R²
				print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train R²: {y_train_pred_in_epoch:.4f}, Loss: {total_loss / len(data_loader):.4f}")

			# 每个 epoch 打印测试集信息
			if self.print_each_epoch_test == 1:
				if X_test is None or y_test is None:
					X_test, y_test = X_train.cpu().numpy(), y_train.cpu().numpy()
				y_test_pred = self.predict(X_test)
				y_test_pred_in_epoch = r2_score(y_test, y_test_pred)
				print(f"---> Epoch [{epoch + 1}/{self.num_epochs}], Test R²: {y_test_pred_in_epoch:.4f}")
				print(f"----------------------------------------")

		# 打印最终结果
		if self.print_log == 1:
			y_train_pred = self.predict(X_train.cpu().numpy())  # 获取最终训练集的预测
			train_r2_in_ALL = r2_score(y_train.cpu().numpy(), y_train_pred)  # 计算 R²
			print(f"=========================")
			print(f"Final Train R²: {train_r2_in_ALL:.4f}")
			print(f"=========================")

	def get_model(self):
		return self.model


if __name__ == '__main__':
	# %% 读取与准备数据
	X_train, X_test, y_train, y_test = sample_data_IN_Other_main()  # 从外部方法加载数据

	# 初始化 Transformer 模型
	Transformer_make = Transformer_reg_mySet({
		'num_heads': 4, 'model_dim': 28, 'num_layers': 2, 'num_epochs': 10,
		'print_each_epoch': 1, 'print_log': 1, 'print_each_epoch_test': 1,
	})

	Transformer_make.fit(X_train, y_train, X_test, y_test)  # 训练模型

	y_test_pred = Transformer_make.predict(X_test)  # 预测测试集
	results = r2_score(y_test, y_test_pred)  # 计算最终 R²
	print(f"Final Test R²: {results:.4f}")  # 输出最终测试集的 R² 分数
