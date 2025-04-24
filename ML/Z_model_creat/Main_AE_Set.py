import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader

from Other.Other_main_1 import Fixed_random_seed_IN_Other_main


class AE_mySet:
	def __init__(self, model_set_dict=None):
		Fixed_random_seed_IN_Other_main()

		# 设备设置：检查是否有可用的 GPU
		device_set = model_set_dict.get('device_set', "cuda")
		self.device = torch.device(device_set)

		self.encoding_dim = model_set_dict.get('encoding_dim', 100)
		self.hidden_layers = model_set_dict.get('hidden_layers', [256, 128, 64])
		self.learning_rate = model_set_dict.get('learning_rate', 0.001)
		self.num_epochs = model_set_dict.get('num_epochs', 100)
		self.batch_size = model_set_dict.get('batch_size', 300)
		self.print_log = model_set_dict.get('print_log', 1)
		self.print_each_epoch = model_set_dict.get('print_each_epoch', 1)
		self.print_each_epoch_test = model_set_dict.get('print_each_epoch', 1)

	def _build_model(self):
		encoder_layers = []
		prev_layer_size = self.input_dim
		for hidden_size in self.hidden_layers:
			encoder_layers.append(nn.Linear(prev_layer_size, hidden_size))
			encoder_layers.append(nn.LeakyReLU(negative_slope=0.01))
			prev_layer_size = hidden_size
		encoder_layers.append(nn.Linear(prev_layer_size, self.encoding_dim))
		self.encoder = nn.Sequential(*encoder_layers)

		decoder_layers = []
		prev_layer_size = self.encoding_dim
		for hidden_size in reversed(self.hidden_layers):
			decoder_layers.append(nn.Linear(prev_layer_size, hidden_size))
			decoder_layers.append(nn.LeakyReLU(negative_slope=0.01))
			prev_layer_size = hidden_size
		decoder_layers.append(nn.Linear(prev_layer_size, self.input_dim))
		decoder_layers.append(nn.Tanh())
		self.decoder = nn.Sequential(*decoder_layers)

		# 把模型移到 GPU 或 CPU
		return nn.ModuleDict({'encoder': self.encoder.to(self.device), 'decoder': self.decoder.to(self.device)})

	def fit(self, Data_load):
		self.input_dim = Data_load.shape[1]
		self.model = self._build_model()

		# 将数据转换为 Tensor 并移到 GPU 或 CPU
		Data_load = self.to_np_AE(Data_load)
		Data_load = torch.tensor(Data_load.copy(), dtype=torch.float32).to(self.device)

		dataset = TensorDataset(Data_load)
		data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

		criterion = nn.MSELoss()
		optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

		for epoch in range(self.num_epochs):
			total_loss = 0.0
			for batch_data in data_loader:
				batch_XRD = batch_data[0]  # 从 DataLoader 中提取 batch 数据
				optimizer.zero_grad()
				encoded = self.model['encoder'](batch_XRD)
				decoded = self.model['decoder'](encoded)
				loss = criterion(decoded, batch_XRD)
				loss.backward()
				optimizer.step()
				total_loss += loss.item()

			if self.print_each_epoch == 1:
				Data_encoded = self.predict(Data_load)
				Data_recovered = self.recover(Data_encoded)

				Data_load_np = Data_load.cpu().numpy() if isinstance(Data_load, torch.Tensor) else Data_load
				Data_recovered_np = np.array(Data_recovered)

				R2_result = r2_score(Data_load_np, Data_recovered_np)
				print(f"Epoch [{epoch + 1}/{self.num_epochs}], R²: {R2_result:.4f}, Loss: {total_loss / len(data_loader):.4f}")

		if self.print_log == 1:
			Data_recovered = self.recover(Data_encoded)
			Data_recovered_np = np.array(Data_recovered)
			Data_load_np = Data_load.cpu().numpy()
			R2_result = r2_score(Data_load_np, Data_recovered_np)
			print(f"=========================")
			print(f"Final R²: {R2_result:.4f}")
			print(f"=========================")

	def predict(self, Data_load):
		Data_load = self.to_np_AE(Data_load)
		Data_load_tensor = torch.tensor(Data_load, dtype=torch.float32).to(self.device)

		with torch.no_grad():
			Data_encoded = self.model['encoder'](Data_load_tensor)
		Data_encoded = Data_encoded.cpu().numpy()  # 返回到 CPU 并转换为 NumPy 数组

		return Data_encoded

	def recover(self, encoded_data):
		encoded_data = torch.tensor(encoded_data, dtype=torch.float32).to(self.device)
		with torch.no_grad():
			recovered_data = self.model['decoder'](encoded_data)
		Data_recovered = recovered_data.cpu().numpy()
		Data_recovered = np.where(Data_recovered < 0.03, 0, Data_recovered)
		Data_recovered = np.where(Data_recovered > 1, 1, Data_recovered)
		return Data_recovered  # 返回到 CPU

	def to_np_AE(self, Data_load):
		if isinstance(Data_load, torch.Tensor):
			return Data_load.cpu().numpy()  # 转换为 NumPy 数组
		elif isinstance(Data_load, list):
			return np.array(Data_load)
		elif isinstance(Data_load, pd.DataFrame):
			return Data_load.values
		return Data_load


if __name__ == '__main__':
	# %% 读取数据
	# 训练模型，不区分测试和训练
	XRD_Data_load = pd.DataFrame([
		[1, 1, 2, 2],
		[2, 2, 3, 3],
		[4, 4, 5, 5],
		[3, 3, 4, 4],
	])
	# 创建自编码器对象
	model = AE_mySet({
		'encoding_dim': 2, 'hidden_layers': [256, 128, 64],
		'num_epochs': 20, 'batch_size': 500,
		'print_each_epoch': 0, 'print_log': 0, 'print_each_epoch_test': 0,
	})

	# 训练模型，不区分测试和训练
	model.fit(XRD_Data_load)

	# %
	temp = pd.DataFrame([
		[5, 5, 6, 6],
		[6, 6, 7, 7],
		[6, 6, 8, 8],
	])
	Data_encoded = model.predict(temp)
	print(Data_encoded)

	temp = pd.DataFrame([
		[6, 6, 8, 8],
		[6, 6, 7, 7],
		[5, 5, 6, 6],
	])
	Data_encoded = model.predict(temp)
	print(Data_encoded)
# Data_recovered = model.recover(Data_encoded)
#
# result = evaluate_model_2_IN_Other_main(XRD_Data_load, Data_recovered, print_result=1, model_name=None)
