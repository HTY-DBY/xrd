import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from Other.GobleD import GobleD
from Other.Other_main_1 import pd_2_np_IN_Other_main, get_ML_data_by_Need_IN_Other_main, Fixed_random_seed_IN_Other_main


class CNN1D_Set_Basic_Form(nn.Module):
	"""定义 1D 卷积神经网络结构。"""

	def __init__(self, input_channels, data_1d_len, kernel_size, out_channels):
		super(CNN1D_Set_Basic_Form, self).__init__()
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.conv1 = nn.Conv1d(input_channels, out_channels[0], kernel_size[0], stride=1, padding=1)
		self.conv2 = nn.Conv1d(out_channels[0], out_channels[1], kernel_size[1], stride=1, padding=1)
		self.conv3 = nn.Conv1d(out_channels[1], out_channels[2], kernel_size[2], stride=1, padding=1)

		# 计算全连接层的输入大小
		self.fin_fc_width = self._get_fc_input_size(data_1d_len, kernel_size, out_channels)
		self.fc = nn.Linear(out_channels[2] * self.fin_fc_width, 1)
		self.model = None

	def _get_fc_input_size(self, data_1d_len, kernel_size, out_channels):
		# 计算每一层输出的长度
		length = data_1d_len

		# 第一层卷积
		length = (length + 2 * 1 - kernel_size[0]) // 1 + 1  # padding=1
		length = length // 2  # MaxPool1d (kernel_size=2, stride=2)

		# 第二层卷积
		length = (length + 2 * 1 - kernel_size[1]) // 1 + 1  # padding=1
		length = length // 2  # MaxPool1d (kernel_size=2, stride=2)

		# 第三层卷积
		length = (length + 2 * 1 - kernel_size[2]) // 1 + 1  # padding=1
		length = length // 2  # MaxPool1d (kernel_size=2, stride=2)

		return length

	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = self.pool(x)
		x = torch.relu(self.conv2(x))
		x = self.pool(x)
		x = torch.relu(self.conv3(x))
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		return self.fc(x)


class CNN_mySet:
	def __init__(self, CNN_set_dict=None):
		Fixed_random_seed_IN_Other_main()
		self.kernel_size = CNN_set_dict.get('kernel_size', [3, 3, 3])
		self.out_channels = CNN_set_dict.get('out_channels', [16, 24, 32])
		self.num_epochs = CNN_set_dict.get('num_epochs', 10)
		self.device = CNN_set_dict.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

		self.print_log = CNN_set_dict.get('print_log', 1)
		self.print_each_epoch = CNN_set_dict.get('print_each_epoch', 1)
		self.print_each_epoch_test = CNN_set_dict.get('print_each_epoch_test', 1)

	def predict(self, X_test):
		# 数据转换为 Tensor，并调整维度
		self.model.eval()
		with torch.no_grad():
			# 判断传入的数据类型
			if isinstance(X_test, pd.DataFrame):
				X_array = pd_2_np_IN_Other_main(X_test)  # 将 DataFrame 转换为 numpy 数组
			elif isinstance(X_test, np.ndarray):
				X_array = X_test  # 如果已经是 numpy 数组，则直接使用
			elif isinstance(X_test, torch.Tensor):
				X_array = X_test.cpu().numpy()  # 如果是 Tensor，则转换为 numpy 数组
			else:
				raise ValueError("Unsupported input type: Expected pd.DataFrame, np.ndarray, or torch.Tensor")

			# 确保形状正确并转换为 Tensor
			X_tensor = torch.tensor(X_array, dtype=torch.float32).to(self.device).view(X_array.shape[0], 1, -1)
			y_test_pred = self.model(X_tensor).cpu().numpy()

		return y_test_pred

	def fit(self, X_train, y_train, X_test=None, y_test=None):
		"""主函数：执行 CNN 模型训练与评估。"""

		X_train, y_train = [torch.tensor(pd_2_np_IN_Other_main(data), dtype=torch.float32).to(self.device) for data in [X_train, y_train]]
		X_train = X_train.view(X_train.size(0), 1, -1)
		y_train = y_train.view(-1, 1)

		self.model = CNN1D_Set_Basic_Form(1, X_train.shape[2], self.kernel_size, self.out_channels).to(self.device)

		if self.print_log == 1:
			print(f"Using device: {self.device}")
			print(f"----------------------------------------")

		data_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)

		criterion = nn.MSELoss()
		optimizer = optim.Adam(self.model.parameters(), lr=0.001)

		for epoch in range(self.num_epochs):
			self.model.train()
			total_loss = 0.0
			for data, targets in data_loader:
				data, targets = data.to(self.device), targets.to(self.device)
				outputs = self.model(data)
				loss = criterion(outputs, targets)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				total_loss += loss.item()

			if self.print_each_epoch == 1:
				y_train_pred = self.predict(X_train.cpu())
				y_train_pred_in_epoch = r2_score(y_train.cpu().numpy(), y_train_pred)
				print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train R²: {y_train_pred_in_epoch:.4f}, Loss: {total_loss / len(data_loader):.4f}")
			if self.print_each_epoch_test == 1:
				if X_test is None or y_test is None:
					X_test, y_test = X_train.cpu().numpy(), y_train.cpu().numpy()
				y_test_pred = self.predict(X_test)
				y_test_pred_in_epoch = r2_score(y_test, y_test_pred)
				print(f"---> Epoch [{epoch + 1}/{self.num_epochs}], Test R²: {y_test_pred_in_epoch:.4f}")
				print(f"----------------------------------------")
		if self.print_log == 1:
			y_train_pred = self.predict(X_train.cpu().numpy())
			train_r2_in_ALL = r2_score(y_train.cpu().numpy(), y_train_pred)
			print(f"=========================")
			print(f"Final Train R²: {train_r2_in_ALL:.4f}")
			print(f"=========================")

	def get_modle(self):
		return self.model


if __name__ == '__main__':
	# 读取与准备数据
	# %%
	ML_outout_input = pd.read_csv(GobleD().Merge_data_for_ML)
	Need_input_in_ML = GobleD().ML_Input_need_in_database_1
	Need_output_in_ML = GobleD().ML_Output_need_in_database_1

	X_train, X_test, y_train, y_test, _ = get_ML_data_by_Need_IN_Other_main(Need_input_in_ML, Need_output_in_ML, MinMax=1)

	CNN_mySet_make = CNN_mySet({'kernel_size': [4, 8, 10], 'out_channels': [116, 197, 187], 'num_epochs': 30,
								'print_each_epoch': 1, 'print_each_epoch_test': 1, 'print_log': 1})
	CNN_mySet_make.fit(X_train, y_train)

	y_test_pred = CNN_mySet_make.predict(X_test)
	results = r2_score(y_test, y_test_pred)
	print(f"Final Test R²: {results:.4f}")
