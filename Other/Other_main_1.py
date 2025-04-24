import json
import math
import os
import random

import joblib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from Other.GobleD import GobleD


def is_number_or_pure_digit_IN_Other_main(value):
	"""检查值是否为数字或纯数字字符串。"""
	return isinstance(value, (int, float)) or (
			isinstance(value, str) and value.isdigit()
	)


def get_all_files_IN_Other_main(directory):
	"""递归获取目录中的所有文件路径。"""
	return [
		os.path.join(root, file)
		for root, _, files in os.walk(directory)
		for file in files
	]


def get_decimal_places_IN_Other_main(num):
	"""获取数字的小数位数。"""
	return len(str(num).split(".")[1]) if "." in str(num) else 0


def check_missing_values_IN_Other_main(data) -> None:
	"""检查 DataFrame 中的空值，并打印包含空值的列名和具体位置。"""
	missing_columns = data.columns[data.isna().any()].tolist()

	if missing_columns:
		print(f"空值在列：{missing_columns}")
		for row, col in zip(*np.where(data.isna())):
			print(f"空值位于：行 {row}，列 '{data.columns[col]}'")
	else:
		print("没有空值")


def save_array_to_csv_IN_Other_main(array, filename, index=False, header=True):
	"""将数组或数据框保存到 CSV 文件。"""
	if isinstance(array, pd.DataFrame):
		array.to_csv(filename, index=index, header=header, encoding="utf_8_sig")
	else:
		pd.DataFrame(array).to_csv(
			filename, index=index, header=False, encoding="utf_8_sig"
		)


def train_test_split_IN_Other_main(X_ML_input, y_ML_output, random_state=None):
	"""拆分数据集为训练集和测试集。"""
	return train_test_split(
		X_ML_input, y_ML_output, test_size=0.1, random_state=random_state, shuffle=True
	)


def get_variable_name_IN_Other_main(var, scope):
	"""
	获取变量名
	:param var: 变量
	:param scope: 作用域字典（如 locals() 或 globals()）
	:return: 变量名列表
	"""
	return [name for name in scope if scope[name] is var]


def XRD_save_IN_Other_main(D_2theta, D_value, file_path, test=False):
	# 保存 XRD 数据到文本文件，并支持生成随机测试数据。

	if test:
		num_points = 100
		value_range = (1000, 5000)
		# 生成随机测试数据
		D_2theta = [round(5 + i * 0.1, 4) for i in range(num_points)]
		D_value = [round(random.uniform(value_range[0], value_range[1]), 4) for _ in range(num_points)]

	if D_2theta is None or D_value is None or len(D_2theta) != len(D_value):
		raise ValueError("D_2theta 和 D_value 必须是长度相等")

	# 保存到文件
	with open(file_path, "w") as f:
		for theta, value in zip(D_2theta, D_value):
			f.write(f"{theta:.4f} {value:.4f}\n")
		print(f"XRD 数据保存到 {file_path} 中")


def XRD_read_IN_Other_main(file_path):
	# 从文本文件读取 XRD 数据
	if os.path.exists(file_path):
		with open(file_path, "r") as f:
			lines = f.readlines()
			lines = [line.strip().split() for line in lines]
			Data_fin = np.array(lines, dtype=float)
			return Data_fin
	else:
		print("file not exist")


def interp_XRD_IN_Other_main(XRD_data, interval):
	# 根据 interval 重新对 XRD_data 插值，例如 每5插值至每0.05
	x = XRD_data[:, 0]
	y = XRD_data[:, 1]
	interpolation_function = interp1d(x, y, kind='linear', fill_value="extrapolate")
	x_new = np.arange(x[0], x[-1] + interval, interval)
	y_new = interpolation_function(x_new)
	XRD_data_new = np.column_stack((x_new, y_new))
	return XRD_data_new


def get_all_in_Merge_data_for_ML(Need_input_in_ML, Need_output_in_ML):
	Merge_data_for_ML_MinMax = pd.read_csv(GobleD().Merge_data_for_ML)
	X_ML_input, y_ML_output = get_InOut_form_NeedCol_IN_Other_main(
		Merge_data_for_ML_MinMax, Need_input_in_ML, Need_output_in_ML
	)
	return X_ML_input, y_ML_output


def get_InOut_form_NeedCol_IN_Other_main(Data, Need_input_in_ML, Need_output_in_ML):
	"""从数据框中获取所需的输入和输出列。"""
	X_ML_input = Data.loc[:, Need_input_in_ML]
	y_ML_output = Data.loc[:, Need_output_in_ML]
	if y_ML_output.ndim == 2 and y_ML_output.shape[1] == 1:
		y_ML_output = y_ML_output.values.ravel()  # 转为一维数组
	return X_ML_input, y_ML_output


def run_sklearn_model_IN_Other_main(model, X_train, X_test=None):
	"""运行 sklearn 模型并返回训练和测试集的预测结果。"""
	y_train_pred = model.predict(X_train)
	if X_test is not None:
		y_test_pred = model.predict(X_test)
		return y_train_pred, y_test_pred
	else:
		return y_train_pred


def pd_2_np_IN_Other_main(data):
	"""将 pandas 数据结构转换为 NumPy 数组。"""
	if isinstance(data, pd.DataFrame):
		return data.values
	elif isinstance(data, pd.Series):
		return data.to_numpy()
	else:
		return np.array(data)


def Fixed_random_seed_IN_Other_main(seed=33):
	"""设置固定的随机种子，确保结果可复现。"""
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def sns_my_yeey_fig_IN_Other_main(g, data, where_y, where_yerr):
	# 主动在 catplot 添加误差棒
	# 获取每个条形的位置
	for bar, y, yerr in zip(g.ax.patches, data[where_y], data[where_yerr]):
		x_position = bar.get_x() + bar.get_width() / 2  # 获取条形的中心位置
		y_position = bar.get_height()  # 获取条形的高度
		plt.errorbar(
			x=x_position, y=y_position, yerr=yerr, fmt="none", color="black", capsize=5
		)


def plt_savefig_IN_Other_main(
		save_name, save_path_dir=GobleD().Fig_ALL_Save, dpi_svg=300, dpi_jpg=400
):
	# 保存图片
	plt.savefig(
		os.path.join(save_path_dir, f"{save_name}.jpg"),
		dpi=dpi_jpg,
		format="jpg",
		transparent=True,
		bbox_inches="tight",
	)
	plt.savefig(
		os.path.join(save_path_dir, f"{save_name}.svg"),
		dpi=dpi_svg,
		format="svg",
		transparent=True,
		bbox_inches="tight",
	)


def get_k_Quantile_IN_Other_main(k_data, quantile_need=None):
	# 计算前 20% 和前 40% 的分位值
	if quantile_need is None:
		quantile_need = [0, 0.2, 0.6, 1]
	need_reverse = [1 - temp for temp in quantile_need]
	# 计算前 20% 和前 40% 的分位值
	percent_0 = k_data.quantile(need_reverse[0])
	percent_20 = k_data.quantile(need_reverse[1])  # 前 20% 的数值
	percent_60 = k_data.quantile(need_reverse[2])
	percent_100 = k_data.quantile(need_reverse[3])
	quantile = [percent_0, percent_20, percent_60, percent_100]
	distance = [
		percent_0 - percent_20,
		percent_20 - percent_60,
		percent_60 - percent_100,
	]
	return [quantile, distance]


def Fig_my_init_IN_Other_main(latex=None, sns_do=None):
	if latex is None:
		latex = False
	if sns_do is None:
		sns_do = True
	Fixed_random_seed_IN_Other_main()
	font_set = "Arial"
	config = {
		"axes.unicode_minus": False,
		# "svg.fonttype": "none",  # 这个开了svg放到word会坏，默认用路径描绘
		"figure.autolayout": True,
		"font.family": font_set,
		"font.size": 16,
	}
	if latex:
		plt.rcParams["text.usetex"] = True  # 启用 LaTeX 渲染
	plt.rcParams.update(config)
	if sns_do:
		sns.set_theme(style="whitegrid", font=font_set, rc=config)


import pandas as pd


def rename_cols_IN_Other_main(df_need_rename):
	"""
	对列名进行重命名并返回新 DataFrame 或 Series。
	"""

	def rename_columns(df, properties, properties_latex):
		"""
		根据指定的原始列名和目标列名重命名 DataFrame 的列，或者如果是 Series 重命名其索引。
		"""
		if isinstance(df, pd.Series):
			# 如果是 Series，则重命名索引
			rename_map = {
				need_change_label: properties_latex[index]
				for index, need_change_label in enumerate(properties)
				if need_change_label in df.index
			}
			return df.rename(index=rename_map)
		else:
			# 如果是 DataFrame，则重命名列
			rename_map = {
				need_change_label: properties_latex[index]
				for index, need_change_label in enumerate(properties)
				if need_change_label in df.columns
			}
			return df.rename(columns=rename_map)

	# 判断传入对象类型
	if isinstance(df_need_rename, pd.Series):
		# 对 Series 进行深拷贝以避免修改原 Series
		df_renamed = df_need_rename.copy()
	else:
		# 对 DataFrame 进行深拷贝以避免修改原 DataFrame
		df_renamed = df_need_rename.copy()

	# 调用重命名逻辑
	df_renamed = rename_columns(
		df_renamed,
		GobleD().pollutant_chemistry_properties,
		GobleD().pollutant_chemistry_properties_latex,
	)
	df_renamed = rename_columns(
		df_renamed,
		GobleD().oxidant_chemistry_properties,
		GobleD().oxidant_chemistry_properties_latex,
	)
	df_renamed = rename_columns(
		df_renamed,
		GobleD().needRead_column_Main,
		GobleD().needRead_column_Main_latex,
	)
	df_renamed = rename_columns(
		df_renamed,
		["pollutant classify index"],
		["pollutant classify"]
	)

	return df_renamed


def evaluate_model_2_IN_Other_main(
		y_real, y_pre, print_result=1, model_name=None
):
	# 计算 Train 集和 Test 集的 R² 和 RMSE
	def calculate_metrics(y_true, y_pred):
		r2 = r2_score(y_true, y_pred)
		rmse = math.sqrt(mean_squared_error(y_true, y_pred))
		return r2, rmse

	r2, rmse = calculate_metrics(y_real, y_pre)

	# 构建结果 DataFrame
	result = pd.DataFrame({
		"R2": [r2],
		"RMSE": [rmse],
		"NAME": [model_name]
	}, index=["Overall"])

	# 是否打印结果
	if print_result == 1:
		print(f"Evaluation Results for Model: {model_name}")
		print(result)

	return result


def evaluate_model_IN_Other_main(
		y_train=None, y_test=None, y_train_pred=None, y_test_pred=None, print_result=1, model_name=None
):
	# 设置默认值
	default_value = [1, 1]
	y_train = y_train if y_train is not None else default_value
	y_test = y_test if y_test is not None else default_value
	y_train_pred = y_train_pred if y_train_pred is not None else default_value
	y_test_pred = y_test_pred if y_test_pred is not None else default_value

	# 处理输入数据（list, np 或 pd 的形式）
	def to_array(data):
		if isinstance(data, list):
			return np.array(data)
		elif isinstance(data, pd.DataFrame):
			return data.values
		return data

	# 转换数据为合适的格式
	y_train = to_array(y_train)
	y_test = to_array(y_test)
	y_train_pred = to_array(y_train_pred)
	y_test_pred = to_array(y_test_pred)

	# 计算 Train 集和 Test 集的 R² 和 RMSE
	def calculate_metrics(y_true, y_pred):
		r2 = r2_score(y_true, y_pred)
		rmse = math.sqrt(mean_squared_error(y_true, y_pred))
		return r2, rmse

	# Train 集的指标
	r2_train, rmse_train = calculate_metrics(y_train, y_train_pred)

	# Test 集的指标
	r2_test, rmse_test = calculate_metrics(y_test, y_test_pred)

	# 计算整体(All)集的 R² 和 RMSE
	y_all = np.concatenate([y_train, y_test])
	y_all_pred = np.concatenate([y_train_pred, y_test_pred])
	r2_all, rmse_all = calculate_metrics(y_all, y_all_pred)

	# 构建结果 DataFrame
	result = pd.DataFrame({
		"R2": [r2_train, r2_test, r2_all],
		"RMSE": [rmse_train, rmse_test, rmse_all],
		"NAME": [model_name, model_name, model_name]
	}, index=["Train", "Test", "Overall"])

	# 是否打印结果
	if print_result == 1:
		print(f"Evaluation Results for Model: {model_name}")
		print(result)

	return result


def entropy_weight_method_IN_Other_main(X):
	"""
	熵权法：基于给定的指标矩阵X计算每个指标的权重
	X: 原始数据矩阵，m x n (m为样本数，n为指标数)
	"""
	# 第一步：标准化
	# 对每列进行标准化，假设矩阵X的行表示样本，列表示指标
	X_norm = X / np.sum(X, axis=0)

	# 避免全为0的列（如果有全0列）
	# if np.any(np.sum(X_norm, axis=0) == 0):
	# 	raise ValueError("数据中存在全0列，无法进行熵权法处理")

	# 第二步：避免对数中零值
	X_norm = np.clip(
		X_norm, 1e-9, None
	)  # 将数据中所有小于1e-9的值替换为1e-9，以防对数运算出错

	# 第三步：计算每个指标的熵值
	m, n = X_norm.shape
	k = 1 / np.log(m)

	# 计算熵值
	E = -k * np.sum(X_norm * np.log(X_norm), axis=0)  # 使用标准化后的X_norm直接计算熵

	# 第四步：计算各个指标的权重
	weights = (1 - E) / np.sum(1 - E)

	return weights


def drop_columns_IN_Other_main(data_df, need_drop_columns):
	"""从多个 DataFrame 中删除指定列，并返回新的 DataFrame 列表。"""
	updated_df = data_df.copy()
	for col in need_drop_columns:
		if col in data_df.columns:
			updated_df.drop(col, axis=1, inplace=True)
	return updated_df


def get_ML_data_by_Need_IN_Other_main(
		Need_input_in_ML, Need_output_in_ML, random_state=82, MinMax=1,
		XRD_Data_encoded=None
):
	"""获取需要的 ML 数据并进行归一化处理。"""
	Merge_data_for_ML_MinMax = pd.read_csv(GobleD().Merge_data_for_ML)
	X_ML_input, y_ML_output = get_InOut_form_NeedCol_IN_Other_main(
		Merge_data_for_ML_MinMax, Need_input_in_ML, Need_output_in_ML
	)
	if XRD_Data_encoded is not None:
		X_ML_input = X_ML_input.drop(GobleD().theta_labels, axis=1)
		temp = pd.DataFrame(XRD_Data_encoded)
		num_columns = temp.shape[1]  # 获取列数
		temp.columns = [f"XRDencode_{i}" for i in range(num_columns)]  # 创建以 'encode_' 开头的列名
		X_ML_input = pd.concat([X_ML_input, temp], axis=1)

	X_train, X_test, y_train, y_test = train_test_split_IN_Other_main(
		X_ML_input, y_ML_output, random_state=random_state
	)

	MinMaxScaler_for_X = MinMaxScaler()
	MinMaxScaler_for_X.fit(X_train)
	if MinMax:
		X_train_MinMax = pd.DataFrame(
			MinMaxScaler_for_X.transform(X_train), columns=X_ML_input.columns
		)
		X_test_MinMax = pd.DataFrame(
			MinMaxScaler_for_X.transform(X_test), columns=X_ML_input.columns
		)
	else:
		X_train_MinMax = X_train
		X_test_MinMax = X_test

	return X_train_MinMax, X_test_MinMax, y_train, y_test, MinMaxScaler_for_X


def pca_set_encode_IN_Other_main(data, n_components, random_state=None, print_result=1):
	# 初始化PCA类，指定降至 n 维
	pca = PCA(n_components=n_components, random_state=random_state)
	data_encoded = pca.fit_transform(data)  # 降维
	# 从降维数据重建
	data_recovered = pca.inverse_transform(data_encoded)
	if print_result:
		print(f'PCA R2: {r2_score(data, data_recovered)}')

	return data_encoded, data_recovered


def Get_XRD_data_orifin_IN_Other_main(database=None):
	if database is None:
		database = pd.read_csv(GobleD().Merge_data_for_ML)
	Need_input_in_ML = GobleD().theta_labels
	Data_load = database[Need_input_in_ML]
	return Data_load


def get_old_best_params_IN_Other_main(json_outPath):
	"""从 JSON 文件中读取旧的最佳超参数，如果文件不存在或为空则返回空字典。"""
	if os.path.exists(json_outPath):
		try:
			json_data_my = (
				{}
				if os.path.getsize(json_outPath) == 0
				else json.load(open(json_outPath, "r", encoding="utf-8"))
			)
		except Exception as e:
			print("读取 原超参数 文件出错, 初始化超参数为空")
			json_data_my = {}
	else:
		json_data_my = {}
	return json_data_my


def save_model_IN_Other_main(model, save_path):
	"""保存训练好的模型到指定路径。"""
	joblib.dump(model, save_path)
	print(f"模型保存到 {save_path}")


def build_list_of_models_NEED_run_IN_Other_main(RUN_FLAGS):
	"""根据运行标志构建需要运行的模型名称列表。"""
	return [
		getattr(GobleD(), f"{model}_modle_name")
		for model, should_run in RUN_FLAGS.items()
		if should_run
	]


def sample_data_IN_Other_main():
	# 创建一个包含1000个样本的数据，假设包含温度、湿度、电压三个特征
	num_samples = 1000
	temperature = np.random.rand(num_samples) * 1  # 温度（范围 0-30度）
	humidity = np.random.rand(num_samples) * 1  # 湿度（范围 0-100%）
	voltage = np.random.rand(num_samples) * 1  # 电压（范围 0-5V）
	degradation_rate = temperature + humidity + voltage  # 降解速率（范围 0-0.1）

	# 将其整合到DataFrame中
	X_ML_input = pd.DataFrame({
		'temperature': temperature,
		'humidity': humidity,
		'voltage': voltage
	})
	y_ML_output = degradation_rate  # 输出值：降解速率

	X_train, X_test, y_train, y_test = train_test_split_IN_Other_main(X_ML_input, y_ML_output)
	return X_train, X_test, y_train, y_test
