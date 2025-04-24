import os
import re

import numpy as np
import pandas as pd
from sklearn import preprocessing

from Other.GobleD import GobleD
from Other.Other_main_1 import save_array_to_csv_IN_Other_main
from Pretreatment.xrd_img_read import XRD_KuoZhan_data, save_to_single_excel_forXRD
from Pretreatment.xrd_img_read_Get_XRD_scope import Get_XRD_scope_ori_dict, Get_XRD_scope_ori_arg

two_theta_name = '2θ'
intensity_name = 'I(f)'


def read_xrd_data(filename):
	"""读取XRD数据文件，返回theta和I(f)值的列表"""
	with open(os.path.join(GobleD().XRD_PDF_save_path, filename), 'r') as file:
		data_txt = file.readlines()
	# 定义正则表达式模式，匹配2θ和I(f)值
	pattern = re.compile(r'\s*(\d+\.\d+)\s+\S+\s+(\d+\.\d+)')
	theta_values = []
	intensity_values = []
	for line in data_txt:
		match = pattern.match(line)
		if match:
			theta_values.append(float(match.group(1)))
			intensity_values.append(float(match.group(2)))
	return theta_values, intensity_values


def process_data(theta_values, intensity_values, interval):
	"""处理数据，返回处理后的theta和I(f)值"""
	num_decimal = str(interval)[::-1].find('.')
	# 创建DataFrame
	intensity_values_df_ori = pd.DataFrame({two_theta_name: theta_values, intensity_name: intensity_values})
	# 四舍五入到靠近0.05的倍数
	intensity_values_df_round = intensity_values_df_ori.copy()
	intensity_values_df_round[two_theta_name] = np.round(interval * np.round(intensity_values_df_round[two_theta_name] / interval), decimals=num_decimal + 1)
	# 使用线性插值来生成新的x值
	x_new = np.round(np.arange(min(intensity_values_df_round[two_theta_name]),
							   max(intensity_values_df_round[two_theta_name]) + interval, interval), decimals=num_decimal + 1)
	# 合并操作
	intensity_values_df_new = pd.merge(pd.DataFrame({two_theta_name: x_new}), intensity_values_df_round, on=two_theta_name, how='left').fillna(0)
	return intensity_values_df_new


def main_readXRD_inPDFtxt(filenames_list):
	# 主程序
	Fig_data_dict = {}  # 存储每个文件的 Fig_data_MinMaxScale 数据

	for filename in filenames_list:
		theta_values, intensity_values = read_xrd_data(filename)
		intensity_values_df_new = process_data(theta_values, intensity_values, GobleD().interval)
		file_name_without_extension, file_extension = os.path.splitext(filename)
		# 保存到字典中
		Fig_data_dict[file_name_without_extension] = intensity_values_df_new

	Fig_data_dict_extended = {}
	_, arg_XRD_round = Get_XRD_scope_ori_arg(Get_XRD_scope_ori_dict())
	for Fig_data_single_dict in Fig_data_dict:
		Fig_data_now_data = np.array(Fig_data_dict[Fig_data_single_dict])
		Fig_data_dict_extended[Fig_data_single_dict] = XRD_KuoZhan_data(Fig_data_now_data, arg_XRD_round, GobleD().interval)  # 扩展范围
		scaler = preprocessing.MinMaxScaler()
		temp_0 = Fig_data_dict_extended[Fig_data_single_dict]
		temp_1 = scaler.fit_transform(temp_0[:, 1].reshape(-1, 1))
		temp_0[:, 1] = temp_1.flatten()
		Fig_data_dict_extended[Fig_data_single_dict] = temp_0

	return Fig_data_dict_extended


# 导出合并后的结果为 Excel 文件
def save_to_single_excel_forXRD(Fig_data_dict, output_path):
	df_temp = pd.DataFrame()
	# 遍历字典，将每个文件的数据作为两列添加
	for filename, data in Fig_data_dict.items():
		# 将 X 和 Y 数据转换为 DataFrame，并使用文件名作为列前缀
		df = pd.DataFrame(data, columns=[f"{filename}-X", f"{filename}-Y"])
		df_temp = pd.concat([df_temp, df], axis=1)  # 横向拼接
	# 保存结果为 Excel 文件
	save_array_to_csv_IN_Other_main(df_temp, output_path)
	print(f"XRD pdf 读取 ok")
	print(f"保存至: {output_path}")


def MAIN_xrd_pdf_read(Use_Test_file_Ind=0):
	print(f"------ XRD pdf 读取")

	if Use_Test_file_Ind == 0:
		filenames_list = [filename for filename in os.listdir(GobleD().XRD_PDF_save_path)]
	else:
		filenames_list = [Test_file_Ind]
	Fig_data_dict = main_readXRD_inPDFtxt(filenames_list)
	save_to_single_excel_forXRD(Fig_data_dict, GobleD().xrd_pdf_read_excel_save_path)
	return Fig_data_dict


Test_file_Ind = f"PDF#00-002-0880.txt"

if __name__ == "__main__":
	Use_Test_file_Ind = 0

	Fig_data_dict = MAIN_xrd_pdf_read(Use_Test_file_Ind=Use_Test_file_Ind)
