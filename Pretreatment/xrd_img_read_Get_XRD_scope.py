import os

import numpy as np
import pandas as pd

from Other.GobleD import GobleD
from Other.Other_main_1 import get_all_files_IN_Other_main, is_number_or_pure_digit_IN_Other_main


def Get_XRD_scope_ori_dict():
	excel_Read_in_FillData_for_database = pd.read_csv(GobleD().FillData_for_database_path)
	XRD_scope_dict = {}
	for _, row in excel_Read_in_FillData_for_database.iterrows():
		photocatalysts_Ind = row['catalyst-Ind']
		# 根据 photocatalysts-Ind 生成键
		if is_number_or_pure_digit_IN_Other_main(photocatalysts_Ind) and photocatalysts_Ind != 0 and row['XRD'] == 1:
			key = f"{row['ref-Ind']}-{photocatalysts_Ind}"
			XRD_scope_dict[key] = [row['X-min'], row['X-max']]

	# 获取文件路径列表
	file_path_list = get_all_files_IN_Other_main(GobleD().XRD_IMG_save_path)
	file_name_list = {os.path.splitext(os.path.basename(path))[0] for path in file_path_list}
	XRD_keys_set = set(XRD_scope_dict.keys())

	# 比较两个列表，找到不一样的地方
	only_in_excel_names = file_name_list - XRD_keys_set  # 在 file_name_list 中但不在 XRD_keys_list 中
	only_in_XRD_keys = XRD_keys_set - file_name_list  # 在 XRD_keys_list 中但不在 file_name_list 中
	if only_in_excel_names == only_in_XRD_keys:
		# 确定数据库没冲突了
		return XRD_scope_dict
	else:
		print(f'XRD=1 的位置，excel数据集和XRD库有冲突')
		print(f'only_in_file_names: {only_in_excel_names}')
		print(f'only_in_XRD_keys: {only_in_XRD_keys}')


def Get_XRD_scope_ori_arg(XRD_ori_scope_list):
	arg_XRD_min = np.mean(np.array(list(XRD_ori_scope_list.values()))[:, 0])
	arg_XRD_max = np.mean(np.array(list(XRD_ori_scope_list.values()))[:, 1])
	### 这里我自定义了一个范围
	arg_XRD_round = [5, 90]
	arg_XRD = [arg_XRD_min, arg_XRD_max]

	return arg_XRD, arg_XRD_round


def Get_theta_scope_in_XRD():
	_, arg_XRD_round = Get_XRD_scope_ori_arg(Get_XRD_scope_ori_dict())
	theta_scope = np.arange(arg_XRD_round[0], arg_XRD_round[1] + GobleD().interval, GobleD().interval)
	return theta_scope


if __name__ == '__main__':
	XRD_ori_scope_list = Get_XRD_scope_ori_dict()
	arg_XRD, arg_XRD_round = Get_XRD_scope_ori_arg(XRD_ori_scope_list)
	theta_scope = Get_theta_scope_in_XRD()
