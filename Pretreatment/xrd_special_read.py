import numpy as np
import pandas as pd

from Other.GobleD import GobleD
from Other.Other_main_1 import save_array_to_csv_IN_Other_main
from Pretreatment.xrd_img_read_Get_XRD_scope import Get_XRD_scope_ori_dict, Get_XRD_scope_ori_arg


def save_to_single_excel_forXRD(Fig_data_dict, output_path):
	df_temp = pd.DataFrame()
	# 遍历字典，将每个文件的数据作为两列添加
	for filename, data in Fig_data_dict.items():
		# 将 X 和 Y 数据转换为 DataFrame，并使用文件名作为列前缀
		df = pd.DataFrame(data, columns=[f"{filename}-X", f"{filename}-Y"])
		df_temp = pd.concat([df_temp, df], axis=1)  # 横向拼接
	# 保存结果为 Excel 文件
	save_array_to_csv_IN_Other_main(df_temp, output_path)
	print(f"XRD special 读取 ok")
	print(f"保存至: {output_path}")


def MAIN_xrd_special_read():
	print(f"------ XRD special 读取")

	_, arg_XRD_round = Get_XRD_scope_ori_arg(Get_XRD_scope_ori_dict())
	Fig_data_dict = {}

	X_need = np.arange(arg_XRD_round[0], arg_XRD_round[1] + GobleD().interval, GobleD().interval)
	Y_need = np.zeros(X_need.shape)

	Fig_data_dict['0-0'] = np.column_stack((X_need, Y_need))

	save_to_single_excel_forXRD(Fig_data_dict, GobleD().xrd_special_read_excel_save_path)

	return Fig_data_dict


if __name__ == '__main__':
	Fig_data_dict = MAIN_xrd_special_read()
