import numpy as np
import pandas as pd

from Other.GobleD import GobleD
from Other.Other_main_1 import get_decimal_places_IN_Other_main, is_number_or_pure_digit_IN_Other_main, save_array_to_csv_IN_Other_main


def Main_merge_XRD_read():
	print(f"------ XRD 分数据合并")

	# 读取 XRD 数据
	excel_AllRead_Main_XRD_inPDF = pd.read_csv(GobleD().xrd_pdf_read_excel_save_path)
	excel_AllRead_Main_XRD_inIMG = pd.read_csv(GobleD().xrd_img_read_excel_save_path)
	excel_AllRead_Main_special_inIMG = pd.read_csv(GobleD().xrd_special_read_excel_save_path)
	excel_AllRead_Main_XRD = pd.concat([excel_AllRead_Main_XRD_inPDF, excel_AllRead_Main_special_inIMG, excel_AllRead_Main_XRD_inIMG], ignore_index=False, axis=1)

	# 获取 Theta 值
	Theta_values = np.round(excel_AllRead_Main_XRD.iloc[:, 0], decimals=get_decimal_places_IN_Other_main(GobleD().interval))
	# 保存 Theta 值
	save_array_to_csv_IN_Other_main(Theta_values, GobleD().arg_XRD_round_MY, header=False)

	# 提取名称列表和强度值
	Name_list = [col.rstrip('-X') for col in excel_AllRead_Main_XRD.columns[::2]]
	Intensity_values = excel_AllRead_Main_XRD.iloc[:, 1::2]

	# 创建字典
	AllRead_Main_XRD_dict = {name: Intensity_values.iloc[:, idx] for idx, name in enumerate(Name_list)}

	# 读取 fill 数据库
	excel_AllRead_Main_FillData_for_databas = pd.read_csv(GobleD().FillData_for_database_path)

	# 构建键
	def build_key(ref_Ind, photocatalysts_Ind, XRD_what):
		if not is_number_or_pure_digit_IN_Other_main(photocatalysts_Ind):
			if photocatalysts_Ind.startswith('#'):
				return photocatalysts_Ind[1:]  # 去掉开头的 '#'
			elif photocatalysts_Ind.startswith('PDF'):
				return photocatalysts_Ind  # 直接使用
			return f'{ref_Ind}-{photocatalysts_Ind}'  # 默认情况
		else:
			if photocatalysts_Ind == 0 or XRD_what == 0:
				return f'0-0'  # 处理特殊情况
			else:
				return f'{ref_Ind}-{photocatalysts_Ind}'  # 默认情况

	EACH_key_list = [
		build_key(ref_Ind, photocatalysts_Ind, XRD_what)
		for ref_Ind, photocatalysts_Ind, XRD_what in zip(
			excel_AllRead_Main_FillData_for_databas['ref-Ind'],
			excel_AllRead_Main_FillData_for_databas['catalyst-Ind'],
			excel_AllRead_Main_FillData_for_databas['XRD'])
	]

	Matched_results = np.empty((len(EACH_key_list), len(Theta_values)), dtype=object)  # 使用 object 类型以适应不同长度的数组

	for ind, key in enumerate(EACH_key_list):
		if key in AllRead_Main_XRD_dict:
			Matched_results[ind, :] = AllRead_Main_XRD_dict[key]
		else:
			Matched_results[ind, :] = None  # 其他处理逻辑

	Make_ALL_XRD_data = pd.DataFrame(Matched_results, columns=[f'2theta-{theta}' for theta in Theta_values])

	# 检查是否有 NaN 值
	if Make_ALL_XRD_data.isna().any().any():
		# 查找 NaN 或 None 的位置
		nan_positions = np.where(Make_ALL_XRD_data.isna())
		printed_entries = set()  # 用于存储已打印的引用和序列组合

		for row in zip(*nan_positions):
			ref_ind = excel_AllRead_Main_FillData_for_databas['ref-Ind'].iloc[row[0]]
			sequence_ind = excel_AllRead_Main_FillData_for_databas['sequence-Ind'].iloc[row[0]]

			entry = (ref_ind, sequence_ind)

			# 仅在尚未打印时打印
			if entry not in printed_entries:
				printed_entries.add(entry)  # 添加到集合中
				print(f"数据集出错，未匹配到XRD数据，位于：{ref_ind}-{sequence_ind}")
	else:
		# 如果没有 None 或 NaN
		output_path = GobleD().XRD_ALL_save_path
		save_array_to_csv_IN_Other_main(Make_ALL_XRD_data, output_path)
		print(f"XRD 分数据合并 ok")
		print(f"保存至: {output_path}")
		return Make_ALL_XRD_data


if __name__ == '__main__':
	Make_ALL_XRD_data = Main_merge_XRD_read()
