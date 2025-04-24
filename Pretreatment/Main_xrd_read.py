import pandas as pd

from Other.GobleD import GobleD
from Other.Other_main_1 import check_missing_values_IN_Other_main, save_array_to_csv_IN_Other_main


def Main_xrd_read():
	print(f"------ 基础数据库和XRD合并")

	# 读取
	excel_Read_in_FillData_for_database = pd.read_csv(GobleD().FillData_for_database_path)
	excel_Read_XRD_All = pd.read_csv(GobleD().XRD_ALL_save_path)
	FillData_and_XRD = pd.concat([excel_Read_in_FillData_for_database, excel_Read_XRD_All], axis=1)

	output_path = GobleD().FillData_merge_XRD
	save_array_to_csv_IN_Other_main(FillData_and_XRD, output_path)

	check_missing_values_IN_Other_main(FillData_and_XRD)

	print(f"基础数据库和XRD合并 ok")
	print(f"保存至: {output_path}")

	return FillData_and_XRD


if __name__ == '__main__':
	ALL_DATA_concat = Main_xrd_read()
