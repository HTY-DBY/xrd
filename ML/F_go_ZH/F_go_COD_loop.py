import os
import numpy as np
import pandas as pd
from ML.F_fin_model_uesd.F_fin_model_uesd_fun_all.F_fin_model_uesd_fun import get_model_need_for_nbg, XGB_using_direct_nbg
from Other.GobleD import GobleD

# 获取模型名称等参数
file_name = os.path.join(GobleD().Main_P, "Database", "Experiment", "go_ZH.xlsx")
data_No_XRD = pd.read_excel(file_name, sheet_name='COD_find', index_col=0).loc[:, 'pollutant conc. (mg/L)':]
data_No_XRD_this = data_No_XRD.iloc[:1, :].reset_index(drop=True)

# 获取需要的列名和模型
columns_need_path = os.path.join(GobleD().TEMP_path, f"XGB_fin_in_4_MinMaxScaler_column_name_nbg_columns_need.csv")
columns_need = pd.read_csv(columns_need_path, header=None).squeeze().tolist()
model_need = get_model_need_for_nbg()

# 结果保存路径
save_path = os.path.join(GobleD().TEMP_path, f"COD_result.csv")
print(f'过滤开始 ->')

# 如果文件已存在，读取文件内容
if os.path.exists(save_path):
	predictions_list = pd.read_csv(save_path)
	processed_files = set(predictions_list['COD_ID'].tolist())  # 用 set 提升查找速度
else:
	predictions_list = pd.DataFrame(columns=['COD_ID', 'k'])
	processed_files = set()  # 空 set

# 获取XRD文件路径
COD_path = r"D:\hty\creat\paper\do\XRD\COD_database\xrd_format"
xrd_file_paths = [f for f in os.listdir(COD_path) if f.endswith('.csv')]

# 过滤已处理的文件（O(1) 查找）
xrd_file_paths = [f for f in xrd_file_paths if int(f.split('.')[0]) not in processed_files]

# 获取总文件数量
total_files = len(xrd_file_paths)

print(f'-> 过滤结束')

# 处理每个XRD文件，获取预测结果
for index, xrd_file_path in enumerate(xrd_file_paths):
	xrd_file_name = int(xrd_file_path.split('.')[0])  # 获取文件名（不包括扩展名）

	print(f"处理 -> : {index + 1}/{total_files} - 当前文件: {xrd_file_name}")

	xrd_file_all_path = os.path.join(COD_path, xrd_file_path)  # 获取文件完整路径

	# 读取XRD数据并合并
	data_XRD = pd.read_csv(xrd_file_all_path, header=None).iloc[:, 1:].T.reset_index(drop=True)
	data_this = pd.concat([data_No_XRD_this, data_XRD], axis=1)

	# 获取预测结果
	y_need_pred, _ = XGB_using_direct_nbg(data_this, model_need, columns_need, gpu=1)

	# 将预测结果与文件名一起存储
	predictions_list = pd.concat([predictions_list, pd.DataFrame([[xrd_file_name, y_need_pred[0]]], columns=['COD_ID', 'k'])], ignore_index=True)

	# 打印处理进度
	progress = (index + 1) / total_files * 100
	print(f"-> 结束: {index + 1}/{total_files} - 当前文件: {xrd_file_name}")

# 保存预测结果
predictions_list.to_csv(save_path, index=False)
