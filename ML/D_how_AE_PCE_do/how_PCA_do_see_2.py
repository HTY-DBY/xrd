import os

import cupy as cupy
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from Other.GobleD import GobleD
from Other.Other_main_1 import evaluate_model_IN_Other_main, run_sklearn_model_IN_Other_main, get_ML_data_by_Need_IN_Other_main, get_old_best_params_IN_Other_main, Get_XRD_data_orifin_IN_Other_main

model_name = GobleD().XGB_modle_name

# 读取数据
Need_input_in_ML = GobleD().ML_Input_need_in_database_1
Need_output_in_ML = GobleD().ML_Output_need_in_database_1
XRD_Data_load = Get_XRD_data_orifin_IN_Other_main()
json_data_my_para = get_old_best_params_IN_Other_main(GobleD().hyper_parameters_path_fin_2)

temp = min(np.size(XRD_Data_load, 0), np.size(XRD_Data_load, 1))
encoding_dim_AE_sets = np.arange(10, temp, 10)
exist_yes = 0
# encoding_dim_AE_sets = [10, 20]
time_data = pd.read_csv(GobleD().how_PCA_time, engine='python')['0'].to_list()
# %%
result_data = []  # 用于存储每次迭代的结果
for idx, encoding_dim_AE_set in enumerate(encoding_dim_AE_sets):
	save_path_PCA = os.path.join(GobleD().how_temp, f"{model_name}_how_PCAmix_PCA_{encoding_dim_AE_set}.joblib")
	save_path_model = os.path.join(GobleD().how_temp, f"{model_name}_how_PCAmix_Follow_{encoding_dim_AE_set}.joblib")

	PCA_model = joblib.load(save_path_PCA)
	model = joblib.load(save_path_model)

	print(f'Running {model_name} {encoding_dim_AE_set}\n--------------------------------')

	XRD_Data_encoded = PCA_model.fit_transform(XRD_Data_load)  # 降维
	XRD_Data_recovered = PCA_model.inverse_transform(XRD_Data_encoded)

	X_train, X_test, y_train, y_test, _ = get_ML_data_by_Need_IN_Other_main(Need_input_in_ML, Need_output_in_ML,
																			XRD_Data_encoded=XRD_Data_encoded)

	if model_name == GobleD().XGB_modle_name:
		X_train_gpu, X_test_gpu = cupy.array(X_train), cupy.array(X_test)
		model.fit(X_train, y_train)
		y_train_pred, y_test_pred = run_sklearn_model_IN_Other_main(model, X_train_gpu, X_test_gpu)
	else:
		model.fit(X_train, y_train)
		y_train_pred, y_test_pred = run_sklearn_model_IN_Other_main(model, X_train, X_test)
	# 评估模型
	results = evaluate_model_IN_Other_main(
		y_train, y_test, y_train_pred, y_test_pred, print_result=0, model_name=model_name
	)
	R2_test = results.loc['Test', 'R2']
	result_AE = r2_score(XRD_Data_load, XRD_Data_recovered)
	print(f'best_value: AE: {result_AE} || Follow: {R2_test}')
	result_data.append({
		'encoding_dim_AE_set': encoding_dim_AE_set,
		'AE': result_AE,
		'Follow': R2_test,
		'time': time_data[idx],
	})

	print(f'--------------------------------')
result_fin = pd.DataFrame(result_data)
result_fin.to_csv(GobleD().how_PCA_result_fin, index=False)
