import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ML.Hyper_parameters.ML_Hyper_parameters_fun import choose_model_and_optimize_hyperparameters_3
from Other.GobleD import GobleD
from Other.Other_main_1 import build_list_of_models_NEED_run_IN_Other_main

# 运行模型标志
RUN_FLAGS = {
	GobleD().XGB_modle_name: 0,
	GobleD().RF_modle_name: 0,
	# GobleD().SVM_modle_name: 0,
	GobleD().MLP_modle_name: 0,
	GobleD().CNN_modle_name: 0,
	GobleD().Transformer_modle_name: 1,
}

# 构建需要运行的模型列表
RUN_models_need = build_list_of_models_NEED_run_IN_Other_main(RUN_FLAGS)
# 超参数优化的次数
n_trials = 100

# 加载数据
Need_input_in_ML = GobleD().ML_Input_need_in_database_1
Need_output_in_ML = GobleD().ML_Output_need_in_database_1
database = pd.read_csv(GobleD().Merge_data_for_ML)


# 定义优化模型的函数
def optimize_model(model_name):
	choose_model_and_optimize_hyperparameters_3({
		'model_name': model_name,
		'n_trials': n_trials,
		'database': database,
		'Need_input_in_ML': Need_input_in_ML, 'Need_output_in_ML': Need_output_in_ML,
		'json_outPath': GobleD().hyper_parameters_path_fin_3,
		'v': 3,
		'n_jobs': 1
	})


# %%
# 使用多线程并行执行模型优化
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
	futures = [executor.submit(optimize_model, model_name) for model_name in RUN_models_need]
	for future in futures:
		future.result()  # 等待所有线程完成
