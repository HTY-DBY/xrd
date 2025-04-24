import pandas as pd

from ML.Hyper_parameters.ML_Hyper_parameters_fun_each_model_def import optimize_mixPCA_In_Hyper_parameters_fun_each_model_def
from Other.GobleD import GobleD

MODEL_NAMES_can_Hyper_parameters = [GobleD().XGB_modle_name, GobleD().RF_modle_name,
									# GobleD().SVM_modle_name,
									GobleD().MLP_modle_name,
									GobleD().CNN_modle_name, GobleD().Transformer_modle_name, GobleD().AE_modle_name]


def choose_model_and_optimize_hyperparameters_3(data_dict):
	model_name = data_dict['model_name']

	# 检查模型名称
	if model_name not in MODEL_NAMES_can_Hyper_parameters:
		print(f'模型名称错误: {model_name}。可用模型: {MODEL_NAMES_can_Hyper_parameters}')
		return
	else:
		print(f"==============================\n对于模型 {model_name}, 开始超参数优化")

		best_params, fixed_params = optimize_mixPCA_In_Hyper_parameters_fun_each_model_def(data_dict)

	return best_params, fixed_params


if __name__ == '__main__':
	# %%
	Need_input_in_ML = GobleD().ML_Input_need_in_database_1
	Need_output_in_ML = GobleD().ML_Output_need_in_database_1
	database = pd.read_csv(GobleD().Merge_data_for_ML)

	best_params, fixed_params = choose_model_and_optimize_hyperparameters_3({'model_name': GobleD().XGB_modle_name,
																			 'n_trials': 3,
																			 'database': database,
																			 'Need_input_in_ML': Need_input_in_ML, 'Need_output_in_ML': Need_output_in_ML,
																			 'json_outPath': GobleD().hyper_parameters_path_fin_3,
																			 'v': 'test',
																			 'n_jobs': 1
																			 })
	combine_params = {**fixed_params, **best_params}
