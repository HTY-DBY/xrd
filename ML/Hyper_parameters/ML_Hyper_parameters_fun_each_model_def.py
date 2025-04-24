import json
import os

import cupy as cp
import numpy as np
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from ML.Z_model_creat.Main_CNN_Set import CNN_mySet
from ML.Z_model_creat.Main_Transformer_reg_Set import Transformer_reg_mySet
from Other.GobleD import GobleD
from Other.Other_main_1 import run_sklearn_model_IN_Other_main, evaluate_model_IN_Other_main, get_old_best_params_IN_Other_main, Fixed_random_seed_IN_Other_main, Get_XRD_data_orifin_IN_Other_main, get_ML_data_by_Need_IN_Other_main, pca_set_encode_IN_Other_main

optuna.logging.set_verbosity(optuna.logging.WARNING)
Now_my_optuna_fun = optuna.samplers.TPESampler()
best_trials_set = -1


# 定义 AE 超参数搜索空间
def define_pca_search_space(XRD_Data_load, trial):
	# num_epochs_AE = trial.suggest_int('num_epochs_AE', 50, 500)
	# encoding_dim_AE = trial.suggest_int('encoding_dim_AE', 100, 1000)
	# hidden_layers_AE = [
	# 	trial.suggest_int('out_channels_1_AE', 10, 200),
	# 	trial.suggest_int('out_channels_2_AE', 10, 200),
	# 	trial.suggest_int('out_channels_3_AE', 10, 200),
	# ]
	# batch_size_AE = trial.suggest_int('batch_size_AE', 32, 512)
	min_temp = np.min([np.size(XRD_Data_load, 0), np.size(XRD_Data_load, 1)])
	n_components = trial.suggest_int('n_components_pca', 2, min_temp)
	random_state = trial.suggest_int('random_states_pca', 0, 100)

	# return num_epochs_AE, encoding_dim_AE, hidden_layers_AE, batch_size_AE
	return n_components, random_state


# AE 模型训练和评估
def train_and_evaluate_pca_model(XRD_Data_load, n_components, random_state):
	XRD_Data_encoded, XRD_Data_recovered = pca_set_encode_IN_Other_main(XRD_Data_load,
																		n_components=n_components,
																		random_state=random_state,
																		print_result=0)
	# XRD_Data_encoded = model.predict(XRD_Data_load)
	# XRD_Data_recovered = model.recover(XRD_Data_encoded)
	result_AE = r2_score(XRD_Data_load, XRD_Data_recovered)
	return result_AE, XRD_Data_encoded


def get_fixed_params_each_model(model_name):
	if model_name == GobleD().XGB_modle_name:
		gpu_do = 1
		fixed_params = {
			'objective': 'reg:squarederror',
			'booster': 'gbtree',
			'verbosity': 1,
			'tree_method': 'hist',
			'device': 'cuda' if gpu_do == 1 else 'cpu'
		}
	elif model_name == GobleD().MLP_modle_name:
		fixed_params = {
			'tol': 1e-4,  # 浮点数，默认=1e-4.优化的公差。当损失或分数至少tol在n_iter_no_change连续迭代中没有改善时，除非learning_rate设置为“自适应”，否则认为已达到收敛并停止训练。
			# 'momentum': trial.suggest_uniform('momentum', 0.0, 1.0),  # 动量参数，默认值=0.9，仅在 solver='sgd' 时使用
			# 'learning_rate_init': 0.0001,  # 浮点型，默认=0.001.使用的初始学习率。它控制更新权重的步长。仅当solver='sgd'或'adam'时使用
			'early_stopping': False,  # 是否使用提前停止，默认值=False
			# 'validation_fraction': 0.1,  # 验证集比例，默认值=0.1，仅在 early_stopping=True 时使用
			'verbose': 0,  # verbose，打印消息的详细程度。有效值为 0（静默）、1（警告）、2（信息）、3（调试）
		}
	elif model_name == GobleD().RF_modle_name:
		fixed_params = {
			"n_jobs": -1,  # default=None，即1。并行job个数。-1意味着使用所有处理器
			"min_weight_fraction_leaf": 0,  # (default=0) 叶子节点所需要的最小权值
			"max_leaf_nodes": None,  # 数值型参数，默认值为None，即不限制最大叶子节点数。这个参数通过限制树的最大叶子数量来防止过拟合，如果设置了一个正整数，则会在建立的最大叶节点内的树中选择最优的决策树
			"bootstrap": True,  # 是否有放回的采样
			"warm_start": False,  # 热启动，决定是否使用上次调用该类的结果然后增加新的。当设置为时True，重用先前调用的解决方案来拟合并向集成添加更多估计器，否则，只需拟合整个新森林
			'verbose': 0,  # verbose，打印消息的详细程度。有效值为 0（静默）、1（警告）、2（信息）、3（调试）
		}
	elif model_name == GobleD().SVM_modle_name:
		fixed_params = {
			'coef0': 0.0,  # 核函数的常数项。对于'poly'和 'sigmoid'有用
			'degree': 3,  # 当指定kernel为 'poly'时，表示选择的多项式的最高次数，默认为三次多项式。 若指定kernel不是'poly',则忽略，即该参数只对'poly'有作用。
			'max_iter': -1,  # 求解器内的迭代次数范围,求解器内迭代的硬性限制, -1 表示无限制。
			'verbose': 0,  # verbose，打印消息的详细程度。有效值为 0（静默）、1（警告）、2（信息）、3（调试）
		}
	else:
		fixed_params = {}
	return fixed_params


# XGBoost模型训练和评估
def train_and_evaluate_follow_XGB(trial, X_train, X_test, y_train, y_test, data_dict=None):
	gpu_do = 1
	if gpu_do == 1:
		X_train = cp.array(X_train)
		X_test = cp.array(X_test)

	model_name = data_dict['model_name']
	fixed_params = get_fixed_params_each_model(model_name)
	params = {
		'random_state': trial.suggest_int('random_state', 0, 100),
		'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
		'max_depth': trial.suggest_int('max_depth', 1, 10),
		'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
		'gamma': trial.suggest_float('gamma', 0, 5),
		'subsample': trial.suggest_float('subsample', 0.5, 1),
		'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
		'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
		'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
		'eta': trial.suggest_float('eta', 0.01, 0.3),
	}

	model = xgb.XGBRegressor(**{**fixed_params, **params})
	model.fit(X_train, y_train)
	y_train_pred, y_test_pred = run_sklearn_model_IN_Other_main(model, X_train, X_test)
	results_temp = evaluate_model_IN_Other_main(y_train, y_test, y_train_pred, y_test_pred, print_result=0)
	return results_temp.loc['Test', 'R2'], fixed_params


def train_and_evaluate_follow_SVM(trial, X_train, X_test, y_train, y_test, data_dict=None):
	model_name = data_dict['model_name']
	fixed_params = get_fixed_params_each_model(model_name)

	params = {
		"kernel": trial.suggest_categorical('kernel', ['rbf', 'linear', 'sigmoid']),  # 核函数类型，选择 'rbf'、'linear'、'sigmoid',算法中所使用的核函数类型
		# 其中有'rbf', 'linear', 'sigmoid', 'poly', 'precomputed'，precompute需要特定的X形式，poly要算很久的
		'C': trial.suggest_float('C', 1e-3, 1e3),  # 正则化参数，使用对数均匀分布以获得广泛的值范围
		'epsilon': trial.suggest_float('epsilon', 0.0, 1.0),  # epsilon值的范围在[0, 1]之间
		'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),  # gamma的选择
		'tol': trial.suggest_float('tol', 1e-5, 1e-2),  # 停止训练的容忍度范围
		'shrinking': trial.suggest_categorical('shrinking', [True, False]),  # 是否使用收缩启发式
		'cache_size': trial.suggest_int('cache_size', 100, 500),  # 指定核函数的缓存大小范围（MB）
	}

	model = SVR(**{**fixed_params, **params})
	model.fit(X_train, y_train)
	y_train_pred, y_test_pred = run_sklearn_model_IN_Other_main(model, X_train, X_test)

	results = evaluate_model_IN_Other_main(y_train, y_test, y_train_pred, y_test_pred, print_result=0)
	return results.loc['Test', 'R2'], fixed_params


def train_and_evaluate_follow_RF(trial, X_train, X_test, y_train, y_test, data_dict=None):
	model_name = data_dict['model_name']
	fixed_params = get_fixed_params_each_model(model_name)

	params = {
		"random_state": trial.suggest_int('random_state', 0, 100),  # 随机数种子，相同的种子可以复现随机结果，常用于参数调优
		"n_estimators": trial.suggest_int('n_estimators', 100, 1000),  # 树的数量，建议范围为100到1000。设置值越大，模型精度越高，但超过某个特定值后，提升效果有限。
		"oob_score": trial.suggest_categorical('oob_score', [True, False]),  # 是否使用袋外（out of bag）样本来估计泛化得分，默认值为False。
		"max_depth": trial.suggest_int('max_depth', 5, 30),  # 树的最大深度，建议范围为5到30。如果为None，节点会扩展至所有叶子节点纯净或所有叶子节点的样本数少于min_samples_split。
		"min_samples_split": trial.suggest_int('min_samples_split', 2, 10),  # 内部节点（非叶子节点）分裂所需的最小样本数，建议范围为2到10。
		"max_features": trial.suggest_categorical('max_features', ['log2', None]),  # 不再包括 'auto'
		"min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 5),  # 叶子节点所需的最小样本数，建议范围为1到5。
	}

	model = RandomForestRegressor(**{**fixed_params, **params})
	model.fit(X_train, y_train)
	y_train_pred, y_test_pred = run_sklearn_model_IN_Other_main(model, X_train, X_test)

	results = evaluate_model_IN_Other_main(y_train, y_test, y_train_pred, y_test_pred, print_result=0)
	return results.loc['Test', 'R2'], fixed_params


def train_and_evaluate_follow_MLP(trial, X_train, X_test, y_train, y_test, data_dict=None):
	model_name = data_dict['model_name']
	fixed_params = get_fixed_params_each_model(model_name)

	# 定义超参数搜索空间
	n_layers = trial.suggest_int('n_layers', 1, 3)  # 隐藏层数，范围 [1, 3]
	n_units = trial.suggest_int('n_units', 50, 200)  # 每层神经元数
	hidden_layer_sizes = tuple([n_units] * n_layers)  # 隐藏层的形状

	params = {
		'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),  # 优化算法，默认值='adam',用于权重优化的求解器。{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
		'hidden_layer_sizes': hidden_layer_sizes,  # 隐藏层的大小，默认值=(100,)，范围取决于 n_layers 和 n_units

		"random_state": trial.suggest_int('random_state', 0, 100),  # 随机数种子，相同的种子可以复现随机结果，常用于参数调优
		'activation': trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu']),  # 激活函数，默认值='relu',['identity', 'logistic', 'tanh', 'relu']
		'alpha': trial.suggest_float('alpha', 1e-5, 1e-2),  # L2 正则化强度，默认值=0.0001，范围 [1e-5, 1e-2],L2 正则化项的强度。添加到损失时，L2 正则化项除以样本大小。
		'batch_size': trial.suggest_categorical('batch_size', ['auto', 32, 64, 128, 256]),  # 批量大小，默认值='auto',随机优化器的小批量大小。如果求解器是“lbfgs”，则回归器将不会使用小批量。当设置为“自动”时，.batch_size=min(200, n_samples)
		'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),  # 学习率调度，默认值='constant'
		# “constant”是由“learning_rate_init”给出的恒定学习率。
		# learning_rate_ “invscaling”使用“power_t”的逆缩放指数逐渐降低每个时间步“t”的学习率。effective_learning_rate = Learning_rate_init / pow(t, power_t)
		# 只要训练损失不断减少，“自适应”就会将学习率保持恒定为“learning_rate_init”。每次两个连续的 epoch 未能将训练损失减少至少 tol，或者未能将验证分数增加至少 tol（如果“early_stopping”打开），当前学习率将除以 5。
		'max_iter': trial.suggest_int('max_iter', 100, 1000),  # 最大迭代次数，默认值=200
		# 求解器迭代直到收敛（由“tol”确定）或迭代次数。对于随机求解器（“sgd”、“adam”），请注意，这决定了历元数（每个数据点将使用多少次），而不是梯度步数。

	}

	model = MLPRegressor(**{**fixed_params, **params})
	model.fit(X_train, y_train)
	y_train_pred, y_test_pred = run_sklearn_model_IN_Other_main(model, X_train, X_test)

	results = evaluate_model_IN_Other_main(y_train, y_test, y_train_pred, y_test_pred, print_result=0)
	return results.loc['Test', 'R2'], fixed_params


def train_and_evaluate_follow_CNN(trial, X_train, X_test, y_train, y_test, data_dict=None):
	Fixed_random_seed_IN_Other_main()
	model_name = data_dict['model_name']
	fixed_params = get_fixed_params_each_model(model_name)

	params = {
		'kernel_size': [trial.suggest_int('kernel_size_1', 1, 10),
						trial.suggest_int('kernel_size_2', 1, 10),
						trial.suggest_int('kernel_size_3', 1, 10),
						],
		'out_channels': [trial.suggest_int('out_channels_1', 10, 200),
						 trial.suggest_int('out_channels_2', 10, 200),
						 trial.suggest_int('out_channels_3', 10, 200),
						 ],
		'num_epochs': trial.suggest_int('num_epochs', 10, 500),
	}

	CNN_mySet_make = CNN_mySet({'kernel_size': params['kernel_size'],
								'out_channels': params['out_channels'],
								'num_epochs': params['num_epochs'],
								'print_each_epoch': 0, 'print_each_epoch_test': 0, 'print_log': 0})
	CNN_mySet_make.fit(X_train, y_train)

	y_test_pred = CNN_mySet_make.predict(X_test)
	result = r2_score(y_test, y_test_pred)

	return result, fixed_params


def train_and_evaluate_follow_Transformer(trial, X_train, X_test, y_train, y_test, data_dict=None):
	Fixed_random_seed_IN_Other_main()
	model_name = data_dict['model_name']
	fixed_params = get_fixed_params_each_model(model_name)

	# 定义超参数搜索空间
	num_heads = trial.suggest_int('num_heads', 2, 16, step=2)  # 仅选择偶数
	model_dim = trial.suggest_int('model_dim', 32, 512)  # 模型维度从 32 到 512

	# 确保 model_dim 可以被 num_heads 整除
	if model_dim % num_heads != 0:
		model_dim = model_dim + (num_heads - model_dim % num_heads)

	num_layers = trial.suggest_int('num_layers', 2, 6)  # 网络层数从 2 到 6 层
	num_epochs = trial.suggest_int('num_epochs', 10, 500)  # 训练轮数从 10 到 100

	model = Transformer_reg_mySet({
		'num_heads': num_heads,
		'model_dim': model_dim,
		'num_layers': num_layers,
		'num_epochs': num_epochs,
		'print_each_epoch': 0, 'print_each_epoch_test': 0, 'print_log': 0
	})

	model.fit(X_train, y_train)

	y_test_pred = model.predict(X_test)
	result = r2_score(y_test, y_test_pred)

	return result, fixed_params


def optimize_mixPCA_In_Hyper_parameters_fun_each_model_def(data_dict):
	Fixed_random_seed_IN_Other_main()
	model_name = data_dict['model_name']
	fixed_params = get_fixed_params_each_model(model_name)
	XRD_Data_load = Get_XRD_data_orifin_IN_Other_main()

	def objective(trial, data_dict):
		# with lock:
		# 获取模型名称和其他需要的数据
		follow_model_name = data_dict['model_name']
		Need_input_in_ML = data_dict['Need_input_in_ML']
		Need_output_in_ML = data_dict['Need_output_in_ML']

		# pca 超参数搜索
		n_components, random_state = define_pca_search_space(XRD_Data_load, trial)

		# pca 模型训练和评估

		result_pca, XRD_Data_encoded = train_and_evaluate_pca_model(XRD_Data_load, n_components, random_state)

		# 获取学习数据
		X_train, X_test, y_train, y_test, _ = get_ML_data_by_Need_IN_Other_main(Need_input_in_ML, Need_output_in_ML, XRD_Data_encoded=XRD_Data_encoded)

		# XGBoost模型训练与评估
		if follow_model_name == GobleD().XGB_modle_name:
			results_follow, fixed_params_new = train_and_evaluate_follow_XGB(trial, X_train, X_test, y_train, y_test, data_dict)
		elif follow_model_name == GobleD().SVM_modle_name:
			results_follow, fixed_params_new = train_and_evaluate_follow_SVM(trial, X_train, X_test, y_train, y_test, data_dict)
		elif follow_model_name == GobleD().RF_modle_name:
			results_follow, fixed_params_new = train_and_evaluate_follow_RF(trial, X_train, X_test, y_train, y_test, data_dict)
		elif follow_model_name == GobleD().MLP_modle_name:
			results_follow, fixed_params_new = train_and_evaluate_follow_MLP(trial, X_train, X_test, y_train, y_test, data_dict)
		elif follow_model_name == GobleD().CNN_modle_name:
			results_follow, fixed_params_new = train_and_evaluate_follow_CNN(trial, X_train, X_test, y_train, y_test, data_dict)
		elif follow_model_name == GobleD().Transformer_modle_name:
			results_follow, fixed_params_new = train_and_evaluate_follow_Transformer(trial, X_train, X_test, y_train, y_test, data_dict)

		return result_pca, results_follow

	return finalize_optimization(objective, fixed_params, data_dict)


def print_best_params_each_time(study, trial, model_name, data_dict):
	# 获取最佳试验的参数和结果值
	best_params = data_dict['best_params']
	best_values = data_dict['best_value']
	best_params_now, best_values_now_this = get_best_trial(study, best_params, best_values)

	# 如果当前的最佳结果与之前不同，则更新并输出
	if best_values_now_this != best_values:
		best_values_now = best_values_now_this

		# 直接修改 data_dict 中的值（通过引用字典内的数据）
		data_dict['best_params'] = best_params_now  # 更新 'best_params'
		data_dict['best_value'] = best_values_now  # 更新 'best_value'

		# 输出
		print(f"\r{model_name} sum: {np.sum(best_values_now)}, best R2:{best_values_now}, best params:{best_params_now}")
	pass


def save_best_params_to_json(model_name, best_params, best_value, fixed_params, json_outPath, Data_v):
	# 确保路径存在
	if not os.path.exists(os.path.dirname(json_outPath)):
		os.makedirs(os.path.dirname(json_outPath))

	if os.path.exists(json_outPath):
		try:
			# 先检查文件是否为空
			if os.path.getsize(json_outPath) == 0:
				json_data_my = {}
			else:
				with open(json_outPath, 'r', encoding='utf-8') as f:
					json_data_my = json.load(f)
		except Exception as e:
			print('原超参数 文件出错，结果另存为')
			json_data_my = {}  # 初始化为空字典
			json_outPath = os.path.join(GobleD().TEMP_path, f'WrongOccur_SaveTemp_FOR_{model_name}_{Data_v}.json')
	else:
		json_data_my = {}

	combined_params = {**best_params, **fixed_params}
	combined_params['best_value'] = best_value
	json_data_my[model_name] = combined_params

	with open(json_outPath, 'w', encoding='utf-8') as f:
		json.dump(json_data_my, f, indent=4, ensure_ascii=False)
		print(f"保存至: {json_outPath}")


def get_best_trial(study, best_params, best_values):
	# 获取所有最佳试验
	best_trials = study.best_trials

	if not best_trials:
		print("没有可用的最佳试验")
		return None

	for index, trial in enumerate(best_trials):
		# 在多目标优化中，使用 trial.values 来访问每个目标值
		if np.sum(trial.values) > np.sum(best_values):
			best_params = trial.params
			best_values = trial.values

	return best_params, best_values


def finalize_optimization(objective, fixed_params, data_dict):
	Fixed_random_seed_IN_Other_main()
	model_name = data_dict['model_name']
	n_trials = data_dict['n_trials']
	json_outPath = data_dict['json_outPath']
	Data_v = str(data_dict['v'])
	n_jobs = data_dict['n_jobs']
	data_dict['best_params'] = None
	old_best_params = None
	data_dict['best_value'] = float('-inf')

	# 加载先前的最优参数
	json_data_my = get_old_best_params_IN_Other_main(json_outPath)

	directions_my = ['maximize', 'maximize']

	study = optuna.create_study(directions=directions_my, sampler=Now_my_optuna_fun)
	try:
		# 尝试加载先前的超参数
		old_best_params = json_data_my[model_name]
		study.enqueue_trial(old_best_params)  # 加载先前的超参数
		print("读取先前的超参数 OK\n"
			  f"old best params: {old_best_params}")
		old_best_values = old_best_params.get('best_value', -100)
		print(f"sum: {np.sum(old_best_values)}, old best value: {old_best_values}")

		data_dict['best_params'] = old_best_params
	# data_dict['best_value'] = old_best_values # 不要使用

	except KeyError:
		print("加载先前的超参数出错，使用新的超参数")

	# 优化过程
	study.optimize(lambda trial: objective(trial, data_dict), n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True,
				   callbacks=[lambda study, trial: print_best_params_each_time(study, trial, model_name, data_dict)])

	# 处理多目标优化的最佳结果
	best_params_now, best_values_now = get_best_trial(study, data_dict['best_params'], data_dict['best_value'])
	if best_params_now == old_best_params:
		print('最终使用了先前的超参数')

	# data_dict['best_params'] = best_params_now
	# data_dict['best_value'] = best_values_now

	if best_params_now != {}: save_best_params_to_json(model_name, best_params_now, best_values_now, fixed_params, json_outPath, Data_v)

	# 特殊处理
	if model_name == GobleD().Transformer_modle_name:
		best_params_now['model_dim'] = best_params_now['model_dim'] - best_params_now['model_dim'] % best_params_now['num_heads']
		save_best_params_to_json(model_name, best_params_now, best_values_now, fixed_params, json_outPath, Data_v)

	print(f"==============================\n{model_name} best_params:", best_params_now)
	print(f"==============================\n{model_name} sum: {np.sum(best_values_now)}, best_value:", 'PCA:', best_values_now[0], '|| Follow:', best_values_now[1])

	return best_params_now, fixed_params


if __name__ == '__main__':
	study = optuna.create_study(sampler=Now_my_optuna_fun)
	print(f"超参数方法目前采用 {study.sampler.__class__.__name__}")
	json_data_my = get_old_best_params_IN_Other_main(GobleD().hyper_parameters_path_fin_1)
