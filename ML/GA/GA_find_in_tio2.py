# %%
import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ML.F_fin_model_uesd.F_fin_model_uesd_fun_all.F_fin_model_uesd_fun import run_ga_fun, get_model_need_for_ga
from Other.GobleD import GobleD

max_iter_set = 10000

# %%
# 载入数据
file_name = os.path.join(GobleD().Main_P, "Database", "Experiment", "ga_tio2.xlsx")
data_No_XRD = pd.read_excel(file_name, sheet_name='ga_E', index_col=None).reset_index(drop=True).loc[:, 'pollutant conc. (mg/L)':]
data_XRD = pd.read_excel(file_name, sheet_name='ga_E_XRD', index_col=None).reset_index(drop=True).loc[:, '2theta-5.0':]
model_need = get_model_need_for_ga()
start_time = time.time()

# %%  无催化剂
# 搜寻设置
need_cow = list(range(0, 4))
data_this = pd.concat([data_No_XRD, data_XRD], axis=1).iloc[need_cow, :]
data_this[data_this == 'opt'] = 0

need_find_col = ['pollutant conc. (mg/L)', 'catalyst conc. (g/L)', 'pH', 'irradiance (W/m2)']
lb = [5, 0.05, 2, 500]
ub = [30, 0.5, 10, 1200]

# %%
# GA 搜寻
best_x_all_1 = []
best_y_all_1 = []
for i in np.arange(len(data_this)):
	data_this_i = data_this.iloc[[i], :]
	print(f'start GA in {i}')
	[best_x, best_y], Y_history = run_ga_fun(data_this_i, need_find_col, lb, ub, model_need, max_iter=max_iter_set, is_show=1)
	print('-' * 30)
	best_x_all_1.append(best_x)
	best_y_all_1.append(best_y)

	plt.plot(Y_history.index, -Y_history.values, '.', color='red')
	plt.show()

best_x_all_1 = np.vstack(best_x_all_1)
best_y_all_1 = np.array(best_y_all_1)
best_y_all_1_relog = 2 ** best_y_all_1
best_x_all_1 = pd.DataFrame(best_x_all_1, columns=need_find_col)
# %%  有催化剂
# 搜寻设置
need_cow = list(range(4, 8))
data_this = pd.concat([data_No_XRD, data_XRD], axis=1).iloc[need_cow, :]
data_this[data_this == 'opt'] = 0

need_find_col = ['pollutant conc. (mg/L)', 'catalyst conc. (g/L)', 'pH', 'irradiance (W/m2)', 'oxidant conc. (mmol/l)']
lb = [5, 0.05, 3, 500, 0.2]
ub = [30, 0.5, 9, 1200, 2]

# %%
# GA 搜寻
best_x_all_2 = []
best_y_all_2 = []
for i in np.arange(len(data_this)):
	data_this_i = data_this.iloc[[i], :]
	print(f'start GA in {i}')
	[best_x, best_y], Y_history = run_ga_fun(data_this_i, need_find_col, lb, ub, model_need, max_iter=max_iter_set, is_show=1)
	print('-' * 30)
	best_x_all_2.append(best_x)
	best_y_all_2.append(best_y)

	plt.plot(Y_history.index, -Y_history.values, '.', color='red')
	plt.show()
best_x_all_2 = np.vstack(best_x_all_2)
best_y_all_2 = np.array(best_y_all_2)
best_y_all_2_relog = 2 ** best_y_all_2
best_x_all_2 = pd.DataFrame(best_x_all_2, columns=need_find_col)

print(f"all time use: {time.time() - start_time} s")
