# %%
import os
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import vstack

from ML.F_fin_model_uesd.F_fin_model_uesd_fun_all.F_fin_model_uesd_fun import get_model_need_for_nbg, XGB_using_direct_nbg
from ML.F_fin_model_uesd.F_go_ZH.any_nongDU_xrd_fun import xrd_get_with_nongDU
from Other.GobleD import GobleD

# 获取需要的列名和模型
columns_need_path = os.path.join(GobleD().TEMP_path, f"XGB_fin_in_4_MinMaxScaler_column_name_nbg_columns_need.csv")
columns_need = pd.read_csv(columns_need_path, header=None).squeeze().tolist()
model_need = get_model_need_for_nbg()

# %%

file_name = os.path.join(GobleD().Main_P, "Database", "Experiment", "go_ZH.xlsx")
data_No_XRD = pd.read_excel(file_name, sheet_name='COD_find', index_col=0).loc[:, 'pollutant conc. (mg/L)':]
data_No_XRD_this = data_No_XRD.iloc[:, :].reset_index(drop=True)

# nongDU_need = 1.1
inter = 0.05
nongDU_needs = np.arange(0, 1 + inter, inter)
y_need_preds = []

for nongDU_need in nongDU_needs:
	preds, r2, theat_data = xrd_get_with_nongDU(nongDU_need, rank=None)
	data_XRD = pd.DataFrame(preds).T.reset_index(drop=True)
	data_this = pd.concat([data_No_XRD_this, data_XRD], axis=1)

	# 获取预测结果
	y_need_pred, _ = XGB_using_direct_nbg(data_this, model_need, columns_need, gpu=1)
	y_need_preds.append(y_need_pred[0])

# %%
temp = np.array(y_need_preds).reshape(-1, 1)
y_need_preds_fin = np.hstack([nongDU_needs.reshape(-1, 1), temp])
