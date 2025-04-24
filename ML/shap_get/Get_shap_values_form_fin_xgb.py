# %% 导入模块
import os
import pickle

import joblib
import pandas as pd
import seaborn as sns

from Other.GobleD import GobleD
from Other.Other_main_1 import Fig_my_init_IN_Other_main, Get_XRD_data_orifin_IN_Other_main, get_InOut_form_NeedCol_IN_Other_main, save_array_to_csv_IN_Other_main

# %% 初始化图形配置
Fig_my_init_IN_Other_main(latex=False)
cmap = sns.color_palette("vlag", as_cmap=True).colors  # 配色

# 获取变量
Need_input_in_ML = GobleD().ML_Input_need_in_database_2
Need_output_in_ML = GobleD().ML_Output_need_in_database_1

XRD_Data_load = Get_XRD_data_orifin_IN_Other_main()
Merge_data_for_ML_MinMax = pd.read_csv(GobleD().Merge_data_for_ML)
X_ML_input, y_ML_output = get_InOut_form_NeedCol_IN_Other_main(
	Merge_data_for_ML_MinMax, Need_input_in_ML, Need_output_in_ML
)

X_all_need = X_ML_input
y_need = y_ML_output

fin_model_path = os.path.join(GobleD().TEMP_path, f"fin_model.joblib")
model = joblib.load(fin_model_path)
y_pre = model.predict(X_all_need)

# %% shap
X_need = pd.DataFrame(model.X_need)
temp = 19
X_need.columns = list(X_all_need.columns[:temp]) + list(X_need.columns[temp:])

# %%
with open(os.path.join(GobleD().TEMP_path, "shap_explainer_xgb.pkl"), 'rb') as f:
	explainer = pickle.load(f)

shap_values = explainer.shap_values(X_need)

# %%
shap_values = pd.DataFrame(shap_values, columns=X_need.columns)
save_path = GobleD().fin_xgb_shap_values
save_array_to_csv_IN_Other_main(shap_values, save_path)
# save_path = GobleD().fin_xgb_shap_values_x_need
# save_array_to_csv_IN_Other_main(X_need, save_path)
