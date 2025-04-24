import os

import pandas as pd


class GobleD:
	def __init__(self, ):
		self.Main_P = r"D:\hty\creat\paper\do\XRD\PY"
		# 原始数据库保存路径
		self.data_ALL = os.path.join(self.Main_P, "Database", "Main_database", "Database.xlsx")
		self.data_fig_ues = r"D:\hty\creat\paper\do\XRD\ttt.xlsx"
		# 临时文件导出位置
		self.TEMP_path = os.path.join(self.Main_P, "Pretreatment", "Temp")
		self.how_temp = os.path.join(self.Main_P, "Pretreatment", "Temp", "how_temp")
		# 图片保存位置
		self.Fig_ALL_Save = os.path.join(self.Main_P, 'Fig_ALL', 'Fig_ALL_Save')
		# 数据文件存储
		self.FillData_for_database_path = os.path.join(self.TEMP_path, "FillData_for_database.csv")
		self.FillData_for_database_path_publish = os.path.join(self.TEMP_path, "FillData_for_database_publish.csv")
		self.xrd_img_read_excel_save_path = os.path.join(self.TEMP_path, 'XRD_IMG_read.csv')
		self.xrd_img_read_excel_save_path_ori = os.path.join(self.TEMP_path, 'XRD_IMG_read_ori.csv')
		self.xrd_img_read_excel_save_path_dictread = os.path.join(self.TEMP_path, 'XRD_IMG_read_dictread.csv')
		self.xrd_img_read_excel_save_path_ori_dictread = os.path.join(self.TEMP_path, 'XRD_IMG_read_ori_dictread.csv')
		self.xrd_pdf_read_excel_save_path = os.path.join(self.TEMP_path, 'XRD_PDF_read.csv')
		self.xrd_special_read_excel_save_path = os.path.join(self.TEMP_path, 'XRD_special_read.csv')
		self.XRD_ALL_save_path = os.path.join(self.TEMP_path, 'XRD_ALL_read.csv')
		self.FillData_merge_XRD = os.path.join(self.TEMP_path, 'FillData_merge_XRD.csv')
		self.arg_XRD_round_MY = os.path.join(self.TEMP_path, 'arg_XRD_round_MY.csv')
		self.Merge_data_for_ML = os.path.join(self.TEMP_path, 'Merge_data_for_ML.csv')
		self.Merge_data_for_ML_with_catalyst_quality = os.path.join(self.TEMP_path, 'Merge_data_for_ML_with_catalyst_quality.csv')
		self.Merge_data_for_ML_use_for_shap = os.path.join(self.TEMP_path, 'Merge_data_for_ML_use_for_shap.csv')
		self.Per_each_ML_fin_in_1 = os.path.join(self.TEMP_path, 'Pre_each_ML_fin_in_1.csv')
		self.Per_each_ML_fin_in_2 = os.path.join(self.TEMP_path, 'Pre_each_ML_fin_in_2.csv')
		self.Per_each_ML_fin_in_3 = os.path.join(self.TEMP_path, 'Pre_each_ML_fin_in_3.csv')
		self.Per_each_ML_fin_in_4 = os.path.join(self.TEMP_path, 'Pre_each_ML_fin_in_4.csv')

		self.XRD_search = os.path.join(self.Main_P, "Database", "XRD_search")
		self.E_data_ALL = os.path.join(self.Main_P, "Database", "Experiment", "E_Database.csv")
		self.E_xrd_data = os.path.join(self.Main_P, "Database", "Experiment", "E_xrd_data.csv")

		# 超参数存放
		self.hyper_parameters_path_fin_1 = os.path.join(self.Main_P, "Other", "ML_Hyper_parameters_fin_1.json")
		self.hyper_parameters_path_fin_2 = os.path.join(self.Main_P, "Other", "ML_Hyper_parameters_fin_2.json")
		self.hyper_parameters_path_fin_3 = os.path.join(self.Main_P, "Other", "ML_Hyper_parameters_fin_3.json")
		self.hyper_parameters_path_fin_4 = os.path.join(self.Main_P, "Other", "ML_Hyper_parameters_fin_4.json")

		### 读取原始数据库

		# 读取 Main 表需要的列
		self.needRead_column_Main = [
			'ref-Ind',  # 文献索引号
			'catalyst-Ind',  # 催化剂索引号
			'sequence-Ind',  # 实验序列索引号
			'k (min-1)',  # 一级反映常数
			# '污染物',  # 污染物名称
			'pollutant',  # 污染物名称，英文
			'pollutant conc. (mg/L)',  # 污染物的浓度
			'band gap (eV)',  # 催化剂的带隙
			'light type',  # 光类型-紫外1；可见光2；太阳光3，无光0
			'is light',  # 1有光，0无光
			'catalyst',  # 催化剂名称
			'catalyst conc. (g/L)',  # 催化剂的浓度
			'pH',  # pH
			'oxidant',  # 额外添加的氧化剂
			'oxidant conc. (mmol/l)',  # 额外添加的氧化剂的浓度
			'temperature (℃)',  # 温度
			'light source power (W)',  # 光源功率
			'irradiance (W/m2)',  # 辐照度
			'X-min',  # XRD x轴下获取的最小值
			'X-max',  # XRD x轴上获取的最大值
			'total reaction time (min)',  # 总反应时间
			'maximum degradation rate (%)',  # 最大降解率
			'XRD', 'XPS', 'Raman', 'UV', 'PL', 'FTIR', 'BET', 'ESR', 'EIS', 'transient photocurrent response'  # 表征方法统计数
		]

		# 读取 Chemistry 表需要的列
		self.chemistry_properties = [
			'EHOMO (eV)',  # HOMO
			'ELUMO (eV)',  # LUMO
			'Egap (eV)',  # LUMO-HOMO
			'single point energy (Eh)',  # 单点能
			'dipole moment (Debye)',  # 偶极矩
			# 'f(-)min (e)',  # 福井指数 原子最小
			# 'f(+)min (e)',  # 福井指数 原子最小
			# 'f(0)min (e)',  # 福井指数 原子最小
			# 'f(-)max (e)',  # 福井指数 原子最大
			# 'f(+)max (e)',  # 福井指数 原子最大
			# 'f(0)max (e)',  # 福井指数 原子最大
		]

		self.Characterization_methods = ['XRD', 'XPS', 'Raman', 'UV', 'PL', 'FTIR', 'BET', 'ESR', 'EIS', 'transient photocurrent response']

		self.needRead_column_Main_latex = [
			'ref-Ind',  # 文献索引号
			'catalyst-Ind',  # 催化剂索引号
			'sequence-Ind',  # 实验序列索引号
			'log$_{2}k$ (min$^{-1}$)',  # 一级反映常数
			# '污染物',  # 污染物名称
			'pollutant',  # 污染物名称，英文
			'pollutant conc. (mg/L)',  # 污染物的浓度
			'band gap (eV)',  # 催化剂的带隙
			'light type',  # 光类型-紫外1；可见光2；太阳光3，无光0
			'is light',  # 1有光，0无光
			'catalyst',  # 催化剂名称
			'catalyst conc. (g/L)',  # 催化剂的浓度
			'pH',  # pH
			'oxidant',  # 额外添加的氧化剂
			'oxidant conc. (mmol/l)',  # 额外添加的氧化剂的浓度
			'temperature (\u00B0C)',  # 温度
			'light source power (W)',  # 光源功率
			'irradiance (W/m$^{2}$)',  # 辐照度
			'X-min',  # XRD x轴下获取的最小值
			'X-max',  # XRD x轴上获取的最大值
			'total reaction time (min)',  # 总反应时间
			'maximum degradation rate (%)',  # 最大降解率
			# --------- 表征方法统计数
		]
		self.needRead_column_Main_latex += [prop for prop in self.Characterization_methods]

		self.chemistry_properties_latex = [
			'$E_{HOMO}$ (eV)',  # HOMO
			'$E_{LUMO}$ (eV)',  # LUMO
			'$E_{gap}$ (eV)',  # LUMO-HOMO
			'single point energy (Eh)',  # 单点能
			'dipole moment (Debye)',  # 偶极矩
			# '$f_{(-)min}$ (e)',  # 福井指数 原子最小
			# '$f_{(+)min}$ (e)',  # 福井指数 原子最小
			# '$f_{(0)min}$ (e)',  # 福井指数 原子最小
			# '$f_{(-)max}$ (e)',  # 福井指数 原子最大
			# '$f_{(+)max}$ (e)',  # 福井指数 原子最大
			# '$f_{(0)max}$ (e)',  # 福井指数 原子最大
		]

		self.pollutant_chemistry_properties = [f'pollutant {prop}' for prop in self.chemistry_properties]
		self.pollutant_chemistry_properties_latex = [f'pollutant {prop}' for prop in self.chemistry_properties_latex]
		self.oxidant_chemistry_properties = [f'oxidant {prop}' for prop in self.chemistry_properties]
		self.oxidant_chemistry_properties_latex = [f'oxidant {prop}' for prop in self.chemistry_properties_latex]

		self.needRead_column_Chemistry = [
			'c-污染物',  # 污染物名称
			'chemistry_name',  # 污染物名称，英文
		]
		self.needRead_column_Chemistry += [prop for prop in self.chemistry_properties]

		# XRD数据位置
		self.XRD_save_path = os.path.join(self.Main_P, "Database", "XRD")

		self.XRD_PDF_save_path = os.path.join(self.XRD_save_path, 'PDF')
		self.XRD_IMG_save_path = os.path.join(self.XRD_save_path, 'IMG')
		self.XRD_IMG_save_path_deict_read = os.path.join(self.XRD_save_path, 'IMG_deict_read')
		self.interval = 0.05

		self.ML_Output_need_in_database_1 = ['k (min-1)', ]  # 一级反应常数

		self.ML_no_chemistry_properties_1 = [
			# 'ref-Ind',  # 文献索引号
			# 'catalyst-Ind',  # 催化剂索引号
			# 'sequence-Ind',  # 实验序列索引号
			# 'k(min-1)',  # 一级反映常数
			# '污染物',  # 污染物名称
			# 'pollutant',  # 污染物名称，英文
			'pollutant conc. (mg/L)',  # 污染物的浓度
			'band gap (eV)',  # 催化剂的带隙
			'light type',  # 光类型-紫外1；可见光2；太阳光3，无光0
			'is light',  # 1有光，0无光
			# 'catalyst',  # 催化剂名称
			'catalyst conc. (g/L)',  # 催化剂的浓度
			'pH',  # pH
			# 'oxidant',  # 额外添加的氧化剂
			'oxidant conc. (mmol/l)',  # 额外添加的氧化剂的浓度
			'temperature (℃)',  # 温度
			# 'light source power (W)',  # 光源功率
			'irradiance (W/m2)',  # 辐照度
			'pollutant classify index',  # 污染物分类
			# 'X-min',  # XRD x轴下获取的最小值
			# 'X-max',  # XRD x轴上获取的最大值
			# 'total reaction time (min)',  # 总反应时间
			# 'maximum degradation rate (%)',  # 最大降解率
			# 'XRD', 'XPS', 'Raman', 'UV', 'PL', 'FTIR', 'BET', 'ESR', 'EIS', 'transient photocurrent response'  # 表征方法统计数
		]
		self.ML_no_chemistry_properties_2 = [
			# 'ref-Ind',  # 文献索引号
			# 'catalyst-Ind',  # 催化剂索引号
			# 'sequence-Ind',  # 实验序列索引号
			# 'k(min-1)',  # 一级反映常数
			# '污染物',  # 污染物名称
			# 'pollutant',  # 污染物名称，英文
			'pollutant conc. (mg/L)',  # 污染物的浓度
			'band gap (eV)',  # 催化剂的带隙
			'light type',  # 光类型-紫外1；可见光2；太阳光3，无光0
			'is light',  # 1有光，0无光
			# 'catalyst',  # 催化剂名称
			'catalyst conc. (g/L)',  # 催化剂的浓度
			'pH',  # pH
			# 'oxidant',  # 额外添加的氧化剂
			'oxidant conc. (mmol/l)',  # 额外添加的氧化剂的浓度
			'temperature (℃)',  # 温度
			# 'light source power (W)',  # 光源功率
			'irradiance (W/m2)',  # 辐照度
			# 'pollutant classify index',  # 污染物分类
			# 'X-min',  # XRD x轴下获取的最小值
			# 'X-max',  # XRD x轴上获取的最大值
			# 'total reaction time (min)',  # 总反应时间
			# 'maximum degradation rate (%)',  # 最大降解率
			# 'XRD', 'XPS', 'Raman', 'UV', 'PL', 'FTIR', 'BET', 'ESR', 'EIS', 'transient photocurrent response'  # 表征方法统计数
		]
		self.pollutant_chemistry_properties_1 = self.pollutant_chemistry_properties
		self.oxidant_chemistry_properties_1 = self.oxidant_chemistry_properties

		self.ML_Input_need_in_database_1 = self.ML_no_chemistry_properties_1 + self.pollutant_chemistry_properties_1 + self.oxidant_chemistry_properties_1

		self.pollutant_chemistry_properties_2 = self.pollutant_chemistry_properties
		self.oxidant_chemistry_properties_2 = self.oxidant_chemistry_properties
		self.chemistry_properties_2 = self.pollutant_chemistry_properties + self.oxidant_chemistry_properties
		self.ML_Input_need_in_database_2 = self.ML_no_chemistry_properties_2 + self.pollutant_chemistry_properties_2 + self.oxidant_chemistry_properties_2

		# elements_to_remove = ['oxidant EHOMO (eV)', 'pollutant EHOMO (eV)']
		# for element in elements_to_remove:
		# 	if element in self.ML_Input_need_in_database_1:
		# 		self.ML_Input_need_in_database_1.remove(element)
		if os.path.exists(self.arg_XRD_round_MY):
			theta_scope = pd.read_csv(self.arg_XRD_round_MY, header=None).values.flatten()
			self.theta_labels = [f'2theta-{prop}' for prop in theta_scope]
			self.ML_Input_need_in_database_1 += self.theta_labels
			self.ML_Input_need_in_database_2 += self.theta_labels

		self.ML_Input_need_in_database_1_with_catalyst_quality = (self.ML_Input_need_in_database_1 +
																  ['catalyst quality', 'pollutant', 'oxidant', 'pollutant classify'])

		# 各个模型的缩写名
		self.XGB_modle_name = 'XGB'
		self.RF_modle_name = 'RF'
		self.SVM_modle_name = 'SVM'
		self.MLP_modle_name = 'MLP'
		self.CNN_modle_name = 'CNN'
		self.Transformer_modle_name = 'Transformer'
		self.AE_modle_name = 'AE'
		self.how_AE_time = os.path.join(self.how_temp, "how_AE_time.csv")
		self.how_PCA_time = os.path.join(self.how_temp, "how_PCA_time.csv")
		self.how_AE_result_fin = os.path.join(self.how_temp, "how_AE_result_fin.csv")
		self.how_PCA_result_fin = os.path.join(self.how_temp, "how_PCA_result_fin.csv")
		self.fin_xgb_shap_values = os.path.join(self.TEMP_path, 'fin_xgb_shap_values.csv')
		self.fin_oxgb_shap_values = os.path.join(self.TEMP_path, 'fin_oxgb_shap_values.csv')


if __name__ == '__main__':
	# %%
	# 初始化对象
	GobleD = GobleD()
	temp = GobleD.ML_Input_need_in_database_1
	print(temp)
	pass
