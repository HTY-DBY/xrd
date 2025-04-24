import os
import re
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import seaborn as sns
from Other.GobleD import GobleD
from Other.Other_main_1 import Fig_my_init_IN_Other_main, plt_savefig_IN_Other_main

Fig_my_init_IN_Other_main(latex=False)


# 化学式转latex格式（如 Fe2O3 -> Fe$_2$O$_3$）
def latexify_formula(chemical_formula: str) -> str:
	# 匹配元素符号后跟整数或小数（如 O2、Sr0.24、La1.76）
	return re.sub(r'([A-Za-z\)\]])(\d+\.?\d*)', r'\1$_{\2}$', chemical_formula)


# 读取并解析cif文件，提取化学名称
def get_COD_name(COD_ID_need):
	path = os.path.join(COD_cif_data_path, f"{COD_ID_need}.cif")
	try:
		with open(path, "r", encoding="utf-8") as f:
			cif_text = f.read()

		match = re.search(r"_chemical_formula_sum\s+(?:'([^']+)'|(\S+))", cif_text)
		if match:
			chemical_name = match.group(1) if match.group(1) else match.group(2)
			return chemical_name

		match = re.search(r"_chemical_name_systematic\s+(?:'([^']+)'|(\S+))", cif_text)
		if match:
			chemical_name = match.group(1) if match.group(1) else match.group(2)
			return chemical_name

		print(f"未找到 chemical_name: {COD_ID_need}")
		return None

	except Exception as e:
		print(f"读取 {COD_ID_need} 时发生错误: {e}")
		return None  # 返回 None，跳过出错的记录


# 使用并行化加速处理
def get_COD_name_with_df(df):
	# 使用线程池并行处理每一行的 COD_ID 查询
	with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
		temp = list(executor.map(get_COD_name, df["COD_ID"]))

	# 将获取的名称添加到 DataFrame
	df["COD_name"] = temp
	df["COD_name_latex"] = df["COD_name"].apply(lambda x: latexify_formula(x) if isinstance(x, str) else "")
	df["COD_name_and_id"] = df["COD_name_latex"] + " (" + df["COD_ID"].astype(str) + ")"
	return df


COD_cif_data_path = r"D:\hty\creat\paper\do\XRD\COD_database\cif_files"

# %% 加载数据
temp = os.path.join(GobleD().TEMP_path, f"COD_result.csv")
COD_result = pd.read_csv(temp)
COD_result = COD_result.sort_values(by=['k'], ascending=False)

# 获取化学名称
COD_result = get_COD_name_with_df(COD_result)
# %%
# 保存结果
save_path = os.path.join(GobleD().TEMP_path, f"COD_result_with_name_all.csv")
COD_result.to_csv(save_path, index=False)
print(f"保存结果到 {save_path}")
