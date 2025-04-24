# %%
import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from Other.GobleD import GobleD
from Other.Other_main_1 import Fig_my_init_IN_Other_main, plt_savefig_IN_Other_main

Fig_my_init_IN_Other_main(latex=False)


# 化学式转latex格式（如 Fe2O3 -> Fe$_2$O$_3$）
def latexify_formula(chemical_formula: str) -> str:
	# 匹配元素符号后跟整数或小数（如 O2、Sr0.24、La1.76）
	return re.sub(r'([A-Za-z\)\]])(\d+\.?\d*)', r'\1$_{\2}$', chemical_formula)


def get_COD_name(COD_ID_need):
	path = os.path.join(COD_cif_data_path, f"{COD_ID_need}.cif")
	with open(path, "r", encoding="utf-8") as f:
		cif_text = f.read()

	match = re.search(r"_chemical_formula_sum\s+(?:'([^']+)'|(\S+))", cif_text)
	if match:
		chemical_name = match.group(1) if match.group(1) else match.group(2)
		return chemical_name

	match = re.search(r"_chemical_name_systematic\s+'(.+?)'", cif_text)
	if match:
		chemical_name = match.group(1)
		return chemical_name

	print(f"未找到 chemical_name: {COD_ID_need}")
	return None


def get_COD_name_with_df(df):
	temp = df["COD_ID"].apply(get_COD_name)
	df["COD_name"] = temp
	df["COD_name_latex"] = df["COD_name"].apply(lambda x: latexify_formula(x) if isinstance(x, str) else "")
	df["COD_name_and_id"] = df["COD_name_latex"] + " (" + df["COD_ID"].astype(str) + ")"
	return df


COD_cif_data_path = r"D:\hty\creat\paper\do\XRD\COD_database\cif_files"

# %%
temp = os.path.join(GobleD().TEMP_path, f"COD_result.csv")
COD_result = pd.read_csv(temp)
COD_result = COD_result.sort_values(by=['k'], ascending=False)

COD_result_f_50 = COD_result.iloc[:50, :].copy()
COD_result_f_50 = get_COD_name_with_df(COD_result_f_50)
COD_result_b_50 = COD_result.sort_values(by='k', ascending=True).iloc[:50, :].copy()
COD_result_b_50 = get_COD_name_with_df(COD_result_b_50)
COD_results = [COD_result_f_50, COD_result_b_50]
COD_results_name = ['f_50', 'b_50']
for i, COD_result_this in enumerate(COD_results):
	# %%
	data = COD_result_this

	fontsize = 14
	x_labels = data["COD_name_and_id"]
	x_pos = np.arange(len(x_labels))

	plt.figure(figsize=(18, 4.5))
	plt.plot(data["COD_name_and_id"], data["k"], marker='o')
	plt.xticks(rotation=90, fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlim(-0.5, len(x_labels) - 0.5)
	plt.ylabel("log$_2k$", fontsize=fontsize)
	plt.grid(axis='x', linestyle='--', alpha=0.5)  # 设置x轴上的网格线（因为已经翻转）
	plt.grid(visible=True, which='both', axis='x', linestyle='--', linewidth=0.7)
	sns.despine(left=True, bottom=True)

	plt_savefig_IN_Other_main(f'{COD_results_name[i]}-{os.path.basename(os.path.abspath(__file__))}')
	plt.show()
