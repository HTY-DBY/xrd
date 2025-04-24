import numpy as np
import pandas as pd

from Other.GobleD import GobleD
from Pretreatment.xrd_img_read import resample_waveform, smooth_waveform, remove_non_peak_data, XRD_KuoZhan_data, flatten_xrd_data, MinMaxScale_myfun


def get_E_xrd_data_IN_Other_main(file_path):
	E_xrd_data = pd.read_csv(file_path, engine='python')
	E_xrd_data_pd = pd.DataFrame()
	for i in np.arange(0, E_xrd_data.shape[1], 2):
		columns_need = E_xrd_data.columns[i + 1]
		x_theta_data = E_xrd_data.iloc[:, i].values
		y_XRD_data = E_xrd_data.iloc[:, i + 1].values
		XRD_ALL_data = np.stack((x_theta_data, y_XRD_data), axis=1)
		XRD_ori_scope = [min(x_theta_data), max(x_theta_data)]
		XRD_new_scope = [5, 90]
		Fig_data_resampled = resample_waveform(XRD_ALL_data, XRD_ori_scope, GobleD().interval)  # 对波形坐标进行重采样
		Fig_data_nosmooth_noflat = MinMaxScale_myfun(Fig_data_resampled)  # 归一化后存储，确保这是未平滑且未拉平的数据

		Fig_data_smooth_noflat = Fig_data_nosmooth_noflat.copy()
		Fig_data_smooth_noflat[:, 1] = smooth_waveform(Fig_data_nosmooth_noflat[:, 1], window_length=10, polyorder=3)  # 平滑处理

		Fig_data_smooth_noflat = flatten_xrd_data(Fig_data_smooth_noflat, method_type=2)  # 拉平数据
		Fig_data_extended = XRD_KuoZhan_data(Fig_data_smooth_noflat, XRD_new_scope, GobleD().interval)  # 扩展范围
		Fig_data_final = MinMaxScale_myfun(Fig_data_extended)  # 归一化
		Fig_data_final = remove_non_peak_data(Fig_data_final, threshold_factor=0.03)  # 使用阈值去除非峰部分

		E_xrd_data_pd[f'{columns_need}'] = Fig_data_final[:, 1]

	E_xrd_data_pd = E_xrd_data_pd.T
	E_xrd_data_pd.columns = GobleD().theta_labels
	E_xrd_data_pd_fin = E_xrd_data_pd.reset_index(names='XRD')
	return E_xrd_data_pd_fin


def get_E_data_ALL_pd_IN_Other_main(file_path):
	E_data_ALL = pd.read_csv(file_path, engine='python')

	need_columns = (['pollutant conc. (mg/L)', 'band gap (eV)', 'light type',
					 'is light', 'catalyst conc. (g/L)', 'pH',
					 'oxidant conc. (mmol/l)', 'temperature (℃)', 'irradiance (W/m2)'] +
					GobleD().pollutant_chemistry_properties + GobleD().oxidant_chemistry_properties +
					['XRD'])

	E_data_ALL_pd = pd.DataFrame({
		column: E_data_ALL[column] for column in need_columns
	})

	return E_data_ALL_pd


def get_MLdata_form_file_pd_IN_Other_main(mian_data_filePath, XRD_data_filePath):
	E_data_ALL_pd = get_E_data_ALL_pd_IN_Other_main(mian_data_filePath)
	E_xrd_data_pd = get_E_xrd_data_IN_Other_main(XRD_data_filePath)

	Need_fin_pd = pd.merge(E_data_ALL_pd, E_xrd_data_pd, on='XRD', how='left')
	XRD_data_pd = Need_fin_pd[GobleD().theta_labels]
	XRD_data_pd.index = Need_fin_pd['XRD']

	mian_data_pd = Need_fin_pd.copy()
	mian_data_pd = mian_data_pd.drop('XRD', axis=1)
	mian_data_pd = mian_data_pd.drop(GobleD().theta_labels, axis=1)

	return mian_data_pd, XRD_data_pd


if __name__ == '__main__':
	E_xrd_data_pd = get_E_xrd_data_IN_Other_main(GobleD().E_xrd_data)
