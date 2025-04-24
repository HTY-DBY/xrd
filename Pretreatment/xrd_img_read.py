import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from BaselineRemoval import BaselineRemoval
from PIL import Image
from pybaselines import Baseline
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn import preprocessing

from Other.GobleD import GobleD
from Other.Other_main_1 import save_array_to_csv_IN_Other_main, get_all_files_IN_Other_main
from Pretreatment.xrd_img_read_Get_XRD_scope import Get_XRD_scope_ori_dict, Get_XRD_scope_ori_arg


def convert_jpg_to_png(file_path):
	"""将文件夹中的所有.jpg文件转换为.png文件"""
	filename = os.path.basename(file_path)
	if filename.endswith(".jpg"):
		old_file_path = file_path
		new_file_path = os.path.join(os.path.dirname(file_path), f"{os.path.splitext(filename)[0]}.png")
		os.rename(old_file_path, new_file_path)
		print(f"已将 {filename} 修改为 {new_file_path}")


def Crop_image(file_path):
	"""裁剪图像的透明部分并保存"""
	filename = os.path.basename(file_path)
	if filename.endswith(".png"):
		old_file_path = file_path
		img = Image.open(old_file_path).convert("RGBA")
		# 把太黑的看不清的换成绿色
		pixels = img.load()
		threshold = 30
		green_color = (0, 255, 0, 255)  # 绿色 (R=0, G=255, B=0, A=255)
		width, height = img.size
		for x in range(width):
			for y in range(height):
				r, g, b, a = pixels[x, y]
				if a > 0 and r < threshold and g < threshold and b < threshold:
					pixels[x, y] = green_color  # 将太黑的像素替换为绿色
		bbox = img.getbbox()  # 获取图像内容的边界框

		save_folder = os.path.join(GobleD().XRD_save_path, 'IMG-pre')
		os.makedirs(save_folder, exist_ok=True)  # 如果文件夹不存在则创建
		file_now_path = os.path.join(save_folder, filename)
		img.crop(bbox).save(file_now_path)  # 保存裁剪后的图像

		return file_now_path


def extract_waveform(image_path, XRD_ori_scope):
	"""从图像中提取波形数据，假设图像纵向为Y轴，横向为X轴"""
	XRD_min = XRD_ori_scope[0]
	XRD_max = XRD_ori_scope[1]
	img = Image.open(image_path).convert("RGBA")
	width, height = img.size
	pixels = img.load()

	x_scale = (XRD_max - XRD_min) / width  # X轴缩放
	waveform_data = []

	for x in range(width):
		for y in range(height):
			r, g, b, a = pixels[x, y]
			if a > 0:  # 检查像素是否非透明
				new_x = x * x_scale + XRD_min  # 计算实际X值
				new_y = height - y  # 计算Y值，原点在左下角
				waveform_data.append([new_x, new_y])

	return np.array(waveform_data)


def resample_waveform(waveform_coordinates, XRD_ori_scope, interval):
	"""对波形坐标进行重采样"""
	XRD_min = XRD_ori_scope[0]
	XRD_max = XRD_ori_scope[1]
	x_original, y_original = waveform_coordinates[:, 0], waveform_coordinates[:, 1]
	unique_x, indices = np.unique(x_original, return_inverse=True)
	y_averaged = np.bincount(indices, weights=y_original) / np.bincount(indices)
	# 使用线性插值来生成新的y值
	x_new = np.arange(XRD_min, XRD_max + interval, interval)
	interpolation_function = interp1d(unique_x, y_averaged, kind='linear', fill_value="extrapolate")
	y_new = interpolation_function(x_new)

	return np.column_stack((x_new, y_new))


def XRD_KuoZhan_data(waveform_coordinates, XRD_new_scope, interval):
	"""扩展波形数据范围"""
	XRD_min_new, XRD_max_new = XRD_new_scope
	x_original, y_original = waveform_coordinates[:, 0], waveform_coordinates[:, 1]
	# 填充左侧和右侧的边界值
	y_fill = min(y_original)
	# 生成扩展后的 x_new
	x_min = min(XRD_min_new, min(x_original)) - 10
	x_max = max(XRD_max_new, max(x_original)) + 10
	x_new = np.arange(x_min, x_max, interval)
	# 初始化 y_new 为填充值，后续覆盖
	y_new = np.full(len(x_new), y_fill)
	# 找到 x_original 在 x_new 中的起始索引
	start_index = np.searchsorted(x_new, x_original[0])
	# 将 y_original 对应的部分填入 y_new
	y_new[start_index:start_index + len(y_original)] = y_original
	# Fin
	start_index_fin = np.searchsorted(x_new, XRD_min_new)
	end_index_fin = np.searchsorted(x_new, XRD_max_new)
	# 截取 y_new 和 x_new
	x_new_fin = x_new[start_index_fin - 1:end_index_fin]
	y_new_fin = y_new[start_index_fin - 1:end_index_fin]
	return np.column_stack((x_new_fin, y_new_fin))


def flatten_xrd_data(waveform_coordinates, method_type=1):
	"""拉平XRD数据"""
	x, y = waveform_coordinates[:, 0], waveform_coordinates[:, 1]
	if np.sum(y) == 0:
		return waveform_coordinates

	if method_type == 1:
		baseObj = BaselineRemoval(y)
		y_flat = baseObj.ZhangFit(lambda_=20, porder=1, repitition=30)
	else:
		baseline_fitter = Baseline(x_data=x)
		y_flat = y - baseline_fitter.arpls(y, lam=5e5)[0]

	return np.column_stack((x, y_flat))


def smooth_waveform(data, window_length=10, polyorder=3):
	"""对Y数据进行平滑处理，使用Savitzky-Golay滤波"""
	window_length = min(window_length, len(data))  # 确保窗口长度不超过数据长度
	return savgol_filter(data, window_length=window_length, polyorder=polyorder)


def MinMaxScale_myfun(Fig_data):
	"""最小最大归一化处理"""
	scaler = preprocessing.MinMaxScaler()
	Fig_data_MinMax = Fig_data.copy()  # 这里要确保你得到的数据是 NumPy 数组
	Fig_data_MinMax[:, 1] = scaler.fit_transform(Fig_data[:, 1].reshape(-1, 1)).flatten()
	return Fig_data_MinMax


def MinMaxScale_myfun_return(Fig_data):
	"""最小最大归一化处理"""
	scaler = preprocessing.MinMaxScaler()
	Fig_data_MinMax = Fig_data.copy()  # 这里要确保你得到的数据是 NumPy 数组
	Fig_data_MinMax[:, 1] = scaler.fit_transform(Fig_data[:, 1].reshape(-1, 1)).flatten()
	return Fig_data_MinMax, scaler


def remove_non_peak_data(waveform_coordinates, threshold_factor):
	"""去除非峰数据"""
	x, y = waveform_coordinates[:, 0], waveform_coordinates[:, 1]
	threshold = threshold_factor  # 定义阈值
	y_filtered = np.where(y < threshold, 0, y)  # 将低于阈值的部分设置为0
	return np.column_stack((x, y_filtered))


def FIG_finally(data_plt, save_folder, file_path, show=0):
	"""绘制并展示最终的波形图"""
	plt.figure()
	plt.plot(data_plt[:, 0], data_plt[:, 1], label='PLT')
	plt.xlabel('2θ (degree)')
	plt.ylabel('Y')
	plt.legend()
	plt.grid(True)

	filename = os.path.basename(file_path)
	plt.savefig(os.path.join(save_folder, 'fin-' + filename))

	if show == 1:
		plt.show()
	plt.close()


def process_single_image(file_path, XRD_ori_scope, XRD_new_scope):
	"""处理单个图像文件"""
	# 将.jpg文件转换为.png文件
	convert_jpg_to_png(file_path)
	# 裁剪图像的透明部分并保存
	file_path = Crop_image(file_path)

	# 提取图像中的波形数据
	Fig_data = extract_waveform(file_path, XRD_ori_scope)  # 从图像中提取波形数据，假设图像纵向为Y轴，横向为X轴
	Fig_data_resampled = resample_waveform(Fig_data, XRD_ori_scope, GobleD().interval)  # 对波形坐标进行重采样
	Fig_data_nosmooth_noflat = MinMaxScale_myfun(Fig_data_resampled)  # 归一化后存储，确保这是未平滑且未拉平的数据

	Fig_data_smooth_noflat = Fig_data_nosmooth_noflat.copy()
	Fig_data_smooth_noflat[:, 1] = smooth_waveform(Fig_data_nosmooth_noflat[:, 1], window_length=10, polyorder=3)  # 平滑处理

	Fig_data_smooth_noflat = flatten_xrd_data(Fig_data_smooth_noflat, method_type=2)  # 拉平数据
	Fig_data_extended = XRD_KuoZhan_data(Fig_data_smooth_noflat, XRD_new_scope, GobleD().interval)  # 扩展范围
	Fig_data_final = MinMaxScale_myfun(Fig_data_extended)  # 归一化
	Fig_data_final = remove_non_peak_data(Fig_data_final, threshold_factor=0.03)  # 使用阈值去除非峰部分

	return file_path, Fig_data_final


def ThreadPool_read_image(file_path, XRD_ori_scope_list, XRD_new_scope_list):
	save_fin_img_path = os.path.join(GobleD().XRD_save_path, 'IMG-fin')
	os.makedirs(save_fin_img_path, exist_ok=True)  # 如果文件夹不存在则创建
	Fig_data_dict = {}  # 存储每个文件的 Fig_data_MinMaxScale 数据

	# 使用线程池并行处理每个图像，限制线程数
	with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
		# 提交所有图像处理任务，并存储未来对象与文件名的映射
		future_to_filename = {
			executor.submit(process_single_image, filename, XRD_ori_scope, XRD_new_scope):
				(filename, XRD_ori_scope, XRD_new_scope) for (filename, XRD_ori_scope, XRD_new_scope) in
			zip(file_path, XRD_ori_scope_list, XRD_new_scope_list)  # 只迭代两次
		}

		for future in future_to_filename:
			file_path_in_future, Fig_data_final = future.result()
			FIG_finally(Fig_data_final, save_fin_img_path, file_path_in_future, show=0)
			filename_in_future = os.path.basename(file_path_in_future)
			ile_name_without_extension, _ = os.path.splitext(filename_in_future)
			Fig_data_dict[ile_name_without_extension] = Fig_data_final  # 存储结果

	return Fig_data_dict


# 导出合并后的结果为 Excel 文件
def save_to_single_excel_forXRD(Fig_data_dict, output_path):
	df_temp = pd.DataFrame()
	# 遍历字典，将每个文件的数据作为两列添加
	for filename, data in Fig_data_dict.items():
		# 将 X 和 Y 数据转换为 DataFrame，并使用文件名作为列前缀
		df = pd.DataFrame(data, columns=[f"{filename}-X", f"{filename}-Y"])
		df_temp = pd.concat([df_temp, df], axis=1)  # 横向拼接
	# 保存结果为 Excel 文件
	save_array_to_csv_IN_Other_main(df_temp, output_path)
	print(f"XRD img 读取 ok")
	print(f"保存至: {output_path}")


def MAIN_xrd_img_read(Use_Test_file_Ind=0):
	print(f"------ XRD img 读取")

	if Use_Test_file_Ind == 0:
		file_path_list = get_all_files_IN_Other_main(GobleD().XRD_IMG_save_path)
		XRD_ori_scope_list = Get_XRD_scope_ori_dict()
		_, arg_XRD_round = Get_XRD_scope_ori_arg(XRD_ori_scope_list)
		XRD_ori_scope_list = list(XRD_ori_scope_list.values())
		XRD_new_scope_list = np.full((np.size(XRD_ori_scope_list), np.size(arg_XRD_round)), arg_XRD_round)
	else:
		file_path_list = [os.path.join(GobleD().XRD_IMG_save_path, filename) for filename in Test_file_Ind]
		XRD_ori_scope_list = Test_XRD_ori_scope_list
		XRD_new_scope_list = Test_XRD_new_scope_list

	Fig_data_dict = ThreadPool_read_image(file_path_list, XRD_ori_scope_list, XRD_new_scope_list)
	save_to_single_excel_forXRD(Fig_data_dict, GobleD().xrd_img_read_excel_save_path)
	return Fig_data_dict


Test_file_Ind = ['ZF4.png']
Test_XRD_ori_scope_list = [[20, 75]]
Test_XRD_new_scope_list = [[5, 90]]

if __name__ == "__main__":
	Use_Test_file_Ind = 1

	Fig_data_dict = MAIN_xrd_img_read(Use_Test_file_Ind=Use_Test_file_Ind)
