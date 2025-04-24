import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import r2_score


def cubic_spline_predict_in_Other_main(X, Y, x_pred=None, verbose=1):
	if X.shape != Y.shape:
		raise ValueError("x_data和y_data的形状必须相同")
	if x_pred is None:
		x_pred = X

	Y_pred_all = []
	Y_fit_all = []

	for i in range(len(X)):
		try:
			cs = PchipInterpolator(X[i], Y[i])
			Y_fit = cs(X[i])
			Y_pred = cs(x_pred[i])

			Y_fit_all.append(Y_fit)
			Y_pred_all.append(Y_pred)

			if verbose:
				print(f"{i}: x_data={X[i]} → y_fit={Y_fit.round(4)}")
				print(f"{i}: x_pred={x_pred[i]} → y_pred={Y_pred.round(4)}")
				print('-' * 40)

		except Exception as e:
			print(f"样本{i + 1}出错: {str(e)}")
			Y_pred_all.append(np.full_like(x_pred[i], np.nan))

	# 计算 R²
	y_true = Y.flatten()
	y_fit = np.array(Y_fit_all).flatten()
	mask = ~np.isnan(y_fit)
	r2 = r2_score(y_true[mask], y_fit[mask]) if np.any(mask) else np.nan
	if verbose: print(f"\n总体R²分数: {r2:.4f}")

	Y_pred_all = np.array(Y_pred_all)
	return Y_pred_all, r2


if __name__ == "__main__":
	# %%
	x_data = np.array([[1, 2, 3]] * 3)
	y_data = np.array([
		[1, -100, 300],
		[2, 3, 4],
		[3, 4, 5]
	])
	x_pred = np.array([[1.5, 2, 2.5]] * 3)

	preds, r2 = cubic_spline_predict_in_Other_main(x_data, y_data, x_pred, verbose=0)
