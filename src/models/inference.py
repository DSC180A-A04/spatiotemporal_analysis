import numpy as np
from src.loss import quantile_err


def conformal_prediction(model, x_cal, y_cal, significance=0.1):

    y_cal_preds = model.predict(x_cal)

    cal_scores = quantile_err(y_cal_preds, y_cal[:, 0])

    nc = {0: np.sort(cal_scores, 0)[::-1]}
    # significance = .1
    # Sort calibration scores in ascending order? TODO make sure this is correct
    # this is the apply_inverse portion of RegressorNC predict function
    nc = np.sort(cal_scores, 0)

    index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
    # find largest error that gets us guaranteed coverage
    index = min(max(index, 0), nc.shape[0] - 1)

    err_dist = np.vstack([nc[index], nc[index]])

    prediction = y_cal_preds

    intervals = np.zeros((x_cal.shape[0], 3))
    # ensure that we want to multiply our error distances by the size of our training set
    err_dist = np.hstack([err_dist] * x_cal.shape[0])

    intervals[:, 0] = prediction[:, 0] - err_dist[0, :]
    intervals[:, 1] = prediction[:, 1]
    intervals[:, 2] = prediction[:, -1] + err_dist[1, :]

    conformal_intervals = intervals
    return conformal_intervals