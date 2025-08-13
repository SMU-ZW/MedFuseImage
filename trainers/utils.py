import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

def evaluate_new(df):
    auroc = roc_auc_score(df['y_truth'], df['y_pred'])
    auprc = average_precision_score(df['y_truth'], df['y_pred'])
    return auprc, auroc



def bootstraping_eval(df, num_iter):
    """This function samples from the testing dataset to generate a list of performance metrics using bootstraping method"""
    auroc_list = []
    auprc_list = []
    for i in tqdm(range(num_iter), desc="Bootstrapping Eval"):
        sample = df.sample(frac=1, replace=True)
        auprc, auroc = evaluate_new(sample)
        auroc_list.append(auroc)
        auprc_list.append(auprc)
    return auprc_list, auroc_list

def computing_confidence_intervals(list_,true_value):
    """This function calcualts the 95% Confidence Intervals"""
    delta = (true_value - list_)
    list(np.sort(delta))
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    # print(f"CI 95% {round(true_value, 3)} ( {round(lower, 3)} , {round(upper, 3)} )")
    return (upper,lower)


# def computing_confidence_intervals(list_, true_value, alpha=0.95):
#     """
#     计算 basic bootstrap 置信区间（返回顺序：upper, lower）
#     list_: 一组bootstrap统计量（list/ndarray）
#     true_value: 原始统计量（float）
#     """
#     arr = np.asarray(list_, dtype=float)

#     # 去掉 NaN/inf
#     arr = arr[np.isfinite(arr)]
#     if arr.size < 2:             # 样本太少无法稳定估计CI
#         return (np.nan, np.nan)

#     # basic bootstrap: delta = theta_hat - theta_boot
#     delta = true_value - arr

#     lower_q = (1 - alpha) / 2 * 100   # e.g. 2.5
#     upper_q = (1 + alpha) / 2 * 100   # e.g. 97.5

#     # 注意你原始代码把 97.5% 给了 delta_lower、2.5% 给了 delta_upper
#     # 下面保持与你原版等价的写法：
#     delta_lower = np.percentile(delta, upper_q)  # 97.5
#     delta_upper = np.percentile(delta, lower_q)  # 2.5

#     upper = true_value - delta_upper
#     lower = true_value - delta_lower
#     return (upper, lower)

def get_model_performance(df):
    test_auprc, test_auroc = evaluate_new(df)
    auprc_list, auroc_list = bootstraping_eval(df, num_iter=1000) #change back to 1000 when run
    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)
    return (test_auprc, upper_auprc, lower_auprc), (test_auroc, upper_auroc, lower_auroc)
