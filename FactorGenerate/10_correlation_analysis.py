import numpy as np
import pandas as pd
from config import load_pickle, dump_pickle
from config import RESULT_PATH


def pearson_correlation(e_i, e_j):
    from scipy.stats import pearsonr
    e_i = np.nan_to_num(e_i, nan=0.0)
    e_j = np.nan_to_num(e_j, nan=0.0)

    c, p = pearsonr(e_i, e_j)
    return c


def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np

    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst = []

    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

    # Length of lists (as real numbers)
    T = float(len(actual_lst))

    # construct d according to crit
    if (crit == "MSE"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append((actual - p1) ** 2)
            e2_lst.append((actual - p2) ** 2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(abs((actual - p1) / actual))
            e2_lst.append(abs((actual - p2) / actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
            e1_lst.append(((actual - p1)) ** (power))
            e2_lst.append(((actual - p2)) ** (power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)

            # Mean of d
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N - k):
            autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
        return (1 / (T)) * autoCov

    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
    V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    DM_stat = V_d ** (-0.5) * mean_d
    harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
    DM_stat = harvey_adj * DM_stat
    # Find p-value
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')

    rt = dm_return(DM=DM_stat, p_value=p_value)

    return rt


if __name__ == '__main__':
    result_dict = dict()

    train_data_list = load_pickle(RESULT_PATH + '06_train_result.pkl')

    for data_list in train_data_list:
        for data_dict in data_list:
            data_dict['result_df'].set_index(['option_code', 'month'], inplace=True)
            labels = (data_dict['label'], data_dict['model'])
            if result_dict.get(labels, None) is None:
                result_dict[labels] = [data_dict['result_df']]
            else:
                result_dict[labels].append(data_dict['result_df'])

    ES1 = ['Lasso', 'Ridge', 'ENet', 'PCR', 'PLS']
    ES2 = ['GBR', 'RF', 'AdaBoost']
    result1 = dict()
    result2 = dict()
    result_list = list()
    for key, value in result_dict.items():
        label, model = key
        result_df = pd.concat(value)
        result_dict = {'label': label, 'model': model, 'result_df': result_df.copy(deep=True)}
        result_list.append(result_dict)
        result_dfs = result_df.copy(deep=True)
        result_dfs.columns = ['next_return_' + model, 'predict_return_' + model]
        if model in ES1:
            if result1.get(label, None) is None:
                result1[label] = [result_dfs]
            else:
                result1[label].append(result_dfs)

        if model in ES2:
            if result2.get(label, None) is None:
                result2[label] = [result_dfs]
            else:
                result2[label].append(result_dfs)
    for key, value in result1.items():
        result_df = pd.concat(value, axis=1)
        result_df['next_return'] = result_df[[col for col in result_df.columns if 'next_return' in col]].mean(axis=1)
        result_df['predict_return'] = result_df[[col for col in result_df.columns if 'predict_return' in col]].mean(axis=1)
        result_list.append({'label': key, 'model': 'L-En', 'result_df': result_df})

    for key, value in result2.items():
        result_df = pd.concat(value, axis=1)
        result_df['next_return'] = result_df[[col for col in result_df.columns if 'next_return' in col]].mean(axis=1)
        result_df['predict_return'] = result_df[[col for col in result_df.columns if 'predict_return' in col]].mean(axis=1)
        result_list.append({'label': key, 'model': 'N-En*', 'result_df': result_df})

    data_list = []
    for i in range(len(result_list)):
        if result_list[i]['label'] == 'All':
            for j in range(len(result_list)):
                if result_list[j]['label'] == 'All':
                    name_i = result_list[i]['model']
                    name_j = result_list[j]['model']
                    print(name_i, name_j)
                    result_df_i = result_list[i]['result_df'].dropna().convert_dtypes()
                    result_df_j = result_list[j]['result_df'].dropna().convert_dtypes()
                    dm, p = dm_test(actual_lst=np.nan_to_num(result_df_i['next_return'].values, nan=0.0),
                                    pred1_lst=np.nan_to_num(result_df_i['predict_return'].values, nan=0.0),
                                    pred2_lst=np.nan_to_num(result_df_j['predict_return'].values, nan=0.0))
                    e_i = result_df_i['predict_return']
                    e_j = result_df_j['predict_return']
                    corr = pearson_correlation(e_i, e_j)

                    data_list.append({'model1': name_i, 'model2': name_j, 'corr': corr, 'dm': dm})

    model_list = ['Lasso', 'Ridge', 'ENet', 'PCR', 'PLS', 'L-En', 'MLP', 'GBR', 'RF', 'AdaBoost', 'N-En*']
    df = pd.DataFrame(data_list).pivot(index='model1', columns='model2', values='corr')
    df = df.reindex(index=model_list, columns=model_list)
    df.to_excel(RESULT_PATH + 'model_comparation.xlsx')
