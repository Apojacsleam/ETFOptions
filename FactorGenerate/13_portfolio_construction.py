import numpy as np
import pandas as pd
from config import load_pickle, dump_pickle
from config import RESULT_PATH
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 20, "mathtext.fontset": "cm"})

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

    model_list = ['Lasso', 'Ridge', 'ENet', 'PCR', 'PLS', 'L-En', 'MLP', 'GBR', 'RF', 'AdaBoost', 'N-En*']

    suffix = 'Put'
    data_dict = dict()
    for i in range(len(result_list)):
        if result_list[i]['label'] == suffix and result_list[i]['model'] in model_list:
            result_df = result_list[i]['result_df'][['next_return', 'predict_return']]
            data_dict[result_list[i]['model']] = result_df

    metrics_list = []
    plt.figure(figsize=(14, 6))
    for name in model_list:
        df = data_dict[name]

        df['predict_success'] = df['next_return'] * df['predict_return'] >= 0.0
        df['profit'] = df.apply(lambda row: abs(row['next_return']) if row['predict_success'] else -abs(row['next_return']), axis=1)

        profit = df.groupby('month')['profit'].mean()
        profit.name = name

        new_index = [pd.to_datetime(a, format='%Y%m') + pd.DateOffset(months=2) for a in profit.index]
        profit.index = new_index

        profit -= 0.005
        annual_return = profit.mean() * 12
        annual_std = profit.std() * np.sqrt(12)

        sharpe = (annual_return - 1.708 / 100) / annual_std

        win = (profit > 0).sum() / len(profit)
        metrics_dict = {'模型': name, '年化收益': annual_return * 100, '年化波动率': annual_std * 100, '夏普比率': sharpe, '策略胜率': win * 100}

        profit[pd.to_datetime('2018-01-01')] = 0.0

        profit.sort_index(inplace=True)
        nav = (1 + profit).cumprod()

        cummax = nav.cummax()
        dd = (nav - cummax) / cummax
        maxdd = - dd.min()
        metrics_dict['最大回撤'] = maxdd * 100

        metrics_list.append(metrics_dict)
        plt.plot(nav.index, nav.values, label=name, linewidth=2)

    plt.ylabel('Net Asset Value')
    plt.xlabel('Time')
    plt.title(f'Net Asset Value of Portfolio on {suffix} Options')
    plt.yticks(np.arange(0.6, 2.21, 0.2))
    plt.ylim([0.6, 2.2])
    plt.axhline(y=1, color='black', linestyle='--', linewidth=3)
    plt.legend(loc='upper left', fontsize=14)
    plt.grid()
    plt.savefig(RESULT_PATH + f'figures/portofolio_{suffix}.jpg', dpi=600)
    plt.show()

    pd.DataFrame(metrics_list).to_excel(RESULT_PATH + f'metrics_{suffix}.xlsx')
