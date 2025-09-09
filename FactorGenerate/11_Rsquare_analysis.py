import numpy as np
import pandas as pd
from config import load_pickle, dump_pickle
from config import RESULT_PATH
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 20, "mathtext.fontset": "cm"})

if __name__ == '__main__':
    result_list = load_pickle(RESULT_PATH + '07_result_list.pkl')
    for data_dict in result_list:
        if data_dict['label'] == 'All' and data_dict['model'] == 'L-En':
            l_en_result_df = data_dict['result_df'].reset_index()
        elif data_dict['label'] == 'All' and data_dict['model'] == 'N-En*':
            n_en_result_df = data_dict['result_df'].reset_index()

    l_en_result_df['month'] = pd.to_datetime(l_en_result_df['month'], format='%Y%m')
    n_en_result_df['month'] = pd.to_datetime(n_en_result_df['month'], format='%Y%m')

    plt.figure(figsize=(9, 6))
    month_list = []
    r2_list = []
    for month, month_df in l_en_result_df.groupby('month'):
        r2 = 1 - ((month_df['next_return'] - month_df['predict_return']) ** 2).sum() / (month_df['next_return'] ** 2).sum()
        month_list.append(month)
        r2_list.append(r2)
    plt.scatter(month_list, r2_list, label='L-En', color='blue', edgecolor='white', alpha=0.6, s=200)
    df1 = pd.DataFrame({'month': month_list, 'L-En': r2_list})

    month_list = []
    r2_list = []
    for month, month_df in n_en_result_df.groupby('month'):
        r2 = 1 - ((month_df['next_return'] - month_df['predict_return']) ** 2).sum() / (month_df['next_return'] ** 2).sum()
        month_list.append(month)
        r2_list.append(r2)
    df2 = pd.DataFrame({'month': month_list, 'N-En*': r2_list})

    plt.scatter(month_list, r2_list, label='N-En*', color='black', edgecolor='white', s=200)
    plt.ylim([-1, 1])
    plt.legend(loc='best')
    plt.ylabel('$R^2_\mathrm{OS}$')
    plt.xlabel('Time')
    plt.legend(loc='lower right', fontsize=14)
    #plt.grid()
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=3)
    plt.savefig(RESULT_PATH + 'figures/LN-EnR2time.pdf', dpi=600)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))
    dfa = pd.merge(df1, df2, on='month', how='inner')
    dfa1 = dfa.copy(deep=True)[dfa['N-En*'] >= dfa['L-En']]
    print(len(dfa1))
    dfa2 = dfa.copy(deep=True)[dfa['N-En*'] < dfa['L-En']]
    print(len(dfa2))
    # 定义两个端点的坐标
    x_coords = [-1, 1]
    y_coords = [-1, 1]

    # 绘制分界线
    plt.plot(x_coords, y_coords, color='black', linestyle='--', linewidth=3)# 定义分界线上方和下方的坐标范围

    plt.scatter(dfa1['L-En'], dfa1['N-En*'], color='lightgreen', edgecolor='white', s=200, label='N-En* better')
    plt.scatter(dfa2['L-En'], dfa2['N-En*'], color='lightcoral', edgecolor='white', s=200, label='L-En better')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('L-En')
    plt.ylabel('N-En*')
    plt.legend(loc='lower right', fontsize=14)
    plt.savefig(RESULT_PATH + 'figures/LN-EnR2compare.pdf', dpi=600)
    plt.tight_layout()

    plt.show()
