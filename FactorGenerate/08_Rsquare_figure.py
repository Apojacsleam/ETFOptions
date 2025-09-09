import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULT_PATH

plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 20, "mathtext.fontset": "cm"})

colors = ['#191928', '#BBC2DE', '#696D95']

if __name__ == '__main__':
    # 创建一个DataFrame（这里是假设的数据）
    algorithm_list = ['Lasso', 'Ridge', 'ENet', 'PCR', 'PLS', 'L-En', 'MLP', 'GBR', 'RF', 'AdaBoost', 'N-En', 'N-En*']
    df_list = []
    df = pd.read_excel(RESULT_PATH + 'Rsquare_OS.xlsx')
    for algorithm_name in algorithm_list:
        df_list.append(df[df['model'] == algorithm_name])
    df = pd.concat(df_list)
    # 根据label和model对Rsquare进行分组并计算平均值
    grouped_df = df
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = grouped_df['label'].unique()
    width = 0.25
    x = range(grouped_df['model'].nunique())

    for i, label in enumerate(labels):
        label_data = grouped_df[grouped_df['label'] == label]
        positions = [pos + i * width for pos in x]
        ax.bar(positions, label_data['Rsquare_OS'], width, label=label, color=colors[i])

    ax.axhline(y=0.0, color='black', linestyle='--', linewidth=3)
    ax.set_xticks([pos + width * (len(labels) - 1) / 2 for pos in x])
    zeros = np.zeros(1000)
    ax.set_xticklabels(grouped_df['model'].unique())
    ax.set_yticks(np.arange(-0.5, 0.51, 0.25))
    ax.set_ylabel('$R^2_\mathrm{OS}$')
    ax.set_xlabel('Model')
    ax.legend(loc='upper left', fontsize=14)
    ax.set_ylim([-0.5, 0.5])
    plt.grid(axis='y')
    plt.savefig(RESULT_PATH + 'figures/Rsquare_OS.pdf', dpi=600)
    plt.tight_layout()
    plt.show()
