import pandas as pd
from config import load_pickle
from config import RESULT_PATH

if __name__ == '__main__':
    result_dict = dict()

    train_data_list = load_pickle(RESULT_PATH + '06_train_result.pkl')

    for data_list in train_data_list:
        for data_dict in data_list:
            labels = (data_dict['label'], data_dict['model'])
            if result_dict.get(labels, None) is None:
                result_dict[labels] = [data_dict['result_df']]
            else:
                result_dict[labels].append(data_dict['result_df'])

    ES1 = ['Lasso', 'Ridge', 'ENet', 'PCR', 'PLS']
    ES2 = ['MLP', 'GBR', 'RF', 'AdaBoost']
    ES3 = ['GBR', 'RF', 'AdaBoost']
    result1 = dict()
    result2 = dict()
    result3 = dict()
    result_list = list()
    for key, value in result_dict.items():
        label, model = key
        result_df = pd.concat(value)
        R_square_os = 1 - ((result_df['next_return'] - result_df['predict_return']) ** 2).sum() / (result_df['next_return'] ** 2).sum()
        result_dict = {'label': label, 'model': model, 'Rsquare_OS': R_square_os}
        result_list.append(result_dict)
        result_dfs = result_df.copy(deep=True).set_index(['option_code', 'month'])
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

        if model in ES3:
            if result3.get(label, None) is None:
                result3[label] = [result_dfs]
            else:
                result3[label].append(result_dfs)

    for key, value in result1.items():
        result_df = pd.concat(value, axis=1)
        next_return = result_df[[col for col in result_df.columns if 'next_return' in col]].mean(axis=1)
        predict_return = result_df[[col for col in result_df.columns if 'predict_return' in col]].mean(axis=1)
        R_square_os = 1 - ((next_return - predict_return) ** 2).sum() / (next_return ** 2).sum()
        result_list.append({'label': key, 'model': 'L-En', 'Rsquare_OS': R_square_os})

    for key, value in result2.items():
        result_df = pd.concat(value, axis=1)
        next_return = result_df[[col for col in result_df.columns if 'next_return' in col]].mean(axis=1)
        predict_return = result_df[[col for col in result_df.columns if 'predict_return' in col]].mean(axis=1)
        R_square_os = 1 - ((next_return - predict_return) ** 2).sum() / (next_return ** 2).sum()
        result_list.append({'label': key, 'model': 'N-En', 'Rsquare_OS': R_square_os})

    for key, value in result3.items():
        result_df = pd.concat(value, axis=1)
        next_return = result_df[[col for col in result_df.columns if 'next_return' in col]].mean(axis=1)
        predict_return = result_df[[col for col in result_df.columns if 'predict_return' in col]].mean(axis=1)
        R_square_os = 1 - ((next_return - predict_return) ** 2).sum() / (next_return ** 2).sum()
        result_list.append({'label': key, 'model': 'N-En*', 'Rsquare_OS': R_square_os})

    pd.DataFrame(result_list).to_excel(RESULT_PATH + 'Rsquare_OS.xlsx', index=False)
