import os
import pandas as pd
import joblib


def load_clear_df(location, file_name):
    print(file_name.replace('.xlsx', ''))
    df = pd.read_excel(location + file_name, converters={'代码': str}, na_values='--')
    df_2 = df.dropna(thresh=2)

    if len(df) - len(df_2) == 2:
        df = df_2
        df.sort_values('证券代码', inplace=True)
        df = df.convert_dtypes()
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        print(len(df), len(df_2))
        print('Error!')


file_dir = './RawData/'
save_dir = '../Data/'
file_list = os.listdir(file_dir)

df = load_clear_df(location=file_dir, file_name='ETF基金基本信息.xlsx')
df.to_excel(save_dir + 'ETF基金基本信息.xlsx', index=False)
