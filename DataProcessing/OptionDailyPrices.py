import os
import pandas as pd
import joblib


def load_clear_df(location, file_name):
    df = pd.read_excel(location + file_name, converters={'合约标的代码': str}, na_values='--')
    df_2 = df.dropna(thresh=2)

    if len(df) - len(df_2) == 2:
        df = df_2
        df.sort_values(['交易日期', '期权代码'], inplace=True)
        df = df.convert_dtypes()
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        print('Error!')


name = '000852.SH'
file_dir = './RawData/股指期权/日收盘行情/' + name + '/'
save_dir = '../Data/'
file_list = os.listdir(file_dir)

data_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(load_clear_df)(file_dir, file) for file in file_list)
data = pd.concat(data_list)
data.sort_values(['期权代码', '交易日期'], inplace=True)
data.to_excel(save_dir + '股指期权日行情' + name + '.xlsx', index=False)
