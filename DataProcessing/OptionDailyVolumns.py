import os
import pandas as pd


def load_clear_df(location, file_name):
    df = pd.read_excel(location + file_name, converters={'合约标的代码': str})
    df_2 = df.dropna(thresh=2)

    if len(df) - len(df_2) == 2:
        df = df_2
        df.sort_values(['日期', '执行价'], inplace=True)
        df = df.convert_dtypes()
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        print('Error!')


file_dir = './RawData/ETF期权/日成交统计（按执行价）/'
save_dir = '../Data/'
file_list = os.listdir(file_dir)

writer = pd.ExcelWriter(save_dir + 'ETF期权日成交量（按执行价）.xlsx')
for file in file_list:
    df = load_clear_df(location=file_dir, file_name=file)
    df.to_excel(writer, sheet_name=file.replace('.xlsx', ''), index=False)
writer.close()
