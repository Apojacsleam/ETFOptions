import time
import pickle

import pandas as pd

UNDERLYING_ASSET_LIST = ['510050.SH', '510300.SH', '510500.SH', '588000.SH', '588080.SH', '159901.SZ', '159915.SZ',
                         '159919.SZ', '159922.SZ']
DATA_PATH = '../Data/'
RESULT_PATH = '../Result/'
RISK_FREE_ASSET = '上证所新质押式国债回购'
MULTIPLIER = 10000
END_MONTH = 202401

def load_pickle(file_name: str):
    start_time = time.time()
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    end_time = time.time()
    print(f'[Load Data]\tPickle data {file_name} loaded.\t\tTime usage:{round(end_time - start_time, 4)} sec.\t\tData type: {type(data)}.')
    print('=' * 200)
    return data


def dump_pickle(file_name: str, data) -> None:
    start_time = time.time()
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    end_time = time.time()
    print(f'[Dump Data]\tPickle data {file_name} dumped.\t\tTime usage:{round(end_time - start_time, 4)} sec.\t\tData type: {type(data)}')
    print('=' * 200)


def display_dataframe(name: str, data: pd.DataFrame) -> None:
    # 设置显示的最大列数和列宽
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print(f'[DataFrame Name]\t {name}\n')
    print(f'[DataFrame Display]\n')
    print(data)
    print(f'[DataFrame Describe]\n')
    print(data.describe().convert_dtypes())
    print(f'\n[DataFrame Dtypes]')
    print(data.dtypes.T)
    print('=' * 200)
