import os
import time
import pandas as pd
from config import DATA_PATH, dump_pickle

if __name__ == '__main__':
    # 获取目录下所有文件夹中的xlsx文件
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.xlsx'):
                start_time = time.time()
                file_path = os.path.join(root, file)

                # 读取xlsx文件
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names

                # 判断sheet数量
                output_filename = str(file_path).replace('.xlsx', '.pkl')
                if len(sheet_names) == 1:
                    df = pd.read_excel(file_path)
                    end_time = time.time()
                    print(f'[Load Data]\tExcel file {file_path} loaded.\t\t Time usage: {round(end_time - start_time, 4)} sec.')
                    dump_pickle(output_filename, df)
                else:
                    ans_dict = dict()
                    for sheet_name in sheet_names:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        ans_dict[sheet_name] = df
                    end_time = time.time()
                    print(f'[Load Data]\tExcel file {file_path} loaded.\t\t Time usage: {round(end_time - start_time, 4)} sec.')
                    dump_pickle(output_filename, ans_dict)
