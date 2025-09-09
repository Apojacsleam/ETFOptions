import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULT_PATH
from config import load_pickle, dump_pickle

if __name__ == '__main__':
    final_df = pd.read_pickle(RESULT_PATH + '00_excess_return.pkl')
    final_df['put'] = final_df['option_code'].apply(lambda x: 'P' in x)
    final_df = final_df[final_df['put'] == 0]
    final_df.describe().T.to_excel(RESULT_PATH + 'excess_return_summary.xlsx')
