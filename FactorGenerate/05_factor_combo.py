import pandas as pd
from config import load_pickle, dump_pickle, display_dataframe
from config import UNDERLYING_ASSET_LIST, DATA_PATH, RESULT_PATH, END_MONTH

if __name__ == '__main__':
    excess_return = pd.read_pickle(RESULT_PATH + '00_excess_return.pkl')

    next_return = excess_return.copy(deep=True)[['option_code', 'month', 'optionmom']]
    next_return['pre_month'] = pd.to_datetime(next_return['month'], format='%Y%m') - pd.DateOffset(months=1)
    next_return['pre_month'] = next_return['pre_month'].dt.strftime('%Y%m').astype(int)
    next_return = next_return[['option_code', 'pre_month', 'optionmom']]
    next_return.columns = ['option_code', 'month', 'next_return']

    final_df = next_return.copy(deep=True)
    final_df = pd.merge(final_df, excess_return, on=['option_code', 'month'], how='inner')

    contract_factors = pd.read_pickle(RESULT_PATH + '01_contract_level_factors.pkl')
    final_df = final_df.merge(contract_factors, on=['option_code', 'month'], how='left')

    contract_factors = pd.read_pickle(RESULT_PATH + '02_bucket_level_factors.pkl')
    final_df = final_df.merge(contract_factors, on=['option_code', 'month'], how='left')

    drop_columns = ['etfname', 'expiration']
    stock_factors = pd.read_pickle(RESULT_PATH + '03_stock_level_factors.pkl')
    stock_factors['etfname'] = stock_factors['etf'].apply(lambda x: x[0:6])

    final_df['etfname'] = final_df['option_code'].apply(lambda x: x[0:6])
    etfs = list(final_df['etfname'].unique())
    print(etfs)
    for i in range(len(etfs)):
        final_df[f'etfidictor{i + 1}'] = (final_df['etfname'] == etfs[i]).astype(int)

    for i in range(1, 13):
        final_df[f'month{i}'] = (final_df['month'] % 100 == i).astype(int)

    final_df = final_df.merge(stock_factors, on=['etfname', 'month'], how='left')

    final_df = final_df.drop(columns=drop_columns)
    final_df.sort_values(by=['option_code', 'month'], inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    final_df = final_df[final_df['next_return'].notna()]

    display_dataframe('final_df', final_df)
    dump_pickle(RESULT_PATH + '04_final_df.pkl', final_df)
