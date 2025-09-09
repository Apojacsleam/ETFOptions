import time
import joblib
import numpy as np
import pandas as pd
from config import UNDERLYING_ASSET_LIST, DATA_PATH, RESULT_PATH, END_MONTH
from config import load_pickle, dump_pickle, display_dataframe

# 设置显示的最大列数和列宽
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def stock_level_factor_generate(param_tuple) -> list[dict]:
    start_time = time.time()
    underlying_asset, month, option_df, etf_df, etf_volume_df, etf_df_all = param_tuple

    final_df = pd.merge(option_df, etf_df, left_on='交易日期', right_on='日期', how='left')
    final_df = final_df[
        ['证券代码', '交易日期', '前结算价', '结算价', '最高价', '最低价', '涨跌(收-结)', '行权价', '成交量(手)',
         '成交额(万元)', '持仓量(手)', 'Delta',
         'Gamma', 'Theta', 'Vega', 'Rho', '隐含波动率', '到期剩余天数', '到期剩余交易日', '到期日', '收盘价(元)']]
    final_df = final_df.convert_dtypes()
    final_df['到期日'] = pd.to_datetime(final_df['到期日'])
    final_df = final_df.rename(
        columns={'交易日期': '日期', '前结算价': '期权前结算价', '结算价': '期权结算价', '收盘价(元)': 'ETF收盘价',
                 '成交量(手)': '期权成交量',
                 '成交额(万元)': '期权成交额', '持仓量(手)': '期权持仓量', '最高价': '期权最高价',
                 '最低价': '期权最低价',
                 '涨跌(收-结)': '收结涨跌幅'})
    # 计算moneyness和实值、虚值、在值期权
    final_df['moneyness'] = final_df.apply(
        lambda row: row['ETF收盘价'] / row['行权价'] if 'C' in row['证券代码'] else row['行权价'] / row['ETF收盘价'],
        axis=1)

    dfs = list()
    for date, option_df_date in final_df.groupby('日期'):
        etf_value = option_df_date['ETF收盘价'].mean()
        high_option_value = option_df_date.loc[option_df_date['行权价'] >= etf_value, '行权价'].min()
        option_df_date.loc[abs(option_df_date['行权价'] - high_option_value) <= 1e-4, 'bucket'] = 'ATM'
        low_option_value = option_df_date.loc[option_df_date['行权价'] <= etf_value, '行权价'].max()
        option_df_date.loc[abs(option_df_date['行权价'] - low_option_value) <= 1e-4, 'bucket'] = 'ATM'
        option_df_date.loc[option_df_date['bucket'].isna() & (option_df_date['moneyness'] > 1), 'bucket'] = 'OTM'
        option_df_date.loc[option_df_date['bucket'].isna() & (option_df_date['moneyness'] < 1), 'bucket'] = 'ITM'
        dfs.append(option_df_date)
    option_df_month = pd.concat(dfs)

    atm_option_df_month = option_df_month[option_df_month['bucket'] == 'ATM']

    # 隐含波动率斜率
    ivLT = atm_option_df_month[atm_option_df_month['到期剩余天数'] == atm_option_df_month['到期剩余天数'].max()][
        '隐含波动率'].mean(skipna=True)
    iv1M = atm_option_df_month[atm_option_df_month['到期剩余天数'] == atm_option_df_month['到期剩余天数'].min()][
        '隐含波动率'].mean(skipna=True)
    ivslope = ivLT - iv1M

    # print(etf_df.columns)

    # 股票与期权的交易量比例
    stock_volume = etf_df['成交量(份)'].sum(skipna=True)
    option_volume = option_df_month['期权成交量'].sum(skipna=True)
    so = stock_volume / option_volume
    lso = np.log(so)

    # 股票与期权的交易额比例
    stock_value = etf_df['成交金额(元)'].sum(skipna=True)
    option_value = option_df_month['期权成交额'].sum(skipna=True)
    dso = stock_value / option_value
    ldso = np.log(dso)

    # pcratio
    pcr_volume = etf_volume_df['成交量PCR'].values[0]
    pcr_value = etf_volume_df['成交额PCR'].values[0]
    pcr_oi = etf_volume_df['持仓量PCR'].values[0]

    ans_dict = {'etf': underlying_asset, 'month': month, 'ivslope': ivslope, 'so': so, 'lso': lso, 'dso': dso,
                'ldso': ldso, 'pcr_volume': pcr_volume, 'pcr_value': pcr_value,
                'pcr_oi': pcr_oi}

    # 股票风险中性动量、波动性、偏度和峰度
    for monthdelta in [1, 3, 6, 9, 12]:
        begin_date = etf_df_all['日期'].max() - pd.DateOffset(months=monthdelta)
        etf_df_delta = etf_df_all.copy(deep=True)[etf_df_all['日期'] > begin_date]
        stock_returns = etf_df_delta['收盘价(元)'].apply(np.log) - etf_df_delta['前收盘价(元)'].apply(np.log)
        ans_dict[f'rns_{monthdelta}m'] = stock_returns.skew(skipna=True)
        ans_dict[f'rnk_{monthdelta}m'] = stock_returns.kurtosis(skipna=True)
        ans_dict[f'retmom_{monthdelta}m'] = stock_returns.mean(skipna=True) * 100
        ans_dict[f'retstd_{monthdelta}m'] = stock_returns.std(skipna=True) * 100

    # 最新收盘价
    ans_dict['stkclose'] = etf_df['收盘价(元)'].iloc[-1]
    ans_dict['stkvol'] = etf_df['成交量(份)'].mean(skipna=True)
    ans_dict['stkdvol'] = np.log(ans_dict['stkvol'])

    ans_dict['stkturnover'] = etf_df['换手率(%)'].mean(skipna=True)
    ans_dict['stkturnoverstd'] = etf_df['换手率(%)'].std(skipna=True)

    ans_dict['stkvolmean'] = etf_df['成交量(份)'].std(skipna=True) / etf_df['成交量(份)'].mean(skipna=True)

    end_time = time.time()
    print(f'[Metrics Calculate]\tThe metrics for ETF {underlying_asset} in month {month} have been calculated.\t Time usage: {round(end_time - start_time, 4)} sec.')
    return ans_dict


def option_data_processing(etf_option: dict, etf: dict, etf_volume_dict: dict) -> list[tuple[int, pd.DataFrame]]:
    start_time = time.time()
    param_list = []
    for underlying_asset in UNDERLYING_ASSET_LIST:
        final_df = etf_option[underlying_asset]
        etf_df = etf[underlying_asset]
        etf_volume = etf_volume_dict[underlying_asset]

        final_df.sort_values(by=['交易日期', '证券代码'], inplace=True)
        final_df['交易日期'] = pd.to_datetime(final_df['交易日期'])
        final_df['month'] = final_df['交易日期'].dt.strftime('%Y%m').astype(int)

        etf_df['month'] = etf_df['日期'].dt.strftime('%Y%m').astype(int)
        etf_df.sort_values(by=['日期'], inplace=True)
        etf_volume.sort_values(by=['日期'], inplace=True)
        etf_volume.rename(columns={'日期': 'month'}, inplace=True)

        etf_df = etf_df.convert_dtypes()
        etf_volume = etf_volume.convert_dtypes()
        final_df = final_df.convert_dtypes()

        for month, final_df_month in final_df.groupby('month'):
            if month <= END_MONTH:
                etf_df_month = etf_df.copy(deep=True)[etf_df['month'] == month]
                etf_volume_month = etf_volume.copy(deep=True)[etf_volume['month'] == month]
                etf_df_all = etf_df.copy(deep=True)[etf_df['month'] <= month]
                param_list.append((underlying_asset, month, final_df_month, etf_df_month, etf_volume_month, etf_df_all))
        """

        etf_df.sort_values(by=['日期'], inplace=True)
        etf_df['ETF后收盘价'] = etf_df['收盘价(元)'].shift(-1)

        
        final_df = pd.merge(final_df, etf_df, left_on='交易日期', right_on='日期', how='left')
        final_df = final_df[['证券代码', '交易日期', '前结算价', '结算价', '最高价', '最低价', '涨跌(收-结)', '行权价', '成交量(手)', '成交额(万元)', '持仓量(手)', 'Delta',
                             'Gamma', 'Theta', 'Vega', 'Rho', '隐含波动率', '到期剩余天数', '到期剩余交易日', '到期日', '收盘价(元)']]
        final_df = final_df.convert_dtypes()
        final_df['到期日'] = pd.to_datetime(final_df['到期日'])
        final_df = final_df.rename(columns={'交易日期': '日期', '前结算价': '期权前结算价', '结算价': '期权结算价', '收盘价(元)': 'ETF收盘价', '成交量(手)': '期权成交量',
                                            '成交额(万元)': '期权成交额', '持仓量(手)': '期权持仓量', '最高价': '期权最高价', '最低价': '期权最低价',
                                            '涨跌(收-结)': '收结涨跌幅'})
        # 计算moneyness和实值、虚值、在值期权
        final_df['moneyness'] = final_df.apply(lambda row: row['ETF收盘价'] / row['行权价'] if 'C' in row['证券代码'] else row['行权价'] / row['ETF收盘价'], axis=1)

        # 计算到期月和起始月
        final_df['month'] = final_df['日期'].dt.strftime('%Y%m').astype(int)
        final_df['起始月'] = final_df.groupby('证券代码')['month'].transform('min')
        final_df['到期月'] = final_df.groupby('证券代码')['month'].transform('max')

        for month, option_df_month in final_df.groupby('month'):
            if month <= END_MONTH:
                param_list.append((month, option_df_month))
        """

    end_time = time.time()
    print(f'[Data Process]\tOption data have been processed.\tTime usage: {round(end_time - start_time, 4)} sec.')
    print('=' * 200)
    return param_list


if __name__ == '__main__':
    print('=' * 200)
    begin_time = time.time()

    # Load Data
    etf_option_dict = dict()
    for etf_name in UNDERLYING_ASSET_LIST:
        etf_option_dict[etf_name] = load_pickle(f'{DATA_PATH}ETFOption/ETF期权日行情{etf_name}.pkl')
    etf_dict = load_pickle(f'{DATA_PATH}/ETF基金日行情.pkl')
    etf_volume_dict = load_pickle(f'{DATA_PATH}ETFOption/ETF期权月成交量.pkl')

    # Data Processing
    params_list = option_data_processing(etf_option=etf_option_dict, etf=etf_dict, etf_volume_dict=etf_volume_dict)

    # Multiprocessing Data
    result_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(stock_level_factor_generate)(params) for params in params_list)


    print('=' * 200)
    result_df = pd.DataFrame(result_list).sort_values(by=['etf', 'month']).reset_index(drop=True).dropna(how='all').convert_dtypes()
    display_dataframe('03_stock_level_factors', result_df)

    dump_pickle(RESULT_PATH + '03_stock_level_factors.pkl', result_df)

    finish_time = time.time()
    print(f'Process finished. Total time usage: {round((finish_time - begin_time) / 60, 4)} minutes.')
