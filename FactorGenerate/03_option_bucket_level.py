import time
import joblib
import pandas as pd
from config import UNDERLYING_ASSET_LIST, DATA_PATH, RESULT_PATH, END_MONTH
from config import load_pickle, dump_pickle, display_dataframe


def contract_level_factor_generate(param_tuple) -> list[dict]:
    start_time = time.time()
    month, option_df_month = param_tuple

    dfs = list()
    for date, option_df_date in option_df_month.groupby('日期'):
        etf_value = option_df_date['ETF收盘价'].mean()
        high_option_value = option_df_date.loc[option_df_date['行权价'] >= etf_value, '行权价'].min()
        option_df_date.loc[abs(option_df_date['行权价'] - high_option_value) <= 1e-4, 'bucket'] = 'ATM'
        low_option_value = option_df_date.loc[option_df_date['行权价'] <= etf_value, '行权价'].max()
        option_df_date.loc[abs(option_df_date['行权价'] - low_option_value) <= 1e-4, 'bucket'] = 'ATM'
        option_df_date.loc[option_df_date['bucket'].isna() & (option_df_date['moneyness'] > 1), 'bucket'] = 'OTM'
        option_df_date.loc[option_df_date['bucket'].isna() & (option_df_date['moneyness'] < 1), 'bucket'] = 'ITM'
        dfs.append(option_df_date)

    option_df_month = pd.concat(dfs)
    result = option_df_month.groupby('证券代码')['bucket'].agg(lambda x: x.value_counts().idxmax()).reset_index()
    option_df_month = option_df_month.merge(result, on='证券代码', suffixes=('', '_new'))
    option_df_month['bucket'] = option_df_month['bucket_new']
    option_df_month.drop(columns=['bucket_new'], inplace=True)

    dfs = list()
    all_volume = option_df_month['期权成交量'].sum(skipna=True)
    all_value = option_df_month['期权成交额'].sum(skipna=True)
    all_oi = option_df_month['期权持仓量'].sum(skipna=True)
    for bucket, option_df_bucket in option_df_month.groupby('bucket'):
        bucket_volume = option_df_bucket['期权成交量'].sum(skipna=True)
        bucket_oi = option_df_bucket['期权成交额'].sum(skipna=True)
        bucket_value = option_df_bucket['期权持仓量'].sum(skipna=True)
        bucket_volume_share = bucket_volume / all_volume
        bucket_value_share = bucket_value / all_value
        bucket_oi_share = bucket_oi / all_oi
        option_df_bucket['bucket_volume'] = bucket_volume
        option_df_bucket['bucket_value'] = bucket_volume
        option_df_bucket['bucket_oi'] = bucket_oi
        option_df_bucket['bucket_volume_share'] = bucket_volume_share
        option_df_bucket['bucket_value_share'] = bucket_value_share
        option_df_bucket['bucket_oi_share'] = bucket_oi_share
        dfs.append(option_df_bucket)
    option_df_month = pd.concat(dfs)

    ans_dict_list = list()
    for code, option_df_code in option_df_month.groupby('证券代码'):
        bucket = option_df_code['bucket'].values[0]

        atm = int(bucket == 'ATM')
        otm = int(bucket == 'OTM')
        itm = int(bucket == 'ITM')

        volume = option_df_code['期权成交量'].sum(skipna=True)
        value = option_df_code['期权成交额'].sum(skipna=True)
        oi = option_df_code['期权持仓量'].sum(skipna=True)

        bucket_volume = option_df_code['bucket_volume'].mean(skipna=True)
        bucket_value = option_df_code['bucket_value'].mean(skipna=True)
        bucket_oi = option_df_code['bucket_oi'].mean(skipna=True)

        bucket_volume_ratio = volume / bucket_volume
        bucket_value_ratio = value / bucket_value
        bucket_oi_ratio = oi / bucket_oi

        bucket_volume_share = option_df_code['bucket_volume_share'].mean(skipna=True)
        bucket_value_share = option_df_code['bucket_value_share'].mean(skipna=True)
        bucket_oi_share = option_df_code['bucket_oi_share'].mean(skipna=True)

        ans_dict = {'option_code': code, 'month': month, 'atm': atm, 'otm': otm, 'itm': itm, 'bucket_volume_ratio': bucket_volume_ratio,
                    'bucket_value_ratio': bucket_value_ratio, 'bucket_oi_ratio': bucket_oi_ratio, 'bucket_oi': bucket_oi, 'bucket_volume': bucket_volume,
                    'ovolume': volume, 'ovalue': value, 'oi': oi, 'bucket_value_share': bucket_value_share, 'bucket_oi_share': bucket_oi_share,
                    'bucket_volume_share': bucket_volume_share}
        ans_dict_list.append(ans_dict)

    end_time = time.time()
    print(f'[Metrics Calculate]\tThe metrics for options in month {month} have been calculated.\t Time usage: {round(end_time - start_time, 4)} sec.')
    return ans_dict_list


def option_data_processing(etf_option: dict, etf: dict) -> list[tuple[int, pd.DataFrame]]:
    start_time = time.time()
    param_list = []
    for underlying_asset in UNDERLYING_ASSET_LIST:
        final_df = etf_option[underlying_asset]
        etf_df = etf[underlying_asset]
        etf_df.sort_values(by=['日期'], inplace=True)
        etf_df['ETF后收盘价'] = etf_df['收盘价(元)'].shift(-1)

        final_df['交易日期'] = pd.to_datetime(final_df['交易日期'])
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

    # Data Processing
    params_list = option_data_processing(etf_option=etf_option_dict, etf=etf_dict)

    for params in params_list:
        contract_level_factor_generate(params)

    # Multiprocessing Data
    lists = joblib.Parallel(n_jobs=-1)(joblib.delayed(contract_level_factor_generate)(params) for params in params_list)

    result_list = []
    for results in lists:
        result_list.extend(results)

    print('=' * 200)
    result_df = pd.DataFrame(result_list).sort_values(by=['option_code', 'month']).reset_index(drop=True).dropna(how='all').convert_dtypes()
    display_dataframe('02_bucket_level_factors', result_df)

    dump_pickle(RESULT_PATH + '02_bucket_level_factors.pkl', result_df)

    finish_time = time.time()
    print(f'Process finished. Total time usage: {round((finish_time - begin_time) / 60, 4)} minutes.')
