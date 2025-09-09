import joblib
import time
import pandas as pd
from config import UNDERLYING_ASSET_LIST, DATA_PATH, RISK_FREE_ASSET, MULTIPLIER, RESULT_PATH, END_MONTH
from config import load_pickle, dump_pickle, display_dataframe


def process_option_excess_return(param_tuple) -> dict:
    start_time = time.time()
    option_code, month, option_df_month = param_tuple
    if len(option_df_month) > 1:
        option_df_month.sort_values(by=['日期'], ascending=True, inplace=True)

        option_price_tau = option_df_month['期权结算价'].iloc[-1] * MULTIPLIER
        option_price_t = option_df_month['期权结算价'].iloc[0] * MULTIPLIER
        delta_t = option_df_month['Delta'].iloc[0]
        stock_price_t = option_df_month['ETF收盘价'].iloc[0] * MULTIPLIER

        option_df_month['An'] = (option_df_month['日期'].shift(-1) - option_df_month['日期']).apply(lambda x: x.days)
        implied_volatility = option_df_month['隐含波动率'].mean(skipna=True)
        absolute_delta = abs(option_df_month['Delta'].mean(skipna=True))
        option_df_month.drop(option_df_month.index[-1], inplace=True)

        delta_hedge = (option_df_month['Delta'] * (
                    option_df_month['ETF后收盘价'] - option_df_month['ETF收盘价']) * MULTIPLIER).sum()
        delta_hedge_cost = (option_df_month['An'] * option_df_month['GC001'] / 100 / 365 * (
                option_df_month['期权结算价'] - option_df_month['Delta'] * option_df_month[
            'ETF收盘价']) * MULTIPLIER).sum()

        pi_t_tau = (option_price_tau - option_price_t - delta_hedge - delta_hedge_cost)
        excess_return_t_tau = pi_t_tau / abs(delta_t * stock_price_t - option_price_t)

        ans_dict = {'option_code': option_code, 'month': month, 'optionmom': excess_return_t_tau,
                    'implied_volatility': implied_volatility,
                    'absolute_delta': absolute_delta, #'call': int('C' in option_code), 'put': int('P' in option_code),
                    # '是否起始月': int(option_df_month['是否起始月'].max()), '是否到期月': int(option_df_month['是否到期月'].max()),
                    'dates': len(option_df_month),
                    # '起始交易日': option_df_month['日期'].min(), '结束交易日': option_df_month['日期'].max(),
                    'remaindates': option_df_month['到期剩余天数'].max(),
                    # '结束至到期日天数': option_df_month['到期剩余天数'].min(),
                    'remaintradedates': option_df_month[
                        '到期剩余交易日'].max()}  # , '结束至到期日交易日数': option_df_month['到期剩余交易日'].min()}
    else:
        ans_dict = {'option_code': option_code, 'month': month}

    end_time = time.time()
    print(
        f'[Metrics Calculate]\tThe metrics for option {option_code} in month {month} have been calculated.\t Time usage: {round(end_time - start_time, 4)} sec.')
    return ans_dict


def option_data_processing(etf_option: dict, etf: dict, risk_free_rate: pd.DataFrame) -> list[
    tuple[str, int, pd.DataFrame]]:
    start_time = time.time()
    param_list = []
    for underlying_asset in UNDERLYING_ASSET_LIST:
        final_df = etf_option[underlying_asset]
        etf_df = etf[underlying_asset]
        etf_df.sort_values(by=['日期'], inplace=True)
        etf_df['ETF后收盘价'] = etf_df['收盘价(元)'].shift(-1)

        final_df['交易日期'] = pd.to_datetime(final_df['交易日期'])
        final_df = pd.merge(final_df, etf_df, left_on='交易日期', right_on='日期', how='left')
        final_df = pd.merge(final_df, risk_free_rate, left_on='交易日期', right_on='日期', how='left')
        final_df = final_df[
            ['证券代码', '交易日期', '前结算价', '结算价', 'Delta', '隐含波动率', '到期剩余天数', '到期剩余交易日',
             '到期日', '收盘价(元)', 'ETF后收盘价',
             'GC001', 'GC002', 'GC003', 'GC007']]
        final_df = final_df.convert_dtypes()
        final_df['到期日'] = pd.to_datetime(final_df['到期日'])
        final_df = final_df.rename(
            columns={'交易日期': '日期', '前结算价': '期权前结算价', '结算价': '期权结算价', '收盘价(元)': 'ETF收盘价'})

        for option_code, option_df in final_df.groupby('证券代码'):
            option_df['month'] = option_df['日期'].dt.strftime('%Y%m').astype(int)
            option_df['是否起始月'] = option_df['month'] == option_df['month'].min()
            option_df['是否到期月'] = option_df['month'] == option_df['month'].max()
            for month, option_df_month in option_df.groupby('month'):
                if month <= END_MONTH:
                    param_list.append((option_code, month, option_df_month))

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
    risk_free_df = load_pickle(f'{DATA_PATH}/无风险利率日行情.pkl')[RISK_FREE_ASSET]

    # Data Processing
    params_list = option_data_processing(etf_option=etf_option_dict, etf=etf_dict, risk_free_rate=risk_free_df)

    # Multiprocessing Data
    result_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_option_excess_return)(params) for params in params_list)
    print('=' * 200)
    result_df = pd.DataFrame(result_list).sort_values(by=['option_code', 'month']).reset_index(
        drop=True).dropna().convert_dtypes()
    display_dataframe('00_excess_return', result_df)

    dump_pickle(RESULT_PATH + '00_excess_return.pkl', result_df)

    finish_time = time.time()
    print(f'Process finished. Total time usage: {round((finish_time - begin_time) / 60, 4)} minutes.')
