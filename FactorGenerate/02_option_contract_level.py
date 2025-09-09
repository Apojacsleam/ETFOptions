import time
import joblib
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from config import UNDERLYING_ASSET_LIST, DATA_PATH, RESULT_PATH, END_MONTH
from config import load_pickle, dump_pickle, display_dataframe


def contract_level_factor_generate(param_tuple) -> dict:
    warnings.filterwarnings("ignore")
    start_time = time.time()
    option_code, month, option_df_month = param_tuple

    if len(option_df_month) > 1:
        # 若期权是认购期权，指示器等于1，否则等于0
        call = 1 if 'C' in option_code else 0
        # 若期权是认沽期权，指示器等于1，否则等于0
        put = 1 if 'P' in option_code else 0
        # 若期权在观察月份内到期，指示器等于1，否则等于0
        expiration = int(month == option_df_month['到期月'].values[0])
        # 若期权在观察月份内上市，指示器等于1，否则等于0
        issue = int(month == option_df_month['起始月'].values[0])
        # 若期权是认购期权，等于S/K；否则等于K/S
        expiration_month = option_df_month['到期月'].values[0]
        ttm = (expiration_month // 100 - month // 100) * 12 + expiration_month % 100 - month % 100
        # 若期权是认购期权，等于S/K；否则等于K/S
        if 'C' in option_code:
            moneyness = (option_df_month['ETF收盘价'] / option_df_month['行权价']).mean(skipna=True)
        else:
            moneyness = (option_df_month['行权价'] / option_df_month['ETF收盘价']).mean(skipna=True)
        # 期权的隐含波动率
        iv = option_df_month['隐含波动率'].mean(skipna=True)
        skewiv = option_df_month['隐含波动率'].skew(skipna=True)
        kurtiv = option_df_month['隐含波动率'].kurtosis(skipna=True)

        # 期权价值对标的资产价格变动的敏感性
        delta = option_df_month['Delta'].mean(skipna=True)
        # Delta对标的资产价格变动的敏感性，乘标的资产价格以在截面上比较
        gamma = (option_df_month['Gamma'] * option_df_month['ETF收盘价']).mean(skipna=True)
        # 期权对隐含波动率变动的敏感性
        theta = option_df_month['Theta'].mean(skipna=True)
        # 期权价值随时间的衰减
        vega = option_df_month['Vega'].mean(skipna=True)
        # 期权价格对利率变动敏感性
        rho = option_df_month['Rho'].mean(skipna=True)
        # 该期权合约的内在杠杆
        embedlev = (option_df_month['ETF收盘价'] / option_df_month['期权结算价'] * option_df_month['Delta']).apply(
            abs).mean(skipna=True)
        # 该期权合约的日持仓量平均值
        oi = option_df_month['期权持仓量'].mean(skipna=True)
        # 该期权合约的日成交额平均值
        avgvolume = option_df_month['期权成交量'].mean(skipna=True)
        # 该期权合约的日成加量平均值
        avgvalue = option_df_month['期权成交额'].mean(skipna=True)
        # 该期权合约的(日最高价-日最低价) / 日最低价的均值
        amp = ((option_df_month['期权最高价'] - option_df_month['期权最低价']) / option_df_month['期权最低价']).mean(
            skipna=True)
        # 该期权合约的(日收盘价-日结算价) / 日结算价的均值
        closediffvwap = (option_df_month['收结涨跌幅'] / option_df_month['期权结算价']).replace([np.inf, -np.inf],
                                                                                                np.nan).mean(
            skipna=True)
        # 该期权合约的(日收盘价-日结算价) / 日结算价的绝对值的均值
        absclosediffvwap = (option_df_month['收结涨跌幅'] / option_df_month['期权结算价']).apply(abs).replace(
            [np.inf, -np.inf], np.nan).mean(skipna=True)
        # illiq流动性指标
        deltapid = option_df_month['期权结算价'].apply(np.log) - option_df_month['期权前结算价'].apply(np.log)
        deltapid = deltapid.replace([np.inf, -np.inf], np.nan).dropna()
        if len(deltapid) <= 2:
            illiq = np.nan
        else:
            x = deltapid[1:].values
            y = deltapid[:-1].values
            illiq = np.nanmean(x * y) - np.nanmean(x) * np.nanmean(y)

        # Roll的日度流动性度量
        return_daily = (option_df_month['期权结算价'] - option_df_month['期权前结算价']) / option_df_month[
            '期权前结算价']
        return_daily = return_daily.replace([np.inf, -np.inf], np.nan).dropna()
        if len(return_daily) <= 2:
            return_cov = np.nan
        else:
            x = return_daily[1:].values
            y = return_daily[:-1].values
            return_cov = np.nanmean(x * y) - np.nanmean(x) * np.nanmean(y)
        roll = 2 * np.sqrt(- return_cov) if return_cov < 0 else 0.0

        # 基于零收益的流动性度量
        zero_return = ((option_df_month['期权结算价'] - option_df_month['期权前结算价']) <= 1e-5).sum()
        pzeros = zero_return / len(option_df_month)

        # 基于零收益的修正流动性度量
        hvol = deltapid.std()
        pfht = 2 * hvol * stats.norm.ppf((1 + pzeros) / 2)

        # 不流动性的Amihud度量
        NAmihud = (option_df_month['期权成交量'] > 0).sum()
        amihud_daily = return_daily.abs() / option_df_month['期权成交量']
        amihud_daily.replace([np.inf, -np.inf], np.nan, inplace=True)
        amihud = amihud_daily.sum() / NAmihud

        # 扩展的Roll度量
        piroll = roll / avgvalue

        # 扩展的FHT度量
        pifht = pfht / avgvalue

        # amihud度量的标准差
        stdamihud = amihud_daily.std()

        # 偏度
        hskew = deltapid.skew()

        # 峰度
        hkurt = deltapid.kurtosis()

        # 持仓量与股票成交量对比
        oistock = (option_df_month['期权持仓量'] / option_df_month['ETF成交量']).mean(skipna=True)

        # 换手率
        turnover_daily = option_df_month['期权成交量'] / option_df_month['期权持仓量']
        turnover_daily.replace([np.inf, -np.inf], np.nan, inplace=True)
        turnover = turnover_daily.mean(skipna=True)

        ans_dict = {'option_code': option_code, 'month': month, 'call': call, 'put': put, 'issue': issue,
                    'expiration': expiration, 'ttm': ttm, 'moneyness': moneyness,
                    'iv': iv, 'skewiv': skewiv, 'kurtiv': kurtiv, 'hvol': hvol, 'delta': delta, 'gamma': gamma,
                    'theta': theta, 'vega': vega, 'rho': rho, 'embedlev': embedlev, 'avgovolume': avgvolume,
                    'avgovalue': avgvalue, 'amp': amp, 'closediffvwap': closediffvwap,
                    'absclosediffvwap': absclosediffvwap, 'illiq': illiq, 'roll': roll, 'pzeros': pzeros,
                    'pfht': pfht, 'amihud': amihud, 'piroll': piroll, 'pifht': pifht, 'stdamihud': stdamihud,
                    'hskew': hskew, 'hkurt': hkurt, 'oistock': oistock,
                    'turnover': turnover}


    else:
        ans_dict = dict()

    end_time = time.time()
    print(
        f'[Metrics Calculate]\tThe metrics for option {option_code} in month {month} have been calculated.\t Time usage: {round(end_time - start_time, 4)} sec.')
    return ans_dict


def option_data_processing(etf_option: dict, etf: dict) -> list[tuple[str, int, pd.DataFrame]]:
    start_time = time.time()
    param_list = []
    for underlying_asset in UNDERLYING_ASSET_LIST:
        final_df = etf_option[underlying_asset]
        etf_df = etf[underlying_asset]
        etf_df.sort_values(by=['日期'], inplace=True)
        etf_df['ETF后收盘价'] = etf_df['收盘价(元)'].shift(-1)

        final_df['交易日期'] = pd.to_datetime(final_df['交易日期'])
        final_df = pd.merge(final_df, etf_df, left_on='交易日期', right_on='日期', how='left')
        final_df = final_df[
            ['证券代码', '交易日期', '前结算价', '结算价', '最高价', '最低价', '涨跌(收-结)', '行权价', '成交量(手)',
             '成交额(万元)', '持仓量(手)', 'Delta', 'Gamma', 'Theta',
             'Vega', 'Rho', '隐含波动率', '到期剩余天数', '到期剩余交易日', '到期日', '收盘价(元)', '成交量(份)',
             '成交金额(元)']]
        final_df = final_df.convert_dtypes()
        final_df['到期日'] = pd.to_datetime(final_df['到期日'])
        final_df = final_df.rename(
            columns={'交易日期': '日期', '前结算价': '期权前结算价', '结算价': '期权结算价', '收盘价(元)': 'ETF收盘价',
                     '成交量(手)': '期权成交量',
                     '成交额(万元)': '期权成交额', '持仓量(手)': '期权持仓量', '最高价': '期权最高价',
                     '最低价': '期权最低价',
                     '涨跌(收-结)': '收结涨跌幅', '成交量(份)': 'ETF成交量', '成交金额(元)': 'ETF成交额'})

        for option_code, option_df in final_df.groupby('证券代码'):
            option_df['month'] = option_df['日期'].dt.strftime('%Y%m').astype(int)
            option_df['起始月'] = option_df['month'].min()
            option_df['到期月'] = option_df['month'].max()
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

    # Data Processing
    params_list = option_data_processing(etf_option=etf_option_dict, etf=etf_dict)

    # Multiprocessing Data
    result_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(contract_level_factor_generate)(params) for params in params_list)
    print('=' * 200)
    result_df = pd.DataFrame(result_list).sort_values(by=['option_code', 'month']).reset_index(drop=True).dropna(
        how='all').convert_dtypes()
    display_dataframe('01_contract_level_factors', result_df)

    dump_pickle(RESULT_PATH + '01_contract_level_factors.pkl', result_df)

    finish_time = time.time()
    print(f'Process finished. Total time usage: {round((finish_time - begin_time) / 60, 4)} minutes.')
