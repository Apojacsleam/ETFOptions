import numpy as np
import pandas as pd
from config import load_pickle, dump_pickle
from config import RESULT_PATH
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 10, "mathtext.fontset": "cm"})

if __name__ == '__main__':
    data_split = load_pickle(RESULT_PATH + '13_importance_analysis.pkl')

    train_data = data_split['train_data']

    for col in train_data.columns:
        if train_data[col].dtype == 'Float64' and col != 'next_return':
            maxv = train_data[col].max()
            minv = train_data[col].min()
            train_data[col] = (train_data[col] - minv) / (maxv - minv)

    features_to_display = ['absolute_delta', 'pfht', 'retmom_3m', 'implied_volatility', 'embedlev', 'retstd_9m', 'retstd_6m', 'bucket_value_share', 'retmom_1m',
                           'so', 'hvol', 'bucket_oi_share', 'moneyness', 'retmom_12m', 'remaindates', 'delta', 'rns_6m', 'theta', 'rho',
                           'next_return', 'closediffvwap']  # , 'ttm','bucket_volume_share', 'remaintradedates', 'next_return']

    train_data = train_data.fillna(0).convert_dtypes(convert_integer='int32').replace([np.inf, -np.inf], 0).fillna(0)
    train_data = train_data[features_to_display]
    train_data = train_data.rename(columns = {'implied_volatility': 'iv'})
    print(train_data.columns)

    train_y = train_data['next_return'].copy(deep=True)

    train_X = train_data.drop(columns=['next_return'])

    # Initialize the XGBoost Regressor
    model = xgb.XGBRegressor(n_estimators=100, tree_method='gpu_hist')

    # Fit the model on training data
    model.fit(train_X, train_y)

    print('process end.')

    # Check if XGBoost is using GPU
    if hasattr(model, 'n_gpus'):
        if model.n_gpus != 0:
            print('XGBoost is running on GPU.')
        else:
            print('XGBoost is not using GPU.')
    else:
        print('Unable to determine if XGBoost is using GPU.')
    # ['next_return',
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(train_X)

    # summarize the effects of all the features
    #plt = shap.plots.beeswarm(shap_values, max_display=20, plot_size=(16, 10), show=False)

    #plt.figure.savefig(RESULT_PATH + 'figures/shap.jpg', dpi=600)

    plt.figure(figsize=(16, 10))
    splt = shap.plots.bar(shap_values * 100, max_display=20, show=False)

    plt.xlabel(r'mean(|(SHAP value)|) $\times 10^2$')
    splt.figure.savefig(RESULT_PATH + 'figures/shap_bar.jpg', dpi=600)


