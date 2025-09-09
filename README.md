## 项目简介

本项目围绕中国 ETF/股指期权的月度收益预测与策略评估，包含：
- 数据清洗与结构化存储（Excel → Pickle）
- 超额收益标签构造（基于 Delta 对冲收益）
- 合约层/桶层/标的层因子工程
- 因子合并与数据集构建
- 机器学习模型训练与滚动预测
- 投资组合构建与绩效评估/可视化

目录位于 `Codes/`，数据位于 `Data/`，结果输出位于 `Result/`。

---

## 数据与路径

- 原始数据：`DataProcessing/RawData/` 下各分类 Excel 文件（ETF 基金、期权、指数、无风险利率等）
- 结构化数据（Pickle）：`Data/` 下对应 `.pkl`
- 结果输出：`Result/` 下各中间/最终结果
- 全局配置：`FactorGenerate/config.py`
  - `UNDERLYING_ASSET_LIST`：标的列表
  - `DATA_PATH` / `RESULT_PATH`：数据与结果相对路径
  - `END_MONTH`：样本截止月份（含）

注意：本项目中脚本默认从 `FactorGenerate/` 目录相对路径读取数据（`../Data/`）并输出结果至 `../Result/`。

---

## 环境要求

- Python 3.9+
- 主要依赖：
  - pandas, numpy, scikit-learn, joblib, tqdm, matplotlib, shap, xgboost

示例安装（建议使用虚拟环境）：
```bash
pip install pandas numpy scikit-learn joblib tqdm matplotlib shap xgboost
```

---

## 处理与建模流水线

以下步骤按推荐顺序执行：

1) Excel → Pickle（可选，若 `Data/` 已存在 `.pkl` 可跳过）
- 脚本：`FactorGenerate/00_excel_to_pickle.py`
- 功能：遍历 `config.DATA_PATH` 下所有 Excel，转换为 `.pkl`
```bash
python FactorGenerate/00_excel_to_pickle.py
```

2) 超额收益与基础标签构造
- 脚本：`FactorGenerate/01_excess_return.py`
- 产出：`Result/00_excess_return.pkl`
- 说明：计算月度 Delta 对冲后的超额收益等标签（如 `optionmom`），会用到无风险利率（`RISK_FREE_ASSET`）及 ETF 日行情。

3) 合约层因子
- 脚本：`FactorGenerate/02_option_contract_level.py`
- 产出：`Result/01_contract_level_factors.pkl`

4) 桶层因子（按行权价/虚实值分组等）
- 脚本：`FactorGenerate/03_option_bucket_level.py`
- 产出：`Result/02_bucket_level_factors.pkl`

5) 标的层因子（关键：严格时序截断）
- 脚本：`FactorGenerate/04_stock_level.py`
- 产出：`Result/03_stock_level_factors.pkl`
- 要点：
  - 传入 `etf_df_all = etf_df[etf_df['month'] <= month]`，保证每个样本 `(etf, month)` 的特征仅使用该月及之前的数据。
  - 计算动量/波动/偏度/峰度等多期统计时，窗尾为样本月最后交易日（`etf_df_all['日期'].max()`），避免未来信息泄露。

6) 因子合并与建模样本集构建
- 脚本：`FactorGenerate/05_factor_combo.py`
- 产出：`Result/04_final_df.pkl`
- 说明：
  - 构造 `next_return`：将 `00_excess_return.pkl` 中的 `optionmom` 向后对齐到次月作为标签；
  - 合并合约层/桶层/标的层因子；
  - 增加月份与 ETF 的哑变量（注意后续可去基避免完全共线）。

7) 训练集/预测集切分（滚动时间外推）
- 产出：`Result/05_train_data_split.pkl` 或 `05_train_data_split2.pkl`
- 说明：仓库未包含显式切分脚本，请确保切分满足：
  - 对于每个预测月 `m_pred`，训练数据的月份均严格小于 `m_pred`；
  - 不同 ETF/合约之间避免交叉污染（必要时按标的隔离或在月维度上严格划分）；
  - 输出结构示例：`{'month': m_pred, 'label': 'Call/Put/All', 'train_data': df_train, 'predict_data': df_os}`。

8) 模型训练与滚动预测
- 脚本：`FactorGenerate/06_machine_learning_train.py`
- 输入：`Result/05_train_data_split2.pkl`
- 产出：`Result/06_train_result.pkl`
- 说明：
  - 内置模型：Lasso/Ridge/ElasticNet/PCR/PLS/RF/AdaBoost/GBR/MLP 等；
  - 列缩放：对训练集按列 Min-Max 缩放，并用同一缩放参数变换预测集；
  - 建议将交叉验证改为 `TimeSeriesSplit`，避免时序数据的乐观偏差；
  - 建议对常数列跳过缩放或统一设为 0；缩放后再次 `fillna(0)`。

9) R^2 分析与可视化
- 脚本：`FactorGenerate/07_rsquare_analysis.py`, `08_Rsquare_figure.py`, `11_Rsquare_analysis.py`
- 产出：`Result/07_result_list.pkl`, `Result/Rsquare_OS.xlsx`, `Result/figures/Rsquare_OS.pdf` 等

10) 相关性与重要性分析
- 相关性：`FactorGenerate/10_correlation_analysis.py`
- 重要性（示例 XGBoost + SHAP）：`FactorGenerate/12_importance_analysis.py`

11) 组合构建与绩效评估
- 脚本：`FactorGenerate/13_portfolio_construction.py`
- 产出：`Result/metrics_*.xlsx`, `Result/figures/portofolio_*.jpg`
- 说明：对不同模型的月度方向/幅度预测构建策略并评估夏普、回撤等；交易成本默认按月固定调整，可参数化。

---

## 运行示例（最小闭环）

按顺序执行关键脚本（若 `Data/` 已准备好，可跳过步骤 1）：
```bash
python FactorGenerate/00_excel_to_pickle.py
python FactorGenerate/01_excess_return.py
python FactorGenerate/02_option_contract_level.py
python FactorGenerate/03_option_bucket_level.py
python FactorGenerate/04_stock_level.py
python FactorGenerate/05_factor_combo.py
python FactorGenerate/06_machine_learning_train.py
python FactorGenerate/13_portfolio_construction.py
```

---

## 时序与数据质量注意事项（强烈建议）

- 严格避免未来信息泄露：
  - 标的层因子：对每个样本 `(etf, month)` 的多期统计以“该月最后交易日”为窗尾，仅回溯历史数据；
  - 切分：`predict_data.month > train_data.month.max()`，并避免同月样本同时出现在训练与预测。
- 交叉验证：使用 `TimeSeriesSplit` 替代普通 `KFold`/`cv=5`。
- 列缩放：对 `max == min` 的列避免除零，统一置 0 或跳过缩放；缩放后再次 `fillna(0)`。
- 列集合与顺序：训练/预测严格按同一列序对齐并断言一致；数据类型显式为 `float64`。
- 特征工程：对 ETF 与月份的哑变量去基，避免完全共线，便于线性模型稳健估计。
- 模型选择：`GaussianRBM` 不适配回归任务，建议移除或替换为更合适的降维/非线性方法。

---

## 结果与产出文件（部分）

- `Result/00_excess_return.pkl`：月度超额收益与相关标签
- `Result/01_contract_level_factors.pkl`：合约层因子
- `Result/02_bucket_level_factors.pkl`：桶层因子
- `Result/03_stock_level_factors.pkl`：标的层因子
- `Result/04_final_df.pkl`：合并后的训练样本基础表
- `Result/05_train_data_split*.pkl`：按月滚动切分的训练与预测数据列表
- `Result/06_train_result.pkl`：各模型滚动预测结果
- `Result/Rsquare_OS.xlsx`、`Result/figures/Rsquare_OS.pdf`：R^2 结果
- `Result/metrics_*.xlsx`、`Result/figures/portofolio_*.jpg`：策略绩效与净值图

---

## FAQ

1) 缺失 `05_train_data_split*.pkl` 怎么办？
- 需要在 `Result/04_final_df.pkl` 基础上，按时间滚动生成若干字典项：`{'month', 'label', 'train_data', 'predict_data'}`，并序列化保存。

2) 如何确认未泄露？
- 抽查若干样本 `(etf, month)`，验证用于特征统计的 `etf_df_all['日期'].max()` 等于该月最后交易日；
- 验证 `predict_data.month > train_data.month.max()`；
- 在训练前后打印列集合与顺序一致性。

3) 如何更换标的或时间范围？
- 修改 `FactorGenerate/config.py` 中的 `UNDERLYING_ASSET_LIST` 与 `END_MONTH`。

---

## 许可证

仅用于学术研究与学习交流，未经许可请勿用于商业用途。


