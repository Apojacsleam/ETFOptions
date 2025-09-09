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

## 许可证

仅用于学术研究与学习交流，未经许可请勿用于商业用途。


