import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import ShuffleSplit, ParameterGrid
from xgboost import XGBRegressor
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from lightgbm import LGBMRegressor

int_rdm = 202512

def start_rfecv(key, clf, df, str_describe, path_data, path_fig, mark_num='', save_file=True):
    """key, clf:遍历字典, str_describe:文件备注用, 会生成表和图"""
    df_rfecv = df.copy()
    min_features_to_select = 25
    X = df_rfecv.drop('value', axis=1)
    y = df_rfecv['value']
    scoring = 'r2'
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=int_rdm)
    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    
    print(f"{key} Optimal number of features: {rfecv.n_features_}")

    selected_features = X.columns[rfecv.support_]
    feature_importances = pd.Series(rfecv.estimator_.feature_importances_, index=selected_features)
    
    all_features_ranking = pd.Series(rfecv.ranking_, index=X.columns)
    
    feature_selection_info = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfecv.support_,
        'Rank': rfecv.ranking_,
    })
    
    if hasattr(rfecv.estimator_, 'feature_importances_'):
        feature_selection_info['Importance'] = feature_selection_info.apply(
            lambda row: feature_importances[row['Feature']] if row['Selected'] else 0, axis=1)
    
    feature_selection_info = feature_selection_info.sort_values(by='Importance', ascending=False)

    if save_file==True:
        feature_selection_info.to_csv(path_data + str_describe + "_rfecv_features_" + key + mark_num+".csv", index=False)
    elif save_file==None:
        pass

    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure(figsize=(6, 4))
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean R2")
    x_plt = range(min_features_to_select, n_scores + min_features_to_select)
    y_plt = rfecv.cv_results_["mean_test_score"]
    line, = plt.plot(x_plt, y_plt, color='C0', label='Mean R2')
    plt.xticks(np.arange(min_features_to_select, n_scores + min_features_to_select, 2))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    yerr=rfecv.cv_results_["std_test_score"]
    y_upper = np.clip(y_plt + yerr, 0, 1)
    y_lower = np.clip(y_plt - yerr, 0, 1)
    plt.fill_between(x_plt, y_lower, y_upper, color=line.get_color(), alpha=0.2, label='±1 std')
    if save_file==True:
        plt.savefig(path_fig + str_describe + "_rfecv_" + scoring +"_"+ key + mark_num+".svg", dpi=300)
    elif save_file==None:
        pass
    df_plot = pd.DataFrame({
        'min_features': range(min_features_to_select, n_scores + min_features_to_select),
        'mean_test_score': rfecv.cv_results_["mean_test_score"],
        'std_test_score': rfecv.cv_results_["std_test_score"]
    })
    df_plot = df_plot.sort_values(by='mean_test_score', ascending=False)
    if save_file==True:
        df_plot.to_csv(path_data + str_describe + "_rfecv_" + scoring +"_" + key + mark_num+".csv", index=False)
    elif save_file==None:
        pass
    return "okk"


def custom_inv(y, list_inv_param, list_treat_value):
    """
    多种逆变换处理
    参数:
        y: 已转换后的值（标量、数组、Series）
        list_inv_param: 三个参数
            zscore: (mean, std, lam)
            minmax: (min, max, lam)
        list_treat_value: 一个或两个元素
            - 如果一个元素: 直接进行box-cox逆变换
            - 两个元素: 第二个元素必须是"zscore"或"minmax"
    返回:
        原始尺度的值
    """
    a = list_inv_param[0]  # mean 或 min
    b = list_inv_param[1]  # std 或 max
    lam = list_inv_param[2]

    if len(list_treat_value) == 1:
        y_bc = y  # 只做 Box-Cox 逆变换
    else:
        transform_type = list_treat_value[1].lower()
        if transform_type == "zscore":
            # 反标准化
            mean = a
            std = b
            y_bc = y * std + mean
        elif transform_type == "minmax":
            # 反归一化
            min_val = a
            max_val = b
            y_bc = y * (max_val - min_val) + min_val
        else:
            raise ValueError("list_treat_value[1] 必须是 'zscore' 或 'minmax'")

    # 反Box-Cox
    if lam == 0:
        return np.exp(y_bc)
    else:
        return np.power(lam * y_bc + 1, 1 / lam)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def grid_search_param_once_cv(df_o, str_describe, path_rfecv_data, model_configs, mark_num='cv', path_part4_temp=None, list_inv_param=None, list_treat_value=None):
    import os
    df_data = df_o.copy()
    results = []
    fold_results = []

    for model_name, config in model_configs.items():
        print(f"{str_describe} {model_name}")
        feature_file = f"{path_rfecv_data}{str_describe}_rfecv_features_{model_name}{mark_num}.csv"
        try:
            selected_features = pd.read_csv(feature_file)
            selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        except FileNotFoundError:
            print(f"特征文件 {feature_file} 未找到，使用所有特征")
            selected_features = df_data.columns.drop('value').tolist()

        X = df_data[selected_features]
        y = df_data['value']

        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=int_rdm)
        param_grid = list(ParameterGrid(config['param_grid']))

        for param_idx, params in enumerate(param_grid):
            print(f"Testing parameters: {params}")
            fold_metrics = []
            pred_records = []  # 用于保存每折的预测和真实值

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                model = config['model_class'](random_state=int_rdm, **config.get('fixed_params', {}), **params)
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # box-cox逆变换
                if list_inv_param is not None:
                    y_test_orig = custom_inv(y_test, list_inv_param, list_treat_value)
                    y_pred_orig = custom_inv(y_pred, list_inv_param, list_treat_value)
                else:
                    y_test_orig = y_test
                    y_pred_orig = y_pred

                # 保存预测结果
                fold_df = pd.DataFrame({
                    'index': X_test.index,
                    'fold': fold_idx,
                    'y_true': y_test.values,
                    'y_pred': y_pred,
                    'y_true_orig': y_test_orig,
                    'y_pred_orig': y_pred_orig
                })
                pred_records.append(fold_df)

                metric_dict = {
                    'model': model_name,
                    'param_idx': param_idx,
                    'fold': fold_idx,
                    **params,
                    'R2': r2_score(y_test, y_pred),
                    'RMSE': mean_squared_error(y_test, y_pred, squared=False),
                    'MSE': mean_squared_error(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'MAPE': mape(y_test, y_pred),
                    'MedAE': median_absolute_error(y_test, y_pred),
                    'R2_orig': r2_score(y_test_orig, y_pred_orig),
                    'RMSE_orig': mean_squared_error(y_test_orig, y_pred_orig, squared=False),
                    'MSE_orig': mean_squared_error(y_test_orig, y_pred_orig),
                    'MAE_orig': mean_absolute_error(y_test_orig, y_pred_orig),
                    'MAPE_orig': mape(y_test_orig, y_pred_orig),
                    'MedAE_orig': median_absolute_error(y_test_orig, y_pred_orig)
                }
                fold_metrics.append(metric_dict)
                fold_results.append(metric_dict.copy())

            if path_part4_temp is not None:
                df_pred = pd.concat(pred_records, ignore_index=True)
                file_name = f"{str_describe}_{model_name}_{param_idx}.csv"
                df_pred.to_csv(os.path.join(path_part4_temp, file_name), index=False)

            mean_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
            mean_metrics.update(params)
            mean_metrics['model'] = model_name
            mean_metrics['param_idx'] = param_idx
            results.append(mean_metrics)

    return pd.DataFrame(results), pd.DataFrame(fold_results)


def get_tree_best_model(df_o):
    df = df_o.copy()
    grouped = df.groupby(['model'])
    results = pd.DataFrame()
    for name, group in grouped:
        max_index = group['R2'].idxmax()
        max_row = group.loc[max_index]
        results = pd.concat([results, max_row.to_frame().T], ignore_index=True)
    return results





def get_forecast_outcome(df_train, df_test, path_rfecv_data, str_describe, mark_num='cv',rd_state=int_rdm):
    """用于填充lr初始文件的sw值"""
    best_params = pd.read_csv(path_rfecv_data + 'ml_cv_best.csv')
    best_params = best_params.sort_values(by='score', ascending=False)
    max_index = best_params['score'].idxmax()
    max_model = best_params.loc[max_index, 'model']
    print(max_model)
    
    if max_model == 'XGBR':
        selected_features = pd.read_csv(
            path_rfecv_data + f"{str_describe}_rfecv_features_XGBR{mark_num}.csv"
        )
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        
        # 读取最优参数
        model_params = best_params[best_params["model"] == "XGBR"].iloc[0]
        if pd.isna(model_params["max_depth"]) or str(model_params["max_depth"]).lower() == 'none':
            param_max_depth = None  # 这里 XGB 也能接受 None
        else:
            param_max_depth = int(float(model_params["max_depth"]))
        select_model = XGBRegressor(
            max_depth=param_max_depth,
            learning_rate=model_params["learning_rate"],
            min_child_weight=int(model_params["min_child_weight"]),
            gamma=int(model_params["gamma"]),
            n_estimators=int(model_params["n_estimators"]),
            random_state=rd_state,
        )
    elif max_model == 'RF':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_RF"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        model_params = best_params[best_params["model"] == "RF"].iloc[0]
        if pd.isna(model_params["max_depth"]) or str(model_params["max_depth"]).lower() == 'none':
            param_max_depth = None
        else:
            param_max_depth = int(float(model_params["max_depth"]))
        
        select_model = RandomForestRegressor(
            max_depth=param_max_depth,
            min_samples_leaf=int(model_params["min_samples_leaf"]),
            min_samples_split=int(model_params["min_samples_split"]),
            n_estimators=int(model_params["n_estimators"]),
            random_state=rd_state
        )
    elif max_model == 'LGBM':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_LGBM"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        model_params = best_params[best_params["model"] == "LGBM"].iloc[0]
        if pd.isna(model_params["max_depth"]) or str(model_params["max_depth"]).lower() == 'none':
            param_max_depth = None
        else:
            param_max_depth = int(float(model_params["max_depth"]))
        select_model = LGBMRegressor(
            learning_rate=model_params["learning_rate"],
            max_depth=param_max_depth,
            min_child_samples=int(model_params["min_child_samples"]),
            num_leaves=int(model_params["num_leaves"]),
            n_estimators=int(model_params["n_estimators"]),
            random_state=rd_state
        )
    X_train = df_train[selected_features]
    y_train = df_train['value']

    select_model.fit(X_train, y_train)
    X_test = df_test[selected_features]
    y_pred = select_model.predict(X_test)
    df_test = df_test.copy()
    df_test['sw_value'] = y_pred
    return df_test


def calc_weighted_score_dict(cv_result, metrics_config):
    """
    根据字典配置，对多个指标进行加权归一化并计算总分，
    返回带分数的结果和每个模型得分最高的行。
    new_cv_result : pd.DataFrame
        增加归一化列与 score 列后的 DataFrame
    best_result : pd.DataFrame
        每种模型 (score最高) 的最佳行
    """
    df = cv_result.copy()
    total_score = 0

    for metric_name, cfg in metrics_config.items():
        min_val, max_val = df[metric_name].min(), df[metric_name].max()
        norm_col = f'{metric_name}_norm'
        
        if max_val != min_val:
            df[norm_col] = (df[metric_name] - min_val) / (max_val - min_val)
        else:
            df[norm_col] = 1  # 如果该列所有值相同

        # 如果是“越小越好”，进行反向处理
        if not cfg['higher_is_better']:
            df[norm_col] = 1 - df[norm_col]

        total_score += cfg['weight'] * df[norm_col]
    
    df['score'] = total_score

    best_result = df.loc[df.groupby('model')['score'].idxmax()].reset_index(drop=True)

    return df, best_result
