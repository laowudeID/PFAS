import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Gaussian
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from scipy import stats
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump, Parallel, delayed, load
import os
import multiprocessing
import random


def start_rfecv(key, clf, df, str_describe, path_data, path_fig, mark_num=''):
    """key, clf:遍历字典, str_describe:文件备注用, 会生成表和图"""
    df_rfecv = df.copy()
    min_features_to_select = 30
    X = df_rfecv.drop('value', axis=1)
    y = df_rfecv['value']
    scoring = 'r2'
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=202406)
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


    feature_selection_info.to_csv(path_data + str_describe + "_rfecv_features_" + key + mark_num+".csv", index=False)


    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean R2")
    plt.errorbar(
        range(min_features_to_select, n_scores + min_features_to_select),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.savefig(path_fig + str_describe + "_rfecv_" + scoring +"_"+ key + mark_num+".svg", dpi=300)


    df_plot = pd.DataFrame({
        'min_features': range(min_features_to_select, n_scores + min_features_to_select),
        'mean_test_score': rfecv.cv_results_["mean_test_score"],
        'std_test_score': rfecv.cv_results_["std_test_score"]
    })
    df_plot = df_plot.sort_values(by='mean_test_score', ascending=False)
    df_plot.to_csv(path_data + str_describe + "_rfecv_" + scoring +"_" + key + mark_num+".csv", index=False)

    return "okk"


def get_var_imp(df_o, str_model, min_feature='NONE'):

    df_data = df_o.copy()
    X = df_data.drop('value', axis=1)
    y = df_data['value']

    if str_model == 'RF':
        rf = RandomForestRegressor(random_state=202406)
        rf.fit(X, y)
        feature_importances = rf.feature_importances_
    elif str_model == 'GBDT':
        gbdt = GradientBoostingRegressor(random_state=202406)
        gbdt.fit(X, y)
        feature_importances = gbdt.feature_importances_

    features_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })


    features_df = features_df.sort_values(by='Importance', ascending=False)
    if min_feature=='NONE':
        pass
    else:
        features_df = features_df.head(min_feature)

    return features_df

def get_rfecv_var(df_o, str_model, min_feature):
    df_rfecv = df_o.copy()
    X = df_rfecv.drop('value', axis=1)
    y = df_rfecv['value']
    if str_model == 'RF':
        clf = RandomForestRegressor(random_state=202406)
    elif str_model == 'GBDT':
        clf = GradientBoostingRegressor(random_state=202406)
    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=10,
        scoring='r2',
        min_features_to_select=min_feature,
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    print(f"Optimal number of features: {rfecv.n_features_}")


    feature_ranks = rfecv.ranking_



    df_features = pd.DataFrame()
    df_features['Feature'] = X.columns
    df_features['Rank'] = feature_ranks

    return df_features



def preprocess(df_data, df_meta, treat_method, list_var=None):
    """sem统一使用raw 然后再使用这个函数去预处理 
        normalization:最大值最小值归一化
        standardization:Z-score标准化
        box_cox:box_cox变换
        其他：正态-z-score  非正-对数
        list_var不填就是对所有列处理 否则只对元素列进行处理"""

    if list_var is None:
        list_var = df_data.columns

    if treat_method == "normalization":

        df_data[list_var] = (df_data[list_var] - df_data[list_var].min()) / (df_data[list_var].max() - df_data[list_var].min())
    elif treat_method == "standardization":

        df_data[list_var] = (df_data[list_var] - df_data[list_var].mean()) / df_data[list_var].std()
    elif treat_method == "box_cox":
        numeric_cols = df_data[list_var].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:



            df_data[col], _ = stats.boxcox(df_data[col])
    elif treat_method == "log":

        df_data[list_var] = np.log10(df_data[list_var])
    elif treat_method == "yeo_johnson":

        numeric_cols = df_data[list_var].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_data[col], _ = stats.yeojohnson(df_data[col])
    elif treat_method == "auto":
        list_yj_meta = df_meta['var_name'][df_meta['var_type0']==1]
        list_bc_meta = df_meta['var_name'][df_meta['var_type0']==0]

        list_var = [item for item in list_var if 'log_' not in item]

        list_yj = list(set(list_yj_meta) & set(list_var))
        list_bc = list(set(list_bc_meta) & set(list_var))
        print('YJ:',list_yj)
        print('BC:',list_bc)

        for col in list_yj:
            try:
                df_data[col], _ = stats.yeojohnson(df_data[col])
            except:
                print('error:',col)
                pass
        for col in list_bc:
            try:
                df_data[col], _ = stats.boxcox(df_data[col])
            except:
                print('error:',col)
                pass
    else:

        for var in list_var:
            p_val = stats.normaltest(df_data[var]).pvalue
            if p_val > 0.05:


                print('normal:',var)
                pass
            else:

                try:
                    df_data[var], _ = stats.yeojohnson(df_data[var])
                except:
                    print('error:',var)
                    pass

    return df_data

def grid_search_param(df_o, str_describe, scoring, path_rfecv_data, model_configs, mark_num=''):
    """
    执行网格搜索以找到最佳超参数

    Parameters:
    -----------
    df_o : DataFrame
        原始数据
    str_describe : str
        描述字符串，用于命名文件
    scoring : str
        评分标准
    path_rfecv_data : str
        RFECV特征存储路径
    model_configs : dict
        模型配置字典，包含模型名称、类和参数网格
    mark_num : str, optional
        标记数字，用于文件命名

    Returns:
    --------
    DataFrame
        所有模型的CV结果
    """
    df_data = df_o.copy()
    cv_results_df = pd.DataFrame()

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


        model_class = config['model_class']
        model_instance = model_class(random_state=202406, **config.get('fixed_params', {}))


        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=202406)
        grid_search = GridSearchCV(model_instance, config['param_grid'], 
                                 cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")


        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results['model'] = model_name
        cv_results_df = pd.concat([cv_results_df, cv_results], ignore_index=True)

    return cv_results_df





def get_tree_best_model(df_o):
    df = df_o.copy()

    grouped = df.groupby(['model'])

    results = pd.DataFrame()

    for name, group in grouped:

        max_index = group['mean_test_score'].idxmax()

        max_row = group.loc[max_index]

        results = pd.concat([results, max_row.to_frame().T], ignore_index=True)
    return results

def glm_inf(df_o, df_vif):
    df_data = df_o.copy()
    list_vif = df_vif['variables'].tolist()
    df_glm_x = df_data[list_vif]
    df_glm_x['value'] = df_data['value']


    print(len(list_vif))
    formula = 'value ~ 1 + ' + ' + '.join(list_vif)

    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    mse_list = []
    r2_list = []
    for train_index, test_index in kf.split(df_glm_x):

        train_data = df_glm_x.iloc[train_index]
        test_data = df_glm_x.iloc[test_index]

        model = glm(formula=formula, data=train_data, family=Gaussian()).fit()

        predictions = model.predict(test_data)
        mse = mean_squared_error(test_data['value'], predictions)
        r2 = r2_score(test_data['value'], predictions)
        mse_list.append(mse)
        r2_list.append(r2)

    average_mse = np.mean(mse_list)
    average_r2 = np.mean(r2_list)
    print(f'Average MSE: {average_mse}')
    print(f'Average R²: {average_r2}')
    return "okk"







def tree_models_evaluation(df_o, selected_features_dict, best_params):
    """
    树模型的交叉验证，支持多种模型，返回每一折的R2和RMSE以及平均值

    参数:
    - df_o: 原始数据框
    - selected_features_dict: 字典，键为模型名称，值为该模型的特征列表
    - best_params: 包含每个模型最佳参数的DataFrame

    返回:
    - DataFrame包含每个模型每一折的R2和RMSE，以及平均值和标准差
    """

    data = df_o.copy()


    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=202406)


    scoring = {
        'r2': make_scorer(r2_score),
        'mse': make_scorer(mean_squared_error)
    }


    models = {}


    if 'GBDT' in selected_features_dict:

        gbdt_params = best_params[best_params["model"] == "GBDT"].iloc[0]


        models['GBDT'] = GradientBoostingRegressor(
            learning_rate=gbdt_params["param_learning_rate"],
            max_depth=None if gbdt_params["param_max_depth"] == 'None' else int(float(gbdt_params["param_max_depth"])),
            min_samples_leaf=int(gbdt_params["param_min_samples_leaf"]),
            min_samples_split=int(gbdt_params["param_min_samples_split"]),
            n_estimators=int(gbdt_params["param_n_estimators"])
        )


    if 'RF' in selected_features_dict:

        rf_params = best_params[best_params["model"] == "RF"].iloc[0]


        models['RF'] = RandomForestRegressor(
            max_depth=None if rf_params["param_max_depth"] == 'None' else int(float(rf_params["param_max_depth"])),
            min_samples_leaf=int(rf_params["param_min_samples_leaf"]),
            min_samples_split=int(rf_params["param_min_samples_split"]),
            n_estimators=int(rf_params["param_n_estimators"])
        )


    if 'LightGBM' in selected_features_dict:

        lgbm_params = best_params[best_params["model"] == "LGBM"].iloc[0]


        models['LightGBM'] = LGBMRegressor(
            learning_rate=lgbm_params["param_learning_rate"],
            max_depth=None if lgbm_params["param_max_depth"] == 'None' else int(float(lgbm_params["param_max_depth"])),
            min_child_samples=int(lgbm_params["param_min_child_samples"]),
            num_leaves=int(lgbm_params["param_num_leaves"]),
            n_estimators=int(lgbm_params["param_n_estimators"])
        )


    y = data['value']


    all_results = []


    for model_name, model in models.items():

        selected_features = [f for f in selected_features_dict[model_name] if f not in ['lon_grid', 'lat_grid']]


        X = data[selected_features]


        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)


        rmse_values = np.sqrt(cv_results['test_mse'])


        for i, r2_value in enumerate(cv_results['test_r2']):
            all_results.append({
                'metric': 'R2',
                'model': model_name,
                f'split{i}_test_score': r2_value
            })


        all_results.append({
            'metric': 'R2',
            'model': model_name,
            'mean_test_score': np.mean(cv_results['test_r2']),
            'std_test_score': np.std(cv_results['test_r2'])
        })


        for i, rmse_value in enumerate(rmse_values):
            all_results.append({
                'metric': 'RMSE',
                'model': model_name,
                f'split{i}_test_score': rmse_value
            })


        all_results.append({
            'metric': 'RMSE',
            'model': model_name,
            'mean_test_score': np.mean(rmse_values),
            'std_test_score': np.std(rmse_values)
        })


        print(f"Average R2 for {model_name}: {np.mean(cv_results['test_r2'])}")
        print(f"Average RMSE for {model_name}: {np.mean(rmse_values)}")


    result_df = pd.DataFrame(all_results)

    return result_df

def additional_validation(df_train, df_test, path_rfecv_data, str_describe, mark_num='cv',rd_state=202406):
    """tree模型在寻找超参等步骤都是只用r2  这里是为了再看看mse 也可以看看别的"""

    best_params = pd.read_csv(path_rfecv_data + 'ml_cv_best.csv')
    best_params = best_params.sort_values(by='mean_test_score', ascending=False)

    max_index = best_params['mean_test_score'].idxmax()
    max_model = best_params.loc[max_index, 'model']
    print(max_model)
    if max_model == 'GBDT':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_GBDT"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        model_params = best_params[best_params["model"] == "GBDT"].iloc[0]

        param_max_depth = None if model_params["param_max_depth"] == 'None' else int(float(model_params["param_max_depth"]))
        select_model = GradientBoostingRegressor(
        learning_rate=model_params["param_learning_rate"],
        max_depth=param_max_depth,
        min_samples_leaf=int(model_params["param_min_samples_leaf"]),
        min_samples_split=int(model_params["param_min_samples_split"]),
        n_estimators=int(model_params["param_n_estimators"]),
        random_state=rd_state
        )
    elif max_model == 'RF':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_RF"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values

        model_params = best_params[best_params["model"] == "RF"].iloc[0]

        param_max_depth = None if model_params["param_max_depth"] == 'None' else int(float(model_params["param_max_depth"]))
        select_model = RandomForestRegressor(
            max_depth=param_max_depth,
            min_samples_leaf=int(model_params["param_min_samples_leaf"]),
            min_samples_split=int(model_params["param_min_samples_split"]),
            n_estimators=int(model_params["param_n_estimators"]),
            random_state=rd_state
        )
    elif max_model == 'LGBM':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_LGBM"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values

        model_params = best_params[best_params["model"] == "LGBM"].iloc[0]

        param_max_depth = None if model_params["param_max_depth"] == 'None' else int(float(model_params["param_max_depth"]))
        select_model = LGBMRegressor(
            learning_rate=model_params["param_learning_rate"],
            max_depth=param_max_depth,
            min_child_samples=int(model_params["param_min_child_samples"]),
            num_leaves=int(model_params["param_num_leaves"]),
            n_estimators=int(model_params["param_n_estimators"]),
            random_state=rd_state
        )

    X_train = df_train[selected_features]
    y_train = df_train['value']

    select_model.fit(X_train, y_train)
    X_test = df_test[selected_features]
    y_test = df_test['value']
    y_pred = select_model.predict(X_test)


    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return r2, mse, rmse



def get_best_model(path_rfecv_data, str_describe, mark_num='cv',rd_state=202406):
    """tree模型在寻找超参等步骤都是只用r2  这里是为了再看看mse 也可以看看别的"""

    best_params = pd.read_csv(path_rfecv_data + 'ml_cv_best.csv')
    best_params = best_params.sort_values(by='mean_test_score', ascending=False)

    max_index = best_params['mean_test_score'].idxmax()
    max_model = best_params.loc[max_index, 'model']
    print(max_model)
    if max_model == 'GBDT':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_GBDT"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        model_params = best_params[best_params["model"] == "GBDT"].iloc[0]

        param_max_depth = None if model_params["param_max_depth"] == 'None' else int(float(model_params["param_max_depth"]))
        select_model = GradientBoostingRegressor(
        learning_rate=model_params["param_learning_rate"],
        max_depth=param_max_depth,
        min_samples_leaf=int(model_params["param_min_samples_leaf"]),
        min_samples_split=int(model_params["param_min_samples_split"]),
        n_estimators=int(model_params["param_n_estimators"]),
        random_state=rd_state
        )
    elif max_model == 'RF':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_RF"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values

        model_params = best_params[best_params["model"] == "RF"].iloc[0]

        param_max_depth = None if model_params["param_max_depth"] == 'None' else int(float(model_params["param_max_depth"]))
        select_model = RandomForestRegressor(
            max_depth=param_max_depth,
            min_samples_leaf=int(model_params["param_min_samples_leaf"]),
            min_samples_split=int(model_params["param_min_samples_split"]),
            n_estimators=int(model_params["param_n_estimators"]),
            random_state=rd_state
        )
    elif max_model == 'LGBM':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_LGBM"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values

        model_params = best_params[best_params["model"] == "LGBM"].iloc[0]

        param_max_depth = None if model_params["param_max_depth"] == 'None' else int(float(model_params["param_max_depth"]))
        select_model = LGBMRegressor(
            learning_rate=model_params["param_learning_rate"],
            max_depth=param_max_depth,
            min_child_samples=int(model_params["param_min_child_samples"]),
            num_leaves=int(model_params["param_num_leaves"]),
            n_estimators=int(model_params["param_n_estimators"]),
            random_state=rd_state
        )
    return selected_features, select_model




def predict_with_stacking_model(new_data, path_rfecv_data, str_describe, model_path=None, mark_num='', rd_state=202406):
    """
    Use trained stacking model to make predictions on new data

    Parameters:
    - new_data: DataFrame containing features for prediction
    - path_rfecv_data: Path to RFECV data
    - str_describe: Description string for feature selection
    - model_path: Path to save/load trained models (if None, models will be trained)
    - mark_num: Additional mark for feature selection
    - rd_state: Random state for reproducibility

    Returns:
    - predictions: Predicted values for new_data
    """



    if model_path is not None and os.path.exists(model_path + 'base_models.joblib') and os.path.exists(model_path + 'meta_model.joblib'):
        print("Loading trained models...")
        base_models = load(model_path + 'base_models.joblib')
        meta_model = load(model_path + 'meta_model.joblib')


        selected_features_gbdt = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_GBDT"+mark_num+".csv")
        selected_features_gbdt = selected_features_gbdt[selected_features_gbdt["Rank"] == 1]["Feature"].values

        selected_features_rf = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_RF"+mark_num+".csv")
        selected_features_rf = selected_features_rf[selected_features_rf["Rank"] == 1]["Feature"].values

        selected_features_lgbm = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_LGBM"+mark_num+".csv")
        selected_features_lgbm = selected_features_rf[selected_features_rf["Rank"] == 1]["Feature"].values


        selected_features = list(set(selected_features_gbdt) & set(selected_features_rf) & set(selected_features_lgbm))

    else:
        print("Training new models...")

        raise ValueError("For new model training, please provide training data. Use the stacking_regression function first to train models.")


    if set(selected_features).issubset(set(new_data.columns)):
        X_pred = new_data[selected_features]
    else:
        missing_features = set(selected_features) - set(new_data.columns)
        raise ValueError(f"New data is missing the following features: {missing_features}")


    meta_features_pred = np.column_stack([model.predict(X_pred) for model in base_models])


    predictions = meta_model.predict(meta_features_pred)

    return predictions

def train_and_save_stacking_model(df_train, path_rfecv_data, str_describe, model_path, mark_num='', rd_state=202406):
    """
    Train and save stacking regression model for later prediction use

    Parameters:
    - df_train: Training dataframe with features and 'value' target
    - path_rfecv_data: Path to RFECV data
    - str_describe: Description string for feature selection
    - model_path: Path to save trained models
    - mark_num: Additional mark for feature selection
    - rd_state: Random state for reproducibility

    Returns:
    - selected_features: List of selected features
    - base_models: List of trained base models
    - meta_model: Trained meta model
    """

    os.makedirs(model_path, exist_ok=True)


    best_params = pd.read_csv(path_rfecv_data + 'ml_cv_best.csv')
    best_params = best_params.sort_values(by='mean_test_score', ascending=False)


    selected_features_gbdt = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_GBDT"+mark_num+".csv")
    selected_features_gbdt = selected_features_gbdt[selected_features_gbdt["Rank"] == 1]["Feature"].values

    selected_features_rf = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_RF"+mark_num+".csv")
    selected_features_rf = selected_features_rf[selected_features_rf["Rank"] == 1]["Feature"].values

    selected_features_lgbm = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_LGBM"+mark_num+".csv")
    selected_features_lgbm = selected_features_rf[selected_features_rf["Rank"] == 1]["Feature"].values


    selected_features = list(set(selected_features_gbdt) & set(selected_features_rf) & set(selected_features_lgbm))


    pd.DataFrame({'Feature': selected_features}).to_csv(model_path + 'selected_features.csv', index=False)


    X_train = df_train[selected_features]
    y_train = df_train['value']



    gbdt_params = best_params[best_params["model"] == "GBDT"].iloc[0]
    param_max_depth = None if gbdt_params["param_max_depth"] == 'None' else int(float(gbdt_params["param_max_depth"]))
    gbdt = GradientBoostingRegressor(
        learning_rate=gbdt_params["param_learning_rate"],
        max_depth=param_max_depth,
        min_samples_leaf=int(gbdt_params["param_min_samples_leaf"]),
        min_samples_split=int(gbdt_params["param_min_samples_split"]),
        n_estimators=int(gbdt_params["param_n_estimators"]),
        random_state=rd_state
    )


    rf_params = best_params[best_params["model"] == "RF"].iloc[0]
    param_max_depth = None if rf_params["param_max_depth"] == 'None' else int(float(rf_params["param_max_depth"]))
    rf = RandomForestRegressor(
        max_depth=param_max_depth,
        min_samples_leaf=int(rf_params["param_min_samples_leaf"]),
        min_samples_split=int(rf_params["param_min_samples_split"]),
        n_estimators=int(rf_params["param_n_estimators"]),
        random_state=rd_state
    )


    lgbm_params = best_params[best_params["model"] == "LGBM"].iloc[0]
    param_max_depth = None if lgbm_params["param_max_depth"] == 'None' else int(float(lgbm_params["param_max_depth"]))
    lgbm = LGBMRegressor(
        n_estimators=int(lgbm_params["param_n_estimators"]),
        learning_rate=lgbm_params["param_learning_rate"],
        max_depth=param_max_depth,
        num_leaves=int(lgbm_params["param_num_leaves"]),
        min_child_samples=int(lgbm_params["param_min_child_samples"]),
        random_state=rd_state
    )


    def generate_meta_features(model):
        return cross_val_predict(model, X_train, y_train, cv=10, method='predict')


    meta_features = Parallel(n_jobs=-1)(delayed(generate_meta_features)(model) 
                                        for model in [gbdt, rf, lgbm])


    meta_features_train = np.column_stack(meta_features)


    def fit_model(model, X, y):
        model.fit(X, y)
        return model


    base_models = Parallel(n_jobs=-1)(delayed(fit_model)(model, X_train, y_train) 
                                      for model in [gbdt, rf, lgbm])


    meta_model = LinearRegression()
    meta_model.fit(meta_features_train, y_train)


    dump(base_models, model_path + 'base_models.joblib')
    dump(meta_model, model_path + 'meta_model.joblib')

    print(f"Models saved to {model_path}")

    return selected_features, base_models, meta_model





def stacking_regression(df_train, df_test, path_rfecv_data, str_describe, mark_num='', rd_state=202406):
    """
    Perform stacking regression with GBDT, RF, and MLR as base models

    Parameters:
    - df_train: Training dataframe
    - df_test: Test dataframe
    - path_rfecv_data: Path to RFECV data
    - str_describe: Description string for feature selection
    - mark_num: Additional mark for feature selection
    - rd_state: Random state for reproducibility

    Returns:
    - r2: R-squared score
    - mse: Mean Squared Error
    """

    best_params = pd.read_csv(path_rfecv_data + 'ml_cv_best.csv')
    best_params = best_params.sort_values(by='mean_test_score', ascending=False)


    selected_features_gbdt = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_GBDT"+mark_num+".csv")
    selected_features_gbdt = selected_features_gbdt[selected_features_gbdt["Rank"] == 1]["Feature"].values

    selected_features_rf = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_RF"+mark_num+".csv")
    selected_features_rf = selected_features_rf[selected_features_rf["Rank"] == 1]["Feature"].values

    selected_features_lgbm = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_LGBM"+mark_num+".csv")
    selected_features_lgbm = selected_features_rf[selected_features_rf["Rank"] == 1]["Feature"].values

    selected_features = list(set(selected_features_gbdt) & set(selected_features_rf) & set(selected_features_lgbm))


    X_train = df_train[selected_features]
    y_train = df_train['value']
    X_test = df_test[selected_features]
    y_test = df_test['value']




    gbdt_params = best_params[best_params["model"] == "GBDT"].iloc[0]
    param_max_depth = None if gbdt_params["param_max_depth"] == 'None' else int(float(gbdt_params["param_max_depth"]))
    gbdt = GradientBoostingRegressor(
        learning_rate=gbdt_params["param_learning_rate"],
        max_depth=param_max_depth,
        min_samples_leaf=int(gbdt_params["param_min_samples_leaf"]),
        min_samples_split=int(gbdt_params["param_min_samples_split"]),
        n_estimators=int(gbdt_params["param_n_estimators"]),
        random_state=rd_state
    )


    rf_params = best_params[best_params["model"] == "RF"].iloc[0]
    param_max_depth = None if rf_params["param_max_depth"] == 'None' else int(float(rf_params["param_max_depth"]))
    rf = RandomForestRegressor(
        max_depth=param_max_depth,
        min_samples_leaf=int(rf_params["param_min_samples_leaf"]),
        min_samples_split=int(rf_params["param_min_samples_split"]),
        n_estimators=int(rf_params["param_n_estimators"]),
        random_state=rd_state
    )

    lgbm_params = best_params[best_params["model"] == "LGBM"].iloc[0]
    param_max_depth = None if lgbm_params["param_max_depth"] == 'None' else int(float(lgbm_params["param_max_depth"]))

    lgbm = LGBMRegressor(
        n_estimators=int(lgbm_params["param_n_estimators"]),
        learning_rate=lgbm_params["param_learning_rate"],
        max_depth=param_max_depth,
        num_leaves=int(lgbm_params["param_num_leaves"]),
        min_child_samples=int(lgbm_params["param_min_child_samples"]),
        random_state=rd_state
    )


    def generate_meta_features(model):
        return cross_val_predict(model, X_train, y_train, cv=10, method='predict')


    meta_features = Parallel(n_jobs=-1)(delayed(generate_meta_features)(model) 
                                        for model in [gbdt, rf, lgbm])


    meta_features_train = np.column_stack(meta_features)


    def fit_model(model, X, y):
        model.fit(X, y)
        return model


    base_models = Parallel(n_jobs=-1)(delayed(fit_model)(model, X_train, y_train) 
                                      for model in [gbdt, rf, lgbm])


    def predict_meta_features(model, X):
        return model.predict(X)


    test_meta_features = Parallel(n_jobs=-1)(delayed(predict_meta_features)(model, X_test) 
                                             for model in base_models)


    meta_features_test = np.column_stack(test_meta_features)


    meta_model = LinearRegression()
    meta_model.fit(meta_features_train, y_train)


    y_pred = meta_model.predict(meta_features_test)


    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return r2, mse

def run_stacking_validation(df_sw_sem_all, df_sw_15, df_sw_other, path_sw_rfecv_data, str_describe, random_seeds):

    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores for parallel processing")


    results = {
        'seed': [],


        'r2_other': [],
        'mse_other': []
    }


    def process_seed(seed):




        r2_other, mse_other = stacking_regression(df_sw_15, df_sw_other, path_sw_rfecv_data, str_describe, 'cv', seed)

        return {
            'seed': seed,


            'r2_other': r2_other,
            'mse_other': mse_other
        }


    parallel_results = Parallel(n_jobs=-1)(delayed(process_seed)(seed) for seed in random_seeds)


    for result in parallel_results:
        for key, value in result.items():
            results[key].append(value)


    df = pd.DataFrame(results)


    mean_values = df.mean()
    std_values = df.std()


    stats_df = pd.DataFrame({
        'type': ['mean', 'std'],


        'r2_other_mean': [mean_values['r2_other'], std_values['r2_other']],
        'mse_other_mean': [mean_values['mse_other'], std_values['mse_other']]
    })

    return df, stats_df