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
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from scipy import stats


def start_rfecv(key, clf, df, str_describe, path_data, path_fig, mark_num=''):
    """
    Key, CLF: traversing the dictionary,
    str: file comment
    ---
    key, clf:遍历字典,
    str_describe:文件备注用
    """
    df_rfecv = df.copy()
    min_features_to_select = 20
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
        n_jobs=10,
    )
    rfecv.fit(X, y)
    
    print(f"{key} Optimal number of features: {rfecv.n_features_}")

    # Get the support array and feature ranking
    selected_features = X.columns[rfecv.support_]
    feature_importances = pd.Series(rfecv.estimator_.feature_importances_, index=selected_features)
    
    # Get the ranking of all features
    all_features_ranking = pd.Series(rfecv.ranking_, index=X.columns)
    
    # Create a DataFrame with feature selection information
    feature_selection_info = pd.DataFrame({
        'Feature': X.columns,
        'Selected': rfecv.support_,
        'Rank': rfecv.ranking_,
    })
    
    # If the estimator has the attribute `feature_importances_`, add it to the DataFrame
    if hasattr(rfecv.estimator_, 'feature_importances_'):
        feature_selection_info['Importance'] = feature_selection_info.apply(
            lambda row: feature_importances[row['Feature']] if row['Selected'] else 0, axis=1)
    
    feature_selection_info = feature_selection_info.sort_values(by='Importance', ascending=False)

    # Save feature selection info
    feature_selection_info.to_csv(path_data + str_describe + "_rfecv_features_" + key + mark_num+".csv", index=False)

    # Plot the results
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

    # Save data for the plot
    df_plot = pd.DataFrame({
        'min_features': range(min_features_to_select, n_scores + min_features_to_select),
        'mean_test_score': rfecv.cv_results_["mean_test_score"],
        'std_test_score': rfecv.cv_results_["std_test_score"]
    })
    df_plot = df_plot.sort_values(by='mean_test_score', ascending=False)
    df_plot.to_csv(path_data + str_describe + "_rfecv_" + scoring +"_" + key + mark_num+".csv", index=False)
    
    return "okk"

# 下面这两个函数应该不需要用
# The following two functions should not be used
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

    # Get the feature ranking (1 is the best, they get "masked" out otherwise)
    feature_ranks = rfecv.ranking_
    #feature_importance = clf.feature_importances_ if hasattr(clf, 'feature_importances_') else None

    # Create a dataframe for the features and their ranks
    df_features = pd.DataFrame()
    df_features['Feature'] = X.columns
    df_features['Rank'] = feature_ranks
    #df_features['Importance'] = feature_importance
    return df_features



def preprocess(df_data, df_meta, treat_method, list_var=None):
    """
    SEM uses raw uniformly and then uses this function to pre-process
    Normalization: Maximum-minimum normalization
    Standardization: Z-score
    Box: Box Transform
    Others: Normal-z-score non-normal-logarithm
    If the list is not filled in, all columns are processed, 
    otherwise only element columns are processed
    ---
    sem统一使用raw 然后再使用这个函数去预处理 
    normalization:最大值最小值归一化
    standardization:Z-score标准化
    box_cox:box_cox变换
    其他：正态-z-score  非正-对数
    list_var不填就是对所有列处理 否则只对元素列进行处理
    """
    
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




def grid_search_param(df_o, str_describe, scoring, path_rfecv_data, param_grid_rf, param_grid_gbdt, mark_num=''):
    df_data = df_o.copy()
    param_grids = {
                'RF': param_grid_rf,
                'GBDT': param_grid_gbdt,
                }
    dict_clf = {
                'RF': RandomForestRegressor(random_state=202406),
                'GBDT': GradientBoostingRegressor(random_state=202406),
                }
    cv_results_df = pd.DataFrame()
    for key, clf in dict_clf.items():
        print(str_describe, key)
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_" + key + mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        X = df_data[selected_features]
        y = df_data['value']
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=202406)
        grid_search = GridSearchCV(clf, param_grids[key], cv=cv, scoring=scoring, n_jobs=10)
        grid_search.fit(X, y)
        print(f"Best parameters for {key}: {grid_search.best_params_}")
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results['model'] = key
        cv_results_df = pd.concat([cv_results_df, cv_results], ignore_index=True)
    return cv_results_df

def get_tree_best_model(df_o):
    df = df_o.copy()
    grouped = df.groupby(['model'])
    results = pd.DataFrame()
    for name, group in grouped:
        max_index = group['mean_test_score'].idxmax()
        max_row = group.loc[max_index]
        results = results.append(max_row)
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


def tree_inf(df_o, selected_features_gbdt, selected_features_rf, best_params):
    """
    Cross-validation of the tree model
    Returns R 2 and MSE for each fold
    ---
    树模型的交叉验证
    返回每一折的R2和MSE
    """

    data = df_o.copy()

    # Get best parameters for GBDT and RF
    gbdt_params = best_params[best_params["model"] == "GBDT"].iloc[0]
    rf_params = best_params[best_params["model"] == "RF"].iloc[0]
    
    # Define models with best parameters
    gbdt = GradientBoostingRegressor(
        learning_rate=gbdt_params["param_learning_rate"],
        max_depth=None if gbdt_params["param_max_depth"] == 'None' else int(float(gbdt_params["param_max_depth"])),
        min_samples_leaf=int(gbdt_params["param_min_samples_leaf"]),
        min_samples_split=int(gbdt_params["param_min_samples_split"]),
        n_estimators=int(gbdt_params["param_n_estimators"])
    )
    
    rf = RandomForestRegressor(
        max_depth=None if rf_params["param_max_depth"] == 'None' else int(float(rf_params["param_max_depth"])),
        min_samples_leaf=int(rf_params["param_min_samples_leaf"]),
        min_samples_split=int(rf_params["param_min_samples_split"]),
        n_estimators=int(rf_params["param_n_estimators"])
    )

    # Remove 'lon_grid' and 'lat_grid' from selected features
    selected_features_gbdt = [f for f in selected_features_gbdt if f not in ['lon_grid', 'lat_grid']]
    selected_features_rf = [f for f in selected_features_rf if f not in ['lon_grid', 'lat_grid']]

    # Prepare data
    X_gbdt = data[selected_features_gbdt]
    X_rf = data[selected_features_rf]
    y = data['value']

    # Define scoring
    scoring = {
        'r2': make_scorer(r2_score),
        'mse': make_scorer(mean_squared_error)
    }
    
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=202406)
    
    # Perform cross-validation for GBDT
    gbdt_results = cross_validate(gbdt, X_gbdt, y, cv=cv, scoring=scoring, return_train_score=False)
    
    # Perform cross-validation for RF
    rf_results = cross_validate(rf, X_rf, y, cv=cv, scoring=scoring, return_train_score=False)

    # Print average results
    print("Average R2 for GBDT:", gbdt_results['test_r2'].mean())
    print("Average MSE for GBDT:", gbdt_results['test_mse'].mean())
    print("Average R2 for RF:", rf_results['test_r2'].mean())
    print("Average MSE for RF:", rf_results['test_mse'].mean())

    # Return lists of results for each fold
    return {
        'gbdt_r2': gbdt_results['test_r2'].tolist(),
        'gbdt_mse': gbdt_results['test_mse'].tolist(),
        'rf_r2': rf_results['test_r2'].tolist(),
        'rf_mse': rf_results['test_mse'].tolist()
    }



def additional_validation(df_train, df_test, path_rfecv_data, str_describe, mark_num=''):
    # Load best model parameters
    best_params = pd.read_csv(path_rfecv_data + 'ml_cv_best.csv')
    best_params = best_params.sort_values(by='mean_test_score', ascending=False)
    # print(best_params.head())
    max_index = best_params['mean_test_score'].idxmax()
    max_model = best_params.loc[max_index, 'model']
    print(max_model)
    if max_model == 'GBDT':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_GBDT"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        model_params = best_params[best_params["model"] == "GBDT"].iloc[0]
    # Load selected features for GBDT and RF
        if model_params["param_max_depth"] == 'None':
            param_max_depth = None
        else:
            param_max_depth = float(model_params["param_max_depth"])
            param_max_depth = int(param_max_depth)
        select_model = GradientBoostingRegressor(
        learning_rate=model_params["param_learning_rate"],
        max_depth=param_max_depth,
        min_samples_leaf=int(model_params["param_min_samples_leaf"]),
        min_samples_split=int(model_params["param_min_samples_split"]),
        n_estimators=int(model_params["param_n_estimators"])
        )
    elif max_model == 'RF':
        selected_features = pd.read_csv(path_rfecv_data + str_describe + "_rfecv_features_RF"+mark_num+".csv")
        selected_features = selected_features[selected_features["Rank"] == 1]["Feature"].values
        # Get best parameters for GBDT and RF
        model_params = best_params[best_params["model"] == "RF"].iloc[0]
        # Define models with best parameters
        if model_params["param_max_depth"] == 'None':
            param_max_depth = None
        else:
            param_max_depth = float(model_params["param_max_depth"])
            param_max_depth = int(param_max_depth)
        select_model = RandomForestRegressor(
            max_depth=param_max_depth,
            min_samples_leaf=int(model_params["param_min_samples_leaf"]),
            min_samples_split=int(model_params["param_min_samples_split"]),
            n_estimators=int(model_params["param_n_estimators"])
        )

    # Prepare data
    X_train = df_train[selected_features]
    y_train = df_train['value']

    select_model.fit(X_train, y_train)
    X_test = df_test[selected_features]
    y_test = df_test['value']
    y_pred = select_model.predict(X_test)

    # Calculate R2 and MSE
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # print(r2, mse)
    return r2, mse

