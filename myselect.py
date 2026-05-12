# 函数准备


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def forward_select_vif(df_features_raw, one_var, max_vif=10):
    """这个跟下面的区别在于这个只能指定一个初始变量"""
    selected_vars = [one_var]
    remaining_vars = [var for var in df_features_raw.columns if var != one_var]
    vif_data = pd.DataFrame(columns=['variables', 'VIF'])

    while remaining_vars:
        temp_vif_data = pd.DataFrame(columns=['variables', 'VIF'])
        for var in remaining_vars:
            temp_df = df_features_raw[selected_vars + [var]]
            temp_vif = [variance_inflation_factor(temp_df.values, i)
                        for i in range(temp_df.shape[1])]
            
            if all(vif < max_vif for vif in temp_vif[:-1]):  # 排除添加的当前变量的VIF
                new_row = pd.DataFrame({'variables': [var], 'VIF': [temp_vif[-1]]})
                temp_vif_data = pd.concat([temp_vif_data, new_row], ignore_index=True)
        
        temp_vif_data['VIF'] = pd.to_numeric(temp_vif_data['VIF'], errors='coerce')

        temp_vif_data = temp_vif_data[temp_vif_data['VIF'] < max_vif]
        if temp_vif_data.empty:
            break
        
        next_var = temp_vif_data.loc[temp_vif_data['VIF'].idxmin(), 'variables']
        selected_vars.append(next_var)
        remaining_vars.remove(next_var)
    
    final_df = df_features_raw[selected_vars]
    vif_data['variables'] = final_df.columns
    vif_data['VIF'] = [variance_inflation_factor(final_df.values, i) 
                       for i in range(final_df.shape[1])]
    
    return vif_data


def forward_select_vif2(df_features_raw, list_var, max_vif=10):
    """通过vif_select_var调用的话用sem开头的数据即可  直接调用的话注意 list_var是初始保留的变量组
        初始变量组只有一个元素无前置条件 但是多个得先通过count_vif函数"""
    remaining_vars = [var for var in df_features_raw.columns if var not in list_var]
    print(len(remaining_vars))
    vif_data = pd.DataFrame(columns=['variables', 'VIF'])

    while remaining_vars:
        temp_vif_data = pd.DataFrame(columns=['variables', 'VIF'])
        for var in remaining_vars:
            temp_df = df_features_raw[list_var + [var]]
            temp_vif = [variance_inflation_factor(temp_df.values, i)
                        for i in range(temp_df.shape[1])]
            
            if all(vif < max_vif for vif in temp_vif[:-1]):  # 排除添加的当前变量的VIF
                # 使用pandas.concat来代替append
                new_row = pd.DataFrame({'variables': [var], 'VIF': [temp_vif[-1]]})
                temp_vif_data = pd.concat([temp_vif_data, new_row], ignore_index=True)
        
        temp_vif_data['VIF'] = pd.to_numeric(temp_vif_data['VIF'], errors='coerce')

        temp_vif_data = temp_vif_data[temp_vif_data['VIF'] < max_vif]
        if temp_vif_data.empty:
            break
        
        next_var = temp_vif_data.loc[temp_vif_data['VIF'].idxmin(), 'variables']
        list_var.append(next_var)
        remaining_vars.remove(next_var)
        # print(remaining_vars)
    
    final_df = df_features_raw[list_var]
    vif_data['variables'] = final_df.columns
    vif_data['VIF'] = [variance_inflation_factor(final_df.values, i) 
                       for i in range(final_df.shape[1])]
    
    return vif_data


def count_vif(df_features_raw, max_vif=10):
    """迭代计算输入的dataframe VIF 每次迭代去除VIF最大的变量 直到所有变量的VIF小于10
        返回一个dataframe 里面包含两列 variables VIF"""
    vif_data = pd.DataFrame()
    vif_data["variables"] = df_features_raw.columns

    while True:
        # 计算VIF
        vif_data["VIF"] = [variance_inflation_factor(df_features_raw.values, i) 
                        for i in range(len(df_features_raw.columns))]
        
        if vif_data['VIF'].max() > max_vif:
            max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'variables']
            df_features_raw = df_features_raw.drop(max_vif_feature, axis=1)
            vif_data = vif_data[vif_data['variables'] != max_vif_feature]
            print('Remove var:',max_vif_feature)
        else:
            break
    print(vif_data.shape)
    return vif_data

def print_equation(df, str_var='NO'):
    """快速生成model中潜在变量的公式 方便写代码
        这个函数直接输出文本 返回的只有okk"""
    df = df.iloc[4:]
    df = df.drop("Meaning", axis=1)
    df.set_index('vars', inplace=True)
    dic_var = {col: [] for col in df.columns}

    for index, row in df.iterrows():
        max_val = row.abs().max()
        if max_val >= 0.3:
            max_col = row.abs().idxmax()
            dic_var[max_col].append(index)

    if str_var != 'NO':
        new_dic_var = {}
        for i, (key, value) in enumerate(dic_var.items()):
            new_key = f'{str_var}_{i}'
            new_dic_var[new_key] = value
        dic_var = new_dic_var

    for key, value in dic_var.items():
        str_value = ' + '.join(value)
        print(f'{key} =~ {str_value}')

    return "okk"


def count_variance(df_sem_data):
    """计算所有变量的方差  返回的是一个df"""
    print(df_sem_data.shape)
    mean = df_sem_data.mean()
    std_dev = df_sem_data.std()
    variance = df_sem_data.var()

    df_results = pd.DataFrame({
        'var': df_sem_data.columns,
        'mean': mean.values,
        'SD': std_dev.values,
        'variance': variance.values
    })

    min_variance = df_results["variance"].min()
    max_variance = df_results["variance"].max()
    print(min_variance, max_variance)
    print(max_variance/min_variance)
    # 显示结果
    return df_results


def vif_select_var_old(df_o, path_2_preanalysis_data, path_rfecv_data, str_describe, cv_marker='' , int_max_vif=10, int_mlr=3, int_ml=10, limit_var=False):
    """前置条件通过R语言的packfor包的sfs+mlr算法筛选变量和rfecv_imp.ipynb筛选变量"""
    df_data = df_o.copy()
    df_mlr = pd.read_csv(path_2_preanalysis_data + 'mlr_'+str_describe+'.csv')
    df_mlr = df_mlr.sort_values(by='R2', ascending=False)
    df_rfecv = pd.read_csv(path_rfecv_data + str_describe + '_merge_rfecv'+cv_marker+'.csv')

    min_features = df_rfecv.loc[0, 'min_features']
    max_model = df_rfecv.loc[0, 'model']
    max_score = round(df_rfecv.loc[0, 'mean_test_score'],3)
    print(f'model:{max_model} min_feature:{min_features} score:{max_score}')
    df_ml  =pd.read_csv(path_rfecv_data + str_describe + '_rfecv_features_'+max_model+'cv'+cv_marker+'.csv')
    df_ml = df_ml[df_ml['Rank']==1]
    if limit_var == False:
        list_mlr_var0 = df_mlr['variables'].values
        list_ml_var0 = df_ml['Feature'].values
    elif limit_var == True:
        list_mlr_var0 = df_mlr['variables'][df_mlr['R2']>=(int_mlr/100)].values
        list_ml_var0 = df_ml['Feature'][df_ml['Importance']>=int_ml/100].values
    list_re_imp0 = list(set((set(list_mlr_var0) | set(list_ml_var0))))
    list_mlr_var1 = df_mlr['variables'][df_mlr['R2']>=(int_mlr/100)].values
    print(f'mlr lrsw:{list_mlr_var1}')
    list_ml_var1 = df_ml['Feature'][df_ml['Importance']>=int_ml/100].values
    print(f'ml lrsw:{list_ml_var1}')
    list_re_imp1 = list(set((set(list_mlr_var1) | set(list_ml_var1))))
    print(f'select var:{len(list_re_imp1)}  var:{list_re_imp1}')

    list_mlr_var2 = df_mlr['variables'][df_mlr['R2']<(int_mlr/100)].values
    list_ml_var2 = df_ml['Feature'][df_ml['Importance']<int_ml/100].values
    list_re_imp2 = list(set((set(list_mlr_var2) | set(list_ml_var2))))
    print(f'other var num:{len(list_re_imp2)}')
    df_data0 = df_data[list_re_imp0]
    df_data1 = df_data[list_re_imp1]
    
    df_vif_sbs = count_vif(df_data1,int_max_vif)
    list_start_var = df_vif_sbs['variables'].tolist()
    print(f'start var:{list_start_var}')
    df_vif_sfs = forward_select_vif2(df_data0,list_start_var,max_vif=int_max_vif)
    list_o = df_data0.columns.to_list()
    list_s = df_vif_sfs["variables"].to_list()
    list_remove  = [i for i in list_o if i not in list_s]
    print(f'SFS select var:{list_s}')
    print(f'var num:{len(list_s)}  remove num:{len(list_remove)}')
    return df_vif_sfs
def vif_select_var(
    df_original,
    path_preanalysis,
    path_rfecv,
    describe_tag,
    cv_marker='',
    max_vif=10,
    start_thresholds=[5, 20, True],   # 高阈值: [MLR阈值%, ML阈值%, use_ml_flag]
    filter_thresholds=[3, 10, True],  # 低阈值: [MLR阈值%, ML阈值%, use_ml_flag]
):
    """
    基于 MLR 与 ML(RFECV) 的变量选择结合 VIF 阈值控制。
    阶段1: 高阈值选核心变量 → SBS剔除高VIF
    阶段2: 低阈值选潜在变量 → SFS逐个添加
    
    参数:
        df_original: 原始数据 DataFrame
        path_preanalysis: 存放 MLR 分析结果的文件夹路径
        path_rfecv: 存放 RFECV(ML) 分析结果的文件夹路径
        describe_tag: 文件名标识符
        cv_marker: 文件名中的CV标识
        max_vif: 最大允许的VIF值
        start_thresholds: 高阈值 [MLR阈值%, ML阈值%, use_ml_flag]
        filter_thresholds: 低阈值 [MLR阈值%, ML阈值%, use_ml_flag]
        
    返回:
        df_vif_sfs: 最终变量集合 DataFrame
    """
    df_data = df_original.copy()
    df_mlr = pd.read_csv(path_preanalysis + f'mlr_{describe_tag}.csv')
    df_mlr = df_mlr.sort_values(by='R2', ascending=False)
    df_ml = None
    if start_thresholds[2] or filter_thresholds[2]:
        df_rfecv_info = pd.read_csv(path_rfecv + f'{describe_tag}_merge_rfecv{cv_marker}.csv')
        best_model = df_rfecv_info.loc[0, 'model']
        print(f'[RFECV] Best model:{best_model}, min_features:{df_rfecv_info.loc[0, "min_features"]}, score:{round(df_rfecv_info.loc[0, "mean_test_score"], 3)}')
        df_ml = pd.read_csv(path_rfecv + f'{describe_tag}_rfecv_features_{best_model}cv{cv_marker}.csv')
        df_ml = df_ml[df_ml['Rank'] == 1]  # 只保留 Rank=1 的特征
    mlr_start_vars = df_mlr['variables'][df_mlr['R2'] >= (start_thresholds[0] / 100)]
    if start_thresholds[2] and df_ml is not None:
        ml_start_vars = df_ml['Feature'][df_ml['Importance'] >= (start_thresholds[1] / 100)]
        core_vars_set = set(mlr_start_vars) | set(ml_start_vars)
    else:
        core_vars_set = set(mlr_start_vars)
    core_vars_list = list(core_vars_set)
    print(f'[Stage1] Core variables before SBS ({len(core_vars_list)}): {core_vars_list}')
    df_core_data = df_data[core_vars_list]
    df_vif_sbs = count_vif(df_core_data, max_vif)  # SBS阶段
    core_vars_after_sbs = df_vif_sbs['variables'].tolist()
    print(f'[Stage1] Core variables after SBS ({len(core_vars_after_sbs)}): {core_vars_after_sbs}')
    mlr_potential_vars = df_mlr['variables'][df_mlr['R2'] >= (filter_thresholds[0] / 100)]
    if filter_thresholds[2] and df_ml is not None:
        ml_potential_vars = df_ml['Feature'][df_ml['Importance'] >= (filter_thresholds[1] / 100)]
        potential_vars_set = set(mlr_potential_vars) | set(ml_potential_vars)
    else:
        potential_vars_set = set(mlr_potential_vars)
    potential_vars_list = [v for v in potential_vars_set if v not in core_vars_after_sbs]
    print(f'[Stage2] Potential variables for SFS ({len(potential_vars_list)}): {potential_vars_list}')
    df_final_sfs = forward_select_vif2(
        df_data[core_vars_after_sbs + potential_vars_list],
        core_vars_after_sbs,
        max_vif=max_vif
    )
    final_vars = df_final_sfs['variables'].tolist()
    print(f'[Stage3] Final selected variables ({len(final_vars)}): {final_vars}')
    return df_final_sfs