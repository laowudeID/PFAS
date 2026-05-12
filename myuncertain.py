
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import myfunction as mf


# list_pfas =['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
# list_pfas_lc = ['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA']
# list_pfas_sc = ['PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
# list_color = ["#4d8cbf", "#4f9c8b", "#555c6c", "#d77563", "#7d84a8", "#84aeb8", "#c3473b", "#89756d","#ffb3cc","#9a7ebf","#ffddb8", "#c4eaff", "#d1c6ff", "#c2ffbf", "#f5f5b0"]
# dic_color = dict(zip(list_pfas,list_color))

path_inf = 'C:/Users/dell/OneDrive/file/'
inf_file = 'inf.xlsx'


def process_pfas_yearly_data(path, list_pfas_base, suffix=''):
    """
    处理PFAS年度数据，计算平均值并合并结果。可选择是否使用'_median'后缀。

    参数:
    path (str): 包含CSV文件的文件夹路径
    suffix (str): 列名后缀，当为'median'时使用'_median'后缀，否则不使用后缀

    返回:
    pd.DataFrame: 包含所有PFAS物质年度平均值的DataFrame
    """
    # 根据suffix参数决定使用哪种列名
    if suffix == 'median':
        list_pfas_merge = [f"{pfas}_median" for pfas in list_pfas_base]
    else:
        list_pfas_merge = list_pfas_base.copy()
    
    list_pfas_merge.append('year')

    df_merge = pd.DataFrame(columns=list_pfas_merge)

    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            file_path = os.path.join(path, filename)
            df = pd.read_csv(file_path, usecols=list_pfas_merge[:-1])
            mean_values = df.mean()
            df_temp = pd.DataFrame(mean_values).transpose()
            df_temp['year'] = filename[3:7]
            df_merge = pd.concat([df_merge, df_temp], ignore_index=True)

    return df_merge




def start_lr_forecast_uncertain(df_raw, lam, best_params, selected_features, path_forecast_input, save_path_lr, forecast_type, list_pfas, star_year=2000, end_year=2020):
    df_data = df_raw.copy()
    remain_col = ['lon_grid', 'lat_grid', 'year', 'value']

    print(len(selected_features))
    model_params = best_params[best_params["model"] == forecast_type].iloc[0]
    if model_params["max_depth"] == 'None':
        param_max_depth = None
    else:
        param_max_depth = float(model_params["max_depth"])
        param_max_depth = int(param_max_depth)

    X_train = df_data[selected_features]
    y_train = df_data['value']

    random_seeds = random.sample(range(202509), 10)
    print("随机生成的种子：", random_seeds)

    for seed in random_seeds:
        if forecast_type == 'RF':
            select_model = RandomForestRegressor(
                max_depth=param_max_depth,
                min_samples_leaf=int(model_params["min_samples_leaf"]),
                min_samples_split=int(model_params["min_samples_split"]),
                n_estimators=int(model_params["n_estimators"]),
                random_state=seed
            )
        elif forecast_type == 'GBDT':
            select_model = GradientBoostingRegressor(
                max_depth=param_max_depth,
                learning_rate=model_params["learning_rate"],
                min_samples_leaf=int(model_params["min_samples_leaf"]),
                min_samples_split=int(model_params["min_samples_split"]),
                n_estimators=int(model_params["n_estimators"]),
                random_state=seed,
                subsample=0.8
            )
        elif forecast_type == 'LGBM':
            select_model = LGBMRegressor(
                max_depth=param_max_depth,
                learning_rate=model_params["learning_rate"],
                min_child_samples=int(model_params["min_child_samples"]),
                num_leaves=int(model_params["num_leaves"]),
                n_estimators=int(model_params["n_estimators"]),
                random_state=seed,
                subsample=0.8,
                subsample_freq=1
            )
        print(select_model.get_params())
        select_model.fit(X_train, y_train)
        for i in range(star_year,end_year+1):
            for pfas in list_pfas:
                safe_pfas = pfas.replace(':', '-').replace(' ', '-').replace('/', '-')
                for sp_index in [0,1,2,3,4]:
                    df_forecast_data = pd.read_csv(path_forecast_input + 'lr_'+str(i)+'_' + safe_pfas +"_"+str(sp_index) +'.csv')
                    X_forecast = df_forecast_data[selected_features]
                    y_pred = select_model.predict(X_forecast)
                    df_forecast_data['value'] = y_pred
                    df_reture = df_forecast_data[remain_col]
                    df_reture['sp_value'] = inv_boxcox(df_reture['value'], lam)

                    save_dir = os.path.join(save_path_lr, f"{seed}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    save_path = os.path.join(save_dir, f"lr_{i}_{safe_pfas}_{sp_index}.csv")
                    df_reture.to_csv(save_path, index=False)
    return print('okk')


def start_sw_forecast_uncertain(df_raw, lam, best_params, selected_features, path_forecast_input, save_path_sw, forecast_type, list_pfas, star_year=2000, end_year=2020):
    df_data = df_raw.copy()
    remain_col = ['lon_grid', 'lat_grid', 'year', 'value']

    print(len(selected_features))
    model_params = best_params[best_params["model"] == forecast_type].iloc[0]
    if model_params["max_depth"] == 'None':
        param_max_depth = None
    else:
        param_max_depth = float(model_params["max_depth"])
        param_max_depth = int(param_max_depth)

    # Prepare data
    X_train = df_data[selected_features]
    y_train = df_data['value']

    random_seeds = random.sample(range(202509), 10)
    print("随机生成的种子：", random_seeds)

    for seed in random_seeds:
        if forecast_type == 'RF':
            select_model = RandomForestRegressor(
                max_depth=param_max_depth,
                min_samples_leaf=int(model_params["min_samples_leaf"]),
                min_samples_split=int(model_params["min_samples_split"]),
                n_estimators=int(model_params["n_estimators"]),
                random_state=seed
            )
        elif forecast_type == 'GBDT':
            select_model = GradientBoostingRegressor(
                max_depth=param_max_depth,
                learning_rate=model_params["learning_rate"],
                min_samples_leaf=int(model_params["min_samples_leaf"]),
                min_samples_split=int(model_params["min_samples_split"]),
                n_estimators=int(model_params["n_estimators"]),
                random_state=seed,
                subsample=0.8
            )
        elif forecast_type == 'LGBM':
            select_model = LGBMRegressor(
                max_depth=param_max_depth,
                learning_rate=model_params["learning_rate"],
                min_child_samples=int(model_params["min_child_samples"]),
                num_leaves=int(model_params["num_leaves"]),
                n_estimators=int(model_params["n_estimators"]),
                random_state=seed,
                subsample=0.8,
                subsample_freq=1
            )
        print(select_model.get_params())
        select_model.fit(X_train, y_train)
        for i in range(star_year,end_year+1):
            for pfas in list_pfas:
                safe_pfas = pfas.replace(':', '-').replace(' ', '-').replace('/', '-')
                read_path = os.path.join(path_forecast_input, f"sw_{i}_{safe_pfas}.csv")
                df_forecast_data = pd.read_csv(read_path)
                X_forecast = df_forecast_data[selected_features]
                y_pred = select_model.predict(X_forecast)
                df_forecast_data['value'] = y_pred
                df_reture = df_forecast_data[remain_col]
                df_reture['sw_value'] = inv_boxcox(df_reture['value'], lam)
                
                save_dir = os.path.join(save_path_sw, f"{seed}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                save_path = os.path.join(save_dir, f"sw_{i}_{safe_pfas}.csv")
                df_reture.to_csv(save_path, index=False)
    return print('okk')



# 得插入一个转换函数，替换符号和没替换的相互转换一下

def create_converter(df, from_col='posname', to_col='posname_rep'):
    """
    创建一个基于DataFrame映射关系的转换器
    
    参数:
    df: 包含映射关系的DataFrame
    from_col: 源列名
    to_col: 目标列名
    
    返回:
    一个转换函数，可以接受单个值或列表
    """
    # 创建双向映射字典
    forward_map = dict(zip(df[from_col], df[to_col]))
    reverse_map = dict(zip(df[to_col], df[from_col]))
    
    full_map = {**forward_map, **reverse_map}
    
    def converter(input_data):
        if isinstance(input_data, str):
            return full_map.get(input_data, input_data)
        elif isinstance(input_data, list):
            return [full_map.get(item, item) for item in input_data]
        else:
            raise ValueError("输入必须是字符串或字符串列表")
    
    return converter

# 注意这里往后所有文件中，pfas名称涉及两类
# 一类是原始名称，一类是替换名称
# 原始名称用于列名，替换名称用于文件名
df_inf = pd.read_excel(path_inf + inf_file, sheet_name='po_treat')
convert = create_converter(df_inf)



def merge_year(path_input, path_output, list_list_pfas,file_prefix, str_y='value', start_year=2000, end_year=2020):
    year_data = {year: [] for year in range(start_year, end_year + 1)}
    # list_pfas =['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
    # list_pfas_lc = ['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA']
    # list_pfas_sc = ['PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
    # 遍历文件
    for filename in os.listdir(path_input):
        if filename.endswith(".csv"):
            _, year, pfas_name = filename[:-4].split("_")
            year = int(year)
            raw_pfas_name = convert(pfas_name)
            if raw_pfas_name in list_list_pfas[0]:
                df = pd.read_csv(os.path.join(path_input, filename))
                df = df[['lon_grid', 'lat_grid',str_y]]
                
                df = df.rename(columns={str_y: raw_pfas_name})
                
                year_data[year].append(df)
    
    for year, data in year_data.items():
        if data:
            # 从第一个数据框开始，逐个合并其他数据框
            df_year = data[0]
            for df in data[1:]:
                df_year = df_year.merge(df, on=['lon_grid', 'lat_grid'], how='outer')
            df_year['value'] = df_year[list_list_pfas[0]].sum(axis=1)
            df_year['lc_value'] = df_year[list_list_pfas[1]].sum(axis=1)
            df_year['sc_value'] = df_year[list_list_pfas[2]].sum(axis=1)
            df_year.to_csv(f"{path_output}/{file_prefix}_{year}.csv", index=False)
    return print('merge over')


def merge_csv_files(base_path, output_path, file_prefix):
    """
    合并指定路径下所有文件夹中的CSV文件，并保存到输出路径。

    参数:
    base_path (str): 包含待合并CSV文件的文件夹的基础路径
    output_path (str): 合并后CSV文件的输出路径

    返回:
    None
    """
    os.makedirs(output_path, exist_ok=True)

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        print(f"Processing folder: {folder_name}")
        merged_data = pd.DataFrame()

        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    
                    df = pd.read_csv(file_path)
                    
                    year = file_name[3:7]
                    df['year'] = year
                    
                    merged_data = pd.concat([merged_data, df], ignore_index=True)

        output_file_name = f'{file_prefix}_{folder_name}.csv'
        output_file_path = os.path.join(output_path, output_file_name)

        merged_data.to_csv(output_file_path, index=False)

        print(f'Merge completed, file saved to: {output_file_path}')
    

def process_csv_files(input_path, output_path, columns, file_prefix):
    """
    处理指定路径下的CSV文件，合并数据并计算统计量。

    参数:
    input_path (str): 输入CSV文件的路径
    output_path (str): 输出CSV文件的路径
    columns (list): 需要处理的列名列表,pfas名称

    返回:
    None
    """
    os.makedirs(output_path, exist_ok=True)

    csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]

    for col in columns:
        merged_df = pd.DataFrame()
        col_names = []  
        safe_pfas_name = convert(col)
        for csv_file in csv_files:
            file_path = os.path.join(input_path, csv_file)
            df = pd.read_csv(file_path)
            file_identifier = csv_file.replace(f'{file_prefix}_', '').replace('.csv', '')
            new_col_name = f"{col}_{file_identifier}"
            col_names.append(new_col_name)  
            
            temp_df = df[['lon_grid', 'lat_grid', 'year', col]].copy()
            temp_df.rename(columns={col: new_col_name}, inplace=True)
            
            if merged_df.empty:
                merged_df = temp_df
            else:
                merged_df = pd.merge(merged_df, temp_df, on=['lon_grid', 'lat_grid', 'year'], how='outer')
        
        print(f"Processing column: {col}")
        print(f"Used column names: {col_names}")

        merged_df[f'{col}_min'] = merged_df[col_names].min(axis=1)
        merged_df[f'{col}_max'] = merged_df[col_names].max(axis=1)
        merged_df[f'{col}_median'] = merged_df[col_names].median(axis=1)
        merged_df[col] = merged_df[col_names].mean(axis=1)
        merged_df[f'{col}_cv'] = merged_df[col_names].std(axis=1) / merged_df[col]

        output_file_name = f'{file_prefix}_{safe_pfas_name}.csv'
        output_file_path = os.path.join(output_path, output_file_name)
        merged_df.to_csv(output_file_path, index=False)

        print(f"Merge completed, file saved to: {output_file_path}")




def process_and_split_pfas_data(input_path, output_path, list_pfas, list_pfas_lc, list_pfas_sc, start_year=2000, end_year=2020, file_prefix='lr', value_type=''):
    """
    处理PFAS数据，合并CSV文件，计算总值，并按年份分割保存。

    参数:
    input_path (str): 输入CSV文件的路径
    output_path (str): 输出CSV文件的路径
    list_pfas (list): 所有PFAS的列表
    list_pfas_lc (list): 长链PFAS的列表
    list_pfas_sc (list): 短链PFAS的列表
    start_year (int): 开始年份（默认为2000）
    end_year (int): 结束年份（默认为2021）

    返回:
    None
    """
    merged_df = pd.DataFrame()
    if value_type == '':
        str_suffix = ''
    else:
        str_suffix = '_' + value_type
    print('suffix:', str_suffix)
    for pfas in list_pfas:
        safe_pfas_name = convert(pfas)
        file_name = f'{file_prefix}_{safe_pfas_name}.csv'
        file_path = os.path.join(input_path, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            columns_to_keep = ['lon_grid', 'lat_grid', 'year', f'{pfas}{str_suffix}']
            df = df[columns_to_keep]
            
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=['lon_grid', 'lat_grid', 'year'], how='outer')

    print(merged_df.columns)
    print('list_pfas:', list_pfas)
    merged_df['value'] = merged_df[[pfas+str_suffix for pfas in list_pfas]].sum(axis=1)
    merged_df['lc_value'] = merged_df[[pfas+str_suffix for pfas in list_pfas_lc]].sum(axis=1)
    merged_df['sc_value'] = merged_df[[pfas+str_suffix for pfas in list_pfas_sc]].sum(axis=1)

    os.makedirs(output_path, exist_ok=True)

    for year in range(start_year, end_year+1):
        year_df = merged_df[merged_df['year'] == year].copy()
        year_df = year_df.drop(columns=['year'])
        
        for col in list_pfas + ['value', 'lc_value', 'sc_value']:
            if col not in year_df.columns:
                year_df[col] = np.nan
        
        columns_order = ['lon_grid', 'lat_grid'] + [pfas+str_suffix for pfas in list_pfas] + ['value', 'lc_value', 'sc_value']
        year_df = year_df[columns_order]
        
        output_file = os.path.join(output_path, f'{file_prefix}_{year}.csv')
        year_df.to_csv(output_file, index=False)

    print("处理完成。所有文件已保存到指定目录。")

# 下面这个只有lr需要，用于处理organ或者说tissue
def process_pfas_data(base_folder_path, output_base_path, list_pfas, start_year=2000, end_year=2020):
    """
    处理PFAS数据，计算平均值并保存结果。

    参数:
    base_folder_path (str): 包含原始数据的基础文件夹路径
    output_base_path (str): 输出结果的基础文件夹路径

    返回:
    None
    """
    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)
        
        if os.path.isdir(folder_path):  # 确保是文件夹
            file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
            
            for year in range(start_year, end_year + 1):
                for pfas in list_pfas:
                    safe_pfas_name = convert(pfas)
                    relevant_files = [file for file in file_paths if file.split('\\')[-1].startswith(f'lr_{year}_{safe_pfas_name}_')]
                    
                    if relevant_files:
                        dfs = [pd.read_csv(file, usecols=['lon_grid', 'lat_grid', 'sp_value']) for file in relevant_files]
                        combined_df = pd.concat(dfs)
                        
                        averaged_df = combined_df.groupby(['lon_grid', 'lat_grid']).mean().reset_index()
                        
                        output_folder_path = os.path.join(output_base_path, folder_name)
                        
                        if not os.path.exists(output_folder_path):
                            os.makedirs(output_folder_path)
                        
                        output_filename = f'lr_{year}_{safe_pfas_name}.csv'
                        output_path = os.path.join(output_folder_path, output_filename)
                        averaged_df.to_csv(output_path, index=False)

    print("处理完成！")
