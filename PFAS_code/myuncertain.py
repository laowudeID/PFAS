
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import myfunction as mf


list_pfas =['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
list_pfas_lc = ['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA']
list_pfas_sc = ['PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
list_color = ["#4d8cbf", "#4f9c8b", "#555c6c", "#d77563", "#7d84a8", "#84aeb8", "#c3473b", "#89756d","#ffb3cc","#9a7ebf","#ffddb8", "#c4eaff", "#d1c6ff", "#c2ffbf", "#f5f5b0"]
dic_color = dict(zip(list_pfas,list_color))



def process_pfas_yearly_data(path, suffix=''):
    """
    Process Pfas annual data, calculate the average and merge the results. Optionally use the” suffix.

    Parameters:
    Path (Str) : the path to the folder containing the CSV file
    Suffix (str) : column name suffix, use” suffix when 'median' , otherwise do not use suffix

    Back:
    Pd. DataFrame: a DataFrame containing the annual average of all PFAS substances
    ---
    处理PFAS年度数据，计算平均值并合并结果。可选择是否使用'_median'后缀。

    参数:
    path (str): 包含CSV文件的文件夹路径
    suffix (str): 列名后缀，当为'median'时使用'_median'后缀，否则不使用后缀

    返回:
    pd.DataFrame: 包含所有PFAS物质年度平均值的DataFrame
    """
    list_pfas_base = ['PFOA', 'PFNA', 'PFDA', 'PFUnDA', 'PFDoDA', 'PFTrDA', 'PFTeDA', 
                      'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA', 'PFBS']

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



def draw_year_gpfas(df_merge, path_save='N', mark_str='sw', y_max=20):
    df_test5 = df_merge.copy()
    if mark_str == 'sw':
        str_unit = 'ng/l'
    if mark_str == 'lr':
        str_unit = 'ng/g'
    df_test5[["year"]] = df_test5[["year"]].astype(float)
    df_test5[["year"]] = df_test5[["year"]].astype(int)
    df_test5 = df_test5.set_index("year")

    df_merge_copy_td_po = df_merge.copy()
    df_merge_copy_od = mf.two_to_one(df_merge_copy_td_po, ['year'])

    df_test5_tb2 = mf.one_to_two(df_merge_copy_od, ['po_name'], 'year')
    df_test5_tb2["color"] = df_test5_tb2["po_name"].map(dic_color)
    df_test5_tb2 = df_test5_tb2.drop(columns=['po_name'])

    df_test5_tb2["sum"] = df_test5_tb2[[str(year) for year in range(2000, 2021)]].sum(axis=1)
    df_test5_tb2 = df_test5_tb2.sort_values(by=("sum"),ascending=False)

    list_po_need = list(df_test5_tb2.index)
    list_clolor_need = list(df_test5_tb2["color"])
    year = list(df_test5.index)
    df_test5 = df_test5[list_po_need]
    dic_data = df_test5.to_dict("list")

    fig, ax = plt.subplots()
    sns.set_theme(style="ticks")
    ax.stackplot(year, dic_data.values(),
                labels=dic_data.keys(), alpha=0.9,colors=list_clolor_need)
    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel(f'PFASs Concentration({str_unit})', fontsize=18)
    ax.set(ylim=(0,y_max))
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16, rotation=90)
    ax.figure.set_size_inches(12,4)
    plt.grid(axis="x")
    ax.xaxis.set_ticks(np.arange(min(year), max(year)+1, 1))
    ax.xaxis.set_ticklabels(np.arange(min(year), max(year)+1, 1))
    plt.legend(bbox_to_anchor=(1.01, -0.1), loc=3,borderaxespad=0)

    plt.subplots_adjust(left=0.1,right=0.85,bottom=0.2,top=0.95)
    if path_save == 'N':
        pass
    else:
        plt.savefig(path_save + "fig5_"+mark_str+"_global_stack.svg", dpi = 300)
    plt.show()
    return 'okk'



def start_lr_forecast_uncertain(df_sem_o, df_raw_o, best_params, selected_features, path_forecast_input, save_path_lr, forecast_type):
    df_sem = df_sem_o.copy()
    df_raw = df_raw_o.copy()
    list_pfas =['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
    remain_col = ['lon_grid', 'lat_grid', 'year_raw', 'sp_value']
    df_raw['value'], fitted_lambdas = stats.boxcox(df_raw['value'])
    scaler = StandardScaler()
    df_raw['value'] = scaler.fit_transform(df_raw['value'].values.reshape(-1, 1))
    mean = scaler.mean_
    std = scaler.scale_

    print(len(selected_features))
    # Get best parameters for GBDT and RF
    model_params = best_params[best_params["model"] == forecast_type].iloc[0]
    # Define models with best parameters    
    if model_params["param_max_depth"] == 'None':
        param_max_depth = None
    else:
        param_max_depth = float(model_params["param_max_depth"])
        param_max_depth = int(param_max_depth)

    # Prepare data
    X_train = df_sem[selected_features]
    y_train = df_sem['value']

    random_seeds = random.sample(range(202406), 10)
    print("随机生成的种子：", random_seeds)

    for seed in random_seeds:
        if forecast_type == 'RF':
            select_model = RandomForestRegressor(
                max_depth=param_max_depth,
                min_samples_leaf=int(model_params["param_min_samples_leaf"]),
                min_samples_split=int(model_params["param_min_samples_split"]),
                n_estimators=int(model_params["param_n_estimators"]),
                random_state=seed
            )
        elif forecast_type == 'GBDT':
            select_model = GradientBoostingRegressor(
                max_depth=param_max_depth,
                learning_rate=model_params["param_learning_rate"],
                min_samples_leaf=int(model_params["param_min_samples_leaf"]),
                min_samples_split=int(model_params["param_min_samples_split"]),
                n_estimators=int(model_params["param_n_estimators"])
            )
        print(select_model.get_params())
        select_model.fit(X_train, y_train)
        for i in range(2000,2021):
            for pfas in list_pfas:
                for sp_index in [0,1,2,3,4,5]:
                    df_forecast_data = pd.read_csv(path_forecast_input + 'lr_'+str(i)+'_' + pfas +"_"+str(sp_index) +'.csv')
                    df_forecast_data = df_forecast_data.rename(columns={'troph_last': 'sp_troph', 'weight_last': 'sp_weight', 'length_last': 'sp_length'})
                    df_forecast_data['year_raw'] = df_forecast_data['year']
                    df_forecast_data['year'] = (df_forecast_data['year'] - 2000) / (2020 - 2000)
                    X_forecast = df_forecast_data[selected_features]
                    y_pred = select_model.predict(X_forecast)
                    df_forecast_data['sp_value'] = y_pred
                    df_reture = df_forecast_data[remain_col]
                    df_reture = df_reture.rename(columns={'year_raw':'year'})

                    df_reture['sp_value'] = df_reture['sp_value'] * std + mean
                    df_reture['value'] = inv_boxcox(df_reture['sp_value'], fitted_lambdas)

                    save_dir = os.path.join(save_path_lr, f"{seed}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    save_path = os.path.join(save_dir, f"lr_{i}_{pfas}_{sp_index}.csv")
                    df_reture.to_csv(save_path, index=False)
    return print('okk')


def start_sw_forecast_uncertain(df_sem_o, df_raw_o, best_params, selected_features, path_forecast_input, save_path_sw, forecast_type):
    df_sem = df_sem_o.copy()
    df_raw = df_raw_o.copy()
    list_pfas =['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
    remain_col = ['lon_grid', 'lat_grid', 'year_raw', 'sw_value']
    df_raw['value'], fitted_lambdas = stats.boxcox(df_raw['value'])
    scaler = StandardScaler()
    df_raw['value'] = scaler.fit_transform(df_raw['value'].values.reshape(-1, 1))
    mean = scaler.mean_
    std = scaler.scale_

    print(len(selected_features))
    # Get best parameters for GBDT and RF
    model_params = best_params[best_params["model"] == forecast_type].iloc[0]
    # Define models with best parameters    
    if model_params["param_max_depth"] == 'None':
        param_max_depth = None
    else:
        param_max_depth = float(model_params["param_max_depth"])
        param_max_depth = int(param_max_depth)

    # Prepare data
    X_train = df_sem[selected_features]
    y_train = df_sem['value']

    random_seeds = random.sample(range(202406), 10)
    print("随机生成的种子：", random_seeds)


    for seed in random_seeds:
        if forecast_type == 'RF':
            select_model = RandomForestRegressor(
                max_depth=param_max_depth,
                min_samples_leaf=int(model_params["param_min_samples_leaf"]),
                min_samples_split=int(model_params["param_min_samples_split"]),
                n_estimators=int(model_params["param_n_estimators"]),
                random_state=seed
            )
        elif forecast_type == 'GBDT':
            select_model = GradientBoostingRegressor(
                max_depth=param_max_depth,
                learning_rate=model_params["param_learning_rate"],
                min_samples_leaf=int(model_params["param_min_samples_leaf"]),
                min_samples_split=int(model_params["param_min_samples_split"]),
                n_estimators=int(model_params["param_n_estimators"])
            )
        print(select_model.get_params())
        select_model.fit(X_train, y_train)
        for i in range(2000,2021):
            for pfas in list_pfas:
                df_forecast_data = pd.read_csv(path_forecast_input + 'sw_'+str(i)+'_' + pfas + '.csv')
                df_forecast_data['year_raw'] = df_forecast_data['year']
                df_forecast_data['year'] = (df_forecast_data['year'] - 2000) / (2020 - 2000)
                # df_fore_output = forecast_start()
                X_forecast = df_forecast_data[selected_features]
                y_pred = select_model.predict(X_forecast)
                df_forecast_data['sw_value'] = y_pred
                df_reture = df_forecast_data[remain_col]
                df_reture = df_reture.rename(columns={'year_raw':'year'})

                df_reture['sw_value'] = df_reture['sw_value'] * std + mean
                df_reture['value'] = inv_boxcox(df_reture['sw_value'], fitted_lambdas)
                
                save_dir = os.path.join(save_path_sw, f"{seed}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"sw_{i}_{pfas}.csv")
                df_reture.to_csv(save_path, index=False)
    return print('okk')






def merge_year(path_input, path_output, file_prefix, str_y='value', max_year=2021):
    year_data = {year: [] for year in range(2000, max_year)}
    list_pfas =['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
    list_pfas_lc = ['PFOA', 'PFNA', 'PFDA', 'PFUnDA','PFDoDA','PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA']
    list_pfas_sc = ['PFBA', 'PFPeA', 'PFHxA', 'PFHpA','PFBS']
    for filename in os.listdir(path_input):
        if filename.endswith(".csv"):
            _, year, pfas_name = filename[:-4].split("_")
            year = int(year)
            if pfas_name in list_pfas:
                df = pd.read_csv(os.path.join(path_input, filename))
                df = df[['lon_grid', 'lat_grid',str_y]]
                df = df.rename(columns={str_y: pfas_name})
                year_data[year].append(df)
    
    for year, data in year_data.items():
        if data:
            df_year = data[0]
            for df in data[1:]:
                df_year = df_year.merge(df, on=['lon_grid', 'lat_grid'], how='outer')
            df_year[str_y] = df_year[list_pfas].sum(axis=1)
            df_year['lc_' + str_y] = df_year[list_pfas_lc].sum(axis=1)
            df_year['sc_' + str_y] = df_year[list_pfas_sc].sum(axis=1)
            df_year.to_csv(f"{path_output}/{file_prefix}_{year}.csv", index=False)
    return print('merge over')


def merge_csv_files(base_path, output_path, file_prefix):
    """
    Merges CSV files in all folders under the specified path and saves them to the output path.

    Parameters:
    Base (Str) : the base path of the folder containing the CSV files to be merged
    Output (STR) : the output path of the merged CSV file

    Back:
        None
    ---
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
    Processes a CSV file in a specified path, merges data, and computes statistics.

    Parameters:
    Input (STR-RRB- : input the path to CSV CSV file
    Output (str) : the path to the output CSV file
    Columns (list) : List of column names to be processed

    Back:
        None
    ---
    处理指定路径下的CSV文件，合并数据并计算统计量。

    参数:
    input_path (str): 输入CSV文件的路径
    output_path (str): 输出CSV文件的路径
    columns (list): 需要处理的列名列表

    返回:
    None
    """
    os.makedirs(output_path, exist_ok=True)

    csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]

    for col in columns:
        merged_df = pd.DataFrame()
        col_names = []  
        
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

        output_file_name = f'{file_prefix}_{col}.csv'
        output_file_path = os.path.join(output_path, output_file_name)
        merged_df.to_csv(output_file_path, index=False)

        print(f"Merge completed, file saved to: {output_file_path}")




def process_and_split_pfas_data(input_path, output_path, list_pfas, list_pfas_lc, list_pfas_sc, start_year=2000, end_year=2021, file_prefix='lr', value_type=''):
    """
    Process Pfas data, merge CSV file, calculate total value, and save by year.

    Parameters:
    Input (STR-RRB- : input the path to CSV CSV file
    Output (str) : the path to the output CSV file
    List (list) : a list of all Pfas
    List (list) : a list of long-chain PFAS
    List (list) : a list of short-chained PFAS
    Start (Int) : Start year (defaults to 2000) 
    End (Int) : end year (defaults to 2021) 

    Back:
        None
    ---
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
        file_name = f'{file_prefix}_{pfas}.csv'
        file_path = os.path.join(input_path, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            columns_to_keep = ['lon_grid', 'lat_grid', 'year', f'{pfas}{str_suffix}']
            df = df[columns_to_keep]
            
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=['lon_grid', 'lat_grid', 'year'], how='outer')

    merged_df['value'] = merged_df[[pfas+str_suffix for pfas in list_pfas]].sum(axis=1)
    merged_df['lc_value'] = merged_df[[pfas+str_suffix for pfas in list_pfas_lc]].sum(axis=1)
    merged_df['sc_value'] = merged_df[[pfas+str_suffix for pfas in list_pfas_sc]].sum(axis=1)

    os.makedirs(output_path, exist_ok=True)

    for year in range(start_year, end_year):
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

def process_pfas_data(base_folder_path, output_base_path):
    """
    Process Pfas data, calculate the average value and save the results.

    Parameters:
    Base (Str) : the base folder path that contains the original data
    Output (STR) : the underlying folder path of the output

    Back:
        None
    ---
    处理PFAS数据，计算平均值并保存结果。

    参数:
    base_folder_path (str): 包含原始数据的基础文件夹路径
    output_base_path (str): 输出结果的基础文件夹路径

    返回:
    None
    """
    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)
        
        if os.path.isdir(folder_path):  
            file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

            for year in range(2000, 2021):
                for pfas in ['PFOA', 'PFNA', 'PFDA', 'PFUnDA', 'PFDoDA', 'PFTrDA', 'PFTeDA', 'PFHxS', 'PFOS', 'FOSA', 'PFBA', 'PFPeA', 'PFHxA', 'PFHpA', 'PFBS']:
                    relevant_files = [file for file in file_paths if file.split('\\')[-1].startswith(f'lr_{year}_{pfas}_')]
                    
                    if relevant_files:

                        dfs = [pd.read_csv(file, usecols=['lon_grid', 'lat_grid', 'value']) for file in relevant_files]
                        combined_df = pd.concat(dfs)
                        averaged_df = combined_df.groupby(['lon_grid', 'lat_grid']).mean().reset_index()
                        output_folder_path = os.path.join(output_base_path, folder_name)

                        if not os.path.exists(output_folder_path):
                            os.makedirs(output_folder_path)
                        
                        output_filename = f'lr_{year}_{pfas}.csv'
                        output_path = os.path.join(output_folder_path, output_filename)
                        averaged_df.to_csv(output_path, index=False)

    print("处理完成！")
