# 函数准备
import os
import pandas as pd
import numpy as np
import xarray as xr
from lightgbm import LGBMRegressor
from scipy.stats import skew, kurtosis, yeojohnson, boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import myfunction as mf
import dask.array as da
from xgboost import XGBRegressor

def normalize_geo_from_csv(df_raw, method='zscore', feature_range=(0, 1),
                           train_years=(2000, 2020), predict_years=(2021, 2050)):
    """
    从一个 CSV 文件加载地理变量，自动按年份分成训练集和预测集，先用训练集拟合 scaler，再对预测集进行相同参数的转换。

    参数
    ----
    csv_file : str
        数据文件路径
    method : str
        'zscore' -> Z-score 标准化
        'minmax' -> 最小值最大值归一化
    feature_range : tuple
        MinMaxScaler 的缩放范围
    TEMP_PATH : str
        存储输出数据和 scaler 的目录
    train_years : tuple
        (start_year, end_year) 训练集年份范围（闭区间）
    predict_years : tuple
        (start_year, end_year) 预测集年份范围（闭区间）
    """
    df = df_raw.copy()

    base_cols = ['lat_grid', 'lon_grid', 'year']
    feature_cols = [col for col in df.columns if col not in base_cols]
    if not feature_cols:
        raise ValueError("数据中没有除 lat_grid、lon_grid、year 之外的特征列")

    # 按年份分成训练集和预测集
    df_train = df[(df['year'] >= train_years[0]) & (df['year'] <= train_years[1])].copy()
    df_predict = df[(df['year'] >= predict_years[0]) & (df['year'] <= predict_years[1])].copy()

    print(f"训练集年份: {train_years[0]} - {train_years[1]}, 样本数: {len(df_train)}")
    print(f"预测集年份: {predict_years[0]} - {predict_years[1]}, 样本数: {len(df_predict)}")

    # 选择归一化方法
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    else:
        raise ValueError("method 参数必须是 'zscore' 或 'minmax'")
    scaler.fit(df_train[feature_cols])
    df_train[feature_cols] = scaler.transform(df_train[feature_cols])
    df_predict[feature_cols] = scaler.transform(df_predict[feature_cols])
    df_combined = pd.concat([df_train, df_predict], ignore_index=True)
    return df_combined, scaler

def apply_random_factors(df_geo, scenario_factors_range, epi_norm_df, seed, 
                        scenario, current_year=2025):
    """
    对数据框应用场景系数和国家系数调整
    
    Parameters:
    -----------
    df_geo : DataFrame
        包含地理数据的数据框，必须有'year', 'lon', 'lat'列
    scenario_factors_range : dict
        场景系数范围字典，格式: {'var_name': {'coef_down': value, 'coef_up': value}}
    epi_norm_df : DataFrame
        国家环境绩效指数，包含'lon', 'lat', 'EPI_norm'列
    seed : int
        随机种子
    scenario : str
        场景类型: 'rs1', 'rs2', 'us', 'baseline'
    current_year : int
        起始年份，默认2025
        
    Returns:
    --------
    df_adjusted : DataFrame
        调整后的数据框
    """
    np.random.seed(seed)
    df_adjusted = df_geo.copy()
    
    # 基准场景：不做调整
    if scenario == 'baseline':
        return df_adjusted
    
    # 合并EPI系数（仅在rs1和rs2场景下使用）
    use_epi = scenario in ['rs1', 'rs2']
    if use_epi:
        df_adjusted = df_adjusted.merge(
            epi_norm_df[['lon_grid', 'lat_grid', 'EPI_norm']], 
            on=['lon_grid', 'lat_grid'], 
            how='left'
        )
        df_adjusted['EPI_norm'].fillna(1.0, inplace=True)
    else:
        # us和baseline场景不使用EPI系数，设为1
        df_adjusted['EPI_norm'] = 1.0
    
    # 为每个变量随机抽取目标场景系数
    target_coefficients = {}
    for var_name, coef_range in scenario_factors_range.items():
        coef_down = coef_range['coef_down']
        coef_up = coef_range['coef_up']
        epsilon = (coef_up - coef_down) * 1e-6
        target_coefficients[var_name] = np.random.uniform(
            coef_down + epsilon, 
            coef_up - epsilon
        )
    
    # 定义需要增加的变量（其他变量都是减少）
    increasing_vars = ['distance_to_sources']
    
    # 对每个需要调整的变量应用系数
    for var_name, target_coef1 in target_coefficients.items():
        if var_name not in df_adjusted.columns:
            continue
            
        # 判断是增加还是减少
        is_increasing = var_name in increasing_vars
        
        # 获取年份数据
        if 'year' in df_adjusted.columns:
            years = df_adjusted['year'].values
        else:
            years = np.arange(current_year, current_year + len(df_adjusted))
        
        # 获取每行的EPI系数
        epi_coefficients = df_adjusted['EPI_norm'].values
        
        # 计算每年的调整系数（结合场景系数和国家系数）
        adjustment_factors = calculate_scenario_factors(
            years, target_coef1, epi_coefficients, scenario, 
            is_increasing, current_year
        )
        # 应用调整系数
        df_adjusted[var_name] = df_adjusted[var_name] * adjustment_factors
    
    # 删除临时列
    df_adjusted.drop('EPI_norm', axis=1, inplace=True)
    
    return df_adjusted


def calculate_scenario_factors(years, target_coef1, epi_coefficients, scenario, 
                               is_increasing, start_year=2025):
    """
    计算每年的调整系数（结合场景系数和国家系数）
    
    Parameters:
    -----------
    years : array
        年份数组
    target_coef1 : float
        场景目标系数（负值表示减少百分比，如-0.02表示减少2%）
    epi_coefficients : array
        每行对应的国家EPI归一化系数
    scenario : str
        场景类型
    is_increasing : bool
        True表示增加，False表示减少
    start_year : int
        起始年份
        
    Returns:
    --------
    factors : array
        每年的调整系数
    """
    factors = np.ones(len(years))
    
    # us和baseline场景不使用国家系数，直接使用原逻辑
    use_epi = scenario in ['rs1', 'rs2']
    
    if scenario == 'rs1':
        # 限制场景1
        for i, year in enumerate(years):
            progress_factor = _piecewise_exponential_progress(
                year, start_year,
                phase1_end=2027, phase2_end=2030, phase3_end=2032
            )
            
            if use_epi:
                combined_coef = abs(target_coef1) * epi_coefficients[i]
            else:
                combined_coef = abs(target_coef1)
            if is_increasing:
                # distance_to_sources: final_factor = 1 + combined_coef * progress
                factors[i] = 1 + combined_coef * progress_factor
            else:
                # 其他变量: final_factor = 1 - combined_coef * progress
                factors[i] = 1 - combined_coef * progress_factor
    
    elif scenario == 'rs2':
        # 限制场景2
        for i, year in enumerate(years):
            progress_factor = _piecewise_exponential_progress(
                year, start_year,
                phase1_end=2027, phase2_end=2037, phase3_end=2039
            )
            
            if use_epi:
                combined_coef = abs(target_coef1) * epi_coefficients[i]
            else:
                combined_coef = abs(target_coef1)
            
            if is_increasing:
                factors[i] = 1 + combined_coef * progress_factor
            else:
                factors[i] = 1 - combined_coef * progress_factor
    
    elif scenario == 'us':
        # 无限制场景：线性变化到2050，不使用EPI系数
        total_change = abs(target_coef1)
        
        for i, year in enumerate(years):
            if year <= start_year:
                factors[i] = 1.0
            elif year >= 2050:
                # 2050年反向达到限制场景的系数
                if is_increasing:
                    factors[i] = 1 - total_change
                else:
                    factors[i] = 1 + total_change
            else:
                # 线性插值
                progress = (year - start_year) / (2050 - start_year)
                if is_increasing:
                    factors[i] = 1 - total_change * progress
                else:
                    factors[i] = 1 + total_change * progress
    
    return factors


def _piecewise_exponential_progress(year, start_year, phase1_end, phase2_end, phase3_end):
    """
    计算分段指数衰减的进度因子（0到1）
    
    阶段1：指数衰减，完成40%
    阶段2：近似常数（保持在40%）
    阶段3：指数衰减，完成剩余60%（从40%到100%）
    
    Parameters:
    -----------
    year : int
        当前年份
    start_year : int
        起始年份
    phase1_end : int
        阶段1结束年份
    phase2_end : int
        阶段2结束年份
    phase3_end : int
        阶段3结束年份
        
    Returns:
    --------
    progress : float
        进度因子（0-1之间）
    """
    if year <= start_year:
        return 0.0
    elif year >= phase3_end:
        return 1.0
    
    # 阶段1：指数衰减到40%
    if year <= phase1_end:
        t = (year - start_year) / (phase1_end - start_year)
        # 指数衰减，达到40%
        progress = 0.4 * (1 - np.exp(-3 * t))
        return progress
    
    # 阶段2：保持在40%
    elif year <= phase2_end:
        return 0.4
    
    # 阶段3：指数衰减，从40%到100%
    else:
        t = (year - phase2_end) / (phase3_end - phase2_end)
        # 指数衰减，从40%增加到100%
        progress = 0.4 + 0.6 * (1 - np.exp(-3 * t))
        return progress


def _piecewise_exponential(year, start_year, total_change, 
                          phase1_end, phase2_end, phase3_end):
    """
    原有的分段指数函数（保留以防需要）
    """
    progress = _piecewise_exponential_progress(
        year, start_year, phase1_end, phase2_end, phase3_end
    )
    return 1 + total_change * progress



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
        y_bc = y 
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



def train_forecast_model(df_raw, best_params, selected_features, best_model, seed):
    df_data = df_raw.copy()

    model_params = best_params[best_params["model"] == best_model].iloc[0]
    if pd.isna(model_params["max_depth"]) or str(model_params["max_depth"]).lower() == 'none':
        param_max_depth = None
    else:
        param_max_depth = int(float(model_params["max_depth"]))

    X_train = df_data[selected_features].copy()
    y_train = df_data['value'].copy()

    if best_model == 'RF':
        select_model = RandomForestRegressor(
            max_depth=param_max_depth,
            min_samples_leaf=int(model_params["min_samples_leaf"]),
            min_samples_split=int(model_params["min_samples_split"]),
            n_estimators=int(model_params["n_estimators"]),
            random_state=seed
        )
    elif best_model == 'XGBR':
        select_model = XGBRegressor(
            max_depth=param_max_depth,
            learning_rate=model_params["learning_rate"],
            min_child_weight=int(model_params["min_child_weight"]),
            gamma=int(model_params["gamma"]),
            n_estimators=int(model_params["n_estimators"]),
            random_state=seed,
            subsample=0.8,
            n_jobs=1
        )
    elif best_model == 'LGBM':
        select_model = LGBMRegressor(
            max_depth=param_max_depth,
            learning_rate=model_params["learning_rate"],
            min_child_samples=int(model_params["min_child_samples"]),
            num_leaves=int(model_params["num_leaves"]),
            n_estimators=int(model_params["n_estimators"]),
            random_state=seed,
            subsample=0.8,
            subsample_freq=1,
            n_jobs=1
        )
    print(select_model.get_params())
    select_model.fit(X_train, y_train)
    return select_model

def sw_forecast_and_save_nc(list_model_build, list_inv_parm, df_input, save_path_sw_nc, 
                            list_list_pfas, dict_inf_po, seed, treat_value=False):
    """
    使用训练好的模型进行预测，并将结果保存为.nc文件。
    """
    all_pfas_data = {}
    list_pfas, list_pfas_all_lc, list_pfas_all_sc = list_list_pfas
    df_raw, best_params, selected_features, best_model = list_model_build
    model = train_forecast_model(df_raw, best_params, selected_features, best_model, seed)

    for pfas in list_pfas:
        safe_pfas = pfas.replace(':', '-').replace(' ', '-').replace('/', '-')
        df_forecast_data = df_input.copy()
        df_forecast_data['posname'] = pfas
        df_forecast_data = mf.append_inf(df_forecast_data, dict_inf_po, treat_value)
        original_year = df_forecast_data['year'].copy()
        df_forecast_data['year'] = (df_forecast_data['year'] - 2000) / (2020 - 2000)
        X_forecast = df_forecast_data[selected_features]
        y_pred = model.predict(X_forecast)
        df_forecast_data['value'] = y_pred
        df_forecast_data['sw_value'] = custom_inv(df_forecast_data['value'], list_inv_parm, treat_value)
        df_forecast_data['year'] = original_year
        df_pfas = df_forecast_data[['lon_grid', 'lat_grid', 'year', 'sw_value']].copy()
        df_pfas = df_pfas.rename(columns={'lon_grid': 'lon', 'lat_grid': 'lat'})
        ds_pfas = df_pfas.set_index(['year', 'lat', 'lon']).to_xarray()
        all_pfas_data[safe_pfas] = ds_pfas['sw_value']
    # 合并并保存
    ds_combined = xr.Dataset(all_pfas_data)
    os.makedirs(save_path_sw_nc, exist_ok=True)

    vars_all = list(ds_combined.data_vars)  # 所有变量
    vars_lc = [pfas.replace(':', '-').replace(' ', '-').replace('/', '-') 
               for pfas in list_pfas_all_lc if pfas in list_pfas]
    vars_sc = [pfas.replace(':', '-').replace(' ', '-').replace('/', '-') 
               for pfas in list_pfas_all_sc if pfas in list_pfas]
    
    vars_lc = [v for v in vars_lc if v in ds_combined.data_vars]
    vars_sc = [v for v in vars_sc if v in ds_combined.data_vars]
    
    ds_combined['value'] = sum([ds_combined[v] for v in vars_all])
    ds_combined['lc_value'] = sum([ds_combined[v] for v in vars_lc]) if vars_lc else xr.zeros_like(ds_combined[vars_all[0]])
    ds_combined['sc_value'] = sum([ds_combined[v] for v in vars_sc]) if vars_sc else xr.zeros_like(ds_combined[vars_all[0]])
    for var in ds_combined.data_vars:
            ds_combined[var] = ds_combined[var].round(4)
    save_path = os.path.join(save_path_sw_nc, f"sw_forecast_{seed}.nc")
    ds_combined.to_netcdf(save_path)
    
    return f"Seed {seed} completed: {save_path}"



def lr_forecast_and_save_nc(list_model_build, list_inv_parm, df_input, save_path_lr_nc, save_path_sw_nc,
                             list_list_pfas, dict_inf_po, df_sp_cluster_treat, seed, treat_value, list_inv_parm_sw, lam_sw=0.0083):
    """
    使用训练好的模型进行 LR 预测，并将结果保存为 .nc 文件。
    
    参数:
        model : 已训练的模型
        lam : Box-Cox 变换参数 λ
        selected_features : 特征列列表
        df_input : 输入的基础地理数据 DataFrame
        save_path_lr_nc : 存放 LR nc 的路径
        save_path_sw_nc : 存放 SW nc 的路径（用于读取水体预测数据）
        list_pfas : PFAS 污染物列表
        dict_inf_po : 附加信息字典
        df_sp_cluster_treat : 物种特征数据 DataFrame（包含 sp_length、sp_weight、sp_troph）
        seed : 随机种子（用于保存文件命名）
    """
    df_raw, best_params, selected_features, best_model = list_model_build
    a = list_inv_parm_sw[0]  # mean 或 min
    b = list_inv_parm_sw[1]  # std 或 max
    list_all_pfas, list_lc_pfas, list_sc_pfas = list_list_pfas
    model = train_forecast_model(df_raw, best_params, selected_features, best_model, seed)
    for j in range(0, 5):
        all_pfas_data = {}
        for pfas in list_all_pfas:
            safe_pfas = pfas.replace(':', '-').replace(' ', '-').replace('/', '-')
            sw_forecast = xr.open_dataset(os.path.join(save_path_sw_nc, f"sw_{safe_pfas}.nc"))
            df_sw_forecast = sw_forecast.to_dataframe().reset_index()
            sw_forecast.close()
            df_sw_forecast = df_sw_forecast.rename(columns={'mean': 'sw_value'})
            df_forecast_data = df_input.copy()
            df_forecast_data['posname'] = pfas
            df_forecast_data = mf.append_inf(df_forecast_data, dict_inf_po, treat_value)
            df_forecast_data = pd.merge(
                df_forecast_data,
                df_sw_forecast[['year', 'lat', 'lon', 'sw_value']],
                left_on=['year', 'lat_grid', 'lon_grid'],
                right_on=['year', 'lat', 'lon'],
                how='left'
            )
            # Box-Cox转换水体值
            original_year = df_forecast_data['year'].copy()
            df_forecast_data['year'] = (df_forecast_data['year'] - 2000) / (2020 - 2000)
            df_forecast_data['sw_value'] = boxcox(df_forecast_data['sw_value'], lmbda=lam_sw)
            if isinstance(treat_value, (list, tuple)) and len(treat_value) > 1:
                if treat_value[1] == 'minmax':
                    # 最小-最大归一化
                    df_forecast_data['sw_value'] = (df_forecast_data['sw_value'] - a) / (b - a)
                elif treat_value[1] == 'zscore':
                    # Z-Score 标准化
                    df_forecast_data['sw_value'] = (df_forecast_data['sw_value'] - a) / b

            sp_data = df_sp_cluster_treat[df_sp_cluster_treat['index'] == j][['sp_length', 'sp_weight', 'sp_troph']].iloc[0]
            df_forecast_data['sp_length'] = sp_data['sp_length']
            df_forecast_data['sp_weight'] = sp_data['sp_weight']
            df_forecast_data['sp_troph'] = sp_data['sp_troph']

            df_forecast_data['organ_muscle'] = 1
            df_forecast_data['organ_liver'] = 0
 
            X_forecast = df_forecast_data[selected_features]

            y_pred = model.predict(X_forecast)
            df_forecast_data['value'] = y_pred

            df_forecast_data['lr_value'] = custom_inv(df_forecast_data['value'], list_inv_parm, treat_value)
            df_forecast_data['year'] = original_year
 
            df_pfas = df_forecast_data[['lon_grid', 'lat_grid', 'year', 'lr_value']].copy()
            df_pfas = df_pfas.rename(columns={'lon_grid': 'lon', 'lat_grid': 'lat'})
            ds_pfas = df_pfas.set_index(['year', 'lat', 'lon']).to_xarray()
            all_pfas_data[safe_pfas] = ds_pfas['lr_value']
        ds_combined = xr.Dataset(all_pfas_data) 
        ds_combined['value'] = sum([ds_combined[v] for v in list_all_pfas])
        ds_combined['lc_value'] = sum([ds_combined[v] for v in list_lc_pfas])
        ds_combined['sc_value'] = sum([ds_combined[v] for v in list_sc_pfas])
        os.makedirs(save_path_lr_nc, exist_ok=True)
        save_path = os.path.join(save_path_lr_nc, f"lr_forecast_{seed}_{j}.nc")
        ds_combined.to_netcdf(save_path)
        ds_combined.close()
        print(f"Saved: {save_path}")


def merge_and_create_seed(raw_dir: str, seed_dir: str, 
                          list_pfas_all, list_pfas_all_lc, list_pfas_all_sc,
                          prefix_parts: int = 3) -> None:
    """
    直接从原始文件合并并生成seed文件，跳过中间步骤
    """
    os.makedirs(seed_dir, exist_ok=True)

    nc_files = [f for f in os.listdir(raw_dir) if f.endswith(".nc")]
    if not nc_files:
        print("未找到任何 nc 文件！")
        return

    # 按前缀分组
    groups = {}
    for fname in nc_files:
        prefix = "_".join(fname.split("_")[:prefix_parts])
        groups.setdefault(prefix, []).append(fname)

    # 遍历每组：合并 + 计算 + 保存
    for prefix, files in groups.items():
        try:
            datasets = []
            for f in files:
                path = os.path.join(raw_dir, f)
                ds = xr.open_dataset(path)
                datasets.append(ds)
            
            combined = xr.concat(datasets, dim="combine_dim")
            mean_ds = combined.mean(dim="combine_dim", keep_attrs=True)

            vars_all = [v for v in list_pfas_all if v in mean_ds.data_vars]
            vars_lc  = [v for v in list_pfas_all_lc if v in mean_ds.data_vars]
            vars_sc  = [v for v in list_pfas_all_sc if v in mean_ds.data_vars]

            mean_ds['value'] = sum([mean_ds[v] for v in vars_all])
            mean_ds['lc_value'] = sum([mean_ds[v] for v in vars_lc])
            mean_ds['sc_value'] = sum([mean_ds[v] for v in vars_sc])

            out_path = os.path.join(seed_dir, f"{prefix}.nc")
            mean_ds.to_netcdf(out_path)
            print(f"[seed] {prefix} -> {out_path}")
            
            mean_ds.close()
            for ds in datasets:
                ds.close()
                
        except Exception as e:
            print(f"[ERROR] 处理分组 {prefix} 出错：{e}")

    print("全部处理完成！")



def from_seed_get_pfas_dask(
    dir_seed, dir_pfas, str_describe,
    chunk_seed=1, chunk_lat=60, chunk_lon=120,
    approx=False
):
    """
    使用 dask 懒加载并计算 NetCDF 多文件统计量。
    仅对 'value', 'lc_value', 'sc_value':
        - 计算 5%, 50%, 95% 分位数
        - 计算基于中位数的 CV_median = MAD / median
    
    参数：
    approx : True 使用近似分位数（快），False 精确分位数（慢）
    """
    os.makedirs(dir_pfas, exist_ok=True)

    seed_files = sorted(
        [os.path.join(dir_seed, f) for f in os.listdir(dir_seed) if f.endswith(".nc")]
    )

    sample_ds = xr.open_dataset(seed_files[0])
    coord_names = set(sample_ds.dims.keys()) | set(sample_ds.coords.keys())
    var_names = [v for v in sample_ds.data_vars if v not in coord_names]
    var_names_sorted = sorted(var_names, key=lambda x: (
        0 if x in ['value', 'lc_value', 'sc_value'] else 1,
        x
    ))
    sample_ds.close()

    ds_all = xr.open_mfdataset(
        seed_files,
        combine='nested',
        concat_dim='seed',
        chunks={'seed': chunk_seed, 'lat': chunk_lat, 'lon': chunk_lon}
    )

    for var in var_names_sorted:
        if var not in ds_all:
            continue

        da_var = ds_all[var]  # Dask-backed DataArray

        stat_min = da_var.min(dim='seed')
        stat_max = da_var.max(dim='seed')
        stat_mean = da_var.mean(dim='seed')
        stat_std = da_var.std(dim='seed')
        stat_cv = stat_std / stat_mean

        stat_dict = {
            'min': stat_min,
            'max': stat_max,
            'mean': stat_mean,
            'std': stat_std,
            'cv': stat_cv
        }

        if var in ['value', 'lc_value', 'sc_value']:
            if approx:
                # 用近似分位数
                q_values_perc = [5, 50, 95]
                seed_axis = da_var.get_axis_num("seed")
                qs = da.percentile(da_var.data, q_values_perc, axis=seed_axis)

                q05 = xr.DataArray(qs[0], dims=[d for d in da_var.dims if d != "seed"],
                                   coords={dim: da_var.coords[dim] for dim in da_var.dims if dim != "seed"})
                median = xr.DataArray(qs[1], dims=[d for d in da_var.dims if d != "seed"],
                                      coords={dim: da_var.coords[dim] for dim in da_var.dims if dim != "seed"})
                q95 = xr.DataArray(qs[2], dims=[d for d in da_var.dims if d != "seed"],
                                   coords={dim: da_var.coords[dim] for dim in da_var.dims if dim != "seed"})

                # MAD 近似版本（基于中位数）
                abs_diff = abs(da_var - median)
                mad_val = xr.DataArray(
                    da.percentile(abs_diff.data, 50, axis=seed_axis),  # 中位数百分位=50
                    dims=[d for d in da_var.dims if d != "seed"],
                    coords={dim: da_var.coords[dim] for dim in da_var.dims if dim != "seed"}
                )
                cv_median = mad_val / median

                stat_dict.update({
                    'q05': q05,
                    'q50': median,
                    'q95': q95,
                    'cv_median': cv_median
                })

            else:
                # 精确分位数
                quantiles = da_var.quantile([0.05, 0.5, 0.95], dim='seed')
                median = quantiles.sel(quantile=0.5)
                # MAD 精确：先求绝对离差，然后求中位数
                abs_diff = abs(da_var - median)
                mad_val = abs_diff.quantile(0.5, dim='seed')
                cv_median = mad_val / median

                stat_dict.update({
                    'q05': quantiles.sel(quantile=0.05),
                    'q50': median,
                    'q95': quantiles.sel(quantile=0.95),
                    'cv_median': cv_median
                })

        stat_ds_computed = xr.Dataset(stat_dict).compute()
        out_path = os.path.join(dir_pfas, f"{str_describe}_{var}.nc")
        stat_ds_computed.to_netcdf(out_path)
        print(f"[final] 保存 {out_path}")

    print("所有统计计算完成！")



def transform_geo_data(
    df_geo_data,
    list_remove,
    df_result=None,
    year_column='year',
    train_end_year=2020
):
    """
    对地理数据进行变换处理，支持重用已计算好的变换方法和参数
    
    参数:
    - df_geo_data: 包含2000-2050年数据的DataFrame
    - list_remove: 要排除的列名列表
    - year_column: 年份列名，默认'year'
    - train_end_year: 训练数据截止年份，默认2020
    - df_result: 已保存的变量统计结果，如果提供则跳过偏度峰度计算，直接应用
    
    返回:
    - df_result: 变量统计结果DataFrame
    - df_transformed: 变换后的完整数据DataFrame
    """

    df_transformed = df_geo_data.copy()

    if df_result is None:
        df_result_list = []
        train_mask = df_geo_data[year_column] <= train_end_year
        df_train = df_geo_data[train_mask].copy()

        cols_to_process = [
            c for c in df_geo_data.select_dtypes(include=[np.number]).columns
            if c not in list_remove and c != year_column
        ]

        transform_params = {}

        # ==== 分析最佳方法 ====
        for col in cols_to_process:
            series_train = df_train[col].dropna()
            if series_train.empty or len(series_train) < 3:
                continue

            min_val = series_train.min()
            max_val = series_train.max()
            skew_orig = skew(series_train)
            kurt_orig = kurtosis(series_train)

            transform_results = {
                "orig": {"skew": skew_orig, "kurt": kurt_orig, "params": None},
                "yj": {"skew": np.nan, "kurt": np.nan, "params": None},
                "bc_or_log": {"skew": np.nan, "kurt": np.nan, "params": None},
                "log": {"skew": np.nan, "kurt": np.nan, "params": None},
                "log1p": {"skew": np.nan, "kurt": np.nan, "params": None}
            }

            # Yeo-Johnson
            if min_val <= 0:
                try:
                    yj_transformed, yj_lambda = yeojohnson(series_train)
                    transform_results["yj"]["skew"] = skew(yj_transformed)
                    transform_results["yj"]["kurt"] = kurtosis(yj_transformed)
                    transform_results["yj"]["params"] = {"lambda": yj_lambda}
                except Exception as e:
                    print(f"Yeo-Johnson failed for {col}: {e}")

            # Box-Cox 或 log
            if min_val > 0:
                try:
                    bc_transformed, bc_lambda = boxcox(series_train)
                    transform_results["bc_or_log"]["skew"] = skew(bc_transformed)
                    transform_results["bc_or_log"]["kurt"] = kurtosis(bc_transformed)
                    transform_results["bc_or_log"]["params"] = {"method": "boxcox", "lambda": bc_lambda}
                except Exception:
                    try:
                        log_transformed = np.log(series_train)
                        transform_results["bc_or_log"]["skew"] = skew(log_transformed)
                        transform_results["bc_or_log"]["kurt"] = kurtosis(log_transformed)
                        transform_results["bc_or_log"]["params"] = {"method": "log"}
                    except Exception as e:
                        print(f"Log (bc fallback) failed for {col}: {e}")

                try:
                    log_transformed = np.log(series_train)
                    transform_results["log"]["skew"] = skew(log_transformed)
                    transform_results["log"]["kurt"] = kurtosis(log_transformed)
                    transform_results["log"]["params"] = {}
                except Exception as e:
                    print(f"Log failed for {col}: {e}")

            if min_val == 0:
                try:
                    log1p_transformed = np.log1p(series_train)
                    transform_results["log1p"]["skew"] = skew(log1p_transformed)
                    transform_results["log1p"]["kurt"] = kurtosis(log1p_transformed)
                    transform_results["log1p"]["params"] = {}
                except Exception as e:
                    print(f"log1p failed for {col}: {e}")

            # 选择最佳方法
            skew_dict = {k: abs(v["skew"]) for k, v in transform_results.items() if not np.isnan(v["skew"])}
            kurt_dict = {k: abs(v["kurt"]) for k, v in transform_results.items() if not np.isnan(v["kurt"])}

            best_skew_method = min(skew_dict, key=skew_dict.get) if skew_dict else "orig"
            best_kurt_method = min(kurt_dict, key=kurt_dict.get) if kurt_dict else "orig"

            transform_params[col] = {
                "method": best_skew_method,
                "params": transform_results[best_skew_method]["params"]
            }

            df_result_list.append({
                "variable": col,
                "min": min_val,
                "max": max_val,
                "skew_orig": skew_orig,
                "kurt_orig": kurt_orig,
                "yj_skew": transform_results["yj"]["skew"],
                "yj_kurt": transform_results["yj"]["kurt"],
                "bc_or_log_skew": transform_results["bc_or_log"]["skew"],
                "bc_or_log_kurt": transform_results["bc_or_log"]["kurt"],
                "log_skew": transform_results["log"]["skew"],
                "log_kurt": transform_results["log"]["kurt"],
                "log1p_skew": transform_results["log1p"]["skew"],
                "log1p_kurt": transform_results["log1p"]["kurt"],
                "best_skew_method": best_skew_method,
                "best_skew_params": transform_results[best_skew_method]["params"],  # 保存参数
                "best_skew_value": transform_results[best_skew_method]["skew"],
                "best_kurt_method": best_kurt_method,
                "best_kurt_value": transform_results[best_kurt_method]["kurt"]
            })

        df_result = pd.DataFrame(df_result_list)

    else:
        transform_params = {
            row['variable']: {
                "method": row['best_skew_method'],
                "params": row['best_skew_params'] if 'best_skew_params' in row else None
            }
            for _, row in df_result.iterrows()
        }

    # ==== 应用最佳变换方法 ====
    for col, info in transform_params.items():
        method = info["method"]
        params = info["params"]
        series_full = df_transformed[col].copy()

        try:
            if method == "orig":
                pass
            elif method == "yj":
                df_transformed[col] = yeojohnson(series_full, lmbda=params["lambda"])
            elif method == "bc_or_log":
                if params["method"] == "boxcox":
                    df_transformed[col] = boxcox(series_full, lmbda=params["lambda"])
                else:
                    df_transformed[col] = np.log(series_full)
            elif method == "log":
                df_transformed[col] = np.log(series_full)
            elif method == "log1p":
                df_transformed[col] = np.log1p(series_full)
        except Exception as e:
            print(f"Transform failed for {col} with method {method}: {e}")

    return df_result, df_transformed


def transform_geo_data_new(
    df_geo_data,
    list_remove,
    df_result=None,
    year_column='year',
    train_end_year=2020
):
    """
    对地理数据进行变换处理，支持重用已计算好的变换方法和参数
  
    参数:
    - df_geo_data: 包含2000-2050年数据的DataFrame
    - list_remove: 要排除的列名列表
    - year_column: 年份列名，默认'year'
    - train_end_year: 训练数据截止年份，默认2020
    - df_result: 已保存的变量统计结果，如果提供则跳过偏度峰度计算，直接应用
  
    返回:
    - df_result: 变量统计结果DataFrame
    - df_transformed: 变换后的完整数据DataFrame
    """

    df_transformed = df_geo_data.copy()

    if df_result is None:
        df_result_list = []
        train_mask = df_geo_data[year_column] <= train_end_year
        df_train = df_geo_data[train_mask].copy()

        cols_to_process = [
            c for c in df_geo_data.select_dtypes(include=[np.number]).columns
            if c not in list_remove and c != year_column
        ]

        transform_params = {}

        # ==== 分析最佳方法 ====
        for col in cols_to_process:
            series_train = df_train[col].dropna()
            if series_train.empty or len(series_train) < 3:
                continue

            zero_per = (series_train == 0).sum() / len(series_train)
            min_val = series_train.min()
            max_val = series_train.max()
            skew_orig = skew(series_train)
            kurt_orig = kurtosis(series_train)

            transform_results = {
                "orig": {"skew": skew_orig, "kurt": kurt_orig, "params": None},
                "yj": {"skew": np.nan, "kurt": np.nan, "params": None},
                "bc": {"skew": np.nan, "kurt": np.nan, "params": None},     # 修改：仅设为 bc
                "log": {"skew": np.nan, "kurt": np.nan, "params": None},
                "log1p": {"skew": np.nan, "kurt": np.nan, "params": None},
                "log10p": {"skew": np.nan, "kurt": np.nan, "params": None}, 
            }

            if zero_per > 0.1:
                try:
                    log1p_transformed = np.log1p(series_train)
                    transform_results["log1p"]["skew"] = skew(log1p_transformed)
                    transform_results["log1p"]["kurt"] = kurtosis(log1p_transformed)
                except Exception as e:
                    print(f"log1p failed for {col}: {e}")
                
                best_skew_method = "log1p"
                best_kurt_method = "log1p"
                transform_params[col] = {
                    "method": "log1p",
                    "params": None
                }

            elif abs(skew_orig) < 2 and abs(kurt_orig) < 7:
                # 已接近正态分布，不做变换
                best_skew_method = "orig"
                best_kurt_method = "orig"
                transform_params[col] = {
                    "method": "orig",
                    "params": None
                }
            else:
                # 执行其他变换候选测试
                # Yeo-Johnson（支持≤0值）
                if min_val <= 0:
                    try:
                        yj_transformed, yj_lambda = yeojohnson(series_train)
                        transform_results["yj"]["skew"] = skew(yj_transformed)
                        transform_results["yj"]["kurt"] = kurtosis(yj_transformed)
                        transform_results["yj"]["params"] = yj_lambda
                    except Exception as e:
                        print(f"Yeo-Johnson failed for {col}: {e}")

                # Box-Cox 或 log（>0值才允许）
                if min_val > 0:
                    try:
                        bc_transformed, bc_lambda = boxcox(series_train)
                        transform_results["bc"]["skew"] = skew(bc_transformed)
                        transform_results["bc"]["kurt"] = kurtosis(bc_transformed)
                        transform_results["bc"]["params"] = bc_lambda
                    except Exception as e:
                        print(f"Box-Cox failed for {col}: {e}")

                    try:
                        log_transformed = np.log(series_train)
                        transform_results["log"]["skew"] = skew(log_transformed)
                        transform_results["log"]["kurt"] = kurtosis(log_transformed)
                    except Exception as e:
                        print(f"Log failed for {col}: {e}")

                # log1p（允许0值）
                try:
                    log1p_transformed = np.log1p(series_train)
                    transform_results["log1p"]["skew"] = skew(log1p_transformed)
                    transform_results["log1p"]["kurt"] = kurtosis(log1p_transformed)
                except Exception as e:
                    print(f"log1p failed for {col}: {e}")

                # log10p
                try:
                    log10p_transformed = np.log10(series_train + 1)
                    transform_results["log10p"]["skew"] = skew(log10p_transformed)
                    transform_results["log10p"]["kurt"] = kurtosis(log10p_transformed)
                except Exception as e:
                    print(f"log10p failed for {col}: {e}")

                # ====== 选择最佳方法 ======
                skew_dict = {k: abs(v["skew"]) for k, v in transform_results.items() if not np.isnan(v["skew"])}
                kurt_dict = {k: abs(v["kurt"]) for k, v in transform_results.items() if not np.isnan(v["kurt"])}

                best_skew_method = min(skew_dict, key=skew_dict.get) if skew_dict else "orig"
                best_kurt_method = min(kurt_dict, key=kurt_dict.get) if kurt_dict else "orig"

                transform_params[col] = {
                    "method": best_skew_method,
                    "params": transform_results[best_skew_method]["params"]
                }

            # ====== 汇总结果 ======
            df_result_list.append({
                "variable": col,
                "min": min_val,
                "max": max_val,
                "zero_per": zero_per,  
                "skew_orig": skew_orig,
                "kurt_orig": kurt_orig,
                "yj_skew": transform_results["yj"]["skew"],
                "yj_kurt": transform_results["yj"]["kurt"],
                "bc_skew": transform_results["bc"]["skew"],        
                "bc_kurt": transform_results["bc"]["kurt"],       
                "log_skew": transform_results["log"]["skew"],
                "log_kurt": transform_results["log"]["kurt"],
                "log1p_skew": transform_results["log1p"]["skew"],
                "log1p_kurt": transform_results["log1p"]["kurt"],
                "log10p_skew": transform_results["log10p"]["skew"],
                "log10p_kurt": transform_results["log10p"]["kurt"],
                "best_skew_method": best_skew_method,
                "best_skew_params": transform_params[col]["params"],
                "best_skew_value": transform_results.get(best_skew_method, {}).get("skew", np.nan),
                "best_kurt_method": best_kurt_method,
                "best_kurt_value": transform_results.get(best_kurt_method, {}).get("kurt", np.nan)
            })

        df_result = pd.DataFrame(df_result_list)

    else:
        transform_params = {
            row['variable']: {
                "method": row['best_skew_method'],
                "params": row.get('best_skew_params', None)
            }
            for _, row in df_result.iterrows()
        }

    # ====  应用最佳变换方法 ====
    for col, info in transform_params.items():
        method = info["method"]
        params = info["params"]
        series_full = df_transformed[col].copy()

        try:
            if method == "orig":
                pass
            elif method == "yj":

                df_transformed[col] = yeojohnson(series_full, lmbda=params)
            elif method == "bc":
                df_transformed[col] = boxcox(series_full, lmbda=params)
            elif method == "log":
                df_transformed[col] = np.log(series_full)
            elif method == "log1p":
                df_transformed[col] = np.log1p(series_full)
            elif method == "log10p":  
                df_transformed[col] = np.log10(series_full + 1)
        except Exception as e:
            print(f"Transform failed for {col} with method {method}: {e}")

    return df_result, df_transformed


def convert_temp(df_o, from_scale, to_scale):

    df = df_o.copy()
    path_file = r'C:/Users/dell/OneDrive/file/'
    meta_file = 'meta_data.csv'
    df_meta = pd.read_csv(path_file + meta_file, encoding="utf-8")
    list_all_var = df_meta['var_name'][(df_meta['var_select_all']==1)&(df_meta['var_temp']==1)].to_list()
    for col in list_all_var:
        if from_scale == "C":
            if to_scale == "F":
                df[col] = (df[col] * 9/5) + 32
            elif to_scale == "K":
                df[col] = df[col] + 273.15
        elif from_scale == "F":
            if to_scale == "C":
                df[col] = (df[col] - 32) * 5/9
            elif to_scale == "K":
                df[col] = ((df[col] - 32) * 5/9) + 273.15
        elif from_scale == "K":
            if to_scale == "C":
                df[col] = df[col] - 273.15
            elif to_scale == "F":
                df[col] = ((df[col] - 273.15) * 9/5) + 32
    return df