
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, triang
import os
import warnings
from tqdm.notebook import tqdm



warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def truncated_normal(mean, cv, lower, upper):
    std = mean * cv
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm(a, b, loc=mean, scale=std)


def generate_consume(consume_series, cv=0.05):
    result = np.zeros(len(consume_series))
    non_zero_mask = consume_series > 0

    if non_zero_mask.any():
        non_zero_values = consume_series[non_zero_mask]
        generated_values = truncated_normal(
            non_zero_values, 
            cv, 
            non_zero_values * 0.8, 
            non_zero_values * 1.2
        ).rvs()
        result[non_zero_mask] = generated_values

    return result


def calculate_thq(tc, fir, ef, ed, rfd, bw, at):
    return (tc * fir * ef * ed) / (rfd * bw * at) 


def run_thq_simulation(source, list_pfas, path_part4_pfas, path_part4_result,
                       df_weight, df_life, df_rfd, convert, n_simulations=1000):
    for pfas in list_pfas:
        print(f'Processing {source} - {pfas}...')
        safe_pfas_name = convert(pfas)
        df_pfas = pd.read_csv(path_part4_pfas + f'{source}_{safe_pfas_name}.csv')
        df_pfas = df_pfas[df_pfas[pfas + '_min'].notna()]
        df_pfas = df_pfas.merge(df_weight, on='country_id', how='left')
        df_pfas = df_pfas.merge(df_life, on=['country_id', 'year'], how='left')
        if source == 'sw':
            df_pfas['sw_consume'] = 2 * 0.2 * 0.5
        rfd_data = df_rfd[df_rfd['PFAS'] == pfas].iloc[0]
        results = df_pfas[['lon', 'lat', 'year']].copy()
        for i in range(n_simulations):


            tc_min = df_pfas[f'{pfas}_min']
            tc_max = df_pfas[f'{pfas}_max']
            tc = np.random.uniform(low=tc_min, high=tc_max) / 1000
            if rfd_data['min'] == rfd_data['max'] == rfd_data['median']:
                rfd = rfd_data['median'] / 1000000
            else:
                min_val = rfd_data['min'] / 1000000
                max_val = rfd_data['max'] / 1000000
                median_val = rfd_data['median'] / 1000000
                c = (median_val - min_val) / (max_val - min_val)
                loc = min_val
                scale = max_val - min_val
                rfd = triang.rvs(c, loc=loc, scale=scale)
            bw = truncated_normal(df_pfas['weight'], 0.05, df_pfas['weight'] * 0.8, df_pfas['weight'] * 1.2).rvs()
            life_ex = truncated_normal(df_pfas['life_ex'], 0.05, df_pfas['life_ex'] * 0.8, df_pfas['life_ex'] * 1.2).rvs()
            ef = 365
            at = 365 * life_ex
            if source == 'lr':
                ff_consume = generate_consume(df_pfas['ff_consume'])
                sf_consume = generate_consume(df_pfas['sf_consume'])
                thq_ff = calculate_thq(tc, ff_consume, ef, life_ex, rfd, bw, at)
                thq_sf = calculate_thq(tc, sf_consume, ef, life_ex, rfd, bw, at)
                thq_fish = thq_ff + thq_sf
                results[f'thq_fish_{i}'] = thq_fish * 0.71
            elif source == 'sw':
                sw_consume = generate_consume(df_pfas['sw_consume'])
                thq_water = calculate_thq(tc, sw_consume, ef, life_ex, rfd, bw, at)
                results[f'thq_water_{i}'] = thq_water
        output_file = path_part4_result + f'{source}_{safe_pfas_name}_thq.csv'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results.to_csv(output_file, index=False)


def merge_prefix_files(prefixes, list_pfas, path_part4_result, path_part4_thq, marker=None):
    for prefix in prefixes:
        merged_df = None
        for middle in list_pfas:
            file_name = f"{prefix}{middle}_thq.csv"
            file_path = os.path.join(path_part4_result, file_name)

            df = pd.read_csv(file_path)


            if merged_df is None:
                merged_df = df
            else:
                merged_df.iloc[:, 3:] += df.iloc[:, 3:]

            del df
        output_file = os.path.join(path_part4_thq, f"{prefix}{marker}merged_thq.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Merged file saved: {output_file}")
    print("All prefix-based files have been processed and merged.")

def combine_lr_sw_results(path_part4_thq, marker=None):
    file_name1 = f'lr_{marker}merged_thq.csv'
    file_name2 = f'sw_{marker}merged_thq.csv'
    df_lr = pd.read_csv(os.path.join(path_part4_thq, file_name1))
    df_sw = pd.read_csv(os.path.join(path_part4_thq, file_name2))
    assert (df_lr[['lon', 'lat', 'year']] == df_sw[['lon', 'lat', 'year']]).all().all(), "lon, lat, year columns do not match"
    df_merged = df_lr[['lon', 'lat', 'year']].copy()
    for i in range(1000):
        df_merged[f'thq_{i}'] = df_lr[f'thq_fish_{i}'] + df_sw[f'thq_water_{i}']
    output_file = os.path.join(path_part4_thq, f"{marker}merged_thq.csv")
    df_merged.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")


def calculate_statistics(df, prefix):
    stats_df = df[['lon', 'lat', 'year']].copy()
    thq_columns = df.columns[3:]

    stats_df[f'{prefix}_min'] = df[thq_columns].min(axis=1)
    stats_df[f'{prefix}_max'] = df[thq_columns].max(axis=1)
    stats_df[f'{prefix}_2.5%'] = df[thq_columns].quantile(0.025, axis=1)
    stats_df[f'{prefix}_50%'] = df[thq_columns].median(axis=1)
    stats_df[f'{prefix}_97.5%'] = df[thq_columns].quantile(0.975, axis=1)
    stats_df[f'{prefix}_mean'] = df[thq_columns].mean(axis=1)
    stats_df[f'{prefix}_SD'] = df[thq_columns].std(axis=1)

    return stats_df






def calculate_statistics2(df, prefix):
    stats_df = df[['lon', 'lat', 'year']].copy()
    thq_columns = df.columns[3:]
    stats_df[f'{prefix}_min'] = df[thq_columns].min(axis=1)
    stats_df[f'{prefix}_max'] = df[thq_columns].max(axis=1)
    stats_df[f'{prefix}_2.5%'] = df[thq_columns].quantile(0.025, axis=1)
    stats_df[f'{prefix}_5%'] = df[thq_columns].quantile(0.05, axis=1)
    stats_df[f'{prefix}_10%'] = df[thq_columns].quantile(0.1, axis=1)
    stats_df[f'{prefix}_20%'] = df[thq_columns].quantile(0.2, axis=1)
    stats_df[f'{prefix}_50%'] = df[thq_columns].median(axis=1)
    stats_df[f'{prefix}_80%'] = df[thq_columns].quantile(0.8, axis=1)
    stats_df[f'{prefix}_90%'] = df[thq_columns].quantile(0.9, axis=1)
    stats_df[f'{prefix}_95%'] = df[thq_columns].quantile(0.95, axis=1)
    stats_df[f'{prefix}_97.5%'] = df[thq_columns].quantile(0.975, axis=1)
    stats_df[f'{prefix}_mean'] = df[thq_columns].mean(axis=1)
    stats_df[f'{prefix}_SD'] = df[thq_columns].std(axis=1)
    return stats_df

def process_impact_population(input_df, prefix_length, pop_wiw_dict):
    """
    处理影响人口的通用函数

    Parameters:
    input_df: 输入的DataFrame
    prefix_length: 列名前缀的长度
    pop_wiw_dict: 包含人口和wiw数据的字典

    Returns:
    DataFrame: 处理后的结果
    """

    thq_columns = input_df.columns[3:]


    new_columns = {}


    idx = list(zip(input_df['lon'], input_df['lat'], input_df['year']))

    for col in tqdm(thq_columns, desc="Processing columns"):
        new_col_name = f'imp_pop_{col[prefix_length:]}'
        mask = input_df[col] >= 1


        values = np.where(mask, 
                         [pop_wiw_dict.get(i, 0) for i in idx],
                         0)
        new_columns[new_col_name] = values


    result_df = pd.DataFrame(new_columns)


    result_df = pd.concat([input_df[['lon', 'lat', 'year']], result_df], axis=1)

    return result_df


def merge_impact_populations(base_df, *other_dfs):
    """
    合并多个影响人口数据框

    Parameters:
    base_df: 基础DataFrame
    *other_dfs: 其他要合并的DataFrames

    Returns:
    DataFrame: 合并后的结果
    """

    result = base_df.copy()


    base_cols = ['lon', 'lat', 'year']


    initial_zeros = (result.iloc[:, 3:] == 0).sum().sum()
    print(f"初始0值数量: {initial_zeros}")


    for idx, df in enumerate(other_dfs, 1):

        cols_to_process = [col for col in df.columns if col not in base_cols]

        for col in cols_to_process:
            if col in result.columns:

                if idx == 1:
                    result[col] = np.maximum(result[col], df[col])
                else:

                    result[col] = np.where(result[col] == 0, df[col], result[col])
            else:

                result[col] = df[col]


        current_zeros = (result.iloc[:, 3:] == 0).sum().sum()
        print(f"合并DataFrame {idx}后的0值数量: {current_zeros}")

    return result

def process_and_save_statistics(df, output_path, prefix):
    """
    处理统计数据并保存

    Parameters:
    df: 要处理的DataFrame
    output_path: 输出路径
    prefix: 文件名前缀
    """

    df.to_csv(os.path.join(output_path, f'{prefix}_imp_pop.csv'), index=False)


    stats = calculate_statistics(df, 'imp_pop')
    stats.to_csv(os.path.join(output_path, f'{prefix}_pop_statistics.csv'), index=False)

