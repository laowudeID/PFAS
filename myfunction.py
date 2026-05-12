
import pandas as pd
import numpy as np
from scipy import stats
import wquantiles
import math
# % matplotlib inline
# DATE:2025-05-27 17:20  

path_input = "C:/Users/dell/OneDrive/file/"
path_one_spdb = 'C:/Users/dell/OneDrive/file/SPDB/'
lr_name = "lr_use.csv"
inf_name = "inf.xlsx"
meta_name = "meta_data.csv"



def get_weight_count(df_o, list_col, methods="median"):
    df = df_o.copy()
    """str_merge_colname = ("~".join(map(str, list_merge_colname))),method:mean|median"""
    if len(list_col) == 1:
        str_merge_colname = list_col[0]
    if len(list_col) > 1:
        str_merge_colname = ("~".join(map(str, list_col)))

    df = merge_col(df, list_col)
    df_gb = df.groupby(str_merge_colname, as_index=False).agg(n=('n', 'sum'))
    list_index = list(df_gb[str_merge_colname].unique())
    print(len(list_index))
    if methods == "median":
        dic_med_v = {}
        dic_mad_v = {}
        dic_num_v = {}
        for i in list_index:
            df_count = df.loc[df[str_merge_colname] == i].copy()
            med_v = wquantiles.median(df_count["value"], df_count["n"])
            dic_med_v[i] = med_v
            # 加权中位数绝对偏差
            df_count.loc[:, "gap"] = df_count["value"] - med_v
            df_count.loc[:, "gap"] = df_count["gap"].abs()
            mad_v = wquantiles.median(df_count["gap"], df_count["n"])
            dic_mad_v[i] = mad_v
            # 实际使用数据
            num_v = len(df_count[["value"]])
            dic_num_v[i] = num_v

        df_gb["value"] = df_gb[str_merge_colname].map(dic_med_v)
        df_gb["MAD"] = df_gb[str_merge_colname].map(dic_mad_v)
        df_gb["a_num"] = df_gb[str_merge_colname].map(dic_num_v)
        df_gb = spilt_col(df_gb, str_merge_colname)
        return df_gb
    if methods == "mean":
        dic_avg_v = {}
        dic_sd_v = {}
        dic_num_v = {}
        for i in list_index:
            # 加权平均值
            df_count = df[df[str_merge_colname] == i].copy()
            avg_v = np.average(df_count["value"], weights=df_count["n"])
            dic_avg_v[i] = avg_v
            # 加权标准差
            df_count.loc[:, "gap"] = (df_count["value"] - avg_v)**2
            sd_v = math.sqrt(np.average(df_count["gap"], weights=df_count["n"]))
            dic_sd_v[i] = sd_v
            # 实际使用数据
            num_v = len(df_count[["value"]])
            dic_num_v[i] = num_v

        df_gb["value"] = df_gb[str_merge_colname].map(dic_avg_v)
        df_gb["SD"] = df_gb[str_merge_colname].map(dic_sd_v)
        df_gb["a_num"] = df_gb[str_merge_colname].map(dic_num_v)
        df_gb = spilt_col(df_gb, str_merge_colname)
        return df_gb

def df_get_ncol(df_provide, df_main, list_col):
    """list_col:{0:prov_colname, 1:connect_colname, 2:new_colname, 3:prov__val[prov_colname:prov__val]"""
    int_long_list = len(list_col)
    if int_long_list == 2:
        dic_povide = df_provide.set_index(list_col[1])[list_col[0]].to_dict()
        df_main[list_col[0]] = df_main[list_col[1]].map(dic_povide)
        return df_main
    elif int_long_list == 3:
        dic_povide = df_provide.set_index(list_col[1])[list_col[0]].to_dict()
        df_main[list_col[2]] = df_main[list_col[1]].map(dic_povide)
        return df_main
    elif int_long_list == 4:
        dic_povide = df_provide.set_index(list_col[0])[list_col[3]].to_dict()
        df_main[list_col[2]] = df_main[list_col[1]].map(dic_povide)
        return df_main
    else:
        print("error")

def df_merge_lat_lon(df_o):
    df = df_o.copy()
    df[["lon", "lat"]] = df[["lon", "lat"]].astype(float)
    df[["lon", "lat"]] = df[["lon", "lat"]].apply(lambda x:round(x, 6))
    df[["lon", "lat"]] = df[["lon", "lat"]].astype(str)
    df["lat_lon"] = df[["lat", "lon"]].apply("_".join, axis=1) # 符号不能修改不然出错，历史遗留问题
    return df

def df_get_more_ncol(df_pro, df_main, list_all, con_col):
    for col in list_all:
        list_col = [col, con_col]
        df_main = df_get_ncol(df_pro, df_main, list_col)
    return df_main

def merge_col(df_input, list_merge_colname, merge_symbol="~"):
    """df:请选择好需要的内容再输入, list_merge_colname: 需要合并的列名， merge_symbol：期望合并的符号，默认~，因为_有时候会用来命名
    str_merge_colname = (merge_symbol.join(map(str, list_merge_colname)))"""
    df = df_input.copy()
    if len(list_merge_colname) > 1:
        list_df_colname = list(df.columns)
        list_nor_merge_colname = list(set(list_df_colname) - set(list_merge_colname))
        print(list_nor_merge_colname)
        str_merge_colname = (merge_symbol.join(map(str, list_merge_colname)))
        print(str_merge_colname)
        df[list_merge_colname] = df[list_merge_colname].astype(str)
        # df[str_merge_colname] = df[list_merge_colname].apply(merge_symbol.join, axis=1)
        df[str_merge_colname] = df[list_merge_colname].apply(lambda x: merge_symbol.join(x), axis=1)
        df.insert(0, str_merge_colname, df.pop(str_merge_colname))
        list_all = [str_merge_colname]
        list_all.extend(list_nor_merge_colname)
        print(list_all)
        df = df[list_all]
        return df
    elif len(list_merge_colname) == 1:
        return df_input

def spilt_col(df, str_merge_colname, spilt_symbol="~", is_index=None):
    """str_merge_colname：要分列的那一列名称（str_merge_colname = (merge_symbol.join(map(str, list_merge_colname)))）
    ,spilt_symbol：分列的符号（默认~）,is_index：这列是否是索引（groupby若没指定as_index=F，则会为index，很多时候为了对齐会不指定））"""
    if is_index is None:
        df.insert(0, str_merge_colname, df.pop(str_merge_colname))
        df_split = df[str_merge_colname].str.split(spilt_symbol, expand=True)
        list_colname = str_merge_colname.split(spilt_symbol)
        df_split.columns = list_colname
        df_new_df = df_split.join(df)
        df_new_df = df_new_df.drop(str_merge_colname, axis=1)
        return df_new_df
    else:
        df = df.rename_axis(str_merge_colname).reset_index()
        df.insert(0, str_merge_colname, df.pop(str_merge_colname))
        df_split = df[str_merge_colname].str.split(spilt_symbol, expand=True)
        list_colname = str_merge_colname.split(spilt_symbol)
        df_split.columns = list_colname
        df_new_df = df_split.join(df)
        df_new_df = df_new_df.drop(str_merge_colname, axis=1)
        return df_new_df

def two_to_one(df_input, list_merge, new_colname="po_name",new_valname="value"):
    """二维数据转一维 list_merge:其他的数据,new_colname = 其他单独成列的列的列名(自己命名,eg:poname),new_valname:这些列的值也要单独成列的列名(自己命名)"""
    df = df_input.copy()
    if len(list_merge)==1:
        str_merge_colname = list_merge[0]
        df_gb_one =  df.melt(id_vars = str_merge_colname,var_name = new_colname, value_name = new_valname)
        return df_gb_one
    elif len(list_merge)>1:
        df_gb = merge_col(df, list_merge) 
        str_merge_colname = ("~".join(map(str, list_merge)))
        df_gb_one =  df_gb.melt(id_vars = str_merge_colname,var_name = new_colname, value_name = new_valname)
        df_gb_one = spilt_col(df_gb_one, str_merge_colname)
        # df_gb_one = df_gb_one.drop(labels=[str_merge_colname],axis=1)
        return df_gb_one

def one_to_two(df, list_merge, new_colname):
    """一维转二维 new_colname应为原sheet中的一个列名"""
    if len(list_merge)==1:
        df_gb = df.copy()
        str_merge_colname = list_merge[0]
        df_gb = df_gb.set_index([str_merge_colname, new_colname])
        df_gb_two = df_gb.unstack()
        # print(df_gb_two.head())
        df_gb_two.columns = df_gb_two.columns.droplevel(0)
        df_gb_two.columns.name = None
        # print(df_gb_two.head())
        df_gb_two[df_gb_two.index.name] = df_gb_two.index
        # df_gb_two = df_gb_two.reset_index()
        return df_gb_two
    elif len(list_merge)>1:
        df_gb = merge_col(df, list_merge)
        str_merge_colname = ("~".join(map(str, list_merge)))
        df_gb = df_gb.set_index([str_merge_colname, new_colname])
        df_gb_two = df_gb.unstack()
        df_gb_two.reset_index()
        # print(df_gb_two.head())
        df_gb_two.columns = df_gb_two.columns.droplevel(0)
        df_gb_two.columns.name = None
        # print(df_gb_two.head())
        df_gb_two[df_gb_two.index.name] = df_gb_two.index
        
        df_gb_two = spilt_col(df_gb_two, str_merge_colname)
        # print(df_gb_two.head())
        df_gb_two = df_gb_two.reset_index()
        return df_gb_two

def col_value(df, col_name):
    col_value = pd.DataFrame({col_name:df[col_name].value_counts().index, "count":df[col_name].value_counts().values})
    col_value["per"] = col_value["count"]/len(df)
    return col_value

def col_describe2(df_main, col_name, col_value):
    """col_name是分类的，col_value是数值，会展示每个分类的数值分位数"""
    df_col_describe = pd.DataFrame()
    for i in list(df_main[col_name].unique()):
        df =  df_main[[col_name,col_value]][df_main[col_name]==i]
        s_df = df[col_value].describe(percentiles=[.05,.25,.5,.75,.95])
        s_df.name = i
        df_col_describe = pd.concat([df_col_describe, s_df.to_frame().T])
        df_col_describe = df_col_describe.sort_values(by="count",ascending=False)
    df_col_describe.index.name = col_name
    df_col_describe = df_col_describe.reset_index()
    return df_col_describe


def col_describe(df_main, col_name):
    """对某一列各元素的描述，自适应value"""
    df_col_describe = pd.DataFrame()
    for i in list(df_main[col_name].unique()):
        df =  df_main[[col_name,"value"]][df_main[col_name]==i]
        s_df = df["value"].describe(percentiles=[.05,.25,.5,.75,.95])
        s_df.name = i
        # df_col_describe = df_col_describe.append(s_df)
        df_col_describe = pd.concat([df_col_describe, s_df.to_frame().T])
    df_col_describe = df_col_describe.sort_values(by="count",ascending=False)
    df_col_describe.index.name = col_name
    df_col_describe = df_col_describe.reset_index()
    return df_col_describe





def id_to_str(df, col_name, sp_name='canonicalName'):
    """col_name:需要一个list, sp_name只需要在spid转换时使用"""
    df_o = df.copy()
    for col in col_name:
        if col in ['genus','family', 'habitat', 'organ', 'class', 
                   'order', 'po_classification','posname']:
            df_col = pd.read_csv(path_one_spdb + col + '.csv')
            df_col = df_col.drop(columns=["50%"])
            
            df_o[col] = df_o[col].astype(str)
            df_col['id'] = df_col['id'].astype(str)

            id_map = dict(zip(df_col['id'], df_col[col]))
            df_o[col] = df_o[col].map(id_map)
        if col in ['spid']:
            df_col = pd.read_excel(path_input + inf_name, sheet_name="sp_pfas")
            df_col = df_col[df_col[sp_name].notna()]
            df_col = df_col[['spid', sp_name]]

            df_o[col] = df_o[col].astype(str)
            df_col['id'] = df_col['spid'].astype(str)
            
            id_map = dict(zip(df_col['id'], df_col[sp_name]))
            df_o[col] = df_o[col].map(id_map)
    return df_o


def preprocess(df_data, treat_method, list_var=None):
    """
    数据预处理函数，支持多种数据转换方法
    
    参数:
    df_data (DataFrame): 待处理的数据框
    treat_method (str): 处理方法，包括：
        - normalization: 最大值最小值归一化
        - standardization: Z-score标准化
        - box_cox: Box-Cox变换（要求数据为正）
        - log: 对数变换（要求数据为正）
        - yeo_johnson: Yeo-Johnson变换（可处理任意实数）
        - auto: 根据元数据文件自动选择变换方法
        - 其他: 自动检测，正态分布的跳过，非正态的根据数据特点选择变换方法
    list_var (list, optional): 需要处理的变量列表，默认为None（处理所有列）
    path_data_raw (str, optional): 元数据文件路径，仅在auto模式下需要
    meta_name (str, optional): 元数据文件名，仅在auto模式下需要
    
    返回:
    DataFrame: 处理后的数据框
    """
    
    df_data = df_data.copy()
    
    if list_var is None:
        list_var = df_data.columns.tolist()
    
    # 确保list_var中的所有列都存在于df_data中
    list_var = [var for var in list_var if var in df_data.columns]
    
    if treat_method == "normalization" or treat_method == "minmax":
        # 最大值最小值归一化
        for col in list_var:
            min_val = df_data[col].min()
            max_val = df_data[col].max()
            if max_val > min_val:  # 避免除以零
                df_data[col] = (df_data[col] - min_val) / (max_val - min_val)
            else:
                print(f"警告: 列 {col} 的最大值等于最小值，无法进行归一化")
                
    elif treat_method == "standardization" or treat_method == "zscore":
        # z-score标准化
        for col in list_var:
            std_val = df_data[col].std()
            if std_val > 0:  # 避免除以零
                df_data[col] = (df_data[col] - df_data[col].mean()) / std_val
            else:
                print(f"警告: 列 {col} 的标准差为0，无法进行标准化")
                
    elif treat_method == "box_cox":
        # Box-Cox变换（要求数据为正）
        list_var = [item for item in list_var if 'log_' not in item]
        numeric_cols = df_data[list_var].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df_data[col] <= 0).any():
                print(f"警告: 列 {col} 包含非正值，无法应用Box-Cox变换")
                continue
            try:
                df_data[col], lambda_param = stats.boxcox(df_data[col])
                print(f"列 {col} 的Box-Cox变换参数 λ: {lambda_param}")
            except Exception as e:
                print(f"处理列 {col} 时出错: {str(e)}")
                
    elif treat_method == "log":
        # 对数变换（要求数据为正）
        list_var = [item for item in list_var if 'log_' not in item]
        for col in list_var:
            if (df_data[col] <= 0).any():
                print(f"警告: 列 {col} 包含非正值，无法应用对数变换")
                continue
            df_data[col] = np.log10(df_data[col])
            
    elif treat_method == "yeo_johnson":
        # Yeo-Johnson变换（可处理任意实数）
        numeric_cols = df_data[list_var].select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                df_data[col], _ = stats.yeojohnson(df_data[col])
            except Exception as e:
                print(f"处理列 {col} 时出错: {str(e)}")
                
    elif treat_method == "auto":
        # 自动根据元数据选择变换方法
        df_meta = pd.read_csv(path_input + meta_name, encoding="utf-8")
        list_yj_meta = df_meta['var_name'][df_meta['var_type0']==1].tolist()
        list_bc_meta = df_meta['var_name'][df_meta['var_type0']==0].tolist()
        
        # 排除变量名中包含"log_"的变量
        list_var = [item for item in list_var if 'log_' not in item]
        
        # 取交集
        list_yj = list(set(list_yj_meta) & set(list_var))
        list_bc = list(set(list_bc_meta) & set(list_var))
        
        print('YJ变换列:', list_yj)
        print('BC变换列:', list_bc)

        for col in list_yj:
            try:
                df_data[col], lambda_param2 = stats.yeojohnson(df_data[col])
                print(f"列 {col} 的Yeo-Johnson变换参数 λ2: {lambda_param2}")
            except Exception as e:
                print(f"处理列 {col} 时出错: {str(e)}")
                
        for col in list_bc:
            if (df_data[col] <= 0).any():
                print(f"警告: 列 {col} 包含非正值，跳过Box-Cox变换")
                continue
            try:
                df_data[col], _ = stats.boxcox(df_data[col])
            except Exception as e:
                print(f"处理列 {col} 时出错: {str(e)}")
                
    else:
        # 根据数据特性自动选择变换方法
        for var in list_var:
            try:
                # 检查是否为正态分布
                p_val = stats.normaltest(df_data[var]).pvalue
                
                if p_val > 0.05:
                    print('正态分布:', var)
                    continue
                else:
                    # 非正态分布，检查是否有负值或零
                    if (df_data[var] < 0).any():
                        # 有负值，使用Yeo-Johnson变换
                        print('非正态(含负值)，应用YJ变换:', var)
                        df_data[var], _ = stats.yeojohnson(df_data[var])
                    else:
                        # 无负值，使用Box-Cox变换
                        print('非正态(全正值)，应用BC变换:', var)
                        # 检查是否有零值
                        if (df_data[var] == 0).any():
                            print(f"警告: 列 {var} 包含零值，无法直接应用Box-Cox变换，先加上微小值")
                            df_data[var] = df_data[var] + 1e-10
                        df_data[var], lambda_param3 = stats.boxcox(df_data[var])
                        print(f"列 {var} 的Box-Cox变换参数 λ3: {lambda_param3}")
            except Exception as e:
                print(f"处理列 {var} 时出错: {str(e)}")
                
    return df_data


def select_data(df_data_raw, path_inf, mark_num):
    df_meta = pd.read_csv(path_inf + 'meta_data.csv', encoding="utf-8")
    list_var_select = df_meta["var_name"][df_meta["var_select_" + mark_num]==1].to_list()
    list_fea = df_data_raw.columns.tolist()
    list_select = list(set(list_fea)&set(list_var_select))
    return df_data_raw[list_select]


def append_inf(df, dict_inf, treat_value=False):
    """
    根据传入的 dict_inf，从对应的 Excel sheet 读取信息并补充到 df 中。
    
    df: 原 DataFrame
    dict_inf: {字段key: 需要匹配的值列表}
    treat_value: False 或 [处理方法1, 处理方法2]
    """

    list_po_auto = [
        "po_m_w", "log_Px", "log_Koc", "log_Kow", "log_Kaw",
        "log_Koa", "log_Koa_wet", "log_KHxd_air", "log_Koil_w", "log_Koil_air", "density",
        "solubility", "log_pKa", "log_D5_5", "log_D7_4"
    ]
    list_sp_auto = ["sp_length", "sp_weight", "sp_troph"]

    sheet_map = {
        "spid":   ("sp_pfas", "spid", list_sp_auto),
        "country_id": ("country", "country_id", None),
        "poid":   ("po_pfas", "poid", list_po_auto),
        "posid":  ("po_pfas", "posid", list_po_auto),
        "posname":("po_pfas", "posname", list_po_auto),
        "paid":   ("pa_pfas", "paid", list_po_auto),
    }

    def merge_info(df_base, key, values, preprocess_methods=None):
        """从对应 sheet 读取数据，并合并到 df_base"""
        sheet, merge_key, auto_cols = sheet_map[key]
        df_info = pd.read_excel(path_input + inf_name, sheet_name=sheet)
        if preprocess_methods and auto_cols:
            df_info = preprocess(df_info, preprocess_methods[0], auto_cols)
            if len(preprocess_methods) > 1:
                df_info = preprocess(df_info, preprocess_methods[1], auto_cols)
        
        return df_get_more_ncol(df_info, df_base, values, merge_key)

    # 主逻辑
    if not treat_value:
        for key, values in dict_inf.items():
            if key in sheet_map:
                df = merge_info(df, key, values)
    else:
        for key, values in dict_inf.items():
            if key in sheet_map:
                df = merge_info(df, key, values, preprocess_methods=treat_value)

    return df




# ！def：从INF添加


# def append_inf(df, dict_inf):
#     """df:, dict_inf:"""
#     if "habit" in dict_inf:
#         list_habit = dict_inf["habit"]
#         df_habit = pd.read_excel(path_input + inf_name, sheet_name="habit")
#         df = df_get_more_ncol(df_habit, df, list_habit, "habit")
#     if "sp" in dict_inf:
#         list_sp = dict_inf["sp"]
#         df_sp = pd.read_excel(path_input + inf_name, sheet_name="sp_pfas")
#         df = df_get_more_ncol(df_sp, df, list_sp, "spid")
#     if "country" in dict_inf:
#         list_country = dict_inf["country"]
#         df_country = pd.read_excel(path_input + inf_name, sheet_name="country")
#         df = df_get_more_ncol(df_country, df, list_country, "country_id")
#     if "po" in dict_inf:
#         list_po = dict_inf["po"]
#         df_po = pd.read_excel(path_input + inf_name, sheet_name="po_pfas")
#         df = df_get_more_ncol(df_po, df, list_po, "poid")
#     if "pos" in dict_inf:
#         list_po = dict_inf["pos"]
#         df_po = pd.read_excel(path_input + inf_name, sheet_name="po_pfas")
#         df = df_get_more_ncol(df_po, df, list_po, "posid")
#     if "pon" in dict_inf:
#         list_po = dict_inf["pon"]
#         # posid会包括一些异构体的数据
#         df_po = pd.read_excel(path_input + inf_name, sheet_name="po_pfas")
#         df = df_get_more_ncol(df_po, df, list_po, "posname")
#     if "pa" in dict_inf:
#         list_po = dict_inf["pa"]
#         # posid会包括一些异构体的数据
#         df_po = pd.read_excel(path_input + inf_name, sheet_name="pa_pfas")
#         df = df_get_more_ncol(df_po, df, list_po, "paid")
#     if "pot" in dict_inf:
#         list_po = dict_inf["pot"]
#         df_po = pd.read_excel(path_input + inf_name, sheet_name="po_treat")
#         df = df_get_more_ncol(df_po, df, list_po, "posname")
#     if "spt" in dict_inf:
#         list_po = dict_inf["spt"]
#         df_po = pd.read_excel(path_input + inf_name, sheet_name="sp_treat")
#         df = df_get_more_ncol(df_po, df, list_po, "spid")
#     return df




if __name__ == "__main__":
    # 这里可以放一些测试代码，或者作为脚本运行时需要执行的代码
    # print("myfunction运行ing")
    pass