
import pandas as pd
import numpy as np

# % matplotlib inline
import wquantiles
import math

# DATE:2024-09 

path_input = "C:/Users/dell/OneDrive/file/"
path_opt = "C:/Users/dell/OneDrive/file/"
path_one_spdb = 'C:/Users/dell/OneDrive/file/SPDB/'

lr_name = "lr_use.csv"
inf_name = "inf.xlsx"


def get_weight_count(df_o, list_col, methods="median"):
    """
    method: mean|median
    ---
    Used to calculate a weighted median or weighted average value, 
    value: the numeric column 
    n: the sample size column 
    list_col to enter the other columns in df_o
    method: mean|median
    ---
    用于计算加权中位数或者加权平均值 
    默认value是数值列 
    n是样本量列 
    list_col需输入df_o中的其他列
    """
    df = df_o.copy()
    if len(list_col) == 1 :
        str_merge_colname = list_col[0]
    if len(list_col) > 1 :
        str_merge_colname = ("~".join(map(str, list_col)))
    df = merge_col(df,list_col)
    df_gb = df.groupby(str_merge_colname, as_index=False).agg(n=('n', 'sum'))
    list_index = list(df_gb[str_merge_colname].unique())
    print(len(list_index))
    if methods == "median":
        dic_med_v = {}
        dic_mad_v = {}
        dic_num_v = {}
        for i in list_index:
            # Weighted median | 加权中位数
            df_count = df.loc[df[str_merge_colname] == i].copy()
            med_v = wquantiles.median(df_count["value"], df_count["n"])
            dic_med_v[i] = med_v
            # Weighted median absolute deviation | 加权中位数绝对偏差
            df_count["gap"] = df_count["value"] - med_v
            df_count["gap"] = df_count["gap"].abs()
            mad_v = wquantiles.median(df_count["gap"], df_count["n"])
            dic_mad_v[i] = mad_v
            # Actual usage data | 实际使用数据
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
            # Weighted average | 加权平均值
            df_count = df[df[str_merge_colname] == i]
            avg_v = np.average(df_count["value"], weights=df_count["n"])
            dic_avg_v[i] = avg_v
            # Weighted standard deviation | 加权标准差
            df_count["gap"] = (df_count["value"] - avg_v)**2
            sd_v = math.sqrt(np.average(df_count["gap"], weights=df_count["n"]))
            dic_sd_v[i] = sd_v
            # Actual usage data | 实际使用数据
            num_v = len(df_count[["value"]])
            dic_num_v[i] = num_v

        df_gb["value"] = df_gb[str_merge_colname].map(dic_avg_v)
        df_gb["SD"] = df_gb[str_merge_colname].map(dic_sd_v)
        df_gb["a_num"] = df_gb[str_merge_colname].map(dic_num_v)
        df_gb = spilt_col(df_gb, str_merge_colname)
        return df_gb



def df_get_ncol(df_provide, df_main, list_col):
    """
    list_col:{0:prov_colname, 1:connect_colname, 2:new_colname, 3:prov__val[prov_colname:prov__val]
    ---
    Matches columns (lists) in df_provide to df_main
    ---
    将df_provide中的列(list_col)匹配到df_main
    """
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
    """
    Merge lon and lat into one column
    ---
    合并lon和lat为一列
    """
    df = df_o.copy()
    df[["lon", "lat"]] = df[["lon", "lat"]].astype(float)
    df[["lon", "lat"]] = df[["lon", "lat"]].apply(lambda x:round(x, 6))
    df[["lon", "lat"]] = df[["lon", "lat"]].astype(str)
    df["lat_lon"] = df[["lat", "lon"]].apply("_".join, axis=1) 
    # 符号不能修改不然出错，历史遗留问题
    # Symbols cannot be modified, otherwise there will be errors and historical legacy issues
    return df

def df_get_more_ncol(df_pro, df_main, list_all, con_col):
    """
    list_all:all columns that need to be added
    con_col: link columns
    ---
    Matches columns (lists) in df_pro to df_main
    ---
    list_all:需要添加的所有列 
    con_col:链接列
    ---
    将df_pro中的列(list_col)匹配到df_main
    """
    for col in list_all:
        list_col = [col, con_col]
        df_main = df_get_ncol(df_pro, df_main, list_col)
    return df_main

def merge_col(df_input, list_merge_colname, merge_symbol="~"):
    """
    Merge multiple columns into one column because using groupby after merging will be faster
    df_input: Please select the required content before entering,
    list_merge_colname: Column names that need to be merged,
    merge_Symbol: The symbol expected to be merged, default~because _ is sometimes used for naming
    ---
    合并多个列为一列 因为合并后再使用groupby会更快
    df_input:请选择好需要的内容再输入, 
    list_merge_colname: 需要合并的列名， 
    merge_symbol: 期望合并的符号，默认~ 因为_有时候会用来命名
    str_merge_colname = (merge_symbol.join(map(str, list_merge_colname)))
    """
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
    """
    Str_marge_colname: The name of the column to be separated
    Spilt_stymbol: default symbol for columns~
    Is_index: Is this column an index
    If 'groupby' is not specified as as_index=F, it will be 'index' and often not specified for alignment purposes
    ---
    str_merge_colname: 要分列的那一列名称
    spilt_symbol: 分列的符号 默认~
    is_index: 这列是否是索引
    groupby若没指定as_index=F 则会为index  很多时候为了对齐会不指定
    str_merge_colname = (merge_symbol.join(map(str, list_merge_colname)))
    """
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
    """
    Convert 2D data to 1D
    List_marge: Other data
    New_colname=the column name of other separate columns (self named, eg: puname)
    New_valname: The values of these columns should also be separately named as column names (self named)
    ---
    二维数据转一维 
    list_merge:其他的数据
    new_colname = 其他单独成列的列的列名(自己命名,eg:poname)
    new_valname:这些列的值也要单独成列的列名(自己命名)
    """
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
    """
    One-dimensional to two-dimensional conversion
    New_comname should be a column name in the original sheet
    ---
    一维转二维 
    new_colname应为原sheet中的一个列名
    """
    if len(list_merge)==1:
        df_gb = df.copy()
        str_merge_colname = list_merge[0]
        df_gb = df_gb.set_index([str_merge_colname, new_colname])
        df_gb_two = df_gb.unstack()
        df_gb_two.columns = df_gb_two.columns.droplevel(0)
        df_gb_two.columns.name = None
        df_gb_two[df_gb_two.index.name] = df_gb_two.index
        # df_gb_two = df_gb_two.reset_index()
        return df_gb_two
    elif len(list_merge)>1:
        df_gb = merge_col(df, list_merge)
        str_merge_colname = ("~".join(map(str, list_merge)))
        df_gb = df_gb.set_index([str_merge_colname, new_colname])
        df_gb_two = df_gb.unstack()
        df_gb_two.reset_index()
        df_gb_two.columns = df_gb_two.columns.droplevel(0)
        df_gb_two.columns.name = None
        df_gb_two[df_gb_two.index.name] = df_gb_two.index
        df_gb_two = spilt_col(df_gb_two, str_merge_colname)
        df_gb_two = df_gb_two.reset_index()
        return df_gb_two

def col_value(df, col_name):
    """
    Create a new DataFrame that contains the unique values of the specified column and their counts
    ---
    创建一个新的DataFrame，其中包含指定列的值和对应的计数
    """
    col_value = pd.DataFrame({col_name:df[col_name].value_counts().index, "count":df[col_name].value_counts().values})
    col_value["per"] = col_value["count"]/len(df)
    return col_value

def col_describe2(df_main, col_name, col_value):
    """
    Col_name is a categorical variable
    Col_value is a numerical value that displays the quantile of each classification
    ---
    col_name是分类的变量
    col_value是数值  会展示每个分类的数值分位数
    """
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
    """
    Describe each element in a column and obtain quantiles
    ---
    对某一列各元素的描述 并获取分位数
    """
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


# ！def：从INF添加

def append_inf(df, dict_inf):
    """
    Df matches the column data corresponding to each key in dict_inf from the inf file
    Habit: The initial data division was too detailed. Here, it has been reclassified as inland water systems and oceans
    Spid: Custom Species Number
    County_id: Custom country code
    POID: Custom Pollutant Number
    Posid: The custom pollutant number contains isomers
    Posname: abbreviation for pollutant
    Paid: Custom data source number
    ---
    df从inf文件中匹配dict_inf中各键对应值的列数据
    habit: 初始数据划分过于细致 这里重新划分为内陆水系和海洋
    spid: 自定义的物种编号
    country_id: 自定义的国家编号
    poid: 自定义的污染物编号
    posid: 自定的污染物编号 包含异构体
    posname: 污染物的简称
    paid: 自定义的数据来源编号
    """
    if "habit" in dict_inf:
        list_habit = dict_inf["habit"]
        df_habit = pd.read_excel(path_input + inf_name, sheet_name="habit")
        df = df_get_more_ncol(df_habit, df, list_habit, "habit")
    if "sp" in dict_inf:
        list_sp = dict_inf["sp"]
        df_sp = pd.read_excel(path_input + inf_name, sheet_name="sp_pfas")
        df = df_get_more_ncol(df_sp, df, list_sp, "spid")
    if "country" in dict_inf:
        list_country = dict_inf["country"]
        df_country = pd.read_excel(path_input + inf_name, sheet_name="country")
        df = df_get_more_ncol(df_country, df, list_country, "country_id")
    if "po" in dict_inf:
        list_po = dict_inf["po"]
        df_po = pd.read_excel(path_input + inf_name, sheet_name="po_pfas")
        df = df_get_more_ncol(df_po, df, list_po, "poid")
    if "pos" in dict_inf:
        list_po = dict_inf["pos"]
        df_po = pd.read_excel(path_input + inf_name, sheet_name="po_pfas")
        df = df_get_more_ncol(df_po, df, list_po, "posid")
    if "pon" in dict_inf:
        list_po = dict_inf["pon"]
        df_po = pd.read_excel(path_input + inf_name, sheet_name="po_pfas")
        df = df_get_more_ncol(df_po, df, list_po, "posname")
    if "pa" in dict_inf:
        list_po = dict_inf["pa"]
        df_po = pd.read_excel(path_input + inf_name, sheet_name="pa_pfas")
        df = df_get_more_ncol(df_po, df, list_po, "paid")
    return df

def id_to_str(df, col_name, sp_name='canonicalName'):
    """
    Convert the species' ID into a name
    Col_name: A list is required
    Sp_name only needs to be used during spid conversion
    ---
    将物种的id转化为名称
    col_name:需要一个list
    sp_name只需要在spid转换时使用
    """
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

