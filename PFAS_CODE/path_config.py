import os


path_file = 'C:/Users/dell/OneDrive/file/'
path_one_spdb = 'C:/Users/dell/OneDrive/file/SPDB/'
path_file_csv = 'C:/Users/dell/OneDrive/file/csv/'
path_file_nc = 'C:/Users/dell/OneDrive/file/nc/'

meta_file = 'meta_data.csv'
inf_file = 'inf.xlsx'


drive_letter = 'E:'
str_path_med = "/wyy/code_project/running_outcome/final_data/SPDB/"

drive_letter_f = 'F:'
str_path_med_f = "/User_file/wyy/SPDB/"


str_path_part = "part_data"
str_path_part0 = "part0_treat"
str_path_part1 = "part1_describe"
str_path_part2 = "part2_analysis"
str_path_part3 = "part3_forecast"
str_path_part4 = "part4_assess"
str_path_part5 = "part5_uncertain"
str_path_parts = "part_sub"

def ensure_trailing_sep(path):
    return path if path.endswith(os.sep) else path + os.sep


path_part = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_part)
)
path_part0 = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_part0)
)
path_part1 = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_part1)
)
path_part2 = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_part2)
)
path_part3 = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_part3)
)
path_part4 = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_part4)
)
path_part5 = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_part5)
)
path_parts = ensure_trailing_sep(
    os.path.join(drive_letter, str_path_med.lstrip('/'), str_path_parts)
)

pathf_part4 = ensure_trailing_sep(
    os.path.join(drive_letter_f, str_path_med_f.lstrip('/'), str_path_part4)
)



path_part0_pre = ensure_trailing_sep(os.path.join(path_part0, "pretreatment"))
path_part0_match = ensure_trailing_sep(os.path.join(path_part0, "match"))
path_part0_temp = ensure_trailing_sep(os.path.join(path_part0, "temp"))
path_part0_fig = ensure_trailing_sep(os.path.join(path_part0, "fig"))

path_part1_data = ensure_trailing_sep(os.path.join(path_part1, "data"))
path_part1_fig = ensure_trailing_sep(os.path.join(path_part1, "fig"))

path_part2_pre = ensure_trailing_sep(os.path.join(path_part2, "preanalysis"))
path_part2_fig = ensure_trailing_sep(os.path.join(path_part2, "fig"))

path_part3_lr = ensure_trailing_sep(os.path.join(path_part3, "lr_forecast"))
path_part3_sw = ensure_trailing_sep(os.path.join(path_part3, "sw_forecast"))
path_part3_lrsw = ensure_trailing_sep(os.path.join(path_part3, "lrsw_forecast"))
path_part3_fig = ensure_trailing_sep(os.path.join(path_part3, "fig"))
path_part3_temp = ensure_trailing_sep(os.path.join(path_part3, "temp"))

path_part4_thq = ensure_trailing_sep(os.path.join(path_part4, "thq"))
path_part4_nc = ensure_trailing_sep(os.path.join(path_part4, "nc"))
path_part4_hi = ensure_trailing_sep(os.path.join(path_part4, "hi"))
path_part4_pop = ensure_trailing_sep(os.path.join(path_part4, "pop"))
path_part4_grid = ensure_trailing_sep(os.path.join(path_part4, "grid"))
path_part4_fig = ensure_trailing_sep(os.path.join(path_part4, "fig"))
path_part4_pfas = ensure_trailing_sep(os.path.join(path_part4, "pfas"))
path_part4_result = ensure_trailing_sep(os.path.join(path_part4, "results"))
path_part4_temp = ensure_trailing_sep(os.path.join(path_part4, "temp"))

path_part5_data = ensure_trailing_sep(os.path.join(path_part5, "data"))
path_part5_fig = ensure_trailing_sep(os.path.join(path_part5, "fig"))

path_parts_data = ensure_trailing_sep(os.path.join(path_parts, "data"))
path_parts_fig = ensure_trailing_sep(os.path.join(path_parts, "fig"))
path_parts_temp = ensure_trailing_sep(os.path.join(path_parts, "temp"))