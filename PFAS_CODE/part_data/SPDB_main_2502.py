import os
import openpyxl
import re
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import coordinate_from_string
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np


# from numpy.distutils.system_info import dfftw_info
# from openpyxl.utils import FORMULAE
# from fuzzywuzzy import process
# from collections import defaultdict
# import sys
''' Bio-pollutant database entry procedure'''



path_file = 'D:/wyy/pyrunning/spdb_sae/fish_pfas/input/'  # 文件路径
path_list = os.listdir(path_file)
path_prepare = 'D:/wyy/pyrunning/spdb_sae/fish_pfas/search/'
path_opt = 'D:/wyy/pyrunning/spdb_sae/fish_pfas/outcome/'
path_temp = 'D:/wyy/pyrunning/spdb_sae/fish_pfas/temp/'
filename_cx = 'cx.xlsx'
filename_lr = 'lr.xlsx'


# 第233行代码，直接指定了污染物类型为PFASs，如果包含别的污染物需要注释掉或者后面在cx文件修改

def str_preprocessing(str_obj, str_type):
    """字符预处理，去除多余的空格，等"""
    if str_type == 'name':
        str_nobj = str_obj.strip()
        # 去除前后的空格，识别和复制时时常多出来
        return str_nobj
    elif str_type == 'data':
        str_nobj = str_obj.replace(" ", "").replace("..", ".")
        # 字符中所有多余的空格去除，..换成.这是由于识别造成的错误
        return str_nobj
    elif str_type == 'soe':
        # 这个可以考虑用pandas分列
        # str_nobj = re.sub(r'[士土+±]', "-", str_obj.replace(" ", ""))
        str_obj = re.split("[士土+±]", str_obj.replace(" ", ""))  # 分开，只取平均值
        return str_obj[0]


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 单独把pa拎出来，因为比较特别
# 直接调用第一个函数就可以了
# paid是直接用文件名称里面的数字

def pa_deal(object_x):
    pa_l = []
    pa_id = []
    for cells in sheet_pa['D2':'D' + str(sheet_pa.max_row)]:
        for cell_pa in cells:
            xy = coordinate_from_string(cell_pa.coordinate)
            # returns ('A',4)，将单元格坐标拆成字母和数字到集合中
            if fuzz.ratio(object_x.lower(), cell_pa.value.lower()) > 90:
                # 模糊匹配，如果匹配率大于百分之90,大小写敏感
                pa_l.append(cell_pa.value)
                pa_id.append(
                    sheet_pa['A' + str(xy[1])].value)
    # 上面遍历进行匹配，匹配后的结果放到两个列表中
    dic_pa = dict(zip(pa_l, pa_id))
    # 两个列表组成字典
    outcome = panbie_pa(len(list(pa_id)), object_x, dic_pa)
    return outcome


def panbie_pa(long_id, object_x, dic):
    # 判别输入的信息，判别后会有ID储存在相对的列表中，同时paper的内容会被要求填充。至于po，sp后续要添加爬虫,panbi_name==search_object
    """这个函数前一个接着的是查询，查询会得到一个包含所查询的字符的列表在XX_l以及XX_id中， long_nm是XX_l包含元素的个数，如果等于0就代表查询的
    文件的没有要查询的字符，直接在录入表中添加相关信息，如果等于或大于一就有，先人为判别一下,pa有则直接结束，po，sp有则用其id，无则同0，最后得到XX_l会被清空，只留对应的唯一ID，pa，sp未补充"""
    if long_id == 0:  # 等于0，即这是新的，需要添加并分配ID
        return panbie_input_pa(object_x)
    elif long_id > 0:
        print(object_x)
        print(dic)
        while True:
            # answer = str(input('是否是同一篇？1为是，2为否'))
            answer = "3"
            if answer == '1':
                print('那到此为止了，')
                return 'NO'
            elif answer == '2':
                return panbie_input_pa(object_x)
            elif answer == "3":
                str_paid = ("-".join(map(str, list(dic.values()))))
                # [1, 2, 123]转变为1-2-123，即列表int转变为一个str
                return panbie_input_pa(object_x, str_paid)
            else:
                print('重新输入')


def panbie_input_pa(object_x, list_paid="NONE"):
    """用来录入新文献的信息 object_x是文献title"""
    sheet_pa['A' + str(int(sheet_pa.max_row) + 1)] = wjm_ID
    sheet_pa['B' + str(sheet_pa.max_row)] = sheet_sjpa['B2'].value
    sheet_pa['C' + str(sheet_pa.max_row)] = sheet_sjpa['C2'].value
    sheet_pa['D' + str(sheet_pa.max_row)] = object_x
    sheet_pa['F' + str(sheet_pa.max_row)] = sheet_sjpa['D2'].value
    sheet_pa['G' + str(sheet_pa.max_row)] = sheet_sjpa['E2'].value
    sheet_pa['H' + str(sheet_pa.max_row)] = sheet_sjpa['F2'].value
    sheet_pa['I' + str(sheet_pa.max_row)] = sheet_sjpa['H2'].value
    # EFG这里需要修改一下，文献标注，230808
    if list_paid != "NONE":
        sheet_pa['E' + str(sheet_pa.max_row)] = list_paid
    else:
        pass
    sheet_sjpa['A1'] = object_x
    return wjm_ID


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 把sp和po的信息先录入，方便后续查询
# 直接调用头两个函数就可以了
def data_lr(data_type, sheet_sj):  # 因为文章里面的数据排版有两种，污染物和生物分别做横纵表头，在这里以此来判断类型。
    # 这里数据的排版被扩展到4种，但是实际使用的应该还是只有两种，这个函数主要就是判断类型，然后返回相应的类型字符
    """data_type是sj表中A1这个单元格的值，这个函数主要用来完成单位的统一和将要需要查询到object全部集中要XX_sj列表中，这里已经调用list_search函数"""
    list_cellvalue = sheet_sj['B1':get_column_letter(sheet_sj.max_column) + '2']
    for cells in list_cellvalue:  # 稍微的预处理一下,把下面的字符全部填充为空
        for cell in cells:
            if cell.value in ['这里写物种名称', '这里写样本数，即：n', '这里写检测位', '单位（根据文献填，e.g.：μg/g）',
                              '这里写采样时间', "这里写备注"]:
                cell.value = None  # 所以openpyxl为了让单元格的值为空，直接等于None就可以了！
    if data_type == 'po':
        if sheet_sj['B2'].value in [None, 'None', '']:  # 当B2没有时，是第一种录入表格，从第六列开始
            int_value = 5
            str_value = 'E'
            list_ot = data_list_1(int_value, str_value, sheet_sj)
            for po in list_ot[0]:
                list_search(po, list_po)
            for sp in list_ot[1]:
                list_search(sp, list_sp)
            return 'po-5'
        else:
            int_value = 1
            str_value = 'A'
            list_ot = data_list_1(int_value, str_value, sheet_sj)
            for po in list_ot[0]:
                list_search(po, list_po)
            for sp in list_ot[1]:
                list_search(sp, list_sp)
            return 'po-1'
    if data_type == 'sp':
        if sheet_sj['B2'].value in [None, 'None', '']:
            int_value = 5
            str_value = 'E'
            list_ot = data_list_1(int_value, str_value, sheet_sj)
            for sp in list_ot[0]:
                list_search(sp, list_sp)
            for po in list_ot[1]:
                list_search(po, list_po)
            return 'sp-5'
        else:
            int_value = 1
            str_value = 'A'
            list_ot = data_list_1(int_value, str_value, sheet_sj)
            for sp in list_ot[0]:
                list_search(sp, list_sp)
            for po in list_ot[1]:
                list_search(po, list_po)
            return 'sp-1'


def sheet_type(data_type):
    list_cellvalue = sheet_sj['B1':get_column_letter(sheet_sj.max_column) + '2']
    for cells in list_cellvalue:  # 稍微的预处理一下,把下面的字符全部填充为空
        for cell in cells:
            if cell.value in ['这里写物种名称', '这里写样本数，即：n', '这里写检测位', '单位（根据文献填，e.g.：μg/g）',
                              '这里写采样时间', "这里写备注"]:
                cell.value = None  # 所以openpyxl为了让单元格的值为空，直接等于None就可以了！
    if data_type == 'po':
        if sheet_sj['B2'].value in [None, 'None', '']:  # 当B2没有时，是第一种录入表格，从第六列开始
            return 'po-5'
        else:
            return 'po-1'
    if data_type == 'sp':
        if sheet_sj['B2'].value in [None, 'None', '']:
            return 'sp-5'
        else:
            return 'sp-1'


def list_search(object_x, list_x):
    """list_x为需要查询的表，object为待查询的字符，是列表中每一个元素；list_search会遍历list_x查询objec，如果有包含的，则将单元格内容及其位置添加到XX_l和XX_id列表当中，接着去判别"""
    sp_l = []
    sp_id = []
    po_l = []
    po_id = []
    if list_x is list_po:
        for cells in sheet_rp['B2':'B' + str(sheet_rp.max_row)]:
            for cell_rp in cells:
                if cell_rp.value not in [None, 'None', '']:
                    if object_x.lower().strip() in cell_rp.value.lower().strip():
                        xy = coordinate_from_string(cell_rp.coordinate)
                        po_l.append(cell_rp.value.strip())
                        po_id.append(
                            sheet_rp['A' + str(xy[1])].value)
        # dic_po = dict(zip(po_id, po_l))
        dic_po = dict(zip(po_l, po_id))
        # print(dic_po)
        # print(str(object_x) + '查询完成')
        panbie(len(list(set(po_id))), len(list(set(po_l))), list_po, object_x, dic_po)
        # print(str(object_x) + '判别完成')
        return "OKK"
    elif list_x is list_sp:
        for cells in sheet_rp['B2':'B' + str(sheet_rp.max_row)]:
            for cell_rp in cells:
                if cell_rp.value not in [None, 'None', '']:
                    if object_x.lower().strip() in cell_rp.value.lower().strip():
                        xy = coordinate_from_string(cell_rp.coordinate)
                        sp_l.append(cell_rp.value.strip())
                        sp_id.append(
                            sheet_rp['A' + str(xy[1])].value)
        dic_sp = dict(zip(sp_l, sp_id))
        # print(str(object_x) + '查询完成')
        panbie(len(list(set(sp_id))), len(list(set(sp_l))), list_sp, object_x, dic_sp)
        # print(str(object_x) + '判别完成')
        return "OKK"


def panbie_input(list_x, object_x):
    '''list_x：用来判断情况，三种:pa，po，sp；object：要添加信息到各表的文献，物种，污染物名称
        这个函数主要处理的是出现新的pa，po，sp需要添加ID和名称到各表
        如果是pa会返回pa的ID'''
    if list_x is list_po:  # 添加id，name到lr（录入）和cx（查询）表中
        int_max_poid = sheet_po['A' + str(sheet_po.max_row)].value
        sheet_po['A' + str(int(sheet_po.max_row) + 1)] = int(int_max_poid) + 1
        sheet_po['D' + str(sheet_po.max_row)] = object_x
        sheet_po['E' + str(sheet_po.max_row)] = 'PFASs'
        sheet_po['F' + str(sheet_po.max_row)] = wjm_ID
        sheet_rp['A' + str(int(sheet_rp.max_row) + 1)] = int(int_max_poid) + 1
        sheet_rp['B' + str(int(sheet_rp.max_row))] = object_x
        sheet_rp['C' + str(int(sheet_rp.max_row))] = "po"
        return 'ok'
    elif list_x is list_sp:  # 添加id，name到lr（录入）和cx（查询）表中
        int_max_spid = sheet_sp['A' + str(sheet_sp.max_row)].value
        sheet_sp['A' + str(int(sheet_sp.max_row) + 1)] = int_max_spid + 1
        sheet_sp['C' + str(sheet_sp.max_row)] = object_x
        sheet_sp['D' + str(sheet_sp.max_row)] = wjm_ID
        # sheet_sp['I' + str(sheet_sp.max_row)] = sheet_sjpa['E2'].value
        # sheet_sp['J' + str(sheet_sp.max_row)] = sheet_sjpa['F2'].value
        # sheet_sp['K' + str(sheet_sp.max_row)] = sheet_sjpa['G2'].value
        sheet_rp['A' + str(int(sheet_rp.max_row) + 1)] = int_max_spid + 1
        sheet_rp['B' + str(int(sheet_rp.max_row))] = object_x
        sheet_rp['C' + str(int(sheet_rp.max_row))] = "sp"
        return 'ok'
    else:
        print('出错了')


def panbie_ans(object_x, dic, list_x, long_id, long_nm, str_x):
    if long_id == '1':
        while True:
            print(object_x)
            print(dic)
            print(wjm)
            answer = str(input('是否是同一？1为是，2为否:'))
            if answer == '1':
                list_rp = (list(dic.values())[0], object_x, str_x)
                sheet_rp.append(list_rp)
                return 'ok'
            elif answer == '2':
                panbie_input(list_x, object_x)
                return 'ok'
            else:
                print('重新输入')
    elif long_id == 'n':
        if long_nm == 1:
            while True:
                print(object_x)
                print(list(dic.keys())[0])
                print(wjm)
                answer = str(input('是否是同一？1为是，2为否:'))
                if answer == '1':
                    list_rp = (list(dic.values())[0], object_x, str_x)
                    sheet_rp.append(list_rp)
                    return 'ok'
                elif answer == '2':
                    panbie_input(list_x, object_x)
                    return 'ok'
                else:
                    print('重新输入')
        elif long_nm > 1:
            while True:
                print(object_x)
                print(dic)
                print(wjm)
                answer = str(input('是否是同一？1为是，2为否:'))
                if answer == '1':
                    list_rp = (input('输入ID:'), object_x, str_x)
                    sheet_rp.append(list_rp)
                    return 'ok'
                elif answer == '2':
                    panbie_input(list_x, object_x)
                    return 'ok'
                else:
                    print('重新输入')


def panbie(long_id, long_nm, list_x, object_x,
           dic):  # 判别输入的信息，判别后会有ID储存在相对的列表中，同时paper的内容会被要求填充。至于po，sp后续要添加爬虫,panbi_name==search_object
    """这个函数前一个接着的是查询，查询会得到一个包含所查询的字符的列表在XX_l以及XX_id中， long_nm是XX_l包含元素的个数，如果等于0就代表查询的
    文件的没有要查询的字符，直接在录入表中添加相关信息，如果等于或大于一就有，先人为判别一下,pa有则直接结束，po，sp有则用其id，无则同0，最后得到XX_l会被清空，只留对应的唯一ID，pa，sp未补充"""
    if long_id == 0:  # 等于0，即这是新的，需要添加并分配ID
        if list_x is list_po:
            return panbie_input(list_po, object_x)
        elif list_x is list_sp:
            return panbie_input(list_sp, object_x)
        else:
            print('出错了')
    elif long_id == 1:
        for k in dic.keys():
            if k.lower().strip() == object_x.lower().strip():
                return "ok"
            else:
                pass
        if list_x is list_po:
            if object_x != list(dic.keys())[0]:
                return panbie_ans(object_x, dic, list_po, '1', long_nm, "po")
            elif object_x == list(dic.keys())[0]:
                return 'ok'
        elif list_x is list_sp:
            if object_x != list(dic.keys())[0]:
                return panbie_ans(object_x, dic, list_sp, '1', long_nm, "sp")
            elif object_x == list(dic.keys())[0]:
                return 'ok'
        else:
            print('出错了')
    elif long_id > 1:
        for k in dic.keys():
            if k.lower().strip() == object_x.lower().strip():
                return "ok"
            else:
                pass
        if list_x is list_po:
            return panbie_ans(object_x, dic, list_po, 'n', long_nm, "po")
        elif list_x is list_sp:
            return panbie_ans(object_x, dic, list_sp, 'n', long_nm, "sp")
        else:
            print('出错了')


def data_list_1(int_value, str_value, sheet_sj):
    """返回需要查询是否重复的的列表"""
    relist_1 = []
    relist_2 = []
    jsq = 0
    while jsq < ((int(sheet_sj.max_column) - int_value) / 5):  # 这里是横向的数据
        str_data1 = sheet_sj[get_column_letter(1 + int_value + 5 * jsq) + '1'].value
        if str_data1 not in [None, 'None', '']:
            relist_1.append(str_data1.strip())
        jsq = jsq + 1
    for cells in sheet_sj[str_value + '4':str_value + str(sheet_sj.max_row)]:  # 列向的数据都从第四行开始
        for str_data2 in cells:
            if str_data2.value not in [None, 'None', '']:
                relist_2.append(str_data2.value.strip())
    relist_11 = list(set(relist_1))  # 把重复的去除
    relist_22 = list(set(relist_2))  # 把重复的去除
    return [relist_11, relist_22]


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 把sp和po的信息先录入，方便后续查询
# 直接调用第一个函数即可
# 这个函数有很多重复的部分，但是真的不想修改了，要传递的参数太多了
def reframe(data_type, re_pa, sheet_sj, sheet_zb, sheet_lrex, sheet_rp):
    list_inf1 = ['NONE']
    list_inf2 = ['NONE']
    zb_1 = []
    sampletime_l = ['NONE']
    remarks_l = ['']
    organ_l = ['NONE']
    n_l = ['NONE']
    danwei_l = ['NONE']
    djsq = 0
    xjsq = 0
    paid = re_pa
    dwww = sheet_sj['A2'].value
    if data_type == 'po-5':
        # print('po-5')
        while djsq < (int(sheet_sj.max_row) - 3):
            if xjsq < ((int(sheet_sj.max_column) - 5) / 5):
                list_inf_value = get_crd("5", sheet_sj, xjsq, djsq, list_inf1, list_inf2, zb_1, sheet_lrex, sheet_rp)
                list_inf1 = list_inf_value[10]
                list_inf2 = list_inf_value[11]
                zb_1 = list_inf_value[12]
                exid, poid, spid, data_ex, data_unit, organ, sampletime, remarks, n = list_inf_value[0], list_inf_value[
                    1], list_inf_value[2], list_inf_value[3], list_inf_value[4], list_inf_value[5], list_inf_value[6], \
                    list_inf_value[7], list_inf_value[
                    8]
                for i in list_inf_value[9]:
                    if i == 'BL':
                        zbjsq = 0
                        while zbjsq < (int(sheet_zb.max_row) - 1):
                            zb_crd = zbjsq + 2
                            list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ, sampletime,
                                                  remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                  danwei_l, remarks_l)
                            sampletime_l = list_xinxi[1]
                            organ_l = list_xinxi[2]
                            n_l = list_xinxi[3]
                            danwei_l = list_xinxi[4]
                            remarks_l = list_xinxi[5]
                            sheet_lrex.append(list_xinxi[0])
                            zbjsq = zbjsq + 1
                            exid = exid + 1
                    else:
                        for cells in sheet_zb['A2':'A' + str(sheet_zb.max_row)]:
                            for cell in cells:
                                if i == str(cell.value):
                                    xy = coordinate_from_string(cell.coordinate)
                                    # returns ('A',4)，将单元格坐标拆成字母和数字到集合中
                                    zb_crd = xy[1]
                                    list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ,
                                                          sampletime,
                                                          remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                          danwei_l, remarks_l)
                                    sampletime_l = list_xinxi[1]
                                    organ_l = list_xinxi[2]
                                    n_l = list_xinxi[3]
                                    danwei_l = list_xinxi[4]
                                    remarks_l = list_xinxi[5]
                                    sheet_lrex.append(list_xinxi[0])
                                    exid = exid + 1
                xjsq = xjsq + 1
            else:
                xjsq = 0
                djsq = djsq + 1
        return "okk"
    elif data_type == 'sp-5':
        # print('sp-5')
        while djsq < (int(sheet_sj.max_row) - 3):
            if xjsq < ((int(sheet_sj.max_column) - 5) / 5):
                list_inf_value = get_crd("5", sheet_sj, xjsq, djsq, list_inf1, list_inf2, zb_1, sheet_lrex, sheet_rp)
                list_inf1 = list_inf_value[10]
                list_inf2 = list_inf_value[11]
                zb_1 = list_inf_value[12]
                exid, spid, poid, data_ex, data_unit, organ, sampletime, remarks, n = list_inf_value[0], list_inf_value[
                    1], list_inf_value[2], list_inf_value[3], list_inf_value[4], list_inf_value[5], list_inf_value[6], \
                    list_inf_value[7], list_inf_value[
                    8]
                for i in list_inf_value[9]:
                    if i == 'BL':
                        zbjsq = 0
                        while zbjsq < (int(sheet_zb.max_row) - 1):
                            zb_crd = zbjsq + 2
                            list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ, sampletime,
                                                  remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                  danwei_l, remarks_l)
                            sampletime_l = list_xinxi[1]
                            organ_l = list_xinxi[2]
                            n_l = list_xinxi[3]
                            danwei_l = list_xinxi[4]
                            remarks_l = list_xinxi[5]
                            sheet_lrex.append(list_xinxi[0])
                            zbjsq = zbjsq + 1
                            exid = exid + 1
                    else:
                        for cells in sheet_zb['A2':'A' + str(sheet_zb.max_row)]:
                            for cell in cells:
                                if i == str(cell.value):
                                    xy = coordinate_from_string(cell.coordinate)
                                    # returns ('A',4)，将单元格坐标拆成字母和数字到集合中
                                    zb_crd = xy[1]
                                    list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ,
                                                          sampletime,
                                                          remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                          danwei_l, remarks_l)
                                    sampletime_l = list_xinxi[1]
                                    organ_l = list_xinxi[2]
                                    n_l = list_xinxi[3]
                                    danwei_l = list_xinxi[4]
                                    remarks_l = list_xinxi[5]
                                    sheet_lrex.append(list_xinxi[0])
                                    exid = exid + 1
                xjsq = xjsq + 1
            else:
                xjsq = 0
                djsq = djsq + 1
        return "okk"
    elif data_type == 'sp-1':
        # print('sp-1')
        while djsq < (int(sheet_sj.max_row) - 3):
            if xjsq < ((int(sheet_sj.max_column) - 1) / 5):
                list_inf_value = get_crd("1", sheet_sj, xjsq, djsq, list_inf1, list_inf2, zb_1, sheet_lrex, sheet_rp)
                list_inf1 = list_inf_value[10]
                list_inf2 = list_inf_value[11]
                zb_1 = list_inf_value[12]
                exid, spid, poid, data_ex, data_unit, organ, sampletime, remarks, n = list_inf_value[0], list_inf_value[
                    1], list_inf_value[2], list_inf_value[3], list_inf_value[4], list_inf_value[5], list_inf_value[6], \
                    list_inf_value[7], list_inf_value[
                    8]
                for zb_i in list_inf_value[9]:
                    if zb_i == 'BL':
                        zbjsq = 0
                        while zbjsq < (int(sheet_zb.max_row) - 1):
                            zb_crd = zbjsq + 2
                            list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ, sampletime,
                                                  remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                  danwei_l, remarks_l)
                            sampletime_l = list_xinxi[1]
                            organ_l = list_xinxi[2]
                            n_l = list_xinxi[3]
                            danwei_l = list_xinxi[4]
                            remarks_l = list_xinxi[5]
                            sheet_lrex.append(list_xinxi[0])
                            zbjsq = zbjsq + 1
                            exid = exid + 1
                    else:
                        for cells in sheet_zb['A2':'A' + str(sheet_zb.max_row)]:
                            for cell in cells:
                                if zb_i == str(cell.value):
                                    xy = coordinate_from_string(cell.coordinate)
                                    # returns ('A',4)，将单元格坐标拆成字母和数字到集合中
                                    zb_crd = xy[1]
                                    list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ,
                                                          sampletime,
                                                          remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                          danwei_l, remarks_l)
                                    sampletime_l = list_xinxi[1]
                                    organ_l = list_xinxi[2]
                                    n_l = list_xinxi[3]
                                    danwei_l = list_xinxi[4]
                                    remarks_l = list_xinxi[5]
                                    sheet_lrex.append(list_xinxi[0])
                                    exid = exid + 1
                xjsq = xjsq + 1
            else:
                xjsq = 0
                djsq = djsq + 1
        return "okk"
    elif data_type == 'po-1':
        # print('po-1')
        while djsq < (int(sheet_sj.max_row) - 3):
            if xjsq < ((int(sheet_sj.max_column) - 1) / 5):
                list_inf_value = get_crd("1", sheet_sj, xjsq, djsq, list_inf1, list_inf2, zb_1, sheet_lrex, sheet_rp)
                list_inf1 = list_inf_value[10]
                list_inf2 = list_inf_value[11]
                zb_1 = list_inf_value[12]
                exid, poid, spid, data_ex, data_unit, organ, sampletime, remarks, n = list_inf_value[0], list_inf_value[
                    1], list_inf_value[2], list_inf_value[3], list_inf_value[4], list_inf_value[5], list_inf_value[6], \
                    list_inf_value[7], list_inf_value[
                    8]
                for zb_i in list_inf_value[9]:
                    if zb_i == 'BL':
                        zbjsq = 0
                        while zbjsq < (int(sheet_zb.max_row) - 1):
                            zb_crd = zbjsq + 2
                            list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ, sampletime,
                                                  remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                  danwei_l, remarks_l)
                            sampletime_l = list_xinxi[1]
                            organ_l = list_xinxi[2]
                            n_l = list_xinxi[3]
                            danwei_l = list_xinxi[4]
                            remarks_l = list_xinxi[5]
                            sheet_lrex.append(list_xinxi[0])
                            zbjsq = zbjsq + 1
                            exid = exid + 1
                    else:
                        for cells in sheet_zb['A2':'A' + str(sheet_zb.max_row)]:
                            for cell in cells:
                                if zb_i == str(cell.value):
                                    xy = coordinate_from_string(cell.coordinate)
                                    # returns ('A',4)，将单元格坐标拆成字母和数字到集合中
                                    zb_crd = xy[1]
                                    list_xinxi = get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ,
                                                          sampletime,
                                                          remarks, n, paid, dwww, sheet_zb, sampletime_l, organ_l, n_l,
                                                          danwei_l, remarks_l)
                                    sampletime_l = list_xinxi[1]
                                    organ_l = list_xinxi[2]
                                    n_l = list_xinxi[3]
                                    danwei_l = list_xinxi[4]
                                    remarks_l = list_xinxi[5]
                                    sheet_lrex.append(list_xinxi[0])
                                    exid = exid + 1
                xjsq = xjsq + 1
            else:
                xjsq = 0
                djsq = djsq + 1
        return "okk"


def get_id(object_x, sheet_rp):
    """在录入过程中，将名称转换为ID"""
    for cells in sheet_rp['B2':'B' + str(sheet_rp.max_row)]:
        for cell_rp in cells:
            if cell_rp.value not in [None, 'None', '']:
                if object_x.lower().strip() == cell_rp.value.lower().strip():
                    return int(sheet_rp['A' + str(coordinate_from_string(cell_rp.coordinate)[1])].value)
    er_text = object_x
    print(object_x)
    return er_text


def get_crd(sheet_type, sheet_sj, xjsq, djsq, list_inf1, list_inf2, zb_1, sheet_lrex, sheet_rp):
    """获取各个信息的位置，并能够直接取值的取值"""
    if sheet_type == "5":
        exid = (int(sheet_lrex['A' + str(sheet_lrex.max_row)].value) + 1)
        # ———————————————我是分割线—————————————————
        # 获取inf1的id，即表格中横向数据
        inf1 = sheet_sj[get_column_letter(6 + xjsq * 5) + '1'].value
        # po-5就是po，sp-5就是sp
        if inf1 not in [None, 'None', '']:
            list_inf1.append(inf1.strip())
        inf1_zz = list_inf1[len(list_inf1) - 1]
        inf1_id = get_id(inf1_zz, sheet_rp)
        # ———————————————我是分割线—————————————————
        # 获取inf2的id，即表格中纵向数据
        inf2 = sheet_sj['E' + str(4 + djsq)].value
        if inf2 not in [None, 'None', '']:
            list_inf2.append(inf2.strip())
        inf2_zz = list_inf2[len(list_inf2) - 1]
        inf2_id = get_id(inf2_zz, sheet_rp)
        # ———————————————我是分割线—————————————————
        # paid = re_pa
        data_ex = sheet_sj[get_column_letter(6 + xjsq * 5) + str(4 + djsq):get_column_letter(9 + xjsq * 5) + str(
            4 + djsq)]  # 这里是区间没打印值出来
        data_unit = sheet_sj[get_column_letter(6 + xjsq * 5) + '2'].value
        organ = sheet_sj['B' + str(4 + djsq)].value
        sampletime = sheet_sj['C' + str(4 + djsq)].value
        remarks = sheet_sj['D' + str(4 + djsq)].value
        n = sheet_sj['A' + str(4 + djsq)].value
        zb = str(sheet_sj[get_column_letter(10 + xjsq * 5) + str(4 + djsq)].value)
        if zb not in [None, 'None', '']:
            zb_1.append(zb)
        zb_2 = re.split("[,，、]", zb_1[len(zb_1) - 1])
        return [exid, inf1_id, inf2_id, data_ex, data_unit, organ, sampletime, remarks, n, zb_2, list_inf1, list_inf2,
                zb_1]
    elif sheet_type == '1':
        exid = (int(sheet_lrex['A' + str(sheet_lrex.max_row)].value) + 1)
        # ———————————————我是分割线—————————————————
        # 获取inf1的id，即表格中横向数据
        inf1 = sheet_sj[get_column_letter(2 + xjsq * 5) + '1'].value
        if inf1 not in [None, 'None', '']:
            list_inf1.append(inf1.strip())
        inf1_zz = list_inf1[len(list_inf1) - 1]
        inf1_id = get_id(inf1_zz, sheet_rp)
        # ———————————————我是分割线—————————————————
        # 获取inf2的id，即表格中纵向数据
        inf2 = sheet_sj['A' + str(4 + djsq)].value
        if inf2 not in [None, 'None', '']:
            list_inf2.append(inf2.strip())
        inf2_zz = list_inf2[len(list_inf2) - 1]
        inf2_id = get_id(inf2_zz, sheet_rp)
        # ———————————————我是分割线—————————————————
        # paid = re_pa
        data_ex = sheet_sj[get_column_letter(2 + xjsq * 5) + str(4 + djsq):get_column_letter(5 + xjsq * 5) + str(
            4 + djsq)]  # 这里是区间没打印值出来
        data_unit = sheet_sj[get_column_letter(2 + xjsq * 5) + '2'].value
        organ = sheet_sj[get_column_letter(6 + xjsq * 5) + '2'].value
        n = sheet_sj[get_column_letter(5 + xjsq * 5) + '2'].value
        zb = str(sheet_sj[get_column_letter(6 + xjsq * 5) + str(4 + djsq)].value)
        sampletime = sheet_sj[get_column_letter(4 + xjsq * 5) + '2'].value
        remarks = sheet_sj[get_column_letter(3 + xjsq * 5) + '2'].value
        if zb not in [None, 'None', '']:  # 如果单元格内有过文字，再去掉的话，不能直接用is not None来判别了
            zb_1.append(zb)
        zb_2 = re.split("[,，、]", zb_1[len(zb_1) - 1])
        return [exid, inf1_id, inf2_id, data_ex, data_unit, organ, sampletime, remarks, n, zb_2, list_inf1, list_inf2,
                zb_1]


def get_data(zb_crd, exid, poid, spid, data_ex, data_unit, organ, sampletime, remarks, n, paid, dwww, sheet_zb,
             sampletime_l, organ_l, n_l, danwei_l, remarks_l):
    zb_3 = []
    xinxi = [poid, spid, paid]
    if sampletime not in [None, 'None', '']:
        sampletime_l.append(sampletime)
    sampletime_zz = sampletime_l[len(sampletime_l) - 1]
    xinxi.append(sampletime_zz)
    for cells in sheet_zb['B' + str(zb_crd):'L' + str(zb_crd)]:
        for cell in cells:
            zb_3.append(cell.value)
    xinxi.insert(0, exid)
    xinxi.extend(zb_3)
    if organ not in [None, 'None', '']:
        organ_l.append(organ)
    organ_zz = organ_l[len(organ_l) - 1]
    xinxi.append(organ_zz)
    for cells in data_ex:
        for cell in cells:
            xinxi.append(cell.value)
    if n not in [None, 'None', '']:
        n_l.append(n)
    n_zz = n_l[len(n_l) - 1]
    xinxi.append(n_zz)
    xinxi.append(dwww)
    if data_unit not in [None, 'None', '']:
        danwei_l.append(data_unit)
    danwei_zz = danwei_l[len(danwei_l) - 1]
    xinxi.append(danwei_zz)
    if remarks not in [None, 'None', '']:
        remarks_l.append(remarks)
    remarks_zz = remarks_l[len(remarks_l) - 1]
    xinxi.append(remarks_zz)
    return [xinxi, sampletime_l, organ_l, n_l, danwei_l, remarks_l]


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 收集LOD，LOQ
def add_lo_to_df(sheet_x, str_dwl, paid, df_x):
    # 把LO表通过list添加到series然后在添加到df
    # ———————————————我是分割线—————————————————
    # 先把表头的列表创建好
    int_max_row = sheet_x.max_row
    int_max_col = sheet_x.max_column
    list_colname = ['paid', 'type', 'unit', 'dwl']
    for cells in sheet_x['B1':get_column_letter(int_max_col) + '1']:
        for cell in cells:
            if cell.value not in [None, 'None', '']:
                po_id = get_id(cell.value, sheet_rp)
                list_colname.append(po_id)
    int_true_col = (len(list_colname) - 3)
    # 这个减几是上面list_colname = ['paid', 'type', 'unit', 'dwl']，里面有四个元素，要保留一个，就是3
    # ———————————————我是分割线—————————————————
    # 获取sheet真实的行数，因为max_row函数有时不准确，在表格写过删除的情况仍会视为整个表格范围内
    list_count_row = []
    for cells in sheet_x['A3':'A' + str(int_max_row)]:
        for cell in cells:
            if cell.value not in [None, 'None', '']:
                list_count_row.append(cell.value)
    int_true_row = (len(list_count_row) + 2)
    # ———————————————我是分割线—————————————————
    # 获取数据
    str_unit = sheet_x['B2'].value
    start_timer = 3
    # 因为从第三行开始是数据
    while start_timer <= int_true_row:
        # 规避虚假的表格范围
        str_type = sheet_x['A' + str(start_timer)].value
        list_data = [paid, str_type, str_unit, str_dwl]
        for cells in sheet_x['B' + str(start_timer):get_column_letter(int_true_col) + str(start_timer)]:
            for cell in cells:
                list_data.append(cell.value)
        # print(list_colname)
        # print(list_data)
        series_data = pd.Series(list_data, index=list_colname)
        # 使用list创建series
        series_data.name = int((str(paid) + str((start_timer - 2))))
        # series要有名称，然后这个名称会成为df中的索引
        # print(list_colname)
        # print(df_x.columns)
        df_x = df_x.append(series_data)
        # series中不能有重复的索引，即list_colname里面不能有重复的元素，否则报错
        start_timer += 1
    return df_x


def get_lo_dwl(list_sheet_name, wb_x):
    # LO表没有细分dwl，因此需要得到sj表的划分，它们是一致的
    if 'sj' in list_sheet_name:
        sheet_sjdwl = wb_x['sj']
        return sheet_sjdwl['A2'].value
    elif 'sj1' in list_sheet_name:
        sheet_sjdwl = wb_x['sj1']
        return sheet_sjdwl['A2'].value


def get_LO():
    df_possibility_all = pd.DataFrame()
    for file_name in path_list[:]:  # 注意不能打开任何一个文件，不然office会自动创建一个隐藏的临时文件,这个临时文件会被打开
        paper_ID = re.split("[-—]", file_name)[0]
        wb_data = openpyxl.load_workbook(path_file + file_name)
        # print(file_name)
        # print('zb:' + str(sheet_zb.max_row))
        # print('zb:' + str(sheet_zb.max_column))
        list_st = wb_data.sheetnames
        if 'LO' in list_st:
            sheet_sjLO = wb_data['LO']
            str_dwl = get_lo_dwl(list_st, wb_data)
            df_possibility_all = add_lo_to_df(sheet_sjLO, str_dwl, paper_ID, df_possibility_all)
            # print(df_possibility_all)
        else:
            pass
    new_row = {'paid': 100001, 'type': 'LOD-T-T', 'unit': 'ng/g'}
    # 先插入一行，以保证后面分列不出错
    df_possibility_all = df_possibility_all.append(new_row, ignore_index=True)
    df_possibility_all.to_csv(path_opt + 'LO.csv', encoding='utf_8_sig', index=False)
    return df_possibility_all


def reset_lo_df(df_lo):
    df_lo = df_lo.reset_index(drop=True)
    df_methos = df_lo['type'].str.split('-', expand=True, n=2)
    # print(df_methos.columns)
    # print(df_methos.head(10))
    df_methos.rename(columns={0: "LO", 1: "condition_1", 2: "condition_2"}, inplace=True)
    # df_methos = df_methos.reindex(['LO', 'condition_1', 'condition_2'], axis='columns')
    # 上面这行修改完，数据就没有了，很奇怪，不知道为什么，所以换了rename函数来更改列索引
    # print(df_methos.head(10))
    # 千万别再拼错columns了！！！
    df_newlo = pd.merge(df_methos, df_lo, how='left', left_index=True, right_index=True)

    # 把一列弹出，插入到df指定列中
    # df_newlo.replace({'LO': {'MLOD': "MDL", 'MLOQ': "MQL", 'IDL': "LOD"}}, inplace=True)
    # LO列内容替换
    df_newlo['unit'].str.strip()
    # unit列字符去除多余空格
    # df_newlo.to_csv(path_opt + 'NLO.csv', encoding='utf_8_sig', index=False)
    # df_newlo.to_excel(path_opt + 'NLO.xlsx', sheet_name='LO', index=False)
    return df_newlo


def change_lo_df_unit(df_lo):
    df_nlo = pd.DataFrame()
    list_col_name = df_lo.columns
    list_po_col_name = list_col_name[10:]
    print(list_po_col_name)
    # list_unit = []
    # for unit in list(set(df_lo['unit'])):
    # if 'l' not in unit.lower():
    # list_unit.append(unit)
    list_unit = list(set(df_lo['unit']))
    for columns in list_po_col_name:
        df_lo[columns] = df_lo[columns].replace({'n.d.': ''})
    df_lo[list_po_col_name] = df_lo[list_po_col_name].apply(pd.to_numeric)
    for unit in list_unit:
        df_nlo_unit = df_lo[df_lo['unit'] == unit].copy()
        # .copy()加这个是因为，单df_nlo_unit只是一个标签，并不是一个df，后面的代码涉及df内容的更改，它是没得改的
        # 不加虽然结果没问题，但是那是pandas自动纠错从原df拷贝了，所有会出现警示
        df_nlo_unit_data = change_lo_df_data(df_nlo_unit, unit, list_po_col_name, 'unit', 'bio')
        df_nlo = df_nlo.append(df_nlo_unit_data)
    df_nlo['unit'][df_nlo['unit'] != 'ng/ml'] = 'ng/g'
    list_change_cdt = list(filter(None, list(set(df_nlo['condition_2']))))
    print(list_change_cdt)
    dic_change_cdt = {}
    for cdt in list_change_cdt:
        int_crd = get_id(cdt, sheet_rp)
        dic_change_cdt[cdt] = int_crd
    print(dic_change_cdt)
    df_nlo['condition_0'] = df_nlo['condition_2'].map(dic_change_cdt)
    df_nlo.insert(3, 'condition_0', df_nlo.pop('condition_0'))
    # 这里插入了一列，注意
    df_nlo.to_csv(path_opt + 'UNLO.csv', encoding='utf_8_sig', index=False)
    return df_nlo


def injudge_df_lo(df_lo):
    if df_lo.empty:
        return True
    else:
        return False


def injudge_s_lo(s_lo):
    if s_lo.empty:
        return True
    else:
        return False


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 低于检出限的数据处理
def get_s_value_first(series_x, paid):
    # 不用调用
    # 从最后确定的series中取值，即已经确定了当前值对应的检出限
    # 不过担心有多个值，所以多写了一些
    list_the_value = list(series_x)
    if len(list_the_value) == 1:
        return list_the_value[0]
    else:
        print('出错paid：' + str(paid))
        print(list_the_value)
        the_value = np.mean(list_the_value)
        return the_value


def get_select_condition(df_x, list_value_all_condition):
    # 不用调用
    # 条件对应的LO值，将通过这里确定是哪一行的LO
    list_cond_0 = list(map(str.lower, list(df_x['condition_0'])))
    # 把初步筛选后的df条件列转成小写，放进列表中
    list_cond_1 = list(map(str.lower, list(df_x['condition_1'])))
    # 同上，初筛后的df应该是确定好了paid和LOQ/LOD的
    dic_cond = dict(zip(list_cond_0, list_cond_1))
    # 准备好初筛后df的条件字典，cond1是确定位置的，即是spid，organ，time还是n
    # print(dic_cond)
    # ~~~~~~~~~~我是分割线~~~~~~~~~~~~~
    list_value_title = ['sp', 'o', 't', 'n']
    list_value = list(map(str.lower, list_value_all_condition))
    # 同上
    dic_value = dict(zip(list_value, list_value_title))
    # 把当前数据的对应的所有内容也形成一个表格，请注意转为str，且小写
    # print(dic_value)
    dic_item = list(set(dic_cond.items()) & set(dic_value.items()))
    # 获取两个字典的交集，就是需要的条件，最后返回是集合，不好取值，因此又转成了list
    return dic_item[0][0]


def lower_val_get_from_unlo(df_lo, paid, poid, LO, list_paid):
    # 这里主要是查检出限表，然后返回它
    if paid in list_paid[0]:
        s_the_value = df_lo[str(poid)][(df_lo['paid'] == paid) & (df_lo['LO'] == LO)]
        if injudge_s_lo(s_the_value):
            return 'NA-' + LO
        else:
            # print(s_the_value)
            the_value = get_s_value_first(s_the_value, paid)
            return the_value
    elif paid in list_paid[1]:
        df_first_seclect_lo = df_lo[['condition_1', 'condition_0', str(poid)]][
            (df_lo['paid'] == paid) & (df_lo['LO'] == LO)]
        select_condition_0 = get_select_condition(df_first_seclect_lo, list_paid[2])
        print(select_condition_0)
        print(paid)
        s_the_value = df_first_seclect_lo[str(poid)][df_first_seclect_lo['condition_0'] == select_condition_0]
        if injudge_s_lo(s_the_value) is False:
            the_value = get_s_value_first(s_the_value, paid)
            return the_value
        else:
            return 'NA-' + LO
    else:
        print(paid + 'unlo表没查到')
        return 'NA-' + LO


def use_pare(df_pr, paid):
    s_type = df_pr['type'][df_pr['id'] == paid]
    return get_s_value_first(s_type, paid)


def isnumber(aString):
    try:
        float(aString)
        return True
    except:
        return False


def get_value(sheet_x, df_lo, df_pr, val_col_crd, int_value):
    for cells in sheet_x[val_col_crd + '2':val_col_crd + str(sheet_x.max_row)]:
        for cell in cells:
            if cell.value not in [None, 'None', '']:
                # 不是数字就开始
                xy = coordinate_from_string(cell.coordinate)
                if isnumber(cell.value) is False:
                    the_lower_value = get_every_list(cell, sheet_x, df_lo, df_pr)
                    if isnumber(the_lower_value):
                        sheet_x[xy[0] + str(xy[1])] = float(the_lower_value) / int_value
                        sheet_x['Y' + str(xy[1])] = float(the_lower_value)
                        sheet_x['Z' + str(xy[1])] = '1'
                        print(str(xy[1]))
                    else:
                        sheet_x[xy[0] + str(xy[1])] = the_lower_value
                        sheet_x['Y' + str(xy[1])] = the_lower_value
                        sheet_x['Z' + str(xy[1])] = '0'
                        print(str(xy[1]))
                else:
                    # 是数字就pass
                    sheet_x['Y' + str(xy[1])] = 'NA'
                    sheet_x['Z' + str(xy[1])] = '0'
                    pass
    return print('okk')


def get_need_LO(list_LO):
    # 不必调用
    # 有一些有LOQ，但是缺的是LOQ，这种时候，用LOQ补，这里会优先返回LOD，没有再返回LOQ
    list_LO = list(set(list_LO))
    for i in list_LO:
        if 'D' in i:
            return i
        elif 'Q' in i:
            return i


def get_lower_value(cell, sheet_x, df_lo, df_pr, list_po_col_name, list_paid_b, list_paid_s=None):
    """当list_paid_b，包含三个列表时，list_paid_s一定得填"""
    # list_paid_b:list_lo_paid, list_pr_all_scope, list_pr_scope
    # list_paid_b：这个表一定包含所有的paid
    # list_paid_s:list_nocond_paid, list_onecond_paid
    # list_paid_s：这个表加起来是list_lo_paid
    # 用于LO
    # 当list_paid_b本身为一个列表时，list_paid_s可以不填
    xy = coordinate_from_string(cell.coordinate)
    poid = sheet_x['B' + str(xy[1])].value
    the_paid = int(sheet_x['D' + str(xy[1])].value)
    object_cell = cell.value
    if str(poid) in list_po_col_name:
        if type(list_paid_b[0]) is list:
            if '<' in object_cell or '＜' in object_cell:
                n_cell_value = cell.value.replace('<', '').replace(' ', '').replace('＜', '')
                if isnumber(n_cell_value):
                    lower_value = n_cell_value
                    return lower_value
                else:
                    if int(the_paid) in list_paid_b[0]:
                        # 有条件，或者有LOD,LOQ
                        # lower_value = lower_val_get_from_unlo(df_lo, the_paid, poid, n_cell_value, sp, o, t, n, list_pr_all)
                        lower_value = lower_val_get_from_unlo(df_lo, the_paid, poid, n_cell_value, list_paid_s)
                        sheet_x['X' + str(xy[1])] = 'ng/g'
                        return lower_value
                    elif int(the_paid) in list_paid_b[1]:
                        # 进入这个表默认没条件，且缺LOD，换为LOQ补齐数据
                        LO = use_pare(df_pr, the_paid)
                        s_the_value = df_lo[str(poid)][(df_lo['paid'] == the_paid) & (df_lo['LO'] == LO)]
                        if injudge_s_lo(s_the_value):
                            sheet_x['X' + str(xy[1])] = 'ng/g'
                            return 'NA-' + LO
                        else:
                            the_value = get_s_value_first(s_the_value, the_paid)
                            sheet_x['X' + str(xy[1])] = 'ng/g'
                            return the_value
                    else:
                        return 'NA-' + object_cell
            elif 'a' in object_cell.lower():
                return object_cell
            elif 'd' in object_cell.lower():
                if int(the_paid) in list_paid_b[1]:
                    LO = use_pare(df_pr, the_paid)
                    s_the_value = df_lo[str(poid)][(df_lo['paid'] == the_paid) & (df_lo['LO'] == LO)]
                    if injudge_s_lo(s_the_value):
                        return 'NA-ND'
                    else:
                        the_value = get_s_value_first(s_the_value, the_paid)
                        sheet_x['X' + str(xy[1])] = 'ng/g'
                        return the_value
                elif int(the_paid) in list_paid_b[2]:
                    return 'NA-ND'
                elif int(the_paid) in list_paid_b[0]:
                    list_LO = list(df_lo['LO'][df_lo['paid'] == the_paid])
                    print(list_LO)
                    str_LO = get_need_LO(list_LO)
                    # lower_value = use_unlo(df_lo, the_paid, poid, str_LO, sp, o, t, n, list_pr_all)
                    lower_value = lower_val_get_from_unlo(df_lo, the_paid, poid, str_LO, list_paid_s)
                    sheet_x['X' + str(xy[1])] = 'ng/g'
                    return lower_value
                else:
                    return 'NA-ND'
            else:
                pass
        else:
            if '<' in object_cell or '＜' in object_cell:
                n_cell_value = object_cell.replace('<', '').replace(' ', '').replace('＜', '')
                if isnumber(n_cell_value):
                    lower_value = n_cell_value
                    return lower_value
                else:
                    if int(the_paid) in list_paid_b:
                        list_LO = list(df_lo['LO'][df_lo['paid'] == the_paid])
                        print(list_LO)
                        str_LO = get_need_LO(list_LO)
                        s_the_value = df_lo[str(poid)][(df_lo['paid'] == the_paid) & (df_lo['LO'] == str_LO)]
                        if injudge_s_lo(s_the_value):
                            return 'NA-ND'
                        else:
                            # print(s_the_value)
                            the_value = get_s_value_first(s_the_value, the_paid)
                            sheet_x['X' + str(xy[1])] = 'ng/g'
                            return the_value
                    else:
                        return 'NA-' + object_cell
            elif 'a' in object_cell.lower():
                return object_cell
            elif 'd' in object_cell.lower():
                if int(the_paid) in list_paid_b:
                    list_LO = list(df_lo['LO'][df_lo['paid'] == the_paid])
                    print(list_LO)
                    str_LO = get_need_LO(list_LO)
                    s_the_value = df_lo[str(poid)][(df_lo['paid'] == the_paid) & (df_lo['LO'] == str_LO)]
                    if injudge_s_lo(s_the_value):
                        return 'NA-ND'
                    else:
                        # print(s_the_value)
                        the_value = get_s_value_first(s_the_value, the_paid)
                        sheet_x['X' + str(xy[1])] = 'ng/g'
                        return the_value
                else:
                    return 'NA-' + object_cell
            else:
                pass
    elif str(poid) not in list_po_col_name:
        if '<' in object_cell or '＜' in object_cell:
            n_cell_value = cell.value.replace('<', '').replace(' ', '').replace('＜', '')
            if isnumber(n_cell_value):
                lower_value = n_cell_value
                return lower_value
            else:
                return object_cell
        else:
            return object_cell


def get_every_list(cell, sheet_x, df_lo, df_pr):
    # 准备好各种list
    xy = coordinate_from_string(cell.coordinate)
    list_pr_scope = list(set(df_pr['id'][df_pr['dwl'].notna()]))
    # 指代缺LO表的sj
    list_pr_all = list(set(df_pr['id'][df_pr['type'].notna()]))
    # 指代pr中记录的缺表或缺LOD有LOQ的sj
    list_pr_all_scope = list(set(list_pr_all) - set(list_pr_scope))
    # 指代数据缺LOD，但只有LOQ，之类相互不匹配的sj
    list_lo_all = list(set(df_lo['paid']))
    # 有LO表所有的sj
    list_lo_paid = list(set(list_lo_all) - set(list_pr_all))
    # 把这两类分隔开，LO表去掉pr表的paid
    sp = sheet_x['C' + str(xy[1])].value
    # spid在lr中的数据
    o = sheet_x['Q' + str(xy[1])].value.strip()
    # organ
    t = sheet_x['E' + str(xy[1])].value
    # time-year
    n = sheet_x['V' + str(xy[1])].value
    # 样本数
    list_value_all_condition = [str(sp), str(o), str(t), str(n)]

    #
    list_col_name = df_lo.columns
    list_po_col_name = list_col_name[8:]
    # 这里去除了不能转换为int的元素，这个列表需要的是pfas的编号

    list_lo_cond_all = list(set(list_lo_all) - set(list_pr_all))
    # lo-pr表
    list_onecond_paid_pr = list(set(df_lo['paid'][df_lo['condition_1'].notna()]))
    #
    list_onecond_paid = list(set(list_onecond_paid_pr) - set(list_pr_all))
    # lo-pr表中一个条件的paid
    list_nocond_paid = list(set(list_lo_cond_all) - set(list_onecond_paid))
    # lo-pr表中无条件的paid
    list_paid_b = [list_lo_paid, list_pr_all_scope, list_pr_scope]
    list_paid_s = [list_nocond_paid, list_onecond_paid, list_value_all_condition]

    lower_val = get_lower_value(cell, sheet_x, df_lo, df_pr, list_po_col_name, list_paid_b, list_paid_s)
    return lower_val


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 转换单位
def change_un(data_self, data_un, data_type):
    if data_type == 'bio':
        if data_un == 'ng/g' or data_un == 'μg/kg' or data_un == 'ug/kg' or data_un == 'µg/kg':
            return [data_self, 'ng/g', 1]
        elif data_un == 'μg/g' or data_un == 'mg/kg' or data_un == 'ug/g':
            data_new = (data_self * 1000)
            return [data_new, 'ng/g', 1000]
        elif data_un == 'mg/g':
            data_new = (data_self * 1000000)
            return [data_new, 'ng/g', 1000000]
        elif data_un == 'pg/g' or data_un == 'ng/kg':
            data_new = (data_self / 1000)
            return [data_new, 'ng/g', 0.001]
        elif data_un == 'ng/ml' or data_un == 'ng/mL':
            return [data_self, 'ng/ml', 1]
        elif data_un == 'pg/ml':
            data_new = (data_self / 1000)
            return [data_new, 'ng/ml', 0.001]
        else:
            return [data_self, data_un, 1]
    elif data_type == 'soe':
        if data_un == 'ng/g' or data_un == 'μg/kg' or data_un == 'ug/kg' or data_un == 'µg/kg':
            return [data_self, 'ng/g', 1]
        elif data_un == 'μg/g' or data_un == 'mg/kg' or data_un == 'ug/g':
            data_new = (data_self * 1000)
            return [data_new, 'ng/g', 1000]
        elif data_un == 'mg/g':
            data_new = (data_self * 1000000)
            return [data_new, 'ng/g', 1000000]
        elif data_un == 'pg/g' or data_un == 'ng/kg':
            data_new = (data_self / 1000)
            return [data_new, 'ng/g', 0.001]
        elif data_un == 'pg/l':
            data_new = (data_self / 1000)
            return [data_new, 'ng/l', 0.001]
        elif data_un == 'ng/l':
            return [data_self, 'ng/l', 1]


def change_lo_df_data(df_nlo_unit, unit, list_po_col_name, df_unit_colname, data_type):
    list_change_un = change_un(1, unit, data_type)
    new_unit = list_change_un[1]
    df_nlo_unit[df_unit_colname] = new_unit
    for col in list_po_col_name:
        df_nlo_unit[col] = df_nlo_unit[col] * (list_change_un[2])
    return df_nlo_unit


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 获取sd和ed

# ~~~~~~~~~~~~~~~小分割线~~~~~~~~~~~~~~~~~~~~~
# 下面先获取各个数据文件中有什么sheet的表格
def get_list_sheetname(paid, list_sheetname, list_colname):
    # 不必调用
    # 获取list，然后转成series
    list_x = [paid]
    for colname in list_colname[1:]:
        if colname in list_sheetname:
            list_x.append('1')
        else:
            list_x.append('0')
    series_data = pd.Series(list_x, index=list_colname)
    series_data.name = 'sheetname'
    return series_data


def get_table_sheetname():
    # 把各个数据文件里面有什么表格做成一张列表以便获取列表指示
    df_sheetname = pd.DataFrame()
    list_colname = ['paid', 'sd', 'ed', 'LO', 'SLO', 'WLO']
    for file in path_list[:]:  # 注意不能打开任何一个文件，不然office会自动创建一个隐藏的临时文件,这个临时文件会被打开
        file_name = file
        paid = re.split("[-—]", file_name)[0]
        # 字段分开后取第一个字符
        wb_sj = openpyxl.load_workbook(path_file + file_name)
        print(file_name)
        list_allsheetname = wb_sj.sheetnames
        df_sheetname = df_sheetname.append(get_list_sheetname(paid, list_allsheetname, list_colname))
    return df_sheetname


# ~~~~~~~~~~~~~~~小分割线~~~~~~~~~~~~~~~~~~~~~
#
def ed_lo_get(df_sheetname):
    # 获取ed的lo表格
    df_soe_lo = pd.DataFrame()
    list_stn = ['SLO', 'WLO']
    dic_rp = get_iddic_from_rp('po')
    for stn in list_stn:
        list_stn_paid = list(df_sheetname['paid'][df_sheetname[stn] == 1])
        for paid in list_stn_paid:
            df_stn = pd.read_excel(path_file + str(paid) + '-sj.xlsx', sheet_name=stn)
            list_df_colname = df_stn.columns
            list_new_colname = []
            for colname in list_df_colname:
                str_new_colname = get_id_from_dic(colname, dic_rp)
                list_new_colname.append(str_new_colname)
            df_stn.columns = list_new_colname
            df_stn['paid'] = paid
            df_soe_lo = df_soe_lo.append(df_stn)
    list_colname = df_soe_lo.columns
    list_po_colname = []
    for colname in list_colname:
        if isnumber(colname):
            list_po_colname.append(colname)
        else:
            pass
    print(list_po_colname)
    list_soe_unit = list(set(df_soe_lo['UNIT']))
    print(list_soe_unit)
    df_last_soe = pd.DataFrame()
    for unit in list_soe_unit:
        df_soe_unit = df_soe_lo[df_soe_lo['UNIT'] == unit].copy()
        df_soe_unit_change = change_lo_df_data(df_soe_unit, unit.lower(), list_po_colname, 'UNIT', 'soe')
        df_last_soe = df_last_soe.append(df_soe_unit_change)
    # 修改列名，
    df_last_soe = df_last_soe.rename(columns={'Compound': 'LO', 'DWL': 'dwl', 'UNIT': 'unit'})
    # 调整 'paid' 列到第二列
    cols = df_last_soe.columns.tolist()
    cols.insert(1, cols.pop(cols.index('paid')))
    df_last_soe = df_last_soe[cols]
    return df_last_soe


def df_get_new_col(df_provide, prov_colname, prov_colname2, df_receive, lr_colname):
    # 从df_provide里面把两列转成dict，补充到另一个df中
    dic_povide = df_provide.set_index(prov_colname)[prov_colname2].to_dict()
    # df两列转dict
    df_receive[prov_colname2] = df_receive[lr_colname].map(dic_povide)
    # df从字典根据一列添加信息
    return 'okk'


def get_iddic_from_rp(rp_type):
    # rp中获取po或者sp的字典，名称：ID
    df_rp = pd.read_excel(path_opt + 'cx.xlsx', sheet_name='rp')
    df_rp = df_rp[df_rp['type'] == rp_type]
    dic_rp = df_rp.set_index('NAME')['ID'].to_dict()
    return dic_rp


def get_id_from_dic(object_x, dic_rp):
    # 获取object对应的ID，object指代污染物或物种名称
    if object_x == 'location（借用zb第一列就可以了）':
        return 'location'
    elif object_x == '干湿重' or object_x == 'dw/ww':
        return 'dwl'
    elif object_x != 'location（借用zb第一列就可以了）':
        for key, val in dic_rp.items():
            if object_x.lower().strip() == key.lower().strip():
                return val
    return object_x


def ed_get_from_sj(df_sheetname):
    # 把各个文件中的ed整合到一个文件中
    df_all_ed_def = pd.DataFrame()
    list_sd_paid = list(df_sheetname['paid'][df_sheetname['ed'] == 1])
    dic_rp = get_iddic_from_rp('po')
    for paid in list_sd_paid:
        df_zb = pd.read_excel(path_file + str(paid) + '-sj.xlsx', sheet_name='zb')
        df_ed = pd.read_excel(path_file + str(paid) + '-sj.xlsx', sheet_name='ed')
        list_df_colname = df_ed.columns
        list_new_colname = []
        for colname in list_df_colname:
            str_new_colname = get_id_from_dic(colname, dic_rp)
            list_new_colname.append(str_new_colname)
        # 把该df列索引中的污染物名称转为ID，方便对齐
        df_ed.columns = list_new_colname
        # 替换列索引为新的
        # print(df_ed.columns)
        list_crd = ['lon', 'lat']
        for crd in list_crd:
            df_get_new_col(df_zb, '采样点', crd, df_ed, 'location')
        # 把坐标信息添加进去
        df_ed['paid'] = paid
        df_all_ed_def = df_all_ed_def.append(df_ed)
    df_all_ed_def.insert(1, 'paid', df_all_ed_def.pop('paid'))
    df_all_ed_def.insert(5, 'lon', df_all_ed_def.pop('lon'))
    df_all_ed_def.insert(6, 'lat', df_all_ed_def.pop('lat'))
    df_all_ed_def.insert(7, 'dwl', df_all_ed_def.pop('dwl'))
    return df_all_ed_def


def sd_get_from_sj(df_sheetname):
    # 把各个文件中的sd整合到一个文件中
    df_all_sd = pd.DataFrame()
    list_sd_paid = list(df_sheetname['paid'][df_sheetname['sd'] == 1])
    for paid in list_sd_paid:
        df_sd = pd.read_excel(path_file + str(paid) + '-sj.xlsx', sheet_name='sd')
        list_colname = df_sd.columns
        list_new_name = []
        for colname in list_colname:
            str_obj = re.split("[(（]", colname)
            list_new_name.append(str_obj[0].strip())
        # 列索引有点小问题，小小处理一下字符
        df_sd.columns = list_new_name
        df_sd['paid'] = paid
        df_sd = df_sd.drop(df_sd.index[[0]])
        # 把第一行去掉，注意是双[[0]]，
        df_all_sd = df_all_sd.append(df_sd)
    df_all_sd['Species'].str.strip()
    # 去掉前后多余空格，但是好像不太管用
    dic_rp = get_iddic_from_rp('sp')
    # 查看df_all_sd的colname，如果'Detection bit'，'water content'，'Carnivorousrbon source'在其中则删除这三列

    cols_to_drop = ['Detection bit', 'water content', 'Carnivorousrbon source']
    df_all_sd = df_all_sd.drop(columns=[col for col in cols_to_drop if col in df_all_sd.columns])
    # 去除多余的列索引，不知道哪个文件了
    df_all_sd['spid'] = df_all_sd['Species'].map(dic_rp)
    df_all_sd = df_all_sd[
        ['Species', 'paid', 'spid', 'remark', 'age', 'sex', 'Status', 'n', 'Weight', 'Length', 'Width',
         'Moisture content', 'δ15N', 'δ13C', 'TL', ]]
    return df_all_sd


def pretreatment_sad(df, str_type):
    # 经过这一步，数据的各种小问题基本搞定，除了num-num的这种
    # df = df.replace({'/': '', '<L0Q': '<LOQ', '-': ''})
    df = df.replace({'/': '', '<L0Q': '<LOQ', '-': ''})

    if str_type == 'sd':
        if 'remark' not in df.columns:
            # 添加一列 'remark'，可以将其初始化为某个默认值或者留空
            df['remark'] = ''

        print(df.iloc[:, 8:].select_dtypes(include=object).columns)
        # 这里写死从第八列开始处理
        for column in df.iloc[:, 8:].select_dtypes(include=object):
            df[column] = df[column].str.split("[±土+（士(]", 1).str[0]
            df[column] = df[column].str.replace(',', '.')
            df[column] = df[column].str.replace(' ', '')
        df_pre = df.copy()
    elif str_type == 'ed':
        # 前8列是基本信息
        # 后面都是PFAS
        # list_df_colname = df.columns
        list_df_col8 = df.columns[:8]
        list_df_coln = df.columns[8:]
        list_paid_colname = []
        for colname in list_df_coln:
            if isnumber(colname):
                list_paid_colname.append(colname)
            else:
                pass
        print(df[list_paid_colname].select_dtypes(include=object).columns)
        for column in df[list_paid_colname].select_dtypes(include=object):
            df[column] = df[column].str.split("[±土+（士(]", 1).str[0]
            df[column] = df[column].str.replace(',', '.')
            df[column] = df[column].str.replace(' ', '')
        df_pre = pd.concat([df[list_df_col8], df[list_paid_colname]], axis=1)
    # df_newlo.to_excel(path_opt + 'NLO.xlsx', encoding='utf_8_sig', sheet_name='LO', index=None)
    else:
        print('----------ERROR----------')
        df_pre = df.copy()
    return df_pre


def sd_data_last_treat(sheet_x):
    # 主要用于处理sd中num-num的这种
    # 这种处理后，按理来说，只用了两个n的数据，所以n应该修改为2，但是代码没写，后面用的时候要修改，20221019
    # H是需要处理数据的第一列
    for cells in sheet_x['H2':'O' + str(sheet_x.max_row)]:
        # H就是第八列
        for cell in cells:
            if cell.value not in [None, 'None', '']:
                if isnumber(cell.value):
                    pass
                else:
                    xy = coordinate_from_string(cell.coordinate)
                    if '-' in cell.value or '–' in cell.value:
                        list_value = re.split("[-—一–]", cell.value)
                        # print(list_value)
                        list_nv = [float(x) for x in list_value]
                        # print(list_nv)
                        true_value = np.mean(list_nv)
                        sheet_x[xy[0] + str(xy[1])] = true_value
                    elif '>' in cell.value:
                        true_value = cell.value.replace('>', '')
                        sheet_x[xy[0] + str(xy[1])] = true_value
                    else:
                        sheet_x[xy[0] + str(xy[1])] = None
    return print('sd data last treat is okk')


def get_lower_ed_value(cell, sheet_x, df_lo, list_paid_b, list_po_col_name):
    #
    df_lo.columns = df_lo.columns.astype(str)
    xy = coordinate_from_string(cell.coordinate)
    poid = sheet_x['I' + str(xy[1])].value
    the_paid = int(sheet_x['B' + str(xy[1])].value)
    print(the_paid)
    object_cell = cell.value
    if poid in list_po_col_name:
        if '<' in object_cell or '＜' in object_cell:
            n_cell_value = object_cell.replace('<', '').replace(' ', '').replace('＜', '')
            if isnumber(n_cell_value):
                lower_value = n_cell_value
                return lower_value
            else:
                if int(the_paid) in list_paid_b:
                    list_LO = list(df_lo['LO'][df_lo['paid'] == int(the_paid)])
                    print(list_LO)
                    str_LO = get_need_LO(list_LO)
                    s_the_value = df_lo[str(poid)][(df_lo['paid'] == int(the_paid)) & (df_lo['LO'] == str_LO)]
                    if injudge_s_lo(s_the_value):
                        return 'NAND'
                    else:
                        # print(s_the_value)
                        the_value = get_s_value_first(s_the_value, int(the_paid))
                        return the_value
                else:
                    return 'NA1' + object_cell
        elif 'a' in object_cell.lower():
            return object_cell
        elif 'd' in object_cell.lower():
            if int(int(the_paid)) in list_paid_b:
                list_LO = list(df_lo['LO'][df_lo['paid'] == int(the_paid)])
                print(list_LO)
                str_LO = get_need_LO(list_LO)
                s_the_value = df_lo[str(poid)][(df_lo['paid'] == int(the_paid)) & (df_lo['LO'] == str_LO)]
                if injudge_s_lo(s_the_value):
                    return 'NAND'
                else:
                    # print(s_the_value)
                    the_value = get_s_value_first(s_the_value, the_paid)
                    return the_value
            else:
                return 'NA2' + object_cell
        else:
            pass
    else:
        if '<' in object_cell or '＜' in object_cell:
            n_cell_value = cell.value.replace('<', '').replace(' ', '').replace('＜', '')
            if isnumber(n_cell_value):
                lower_value = n_cell_value
                return lower_value
            else:
                return object_cell
        else:
            return object_cell


def ed_data_last_treat(sheet_x, df_ed_lo, int_value, list_colname):
    # C需要是unit列，H需要是dwl列
    # I需要是数据的第一列
    df_ed_lo_w = df_ed_lo[~df_ed_lo['dwl'].notna()]
    df_ed_lo_s = df_ed_lo[df_ed_lo['dwl'].notna()]
    list_w_paid = list(set(df_ed_lo_w['paid']))
    list_s_paid = list(set(df_ed_lo_s['paid']))
    # print(sheet_x['J1'].value)
    for cells in sheet_x['J2':'J' + str(sheet_x.max_row)]:
        for cell in cells:
            if cell.value not in [None, 'None', '']:
                xy = coordinate_from_string(cell.coordinate)
                if isnumber(cell.value):
                    if sheet_x['C' + str(xy[1])].value in ['pg/L', 'pg/l']:
                        # merge_data = str(float(cell.value) / 1000) + '-NONE-0'
                        sheet_x[xy[0] + str(xy[1])] = (float(cell.value) / 1000)
                        sheet_x['C' + str(xy[1])] = 'ng/L'
                        sheet_x['K' + str(xy[1])] = ''
                        sheet_x['L' + str(xy[1])] = '0'
                    elif sheet_x['C' + str(xy[1])].value in ['pg/g']:

                        sheet_x[xy[0] + str(xy[1])] = (float(cell.value) / 1000)
                        sheet_x['C' + str(xy[1])] = 'ng/g'
                        sheet_x['K' + str(xy[1])] = ''
                        sheet_x['L' + str(xy[1])] = '0'
                    else:
                        sheet_x['K' + str(xy[1])] = ''
                        sheet_x['L' + str(xy[1])] = '0'
                        pass
                else:
                    print(sheet_x['H1'].value)
                    if sheet_x['H' + str(xy[1])].value not in [None, 'None', '']:
                        # 上面的列应该是dwl那一列，这里是沉积物和土的数据
                        the_lower_value = get_lower_ed_value(cell, sheet_x, df_ed_lo_s, list_s_paid, list_colname)
                        if isnumber(the_lower_value):
                            # merge_data = str(float(the_lower_value) / int_value) + '-' + str(the_lower_value) +'-1'
                            sheet_x[xy[0] + str(xy[1])] = float(the_lower_value) / int_value
                            sheet_x['K' + str(xy[1])] = float(the_lower_value)
                            sheet_x['L' + str(xy[1])] = '1'
                        else:
                            sheet_x[xy[0] + str(xy[1])] = the_lower_value
                            sheet_x['K' + str(xy[1])] = the_lower_value
                            sheet_x['L' + str(xy[1])] = '0'
                        sheet_x['C' + str(xy[1])] = 'ng/g'
                    else:
                        # 这里是水的数据
                        the_lower_value = get_lower_ed_value(cell, sheet_x, df_ed_lo_w, list_w_paid, list_colname)
                        if isnumber(the_lower_value):
                            sheet_x[xy[0] + str(xy[1])] = float(the_lower_value) / int_value
                            sheet_x['K' + str(xy[1])] = float(the_lower_value)
                            sheet_x['L' + str(xy[1])] = '1'
                        else:
                            # merge_data = str(the_lower_value) + '-NONE-0'
                            sheet_x[xy[0] + str(xy[1])] = the_lower_value
                            sheet_x['K' + str(xy[1])] = the_lower_value
                            sheet_x['L' + str(xy[1])] = '0'
                        sheet_x['C' + str(xy[1])] = 'ng/L'
    return print('ed data last treat is okk')


def transform_excel_data(file_path):
    """
    Transform Excel data by keeping first 8 columns unchanged and melting the remaining columns.

    Args:
        file_path (str): Path to the Excel file

    Returns:
        pandas.DataFrame: Transformed dataframe
    """
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name='all_ed')
    df = df[df['lon'].notna()]
    # Get the first 8 columns that should remain unchanged
    id_vars = ['type', 'paid', 'unit', 'location', 'lon', 'lat', 'time', 'dwl']

    # Melt the dataframe
    melted_df = pd.melt(
        df,
        id_vars=id_vars,
        var_name='poid',
        value_name='value'
    )

    # Sort the dataframe to keep related rows together
    melted_df = melted_df.sort_values(id_vars)

    return melted_df

# def sad_get_data(df_sd, df_ed, df_ed_lo):
#     df_sd = pretreatment_sad(df_sd, 'sd')
#     df_ed = pretreatment_sad(df_ed, 'ed')
#     # 处理数据的各种问题
#     df_sd.to_excel(path_temp + 'all_sd.xlsx', sheet_name='all_sd', index=False)
#     df_ed.to_excel(path_temp + 'all_ed.xlsx', sheet_name='all_ed', index=False)
#     wb_ed = openpyxl.load_workbook(path_temp + 'all_ed.xlsx')
#     sheet_ed = wb_ed['all_ed']
#     wb_ed.save(path_opt + 'all_ed.xlsx')
#     wb_sd = openpyxl.load_workbook(path_temp + 'all_sd.xlsx')
#     sheet_sd = wb_sd['all_sd']
#
#     list_colname = list(df_ed_lo.columns)
#     # print(sheet_ed['L1'].value)
#     ed_data_last_treat(sheet_ed, df_ed_lo, 2, list_colname)
#     sd_data_last_treat(sheet_sd)
#     wb_sd.save(path_opt + 'all_sd.xlsx')
#     return 'okk'


# —————————————————————————————————————————————————我是分割线——————————————————————————————————————————————————————————
# 运行


while True:
    wb_lr = openpyxl.load_workbook(path_prepare + filename_lr)
    sheet_lrex = wb_lr['ex']
    print("1:污染物及物种登记\n2:数据库格式\n3：整理单位\n4：整理LO表\n5：取半检出\n6：获取sd-ed\n7：整理sd-ed\n8：去除dup物种")
    # 先12，然后45，接着3，67
    # 8开始前有个前提条件，目前还没写GBIF api的调用所以需要手动
    str_pg = str(input("输入："))
    if str_pg == "1":
        wb_cx = openpyxl.load_workbook(path_prepare + filename_cx)
        sheet_po = wb_cx['po']
        sheet_pa = wb_cx['pa']
        sheet_sp = wb_cx['sp']
        sheet_rp = wb_cx['rp']
        list_pa = sheet_pa['D2':'D' + str(sheet_pa.max_row)]
        list_po = sheet_po['C2':'D' + str(sheet_po.max_row)]
        list_sp = sheet_sp['B2':'D' + str(sheet_sp.max_row)]
        # 最后查的还是rp表，一定要保证rp表中包含po和sp表的内容
        for i in path_list[:]:
            # 注意不能打开任何一个文件，不然office会自动创建一个隐藏的临时文件,这个临时文件会被打开
            # 先把sp，po录进去
            wjm = i
            wjm_ID = re.split("[-—]", wjm)[0]
            # 获取pa的编号
            wb_sj = openpyxl.load_workbook(path_file + wjm)
            # 打开excel文件
            print(wjm)
            list_sheet = wb_sj.sheetnames
            # excel文件中的表格目录
            for j in list_sheet:
                if 'sj' in j and 'm' not in j:
                    sheet_sj = wb_sj[j]
                    # 访问数据表格
                    print('sheet:' + str(sheet_sj.max_row))
                    print('sheet:' + str(sheet_sj.max_column))
                    st_tp = data_lr(sheet_sj['A1'].value, sheet_sj)
                    # 完成该表格的污染物及物种的查询及录入
                    print('sp-po search over.')
            wb_sj.close()
            # 关闭数据表格
        wb_cx.save(path_opt + filename_cx)
        # cx文件中包括po，pa，sp，rp
        wb_cx.close()
    elif str_pg == "2":
        # 若有物种或者污染物没有出现在cx中，会在lr中直接出现名称
        wb_cx = openpyxl.load_workbook(path_opt + filename_cx)
        sheet_po = wb_cx['po']
        sheet_pa = wb_cx['pa']
        sheet_sp = wb_cx['sp']
        sheet_rp = wb_cx['rp']
        list_pa = sheet_pa['D2':'D' + str(sheet_pa.max_row)]
        list_po = sheet_po['C2':'D' + str(sheet_po.max_row)]
        list_sp = sheet_sp['B2':'D' + str(sheet_sp.max_row)]
        for i in path_list[:]:  # 注意不能打开任何一个文件，不然office会自动创建一个隐藏的临时文件,这个临时文件会被打开
            wjm = i
            wjm_ID = re.split("[-—]", wjm)[0]
            # 获取pa的编号
            wb_sj = openpyxl.load_workbook(path_file + wjm)
            # 打开excel文件
            print(wjm)
            # print('zb:' + str(sheet_zb.max_row))
            # print('zb:' + str(sheet_zb.max_column))
            sheet_sjpa = wb_sj['pa']
            # 打开pa这个表格
            str_title = sheet_sjpa['A2'].value
            # 获取pa的title
            str_title = str_title.replace('\n', ' ').replace('\r', ' ')
            # 换行符替代掉
            print(str_title)
            print('Title search over.')
            re_pa = pa_deal(str_title)
            # 根据title判断pa是否重复，重复则返回NO，不重复则录入
            if re_pa == 'NO':
                print('Duplicate paper,  Over——' + wjm_ID)
                wb_sj.close()
                # 重复文献，跳过到下一篇
            else:
                # 这时候re_pa应该等于wjm_ID
                list_sheet = wb_sj.sheetnames
                # list_sheet所有的sheet名称列表
                if "rpzb" in list_sheet:
                    # rpzb是用于将多个地点的混样数据，合并到一个坐标点
                    # 之前录入的做法是多个地点的混样数据每个都重复一遍，这不合理
                    sheet_zb = wb_sj['rpzb']
                    for j in list_sheet:
                        if 'sj' in j and 'm' not in j:
                            sheet_sj = wb_sj[j]
                            # 访问数据表格
                            # print('sheet:' + str(sheet_sj.max_row))
                            # print('sheet:' + str(sheet_sj.max_column))
                            st_tp = sheet_type(sheet_sj['A1'].value)
                            # st_tp表格类型
                            reframe(st_tp, re_pa, sheet_sj, sheet_zb, sheet_lrex, sheet_rp)
                            print(wjm)
                            print(str_title)
                            print('Record over.')
                    wb_sj.close()
                else:
                    sheet_zb = wb_sj['zb']
                    for j in list_sheet:
                        if 'sj' in j and 'm' not in j:
                            sheet_sj = wb_sj[j]
                            # print('sheet:' + str(sheet_sj.max_row))
                            # print('sheet:' + str(sheet_sj.max_column))
                            st_tp = sheet_type(sheet_sj['A1'].value)
                            reframe(st_tp, re_pa, sheet_sj, sheet_zb, sheet_lrex, sheet_rp)
                            print(wjm)
                            print(str_title)
                            print('Record over.')
                    wb_sj.close()
        wb_lr.save(path_opt + filename_lr)
        wb_cx.save(path_opt + filename_cx)
        wb_cx.close()
        wb_lr.close()
    elif str_pg == "3":
        # 转换单位，用openpyxl写的有部分会出错，所以换成pandas重新写了
        df_lr_pfas = pd.read_csv(path_opt + 'ex_lower_pfas.csv', index_col='id')
        # 这个文件我也忘了是干嘛的，哦哦好像是以及去了2分之一的检出限了，所以这个使用次序在4和5之后
        list_unit = list(set(df_lr_pfas['unit']))
        df_new_ex = pd.DataFrame()
        list_new_ex_col_name = ['avg']
        for unit in list_unit:
            df_new_ex_unit = df_lr_pfas[df_lr_pfas['unit'] == unit].copy()
            df_new_ex_unit_change = change_lo_df_data(df_new_ex_unit, unit, list_new_ex_col_name, 'unit', 'bio')
            df_new_ex = df_new_ex.append(df_new_ex_unit_change)
        df_new_ex.to_csv(path_opt + 'last_lr.csv', encoding='utf_8_sig')
    elif str_pg == "4":
        # 准备Lo文件
        wb_cx = openpyxl.load_workbook(path_prepare + filename_cx)
        sheet_rp = wb_cx['rp']
        # 上面的需要更换，这个是没录入前的rp表
        get_LO()
        df_olo = pd.read_csv(path_opt + 'LO.csv')
        df_n_lo = reset_lo_df(df_olo)
        change_lo_df_unit(df_n_lo)
        # 最后会得到UNLO文件，这个文件是统一单位到ng/g
        # 替换时会把从这个表里取得的数据那行单位也换成ng/g
        # 如果是<号的，单位不会变化，而是统一留到下一步转换单位
    elif str_pg == "5":
        # 这段是把低于检测限的值取二分之一LOD
        df_lo = pd.read_csv(path_opt + 'UNLO.csv')
        df_pr = pd.read_csv(path_prepare + 'paper record.csv')

        # df_lr = pd.read_excel(path_opt + 'lr.xlsx', sheet_name='ex', index_col='id')
        wb_lr_ex = openpyxl.load_workbook(path_opt + 'lr.xlsx')
        # ——————————————————————————我是分割线——————————————————————————————————
        # 这里是将低于检出限的值具体化
        sheet_lr_ex = wb_lr_ex['ex']
        get_value(sheet_lr_ex, df_lo, df_pr, 'T', 2)
        wb_lr_ex.save(path_opt + 'ex_lower.xlsx')
        wb_lr_ex.close()
        # ——————————————————————————我是分割线——————————————————————————————————
        # 这里是筛掉不能用的值，以及筛选出PFAS
        df_lr_ex = pd.read_excel(path_opt + 'ex_lower.xlsx', sheet_name='ex', index_col='id')
        df_po = pd.read_excel(path_prepare + 'cx.xlsx', sheet_name='po')
        # 这个也得注意，新录入的肯定还没划分类别
        list_pfas = list(set(df_po['poid'][df_po['po_type'] == 'PFASs']))
        print(list_pfas)
        list_avg_value = list(set(df_lr_ex['avg'][df_lr_ex['avg'].notna()]))
        list_avg_notval = []
        for i in list_avg_value:
            if isnumber(i):
                pass
            else:
                list_avg_notval.append(i)
        df_lr_ex = df_lr_ex[df_lr_ex['avg'].notna()]
        df_lr_pfas = df_lr_ex[df_lr_ex['poid'].isin(list_pfas)]
        df_lr_pfas = df_lr_pfas[~df_lr_pfas['avg'].isin(list_avg_notval)]
        df_lr_pfas = df_lr_pfas[(df_lr_pfas['avg'] != 0) & (df_lr_pfas['n'] != 0)]
        df_lr_pfas.to_csv(path_opt + 'ex_lower_pfas.csv', encoding='utf_8_sig')
    elif str_pg == '6':
        # 这块代码可能存在一些问题2024.03.11
        # 一定得先保存再读取，不然出错
        # 主要用来获取环境数据和生物信息
        # 没有多余的代码被注释，这里是后面只需要补充ed数据，就把sd的部分注释一下
        # 这里的代码关于数据的处理范围写死了，如果列表列名有变一定要记得修改
        # 默认数据从第九列开始，主要是下面这三个函数
        # pretreatment_sad，sd_data_last_treat，ed_data_last_treat
        df_sheetnme = get_table_sheetname()
        df_sheetnme.to_csv(path_opt + 'sheetname.csv',encoding='utf_8_sig', index=False)
        df_sheetnme = pd.read_csv(path_opt + 'sheetname.csv')
        # df_all_sd = sd_get_from_sj(df_sheetnme)
        # df_all_sd.to_csv(path_temp + 'all_sd.csv', encoding='utf_8_sig', index=False)
        df_all_ed = ed_get_from_sj(df_sheetnme)
        df_all_ed.to_csv(path_temp + 'ed_raw.csv', encoding='utf_8_sig', index=False)
        df_last_soe = ed_lo_get(df_sheetnme)
        df_last_soe.to_csv(path_opt + 'soe_lo.csv', encoding='utf_8_sig', index=False)

    elif str_pg == '7':
        # df_all_sd = pd.read_csv(path_temp + 'all_sd.csv')
        # df_sd = pretreatment_sad(df_all_sd, 'sd')
        # df_sd.to_excel(path_temp + 'all_sd.xlsx', sheet_name='all_sd', index=False)

        df_all_ed = pd.read_csv(path_temp + 'ed_raw.csv')
        df_ed = pretreatment_sad(df_all_ed, 'ed')
        df_ed.to_excel(path_temp + 'ed_treat.xlsx', sheet_name='all_ed', index=False)

        file_path = r"D:\wyy\pyrunning\spdb_sae\fish_pfas\temp\ed_treat.xlsx"
        result_df = transform_excel_data(file_path)
        result_df = result_df[result_df['value'].notna()]
        # Save the transformed data to a new Excel file

        output_path = file_path.replace('.xlsx', '_re.xlsx')
        result_df.to_excel(output_path, index=False)

        df_last_soe = pd.read_csv(path_opt + 'soe_lo.csv')

        wb_ed = openpyxl.load_workbook(path_temp + 'ed_treat_re.xlsx')
        sheet_ed = wb_ed['Sheet1']

        # wb_sd = openpyxl.load_workbook(path_temp + 'all_sd.xlsx')
        # sheet_sd = wb_sd['all_sd']
        list_colname = list(df_last_soe.columns)

        ed_data_last_treat(sheet_ed, df_last_soe, 2, list_colname)
        # sd_data_last_treat(sheet_sd)
        sheet_ed['K1'] = 'limit_value'
        sheet_ed['L1'] = 'type'
        sheet_ed['A1'] = 'water_type'
        wb_ed.save(path_opt + 'all_ed.xlsx')
        # wb_sd.save(path_opt + 'all_sd.xlsx')
    elif str_pg == '8':
        # 可以尝试调用GBIF的API接口以匹配Latin name
        # 这里还是手动匹配后再运行这段代码
        # 这步运行完成后，手动合并一下文件
        # 1. 读取sp表，命名为df_sp，新建一列"sp_d"，该列每行的数据为同行的canonicalName列的物种名称，在整个canonicalName列中的重复次数
        path_re_dup = 'D:/wyy/pyrunning/spdb_sae/re_dup_outcome/'
        df_sp = pd.read_excel(path_opt + 'cx.xlsx', sheet_name='sp')
        df_sp['id'] = df_sp['spid'] + 10000
        df_sp['sp_d'] = df_sp.groupby('canonicalName')['canonicalName'].transform('count')
        df_rp = pd.read_excel(path_opt + 'cx.xlsx', sheet_name='rp')
        df_rp_sp = df_rp[df_rp["type"] == "sp"]
        df_lr = pd.read_csv(path_opt + 'last_lr.csv')
        df_sd = pd.read_excel(path_opt + 'all_sd.xlsx', sheet_name='all_sd')
        # 2. 筛选出重复次数大于的1的df_sp，将canonicalName列不重复的元素放入list_dups_sp
        df_sp_dups = df_sp[df_sp['sp_d'] > 1]
        list_dups_sp = df_sp_dups['canonicalName'].unique().tolist()
        # 3. 新建一个dataframe，命名为df_change，新建两列，一列命名为o_id，一列命名为n_id
        df_change = pd.DataFrame(columns=['o_id', 'n_id'])

        # 4. 遍历list_dups_sp的元素，筛选df_sp，条件为canonicalName列等于遍历的list_dups_sp元素
        # 将筛选后df_sp的spid列数据存入df_change的o_id列，存入的那些行对应的n_id为当前筛选后df_sp的spid列最小的值。
        for name in list_dups_sp:
            df_temp = df_sp[df_sp['canonicalName'] == name]
            min_id = df_temp['spid'].min()
            df_change = df_change.append(
                pd.DataFrame({'o_id': df_temp['spid'].tolist(), 'n_id': [min_id] * len(df_temp)}), ignore_index=True)
        dic_id = df_change.set_index('o_id')['n_id'].to_dict()
        # 5. 开始替换重复的spid
        df_sp['spid'] = df_sp['spid'].map(lambda x: dic_id.get(x, x))
        df_sp = df_sp.loc[df_sp.groupby('spid')['id'].idxmin()]
        df_sp.drop('sp_d', axis=1, inplace=True)
        df_lr["spid"] = df_lr["spid"].map(lambda x: dic_id.get(x, x))
        df_rp_sp["ID"] = df_rp_sp["ID"].map(lambda x: dic_id.get(x, x))
        df_sd["spid"] = df_sd["spid"].map(lambda x: dic_id.get(x, x))

        df_n_rp = pd.concat([df_rp_sp, df_rp[df_rp["type"] == "po"]], axis=0)
        # 6. 保存各个文件
        df_sp.to_csv(path_re_dup + "sp_re.csv", index=False, encoding='utf-8-sig')
        df_lr.to_csv(path_re_dup + "lr_re.csv", index=False, encoding='utf-8-sig')
        df_n_rp.to_csv(path_re_dup + "rp_re.csv", index=False, encoding='utf-8-sig')
        df_change.to_csv(path_re_dup + "spid_change.csv", index=False, encoding='utf-8-sig')
        df_sd.to_csv(path_re_dup + "sd_re.csv", index=False, encoding='utf-8-sig')
    elif str_pg == "N":
        print("后续处理转到SPDB_species.ipynb")
        break
    else:
        print("Again")
