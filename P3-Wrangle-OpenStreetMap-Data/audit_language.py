# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 14:33:25 2016

@author: JackyDocky
"""

import codecs
import unicodecsv as csv 
import json
import pprint
import uniout
import re
from hanziconv import HanziConv

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:41:41 2016

@author: JackyDocky
"""

FIELDNAME = ['id', 'key', 'value', 'type']

#检查字段是否为英文
#注释:ctrl + 1
#def isEnglish(s):
#    try:
#        s.decode('ascii') 
#    except UnicodeDecodeError:
#        return False
#    else:
#        return True
        
#修改后的行列表
row_list = []

#查看字符串是否存在中文
def is_chinese(s):
    for c in s:
        if c >= u'\u4e00' and c <= u'\u9fa5':
            return True
    return False
    
trans_dic = {'KFC':'肯德基',
 "McDonald's":'麦当劳',
 "McDonald‘s":'麦当劳',
 "McDonald´s":'麦当劳',
 'Hospital of HAU': '华中农业大学医院',
 'JK Mall': '金凯购物中心',
 'Pizza Hut':'必胜客',
 'Crown Bakery': '皇冠蛋糕',
 'luojia moutain water tower': '珞珈山水塔',
 'Giant':'捷安特专卖店',
 'Bank of China':'中国银行',
 'Bank of China(CCNU)':'中国银行',
 'Aloha':'爱乐好夏威夷西餐厅',
 'Westin':'万达威斯汀酒店',
 'Woodcutter Pavilion':'樵夫亭',
 'Citadine':'馨乐庭沌口服务公寓',
 'AYD':'欧亚达建材家居生活广场',
 'Citroen':'东风雪铁龙',
 'Merida':'美利达专卖店',
 'MUSE':'MUSE酒吧',
 'French consulate':'法国驻武汉总领事馆',
 'Four Season':'四季恋餐厅',
 'Sanjiaohu Primary School':'三角湖小学',
 'Starbucks Coffee':'星巴克',
 'Feeling':'Feeling啤酒屋',
 'Hubu Alley':'户部巷',
 'Rue piétonne':'汉江路步行街',
 'DPCA Wuhan 1':'神龙汽车有限公司',
 'DPCA Wuhan 2':'神龙汽车有限公司',
 'DPCA Wuhan 3':'神龙汽车有限公司',
 'Dongfeng Johnson Controls Automotive Seating Co. Ltd.':'东风江森汽车座椅有限公司',
 'Dongfeng Passenger Vehicle Compagny':'东风乘用车公司',
 'VOX Livehouse':'VOX酒吧',
 'Wuhan Museum':'武汉博物馆',
 'Pathfinder Youth Hostel':'探路者国际年旅舍',
 'Hubei Institute of Fine Arts':'湖北美术学院',
 'WHUT Wuhan University of Technology, East Campus, Teaching Building 4':'武汉理工大学东校区教学楼4',
 'WHUT Wuhan University of Technology International Students Dormitory Building D4':'武汉理工大学留学生宿舍D4',
 'Wanda realm hotel':'万达瑞华酒店',
 'Wuhan Irivet Machinery Co., Ltd.':'武汉制药机械有限责任公司',
 'GeoStar':'武大吉奥信息技术有限公司',
 'hanting hotel':'汉庭酒店',
 'Wanda plaza':'万达广场',
 '111':'钢花新村111小区'
 }


def check_row(row):
    return (row['type'] == 'name' and row['key'] == 'zh' or row['key'] == 'name')
 
 
def audit_language(filename):
    with open(filename, 'rb') as f:
        reader = csv.DictReader(f)
        reader.next()
        for row in reader:
            if check_row(row) == True and row['value'] in trans_dic:
                row['value'] = trans_dic[row['value']]
            row_list.append(row)
        
        
        with open(filename, 'wb') as f:
            writer = csv.DictWriter(f, delimiter = ',', fieldnames = ['id', 'key', 'value', 'type'])
            writer.writeheader()
            writer.writerows(row_list)


audit_language('nodes_tags.csv')
 
 
 
 
 



                