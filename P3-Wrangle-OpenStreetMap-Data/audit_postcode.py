# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 11:04:45 2016

@author: JackyDocky
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 10:35:01 2016

@author: JackyDocky
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:41:41 2016

@author: JackyDocky
"""

import codecs
import unicodecsv as csv 
import uniout
import re

FIELDNAME = ['id', 'key', 'value', 'type']

#检查各式函数：是否可转化为数字、是否长度为6、是否以‘43’开头
def check_postcode(code):
    return code.isdigit() and len(code) == 6 and code.startswith('43')
to_edit = []
#处理邮编
def audit_postcode(filename):
    with open(filename, 'rb') as f:
        reader = csv.DictReader(f)
        reader.next()
        for row in reader:
            if row['key'] == 'postcode':                
                if not check_postcode(row['value']):
                    to_edit.append(row)
        
        
        
audit_postcode('ways_tags.csv')
audit_postcode('nodes_tags.csv')

#打印待纠正列表
print to_edit

                