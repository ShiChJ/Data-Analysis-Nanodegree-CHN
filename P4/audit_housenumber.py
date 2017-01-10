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

#处理housenumber
def audit_housenumber(filename):
    row_list = []
    with open(filename, 'rb') as f:
        reader = csv.DictReader(f)
        reader.next()
        for row in reader:
            if row['key'] == 'housenumber':
                #正则表达式：提取字符串中的所有数字
                possible_numbers = re.findall(r'[0-9]+', row['value'])
                #如果只提取了一个，说明该数字是正确的房号，否则不能确定，需要手动修改。
                if len(possible_numbers) == 1:
                    row['value'] = possible_numbers[0]
                    
            row_list.append(row)
        
    with open(filename, 'wb') as f:
        writer = csv.DictWriter(f, delimiter = ',', fieldnames = ['id', 'key', 'value', 'type'])
        writer.writeheader()
        writer.writerows(row_list)
        
        
audit_housenumber('ways_tags.csv')
audit_housenumber('nodes_tags.csv')
                