# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 22:41:41 2016

@author: JackyDocky
"""

import codecs
import unicodecsv as csv 
import json
import pprint
import uniout
import re
from hanziconv import HanziConv

FIELDNAME = ['id', 'key', 'value', 'type']


#将繁体中文转换为简体
def audit_traditional(filename):
    row_list = []
    with open(filename, 'rb') as f:
        reader = csv.DictReader(f)
        reader.next()
        for row in reader:
            if row['type'] == 'name' and row['key'] == 'zh' or row['key'] == 'name':
                row['value'] = HanziConv.toSimplified(row['value'])
            row_list.append(row)
        
    with open(filename, 'wb') as f:
        writer = csv.DictWriter(f, delimiter = ',', fieldnames = ['id', 'key', 'value', 'type'])
        writer.writeheader()
        writer.writerows(row_list)
        
        
audit_traditional('ways_tags.csv')
audit_traditional('nodes_tags.csv')
                