import math
import psycopg2
from time import time

from globalmercator import GlobalMercator
from stats import *
from sql import *
from tiles import *

def get_nhd_where(type):
    if type == "drainage":
        return "type='flowline' and upstream < 4"
#        return "type='flowline' and (fcode=46007 or fcode=46003)"
    if type == "stream":
        return "type='flowline' and upstream >= 4"
    if type == "bigstream":
        return "type='flowline' and upstream >= 10"
#        return "type='flowline' and (fcode=46000 or fcode=46006)"
    if type == "lake":
        return "type='waterbody' and (ftype=390 or fcode=56600 or ftype=436 or fcode=44500)"
    if type == "river":
        return "type='waterbody' and (ftype=460 or ftype=336)"
    return None
    
def get_nhd_offset(ll, type):
    return get_offset_linear(ll, [{'table':'nhd', 'where':get_nhd_where(type), 'col':'geom'}])

def get_nhd_offsets(lls, type):
    return get_offsets_linear(lls, [{'table':'nhd', 'where':get_nhd_where(type), 'col':'geom'}])

def get_nhd_stats(meters, type):
    return sql_stats(meters, min(range, 800), [{'table': 'nhd', 'where': get_nhd_where(type), 'col': 'geom'}])