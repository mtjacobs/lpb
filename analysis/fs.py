import math
import psycopg2
from time import time

from globalmercator import GlobalMercator
from stats import *
from sql import *
from tiles import *

def get_fsline_where(type):
    where = "fcsubtype != 107 and fcsubtype != 525"
    if type=="trail":
        where = "fcsubtype = 107 or fcsubtype = 525"
    return 'fsline', where


def get_fsline_offset(ll):
    return get_offset_linear(ll, [{'table': 'fsline', 'where': None, 'col': 'geom'}])

def get_fsline_offsets(lls, type=None):
    return get_offsets_linear(lls, [{'table': 'fsline', 'where': None, 'col': 'geom'}])
    