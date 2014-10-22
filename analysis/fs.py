import math
import psycopg2
from time import time

from globalmercator import GlobalMercator
from stats import *
from sql import *
from tiles import *

def get_fsline_offset(ll):
    return get_offset_linear(ll, 'fsline', None, 'geom')

def get_fsline_offsets(lls, type=None):
    return get_offsets_linear(lls, 'fsline', None, 'geom')
    