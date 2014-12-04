import math
import psycopg2
from time import time

from globalmercator import GlobalMercator
from stats import *
from sql import *
from tiles import *

def get_osm_within_offset_poly(lls, max_range, type=None):
    return find_within_offset_linear(lls, max_range, 'planet_osm_line', get_osm_where(type), 'way')

def get_osm_where(type):
    where = "highway is not null or route='foot' or power='line'"
    table = "planet_osm_line"
    if type=="river":
        where = "water='river' or water='stream;river' or waterway='river' or waterway='riverbank' or waterway='stream'"
        table = 'planet_osm_polygon'
    elif type=="lake":
        where ="\"natural\"='lake' or \"natural\"='bay' or water='pond' or water='lake' or water='lake;pond' or water='reservoir'"
        table = 'planet_osm_polygon'
    if type == "road":
        where = "highway is not null and highway != 'footway' and highway != 'path'"
    elif type == "trail":
        where = "(highway='footway' or highway='path' or highway='trail' or route='foot') and ref!='PCT'"
    elif type == "power":
        where = "power='line'"
    return table, where

def get_osm_offset(ll, type):
    table, where = get_osm_where(type)
    return get_offset_linear(ll, [{'table': table, 'where': where, 'col': 'way'}])

def get_osm_offsets(lls, type):
    table, where = get_osm_where(type)
    return get_offsets_linear(lls, [{'table': table, 'where': where, 'col': 'way'}])

def get_osm_pcts(meters, radius, type):
    table, where = get_osm_where(type)
    return sql_stats(meters, radius, [{'table': table, 'where': where, 'col': 'way'}])