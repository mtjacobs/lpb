import json
import math
import os
import struct
import urllib

from stats import *
from tiles import *

from globalmercator import GlobalMercator
from osgeo import gdal

landcover_key = { 11: 'water', 12: 'snow', 21: 'developed', 22: 'developed', 23: 'developed', 24: 'developed', 31: 'barren', 41: 'deciduous', 42: 'evergreen', 43: 'mixed', 52: 'shrub', 71: 'grassland', 81: 'farm', 82: 'farm', 90: 'wetland', 95: 'wetland'}

def nlcd_avg_landcover(stats):
    landcover_avg = 'none'
    if stats['landcover']['developed'] > 0.5:
        landcover_avg = 'urban'
    elif stats['landcover']['developed'] > 0.25:
        landcover_avg = 'interface'
    elif stats['landcover']['forest'] > 0.5:
        landcover_avg = 'forest'
    elif stats['landcover']['open'] > 0.5:
        landcover_avg = 'open'
    elif stats['landcover']['open'] > 0.25 and stats['landcover']['forest'] > 0.25:
        landcover_avg = 'mixed'
    return landcover_avg


def nlcd_px(p, z):
    return {'landcover': get_pixel('http://s3-us-west-1.amazonaws.com/ctslope/landcover', z, p[0], p[1]), 'canopy': get_pixel('http://s3-us-west-1.amazonaws.com/ctslope/canopy', z, p[0], p[1])}

def nlcd_ll(coord):
    gm = GlobalMercator()
    m = gm.LatLonToMeters(coord[0], coord[1])
    p = gm.MetersToPixels(m[0], m[1], 12)
    return nlcd_px(p, 12)
    
def nlcd_is_open(val):
    return val == 12 or val == 31 or val == 52 or val == 71 or val == 81 or val == 82
    
def nlcd_is_developed(val):
    return val >= 21 and val <= 24

def nlcd_is_water(val):
    return val == 11 or val == 90 or val == 95
    
def nlcd_is_forest(val):
    return val >= 40 and val <= 44

def nlcd_statistics(center, radius):
    canopy_list = []
    
    count_open = 0
    count_developed = 0
    count_forest = 0
    count_water = 0

    z = best_zoom(center, radius, 12)
    pixels = pixels_in_range(center, radius, z)

    for px_sample in pixels:
        nlcd_sample = nlcd_px(px_sample, z)
        canopy_list.append(nlcd_sample['canopy'])

        if nlcd_is_open(nlcd_sample['landcover']):
            count_open += 1
        if nlcd_is_developed(nlcd_sample['landcover']):
            count_developed += 1
        if nlcd_is_water(nlcd_sample['landcover']):
            count_water += 1
        if nlcd_is_forest(nlcd_sample['landcover']):
            count_forest += 1

    count_open = float(count_open) / len(pixels)
    count_developed = float(count_developed) / len(pixels)
    count_forest = float(count_forest) / len(pixels)
    count_water = float(count_water) / len(pixels)
    
    return {'landcover': {'open': count_open, 'developed': count_developed, 'forest': count_forest, 'water': count_water}, 'canopy': generate_stats(canopy_list)}
    