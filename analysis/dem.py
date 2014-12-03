import json
import math
import os
import struct
import urllib

from globalmercator import GlobalMercator
from osgeo import gdal

from stats import *
from tiles import *

gm = GlobalMercator()

def dem_px(p, z):
    return {'elevation': get_pixel('http://s3-us-west-1.amazonaws.com/ctslope/elevation', z, p[0], p[1]), 'slope': get_pixel('http://s3-us-west-1.amazonaws.com/ctslope/slope', z, p[0], p[1]), 'aspect': get_pixel('http://s3-us-west-1.amazonaws.com/ctslope/aspect', z, p[0], p[1])}

def dem_ll(coord):
    gm = GlobalMercator()
    m = gm.LatLonToMeters(coord[0], coord[1])
    p = gm.MetersToPixels(m[0], m[1], 14)
    return dem_px(p, 14)
    
def delta_elevation(start, end):
    m_start = dem_ll(start)['elevation']
    m_end = dem_ll(end)['elevation']
    
    return (m_end-m_start)*3.28084
    
def dem_stats(meters, radius):
    cache_e = {}
    cache_c = {}
    cache_t = {}
    
    r = {'e': [], 'c': [], 't': []}
    
    for m in meters:
        px_center = gm.MetersToPixels(m[0], m[1], 14)
        elevation_center = dem_px(px_center, 14)['elevation']
        conv_center = dem_convergence(px_center)
        tpi_center = dem_tpi(px_center)
    
        points = points_within_radius(gm.MetersToLatLon(m[0], m[1]), radius)[0]
        elevations = []
        convs = []
        tpis = []
        for mp in points:
            px = gm.MetersToPixels(mp[0], mp[1], 14)
            p = [int(round(px[0])), int(round(px[1]))]
            key = str(p[0]) + '_' + str(p[1])
            if not key in cache_e:
                cache_e[key] = dem_px(p, 14)['elevation']
            if not key in cache_c:
                cache_c[key] = dem_convergence(p)
            if not key in cache_t:
                cache_t[key] = dem_tpi(p)
            elevations.append(cache_e[key])
            convs.append(cache_c[key])
            tpis.append(cache_t[key])
        
        r['e'].append(percentile(elevations, elevation_center))
        r['c'].append(percentile(convs, conv_center))
        r['t'].append(percentile(tpis, tpi_center))
    
    return r
        
        
def dem_statistics(center, radius):
    gm = GlobalMercator()
    elevation_list = []

    slope_list = []

    z = best_zoom(center, radius, 14)
    wm_center = gm.LatLonToMeters(center[0], center[1])
    pixels = pixels_in_range(center, radius, z)

    for px_sample in pixels:
        dem_sample = dem_px(px_sample, z)
        elevation_list.append(dem_sample['elevation'])
        slope_list.append(dem_sample['slope'])
        
    return {'elevation': generate_stats(elevation_list), 'slope': generate_stats(slope_list)}

def dem_tpi(px_center):
    dem_cell = dem_px(px_center, 14)
    elevation_total = 0

    offset=2
    samples = math.pow(2*offset+1, 2) - 1
    for xoff in range(-1*offset,offset+1):
        for yoff in range(-1*offset,offset+1):
            if xoff != 0 or yoff != 0:
                dem_neighbor = dem_px([px_center[0] + xoff, px_center[1] + yoff], 14)
                elevation_total += dem_neighbor['elevation']
    
    elevation_avg = float(elevation_total) / samples
    return dem_cell['elevation'] - elevation_avg

def dem_convergence(px_center):
    total_conv = 0
    for xoff in range(-1, 2, 1):
        for yoff in range(-1, 2, 1):
            total_conv += dem_raw_convergence([px_center[0] + xoff, px_center[1] + yoff])
    return total_conv / 9

def dem_raw_convergence(px_center):
    dem_cell = dem_px(px_center, 14)
    elevation_total = 0
    dot_total = 0

    meters = gm.PixelsToMeters(px_center[0], px_center[1], 14)
    wm_resolution = gm.Resolution(14)
    wm_scalefactor = 1/math.cos(gm.MetersToLatLon(meters[0], meters[1])[0]*math.pi/180)
    
    offset=1
    samples = math.pow(2*offset+1, 2) - 1
    for xoff in range(-1*offset,offset+1):
        for yoff in range(-1*offset,offset+1):
            if xoff != 0 or yoff != 0:
                dem_neighbor = dem_px([px_center[0] + xoff, px_center[1] + yoff], 14)
                aspect_neighbor = dem_neighbor['aspect']*math.pi/180
                slope_neighbor = dem_neighbor['slope']*math.pi/180

                z1 = (dem_cell['elevation'] - dem_neighbor['elevation'])/3.28084
                x1 = -1*xoff*wm_resolution/wm_scalefactor
                y1 = -1*yoff*wm_resolution/wm_scalefactor
                length1 = math.sqrt(x1*x1 + y1*y1 + z1*z1)
                
                z2 = math.sin(math.pi/2 - slope_neighbor)
                x2 = math.sin(aspect_neighbor)*math.sin(slope_neighbor)
                y2 = math.cos(aspect_neighbor)*math.sin(slope_neighbor)
                length2 = math.sqrt(x2*x2 + y2*y2 + z2*z2)
                
                dot_total += (x1*x2 + y1*y2 + z1*z2) / (length1*length2)
    
    return float(dot_total) / samples
        
def is_drainage(px_center):
    dem_cell = dem_px(px_center, 14)
    dot_total = 0
    slope_total = 0

    offset=2
    for xoff in range(-1*offset,offset+1):
        for yoff in range(-1*offset,offset+1):
            if xoff != 0 or yoff != 0:
                dem_neighbor = dem_px([px_center[0] + xoff, px_center[1] + yoff], 14)
                theta = math.pi / 2 - math.atan2(-1*yoff, -1*xoff) - dem_neighbor['aspect']*math.pi/180 # y pixels are inverse compared to tile, i.e. +y = upward
                dot_total += math.cos(theta)
                slope_total += dem_neighbor['slope']
            
    dot_avg = float(dot_total) / 24.0
    slope_avg = float(slope_total) / 24.0
    return slope_avg >= 10 and dot_avg > 0.2
