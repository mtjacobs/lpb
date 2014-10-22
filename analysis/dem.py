import json
import math
import os
import struct
import urllib

from globalmercator import GlobalMercator
from osgeo import gdal

from stats import *
from tiles import *

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
    
def fill_grid(grid, px, val, px_radius):
    for dx in range(-1*px_radius,px_radius+1):
        for dy in range(-1*px_radius,px_radius+1):
            if math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)) > px_radius:
                next
            try:
                if grid[px[0] + dx][px[1] + dy] == None:
                    grid[px[0] + dx][px[1] + dy] = val
            except IndexError:
                pass
    
def drainage_percentile(center, radius):
    gm = GlobalMercator()
    scale_factor = math.cos(center[0]*math.pi/180)
    offset_drainage = get_drainage_offset(center)
 
    z = best_zoom(center, radius, 14)
    resolution=gm.Resolution(z)
    wm_radius = radius / math.cos(center[0]*math.pi/180) # scale radius to web mercator meters
    wm_center = gm.LatLonToMeters(center[0], center[1])
    px_center = gm.MetersToPixels(wm_center[0], wm_center[1], z)
    px_radius = int(wm_radius / gm.Resolution(z))
        
    t = 0
    f = 0
#    print px_radius, z, offset_drainage
    for dx in range(-1*px_radius,px_radius+1):
        for dy in range(-1*px_radius,px_radius+1):
            if math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)) > px_radius:
                next
            
            wm_sample = gm.PixelsToMeters(px_center[0] + dx, px_center[1] + dy, z);
            offset = get_drainage_offset(gm.MetersToLatLon(wm_sample[0], wm_sample[1]), limit=offset_drainage+20)
            
            if offset <= offset_drainage:
                t+=1
            else:
                f+=1
    
#    print float(t) / (t+f)
    return float(t) / (t + f)
    
def get_drainage_offset(ll, limit=800):
    gm = GlobalMercator()
    scale_factor = math.cos(ll[0]*math.pi/180)
    
    wm_center = gm.LatLonToMeters(ll[0], ll[1])
    px_center = gm.MetersToPixels(wm_center[0], wm_center[1], 14)
    
    return get_drainage_offset_px(px_center, scale_factor, limit)
    
def get_drainage_offset_px(px_center, scale_factor, limit):
    resolution=9.5546285356470317
    px_radius = int(limit / (scale_factor * resolution))

    offset = px_radius
    for dx in range(0, px_radius):
        for dy in range(0, px_radius):
            offset_sample = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
            for xsign in (-1, 1):
                for ysign in (-1, 1):
                    if offset_sample < offset:
                        if is_drainage([px_center[0] + dx*xsign, px_center[1] + dy*ysign]):
                            offset = offset_sample

    return offset * resolution * scale_factor
    
def is_drainage(px_center):
    dem_cell = dem_px(px_center, 14)
    dot_total = 0
    slope_total = 0

    for xoff in range(-2,3):
        for yoff in range(-2,3):
            if xoff != 0 or yoff != 0:
                dem_neighbor = dem_px([px_center[0] + xoff, px_center[1] + yoff], 14)
                theta = math.pi / 2 - math.atan2(-1*yoff, -1*xoff) - dem_neighbor['aspect']*math.pi/180 # y pixels are inverse compared to tile, i.e. +y = upward
                dot_total += math.cos(theta)
                slope_total += dem_neighbor['slope']
            
    dot_avg = float(dot_total) / 24.0
    slope_avg = float(slope_total) / 24.0
    return slope_avg >= 10 and dot_avg > 0.2
