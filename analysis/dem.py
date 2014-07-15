import json
import math
import os
import struct
import urllib

from globalmercator import GlobalMercator
from osgeo import gdal

dem_cache = {'slope': [None]*15, 'aspect': [None]*15, 'elevation': [None]*15}

def fetch_cache_tile(layer, z, x, y):
    dest = "cache/dem/" + layer + "/" + str(z) + "/" + str(x) + "/" + str(y) + ".png"
    if not os.path.exists(dest):
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        res = urllib.urlretrieve("http://s3-us-west-1.amazonaws.com/ctslope/" + layer + "/" + str(z) + "/" + str(x) + "/" + str(y) + ".png", dest)
        if not(res[1].type=="image/png"):
            os.remove(dest)
    
    return gdal.Open(dest, gdal.GA_ReadOnly)

def fetch_grid(layer, z, x, y):
    try:
        cached = dem_cache[layer][z][x][y]
        if cached != None:
            return cached
    except:
        pass
    tile = fetch_cache_tile(layer, z, x, y)
    band = tile.GetRasterBand(1)
    data = []
    for xpixel in range(0, tile.RasterXSize):
        data.append(struct.unpack('i' * tile.RasterYSize, band.ReadRaster(xpixel, 0, 1, tile.RasterYSize, 1, tile.RasterYSize, gdal.GDT_Int32)))
        
    if layer == 'aspect':
        adjusted_data = []
        for px in range(0, tile.RasterXSize):
            data[px] = [int(data[px][py]*1.5) for py in range(0, tile.RasterYSize)]

    if dem_cache[layer][z] == None:
        dem_cache[layer][z] = [None]*(2**z)
    if dem_cache[layer][z][x] == None:
        dem_cache[layer][z][x] = [None]*(2**z)
    dem_cache[layer][z][x][y] = data

    return data
    
def get_pixel(layer, z, px, py):
    tx = px / 256.0
    ty = py / 256.0
    
    ty = 2**z - ty
    px = int(min(round((tx-math.floor(tx))*256), 255))
    py = int(min(round((ty-math.floor(ty))*256), 255))
    
    tx = int(math.floor(tx))
    ty = int(math.floor(ty))
    
    grid = fetch_grid(layer, z, tx, ty)
    return grid[px][py]

def dem_px(p):
    return {'elevation': get_pixel('elevation', 14, p[0], p[1]), 'slope': get_pixel('slope', 14, p[0], p[1]), 'aspect': get_pixel('aspect', 14, p[0], p[1])}

def dem_ll(coord):
    gm = GlobalMercator()
    m = gm.LatLonToMeters(coord[0], coord[1])
    p = gm.MetersToPixels(m[0], m[1], 14)
    return dem_px(p)
    
def dem2(coord):
    res = urllib.urlretrieve("http://caltopo.com/resource/dem?locations=" + str(coord[0]) + "," + str(coord[1]), "elevation.json")
    json_data=open("elevation.json")
    data = json.load(json_data)
    os.remove("elevation.json")
    
    return {'elevation': data["results"][0]["elevation"], 'slope': data["results"][0]["slope"], 'aspect': data["results"][0]["aspect"]}

def delta_elevation(start, end):
    m_start = dem_ll(start)['elevation']
    m_end = dem_ll(end)['elevation']
    
    return (m_end-m_start)*3.28084
    
def pixels_in_range(center, radius):
    gm = GlobalMercator()
    wm_radius = radius / math.cos(center[0]*math.pi/180) # scale radius to web mercator meters
    wm_center = gm.LatLonToMeters(center[0], center[1])
    px_center = gm.MetersToPixels(wm_center[0], wm_center[1], 14)
    px_radius = wm_radius / gm.Resolution(14)
    
    pixels = []
    for dx in range(-1*int(px_radius), int(px_radius)+1):
        for dy in range(-1*int(px_radius), int(px_radius)+1):
            if math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)) <= px_radius:
               pixels.append([px_center[0] + dx, px_center[1] + dy]) 

    return pixels

def statistics(center, radius):
    slope_total = 0
    slope_range = [99999, 0]
    elevation_total = 0
    elevation_range = [99999, 0]    

    pixels = pixels_in_range(center, radius)
    for px_sample in pixels:
        dem_sample = dem_px(px_sample)
        elevation_total += dem_sample['elevation']
        elevation_range[0] = min(elevation_range[0], dem_sample['elevation'])
        elevation_range[1] = max(elevation_range[1], dem_sample['elevation'])
        slope_total += dem_sample['slope']
        slope_range[0] = min(slope_range[0], dem_sample['slope'])
        slope_range[1] = max(slope_range[1], dem_sample['slope'])
        
    slope_total = slope_total / len(pixels)
    elevation_total = elevation_total / len(pixels)
    
    return {'elevation': (elevation_total, elevation_range), 'slope': (slope_total, slope_range)}
        
#coord=[36.5788,-118.2922]
#stats = statistics(coord, 500)
#print stats