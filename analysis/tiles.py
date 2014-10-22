import json
import math
import os
import struct
import urllib

from globalmercator import GlobalMercator
from osgeo import gdal

tile_cache = {}

def clear_tile_cache():
    tile_cache.clear()

def fetch_cache_tile(url, z, x, y):
    layer = url.split("/")[-1]
    dest = "cache/" + layer + "/" + str(z) + "/" + str(x) + "/" + str(y) + ".png"
    if not os.path.exists(dest):
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        res = urllib.urlretrieve(url + "/" + str(z) + "/" + str(x) + "/" + str(y) + ".png", dest)
        if not(res[1].type=="image/png"):
            os.remove(dest)
    
    return gdal.Open(dest, gdal.GA_ReadOnly)

def fetch_cache_grid(url, z, x, y):
    try:
        cached = tile_cache[url][z][x][y]
        if cached != None:
            return cached
    except:
        pass

    tile = fetch_cache_tile(url, z, x, y)
    data = []
    try:
        band = tile.GetRasterBand(1)
        for xpixel in range(0, tile.RasterXSize):
            data.append(struct.unpack('i' * tile.RasterYSize, band.ReadRaster(xpixel, 0, 1, tile.RasterYSize, 1, tile.RasterYSize, gdal.GDT_Int32)))
    except:
        # near the coast, some tiles won't exist.  use 0s
        for xpixel in range(0, 256):
            data.append([0]*256)
        
    if url.split("/")[-1] == 'aspect':
        for px in range(0, 256):
            data[px] = [int(data[px][py]*1.5) for py in range(0, 256)]
    
    if url.split("/")[-1] == 'elevation':
        for px in range(0, 256):
            data[px] = [int(data[px][py]*3.28084) for py in range(0, 256)] # convert to feet

    if not(tile_cache.has_key(url)):
        tile_cache[url] = [None]*16
    if tile_cache[url][z] == None:
        tile_cache[url][z] = [None]*(2**z)
    if tile_cache[url][z][x] == None:
        tile_cache[url][z][x] = [None]*(2**z)
    tile_cache[url][z][x][y] = data

    return data

def get_pixel(url, z, px, py):
    tx = px / 256.0
    ty = py / 256.0
    
    ty = 2**z - ty
    px = int(min(round((tx-math.floor(tx))*256), 255))
    py = int(min(round((ty-math.floor(ty))*256), 255))
    
    tx = int(math.floor(tx))
    ty = int(math.floor(ty))
    
    grid = fetch_cache_grid(url, z, tx, ty)
    return grid[px][py]

def points_within_radius(center, radius):
    wm_radius = radius / math.cos(center[0]*math.pi/180)
    gm = GlobalMercator()
    z = 16
    while gm.Resolution(z) < wm_radius / 10:
        z-=1

    wm_center = gm.LatLonToMeters(center[0], center[1])
    px_center = gm.MetersToPixels(wm_center[0], wm_center[1], z)
    px_radius = wm_radius / gm.Resolution(z)
    
    meters = []
    lls = []
    for dx in range(-1*int(px_radius), int(px_radius)+1):
        for dy in range(-1*int(px_radius), int(px_radius)+1):
            if (dx != 0 or dx != 0) and math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)) <= px_radius:
                m = gm.PixelsToMeters(px_center[0] + dx, px_center[1] + dy, z)
                meters.append(m)
                lls.append(gm.MetersToLatLon(m[0], m[1]))

    return meters, lls
    