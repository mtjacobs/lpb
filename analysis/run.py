import csv
import sys
from loader import *
from geo import distance
from dem import *
from fs import *
from nhd import *
from nlcd import *
from osm import *
from stats import *
from sql import *
from time import time
from tiles import clear_tile_cache
from globalmercator import GlobalMercator

filename = 'or-sample.csv'
if len(sys.argv) > 1:
    filename = sys.argv[1]
#data = load_isrid('data/' + filename)
data = load_processed(filename)

def csv2ll(col):
    try:
        return [float(col.split(',')[0].strip()), float(col.split(',')[1].strip())]
    except:
        return col
    
def fill_nodata(row, cols):
    for col in cols:
        row[col] = -999
        
def add_ranges(row, ranges):
    for range in ranges:
        m, l = points_within_radius(csv2ll(row['find']), range)
        row['m_' + str(range)] = m
        row['l_' + str(range)] = l

def add_ipp_ranges(row, ranges):
    for range in ranges:
        m, l = points_within_radius(csv2ll(row['ipp']), range)
        row['ipp_m_' + str(range)] = m
        row['ipp_l_' + str(range)] = l

def read_ranges(row, ranges):
    for range in ranges:
        if not row.has_key('m_' + str(range)):
            continue
        if not isinstance(row['m_' + str(range)], basestring):
            continue
        row['m_' + str(range)] = eval(row['m_' + str(range)])
        lls = []
        for m in row['m_' + str(range)]:
            lls.append(gm.MetersToLatLon(m[0], m[1]))
        row['l_' + str(range)] = lls

def refresh_dem(row, ranges):
    find = csv2ll(row['find'])
    dem_find = dem_ll(find)
    row['elevation'] = dem_find['elevation']
    row['slope'] = dem_find['slope']
    row['aspect'] = dem_find['aspect']

    gm = GlobalMercator()
    for range in ranges:
        key = str(range)
        elevations = []
        slopes = []
        aspects = []
        
        for m in row['m_' + key]:
            dem_sample = dem_px(gm.MetersToPixels(m[0], m[1], 14), 14)
            elevations.append(dem_sample['elevation'])
            slopes.append(dem_sample['slope'])
            aspects.append(dem_sample['aspect'])
        
        row['elevation_' + key] = elevations
        row['slope_' + key] = slopes
        row['aspect_' + key] = aspects
        
def refresh_terrain(row, ranges):
    find = csv2ll(row['find'])
    gm = GlobalMercator()
    m = gm.LatLonToMeters(find[0], find[1])
    p = gm.MetersToPixels(m[0], m[1], 14)
    row['tpi'] = dem_tpi(p)
    row['conv'] = dem_convergence(p)
    
    for range in ranges:
        key = str(range)
        tpis = []
        convs = []
        for m in row['m_' + key]:
            p = gm.MetersToPixels(m[0], m[1], 14)
            tpis.append(dem_tpi(p))
            convs.append(dem_convergence(p))
        row['tpi_' + key] = tpis
        row['conv_' + key] = convs
        
        
def refresh_terrain_pcts(row, ranges):
    for range in ranges:
        key = str(range)
        stats = dem_stats(row['m_' + key], min(range, 800))
        row['elevation_pct_' + key] = stats['e']
        row['tpi_pct_' + key] = stats['t']
        row['conv_pct_' + key] = stats['c']
    
def refresh_nlcd(row, ranges):
    find = csv2ll(row['find'])
    nlcd_find = nlcd_ll(find)
    row['canopy'] = nlcd_find['canopy']
    row['landcover'] = nlcd_find['landcover']

    gm = GlobalMercator()
    for range in ranges:
        key = str(range)
        canopies = []
        landcovers = []
        for m in row['m_' + key]:
            nlcd_sample = nlcd_px(gm.MetersToPixels(m[0], m[1], 12), 12)
            canopies.append(nlcd_sample['canopy'])
            landcovers.append(nlcd_sample['landcover'])

        row['canopy_' + key] = canopies
        row['landcover_' + key] = landcovers

def refresh_ipp(row):
    ipp = csv2ll(row['ipp'])
    ranges = (800,)

    table1, where1 = get_osm_where('road')
    table2, where2 = get_fsline_where('road')
    row['ipp_offset_road'] = get_offset_linear(ipp, [{'table': table1, 'where': where1, 'col': 'way'}, {'table': table2, 'where': where2, 'col': 'geom'}])

    table3, where3 = get_osm_where('trail')
    table4, where4 = get_fsline_where('trail')
    row['ipp_offset_trail'] = get_offset_linear(ipp, [{'table': table3, 'where': where3, 'col': 'way'}, {'table': table4, 'where': where4, 'col': 'geom'}])

    row['ipp_offset_lake'] = get_nhd_offset(ipp, 'lake')
    row['ipp_offset_river'] = get_osm_offset(ipp, 'river')
    row['ipp_offset_stream'] = get_nhd_offset(ipp, 'stream')
    row['ipp_offset_bigstream'] = get_nhd_offset(ipp, 'bigstream')
    row['ipp_offset_drainage'] = get_nhd_offset(ipp, 'drainage')

    dem_ipp = dem_ll(ipp)
    row['ipp_elevation'] = dem_ipp['elevation']

    nlcd_ipp = nlcd_ll(ipp)
    row['ipp_canopy'] = nlcd_ipp['canopy']
    row['ipp_landcover'] = nlcd_ipp['landcover']

    gm = GlobalMercator()
    m = gm.LatLonToMeters(ipp[0], ipp[1])
    p = gm.MetersToPixels(m[0], m[1], 14)
    row['ipp_tpi'] = dem_tpi(p)
    row['ipp_conv'] = dem_convergence(p)

    for range in ranges:
        key = str(range)
        elevations = []
        tpis = []
        convs = []
        
        for m in row['ipp_m_' + key]:
            dem_sample = dem_px(gm.MetersToPixels(m[0], m[1], 14), 14)
            elevations.append(dem_sample['elevation'])

            p = gm.MetersToPixels(m[0], m[1], 14)
            tpis.append(dem_tpi(p))
            convs.append(dem_convergence(p))
        
        row['ipp_elevation_' + key] = elevations
        row['ipp_tpi_' + key] = tpis
        row['ipp_conv_' + key] = convs

        stats = dem_stats(row['ipp_m_' + key], min(range, 800))
        row['ipp_elevation_pct_' + key] = stats['e']
        row['ipp_tpi_pct_' + key] = stats['t']
        row['ipp_conv_pct_' + key] = stats['c']
    


def refresh_manmade_linear(row, ranges):
    find = csv2ll(row['find'])

    start = time()
    
    table1, where1 = get_osm_where('road')
    table2, where2 = get_fsline_where('road')
    row['offset_road'] = get_offset_linear(find, [{'table': table1, 'where': where1, 'col': 'way'}, {'table': table2, 'where': where2, 'col': 'geom'}])

    table3, where3 = get_osm_where('trail')
    table4, where4 = get_fsline_where('trail')
    row['offset_trail'] = get_offset_linear(find, [{'table': table3, 'where': where3, 'col': 'way'}, {'table': table4, 'where': where4, 'col': 'geom'}])
    
    for range in ranges:
        key = str(range)
        lls = row['l_' + key]
        meters = row['m_' + key]
        
        row['offset_road_' + key] = get_offsets_linear(lls, [{'table': table1, 'where': where1, 'col': 'way'}, {'table': table2, 'where': where2, 'col': 'geom'}])
        row['offset_trail_' + key] = get_offsets_linear(lls, [{'table': table3, 'where': where3, 'col': 'way'}, {'table': table4, 'where': where4, 'col': 'geom'}])

        row['offset_road_pct_' + key] = sql_stats(meters, min(range, 800), [{'table': table1, 'where': where1, 'col': 'way'}, {'table': table2, 'where': where2, 'col': 'geom'}])
        row['offset_trail_pct_' + key] = sql_stats(meters, min(range, 800), [{'table': table3, 'where': where3, 'col': 'way'}, {'table': table4, 'where': where4, 'col': 'geom'}])
    
def refresh_water(row, ranges):
    find = csv2ll(row['find'])

    row['offset_lake'] = get_nhd_offset(find, 'lake')
    row['offset_river'] = get_osm_offset(find, 'river')
    row['offset_stream'] = get_nhd_offset(find, 'stream')
    row['offset_bigstream'] = get_nhd_offset(find, 'bigstream')
    row['offset_drainage'] = get_nhd_offset(find, 'drainage')
    
    for range in ranges:
        key = str(range)
        lls = row['l_' + key]
        meters = row['m_' + key]
        
        row['offset_lake_' + key] = get_nhd_offsets(lls, 'lake')
        row['offset_river_' + key] = get_osm_offsets(lls, 'river')
        row['offset_stream_' + key] = get_nhd_offsets(lls, 'stream')
        row['offset_bigstream_' + key] = get_nhd_offsets(lls, 'bigstream')
        row['offset_drainage_' + key] = get_nhd_offsets(lls, 'drainage')
        
        row['offset_stream_pct_' + key] = get_nhd_stats(meters, 'stream')
        row['offset_drainage_pct_' + key] = get_nhd_stats(meters, 'drainage')

def refresh_water_pcts(row, ranges):
    find = csv2ll(row['find'])
    
    for range in ranges:
        key = str(range)
        lls = row['l_' + key]
        meters = row['m_' + key]
        
        row['offset_stream_pct_' + key] = get_nhd_stats(meters, 'stream')

gm = GlobalMercator()
raster_ranges = [250,800,2000]
sql_ranges = [800,2000]
ranges = raster_ranges
for r in sql_ranges:
    if not r in ranges:
        ranges.append(r)
max_range = 2000

points = []

# a = average, p = percentile, d = decimal, f = from find location only, s = search area, between IPP and find location 
keys = ['id','ipp','find','type','terrain','category','age','sex','status','time','weather','ecoregion','manhours','signalling','elevation','slope','aspect','tpi','conv','canopy','landcover','offset_road','offset_trail','offset_lake','offset_river','offset_bigstream','offset_stream','offset_drainage','comments']
keys.extend(['ipp_elevation','ipp_tpi','ipp_conv','ipp_canopy','ipp_landcover','ipp_offset_road','ipp_offset_trail','ipp_offset_lake','ipp_offset_river','ipp_offset_bigstream','ipp_offset_stream','ipp_offset_drainage'])

for range in ranges:
    keys.append('m_' + str(range))
        
for range in raster_ranges:
    key = str(range)
    keys.extend(['elevation_' + key, 'slope_' + key, 'aspect_' + key, 'tpi_' + key, 'conv_' + key, 'elevation_pct_' + key, 'tpi_pct_' + key, 'conv_pct_' + key, 'canopy_' + key, 'landcover_' + key])
    if range == 800:
        keys.extend(['ipp_elevation_' + key, 'ipp_tpi_' + key, 'ipp_conv_' + key, 'ipp_elevation_pct_' + key, 'ipp_tpi_pct_' + key, 'ipp_conv_pct_' + key])
    
for range in sql_ranges:
    key = str(range)
    keys.extend(['offset_road_' + key, 'offset_trail_' + key, 'offset_lake_' + key, 'offset_river_' + key, 'offset_bigstream_' + key, 'offset_stream_' + key, 'offset_stream_pct_' + key, 'offset_drainage_' + key])
        
writer = csv.DictWriter(sys.stdout, keys, delimiter='\t', quotechar='"', extrasaction='ignore')
writer.writer.writerow(keys)

for row in data:
    if row['find'] == None:
        continue
        
    add_ranges(row, [250,800,2000])
    read_ranges(row, ranges)

    refresh_dem(row, raster_ranges)
    refresh_terrain(row, raster_ranges)
    refresh_terrain_pcts(row, raster_ranges)
    refresh_nlcd(row, raster_ranges)
    refresh_water(row, sql_ranges)
    refresh_manmade_linear(row, sql_ranges)
    refresh_water_pcts(row, sql_ranges)

    if row['ipp'] != None and len(row['ipp']) > 0:
        add_ipp_ranges(row, [800,])
        refresh_ipp(row)
        
    if isinstance(row['find'], list):
        row['find'] = str(row['find'][0]) + ', ' + str(row['find'][1])
    if isinstance(row['ipp'], list):
        row['ipp'] = str(row['ipp'][0]) + ', ' + str(row['ipp'][1])
    clear_tile_cache()
    writer.writerow(row)
