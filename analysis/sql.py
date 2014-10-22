import math
import psycopg2
from time import time

from tiles import *
from globalmercator import GlobalMercator

max_range = 2000

dbconn = psycopg2.connect("dbname='osm' user='postgres' host='localhost' password='postgres' port='5433'")

def dedupe(ll_list):
    points = []
    
    for lls in ll_list:
        if len(points) == 0:
            points.extend(lls)
            next
        
        for ll in lls:
            match = False
            for p in points:
                if abs(ll[0] - p[0]) < 0.00001 and abs(ll[1] - p[1]) < 0.00001:
                    match = True
            if not(match):
                p.append(ll)

    return points

def get_pct(lls, filtered, nodata=False):
    if len(filtered) == 0 and nodata:
        return -999
    return float(len(filtered)) / len(lls)
    

def meters_to_degrees(lat, meters):
    rlat = lat*math.pi/180
    mperdeg = 111412.84 * math.cos(rlat) - 93.5 * math.cos(3*rlat)
    return meters / mperdeg
    
def get_offset_linear(ll, table, where, geom_col, cursor=None):
    if cursor == None:
        cursor = dbconn.cursor()
        
    increment = 1.2 * meters_to_degrees(ll[0], max_range)

    sql = "select least(min(st_distance(st_setsrid(st_point(" + str(ll[1]) + ", " + str(ll[0]) + "), 4326)::geography, " + geom_col + "::geography)), 2000)::int as dist from " + table + " where "
    if where != None:
        sql += "(" + where + ") and "
    sql += geom_col + " && st_makebox2d(st_setsrid(st_point(" + str(ll[1]-increment) + ", " + str(ll[0]-increment) + "), 4326), st_setsrid(st_point(" + str(ll[1]+increment) + ", " + str(ll[0]+increment) + "), 4326))"
    
    cursor.execute(sql)
    rows = cursor.fetchall()
    if len(rows) == 0:
        return max_range
    if rows[0][0] == None:
        return max_range
    return min(max_range, rows[0][0])
    

def get_offsets_linear(lls, table, where, geom_col, cursor=None):
    if cursor == None:
        cursor = dbconn.cursor()

    increment = 1.2 * meters_to_degrees(lls[0][0], max_range)

    sql = "select s.idx, least(min(st_distance(s.g, t." + geom_col + "::geography)), " + str(max_range) + ")::int from ("
    lat_range = [lls[0][0], lls[0][0]]
    lng_range = [lls[0][1], lls[0][1]]
    idx = 0
    for i in range(0, len(lls)):
        ll = lls[i]
        lat_range = [min(lat_range[0], ll[0]), max(lat_range[1], ll[0])]
        lng_range = [min(lng_range[0], ll[1]), max(lng_range[1], ll[1])]
        sql += "select " + str(idx) + " as idx, " + str(ll[0]) + " as lat, " + str(ll[1]) + " as lng, st_setsrid(st_point(" + str(ll[1]) + ", " + str(ll[0]) + "), 4326)::geography as g "
        if i < len(lls) - 1:
            sql += "union "
        idx += 1
    sql += ") s left outer join " + table + " t on "
    if where != None:
        sql += "(" + where + ") and "
    sql += "t." + geom_col + " && st_makebox2d(st_setsrid(st_point(" + str(lng_range[0]-increment) + ", " + str(lat_range[0]-increment) + "), 4326), st_setsrid(st_point(" + str(lng_range[1]+increment) + ", " + str(lat_range[1]+increment) + "), 4326)) "
    sql += "group by s.idx order by s.idx asc"
    
    results = []
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        results.append(row[1])
    
    return results
