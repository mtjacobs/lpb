import math
import psycopg2
import shapefile

conn = psycopg2.connect("dbname='osm' user='postgres' host='localhost' password='postgres' port='5433'")
cursor = conn.cursor()

def import_waterbody():
    sf=shapefile.Reader("waterbody")
    shapes=sf.iterShapes()
    records=sf.iterRecords()

    for i in range(0, sf.numRecords):
        shape = shapes.next()
        record = records.next()

        geometry = []
        for point in shape.points:
            geometry.append(str(point[0]) + " " + str(point[1]))

        cursor.execute("insert into nhd (type, geom, ftype, fcode) values('waterbody', st_geomfromtext('LINESTRING(" + ", ".join(geometry) + ")', 4326), " + str(record[0]) + ", " + str(record[1]) + ")")
    
    conn.commit()
    print "Water bodies done . . ."

def import_flowlines():
    sf=shapefile.Reader("flowline")
    shapes=sf.iterShapes()
    records=sf.iterRecords()

    for i in range(0, sf.numRecords):
        shape = shapes.next()
        record = records.next()

        geometry = []
        for point in shape.points:
            geometry.append(str(point[0]) + " " + str(point[1]))

        cursor.execute("insert into nhd (type, geom, ftype, fcode) values('flowline', st_geomfromtext('LINESTRING(" + ", ".join(geometry) + ")', 4326), " + str(record[0]) + ", " + str(record[1]) + ")")
    conn.commit()
    print "Flow lines done . . ."

import_waterbody()
#import_flowlines()

def linestring_to_coords(ls):
    text_coords = ls.replace("LINESTRING(", "").replace(")","").split(",")
    coords = []
    for tc in text_coords:
        coords.append([float(tc.split(" ")[0]), float(tc.split(" ")[1])])
    return coords
    
def coords_to_linestring(coords):
    as_strs = []
    for c in coords:
        as_strs.append(str(c[0]) + ' ' + str(c[1]))
    
    return 'LINESTRING(' + ','.join(as_strs) + ')'

def get_flowline(id):
    if id == None:
        return None
    cursor.execute("select id, st_astext(geom) from nhd where type='flowline' and id=%s", [str(id)])
    row = cursor.fetchall()[0]
    return {'id': id, 'geom': linestring_to_coords(row[1])}
    
def find_upstream_flowlines(id):
    cursor.execute("select n2.id from nhd n1, nhd n2 where n1.id=%s and n2.id!=%s and n2.type='flowline' and st_touches(st_pointn(n1.geom, 1), n2.geom)", [str(id), str(id)])
    rows = cursor.fetchall()
    ids = []
    for row in rows:
        ids.append(row[0])
    return ids

def join_flowlines():
    cursor.execute("select nhd1.id, count(*) as cnt from nhd nhd1, nhd nhd2 where nhd1.type='flowline' and nhd2.type='flowline' and nhd1.id != nhd2.id and st_touches(st_pointn(nhd1.geom, 1), nhd2.geom) group by nhd1.id order by cnt asc")
    rows = cursor.fetchall()
    for row in rows:
        if row[1] == 1:
            downstream_id = row[0]
            up = find_upstream_flowlines(row[0])
            if len(up) != 1:
                print "WARNING!  " + str(len(up)) + " lines upstream of " + str(downstream_id)
                continue
            upstream_id = up[0]
            print "joining " + str(downstream_id) + " with " + str(upstream_id)
            downstream_line = get_flowline(downstream_id)
            upstream_line = get_flowline(upstream_id)
            
            upstream_line['geom'].extend(downstream_line['geom'])
            
            cursor.execute("delete from nhd where id=%s and type='flowline'", [str(downstream_id)])
            cursor.execute("update nhd set geom=st_geomfromtext('" + coords_to_linestring(upstream_line['geom']) + "', 4326) where id=%s and type='flowline'", [str(upstream_id)])
            conn.commit()

"""
update nhd set upstream=0;

update nhd n set upstream=1 from (
select n1.id as id, count(*) as cnt from nhd n1, nhd n2 where n1.type='flowline' and n2.type='flowline' and n1.id != n2.id and st_touches(st_pointn(n1.geom, 1), n2.geom) group by n1.id
) q where n.id=q.id;

update nhd n set upstream=2 from (
select n1.id as id, count(*) as cnt from nhd n1, nhd n2 where n1.type='flowline' and n2.type='flowline' and n2.upstream=1 and n1.id != n2.id and st_touches(st_pointn(n1.geom, 1), n2.geom) group by n1.id
) q where n.upstream=1 and n.id=q.id;

update nhd n set upstream=3 from (
select n1.id as id, count(*) as cnt from nhd n1, nhd n2 where n1.type='flowline' and n2.type='flowline' and n2.upstream=2 and n1.id != n2.id and st_touches(st_pointn(n1.geom, 1), n2.geom) group by n1.id
) q where n.upstream=2 and n.id=q.id;

update nhd n set upstream=4 from (
select n1.id as id, count(*) as cnt from nhd n1, nhd n2 where n1.type='flowline' and n2.type='flowline' and n2.upstream=3 and n1.id != n2.id and st_touches(st_pointn(n1.geom, 1), n2.geom) group by n1.id
) q where n.upstream=3 and n.id=q.id;
""""