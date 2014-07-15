from loader import load
from geo import distance
from dem import dem_ll
from dem import delta_elevation
from dem import statistics
from globalmercator import GlobalMercator

data = load('../data/or-sample.csv')

gm = GlobalMercator()

print "ID\tIPP\tFind\tDistance\tElevation Change (m)\tElevation Change (pct)\tSlope (deg)\tSlope (pct)"
for row in data:
    if row['ipp'] != None:
        wm_ipp = gm.LatLonToMeters(row['ipp'][0], row['ipp'][1])
        wm_find = gm.LatLonToMeters(row['find'][0], row['find'][1])
        
        wm_center = [(wm_ipp[0] + wm_find[0])/2.0, (wm_ipp[1] + wm_find[1])/2.0] 
        ll_center = gm.MetersToLatLon(wm_center[0], wm_center[1])

        m_distance = distance(row['ipp'], row['find'])
        dem_find = dem_ll(row['find'])

        m_elevation_ipp = dem_ll(row['ipp'])['elevation']
        m_elevation_find = dem_find['elevation']
        m_elevation_delta = m_elevation_find - m_elevation_ipp
        
        deg_slope_find = dem_find['slope']
        
        if m_distance > 2000:
            continue

        dem_stats = statistics(ll_center, m_distance*0.5)
        
        m_elevation_available = 1000
        if m_elevation_delta > 0:
            m_elevation_available = dem_stats['elevation'][1][1] - m_elevation_ipp
        if m_elevation_delta < 0:
            m_elevation_available = m_elevation_ipp - dem_stats['elevation'][1][0]
        
        pct_elevation = 0
        if m_elevation_available != 0:
            pct_elevation = float(m_elevation_delta) / float(m_elevation_available)
        
        pct_slope_find = 0
        if dem_stats['slope'][0] > 0:
            pct_slope_find = float(deg_slope_find) / float(dem_stats['slope'][0])
        
        print row['id'] + "\t" + "%.4f"%row['ipp'][0] + ", " + "%.4f"%row['ipp'][1] + "\t" + "%.4f"%row['find'][0] + ", " + "%.4f"%row['find'][1] + "\t" + \
            str(int(m_distance)) + "\t" + str(int(m_elevation_delta)) + "\t" + "%.2f"%pct_elevation + "\t" + str(deg_slope_find) + "\t" + "%.2f"%pct_slope_find