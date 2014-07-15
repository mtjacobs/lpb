import math

class GlobalMercator(object):

    def __init__(self, tileSize=256):
        self.tileSize = tileSize
        self.initialResolution = 2 * math.pi * 6378137 / self.tileSize
        self.originShift = 2 * math.pi * 6378137 / 2.0

    def LatLonToMeters(self, lat, lon ):
        mx = lon * self.originShift / 180.0
        my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)
        my = my * self.originShift / 180.0
        return mx, my

    def MetersToLatLon(self, mx, my ):
        lon = (mx / self.originShift) * 180.0
        lat = (my / self.originShift) * 180.0
        lat = 180 / math.pi * (2 * math.atan( math.exp( lat * math.pi / 180.0)) - math.pi / 2.0)
        return lat, lon

    def PixelsToMeters(self, px, py, zoom):
        res = self.Resolution( zoom )
        mx = px * res - self.originShift
        my = py * res - self.originShift
        return mx, my
        
    def MetersToPixels(self, mx, my, zoom):
        res = self.Resolution( zoom )
        px = (mx + self.originShift) / res
        py = (my + self.originShift) / res
        return px, py
    
    def PixelsToTile(self, px, py):
        tx = int( math.ceil( px / float(self.tileSize) ) - 1 )
        ty = int( math.ceil( py / float(self.tileSize) ) - 1 )
        return tx, ty
        
    def MetersToTile(self, mx, my, zoom):
        px, py = self.MetersToPixels( mx, my, zoom)
        return self.PixelsToTile( px, py)

    def Resolution(self, zoom ):
        return self.initialResolution / (2**zoom)
        
    def GoogleTile(self, tx, ty, zoom):
        return tx, (2**zoom - 1) - ty
