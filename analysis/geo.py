import math

def distance(fr, to):
    dLat = (to[0]-fr[0])/180*math.pi
    dLon = (to[1]-fr[1])/180*math.pi
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(fr[0]/180*math.pi) * math.cos(to[0]/180*math.pi) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return c*6378137