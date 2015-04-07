import math
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

def percentagebycolumn(dataset, col):
    labels=[]
    pcts=[]
    for key, grp in dataset.groupby(col):
        labels.append(key)
        pcts.append(float(len(grp))/len(dataset))
    return labels, pcts

def piechart(dataset, col, table=False):
    a,b=percentagebycolumn(dataset, col)
    if not table:
        plt.pie(pd.DataFrame(b),labels=a)
        return
    
    for i in range(0, len(a)):
        print a[i] + '\t' + '%.2f' % b[i]

colors = {
    True: "r",
    False: "b",
}

def distinct(df, col):
    values = []
    for val in df[col]:
        if not val in values:
            values.append(val)
    values.sort()
    return values
    
def nonan(l):
    return [x for x in l if not math.isnan(x)]

def coalesce(data):
    values = []
    for row in data:
        for val in row:
            if val != -999:
                values.append(val)
    return values
    
def hist_dual(series1, series2, bins=20, normal=True, histtype='stepfilled', alpha=0.4, name='', s1='Series 1', s2='Series 2'):
    f=plt.figure()
    plt.title(name)
    bins = numpy.linspace(min( min(nonan(series1)), min(nonan(series1))), max( max(nonan(series1)), max(nonan(series2))), bins)
    plt.hist(series1, bins, alpha=alpha, label=s1,normed=normal,histtype=histtype)
    plt.hist(series2, bins, alpha=alpha, label=s2,normed=normal,histtype=histtype)
    plt.legend(loc='upper right')
    plt.show()
    
    
def hist_by(df, col, by, bins=20, normal=True, histtype='stepfilled', alpha=0.4, name=''):
    f=plt.figure()
    plt.title(name + ' ' + col + ' by ' + str(by))
    bins = numpy.linspace(min(nonan(df[col].values)), max(nonan(df[col].values)), bins)
    for val in distinct(df, by):
        plt.hist(df[df[by]==val][col].values, bins, alpha=alpha, label=str(val),normed=normal,histtype=histtype)
    plt.legend(loc='upper right')
    plt.show()
    
def to_bar_plot(series):
    keys = series.keys()
    data = []
    offsets = []
    offsets.append(0)
    data.append(series[keys[0]])
    for i in range(0, len(keys)):
        key = keys[i]
        if i > 0:
            offsets.append(keys[i-1]+0.0001)
            data.append(series[key])
        offsets.append(key)
        data.append(series[key])
    
    return pd.Series(data, index=offsets)
    
def eq(df, attr, values):
    # cannot figure this out
    if isinstance(values, str) or isinstance(values, bool) or isinstance(values, numpy.bool_):
        return df[df[attr]==values]
    if len(values) == 1:
        return df[df[attr]==values[0]]
    if len(values) == 2:
        return df[(df[attr]==values[0]) | (df[attr]==values[1])]
    if len(values) == 3:
        return df[(df[attr]==values[0]) | (df[attr]==values[1]) | (df[attr]==values[2])]
    if len(values) == 4:
        return df[(df[attr]==values[0]) | (df[attr]==values[1]) | (df[attr]==values[2]) | (df[attr]==values[3])]
    if len(values) == 5:
        return df[(df[attr]==values[0]) | (df[attr]==values[1]) | (df[attr]==values[2]) | (df[attr]==values[3]) | (df[attr]==values[4])]
    if len(values) == 6:
        return df[(df[attr]==values[0]) | (df[attr]==values[1]) | (df[attr]==values[2]) | (df[attr]==values[3]) | (df[attr]==values[4]) | (df[attr]==values[4])]

def gt(df, attr, value):
    return df[df[attr] > value]
    
def lt(df, attr, value):
    return df[df[attr] <= value]
    
def within(col, value):
    return map(lambda(x): percentile(x, value), col)
    
def pref(df, col1, col2, offset, buf=None, inv=False):
    pct_area = sum(within(df[col2], offset))
    pct_finds = len(lt(df, col1, offset))
    if buf != None:
        pct_area = pct_area - sum(within(df[col2], offset-buf))
        pct_finds = pct_finds - len(lt(df, col1, offset-buf))
    elif inv:
        pct_area = len(df) - pct_area
        pct_finds = len(df) - pct_finds
    return pct_finds / float(pct_area)

def pref_prime(df, col1, col2, offset, buf=None, inv=False):
    pct_area = sum(within(df[col2], offset))
    pct_finds = len(lt(df, col1, offset))
    if buf != None:
        pct_area = pct_area - sum(within(df[col2], offset-buf))
        pct_finds = pct_finds - len(lt(df, col1, offset-buf))
    elif inv:
        pct_area = len(df) - pct_area
        pct_finds = len(df) - pct_finds
    return (len(df) - pct_finds) / float(len(df) - pct_area)

def prefs(df, col, col2, type='pden', buf=None, min=5, max=200, inv=False):
    data = []
    offsets = numpy.arange(min, max+min, min)
    if buf != None:
        offsets = numpy.arange(buf, max+buf, buf)
        
    for offset in offsets:
        try:
            if type == 'pden':
                data.append(pref(df, col, col2, offset, buf=buf, inv=inv))
            elif type == 'pct_area':
                pct_area = sum(within(df[col2], offset))
                if buf != None:
                        pct_area = pct_area - sum(within(df[col2], offset-buf))
                elif inv:
                    pct_area = len(df) - pct_area
                data.append(pct_area / float(len(df)))
            elif type == 'pden_prime':
                data.append(pref_prime(df, col, col2, offset, buf=buf, inv=inv))
            elif type == 'ratio':
                data.append(pref(df, col, col2, offset, buf=buf, inv=inv) / pref_prime(df, col, col2, offset, buf=buf))
        except:
            data.append(0)
    
    return pd.Series(data, index=offsets)
    
def plot_prefs(df, attr, by, radius=800, name='', logx=False, type='pden', buf=None):
    f=plt.figure()
    title = name + ' ' + attr + ' ' + type + ' by ' + str(by)
    if buf != None:
        title = title + ' buf ' + str(buf)
    plt.title(title)
    plt.xlabel("Track Offset (m)")
    plt.ylabel("PDEN")
    
    for val in distinct(df, by):
        if val!=val:
            continue
        label=str(val)
        if val == False:
            label = "not " + by
        if val == True:
            label = by
        p=prefs(eq(df, by, [val]), attr, attr + '_' + str(radius), type=type, buf=buf)
        if buf != None:
            p=to_bar_plot(p)
        if val in colors:
            p.plot(label=label, color=colors[val], logx=logx)
        else:
            p.plot(label=label, logx=logx)
	f.legend(f.get_axes()[0].get_lines(), [l.get_label() for l in f.get_axes()[0].get_lines()], loc='upper right')


def plot_pden(df, attr, radius=800, name='', logx=False, buf=None):
    f=plt.figure()
    title = name + ' ' + attr + ' pden '
    if buf != None:
        title = title + ' buf ' + str(buf)
    plt.title(title)
    plt.xlabel("Track Offset (m)")

    col2 = attr + '_' + str(radius)

    def div(a, b):
        if b == 0:
            return 0
        return a / b
        
    s_pden = prefs(df, attr, col2, type='pden', buf=buf)
    s_pden_prime = prefs(df, attr, col2, type='pden_prime', buf=buf)
    s_pden_ratio = prefs(df, attr, col2, type='ratio', buf=buf)
    s_pct_area = prefs(df, attr, col2, type='pct_area', buf=buf)
    
    if buf != None:
        s_pden=to_bar_plot(s_pden)
        s_pden_prime=to_bar_plot(s_pden_prime)
        s_pden_ratio=to_bar_plot(s_pden_ratio)
        s_pct_area=to_bar_plot(s_pct_area)
    
    s_pden.plot(label='pden', color='r', logx=logx)
    s_pden_prime.plot(label='pden prime', color='b', logx=logx)
    s_pden_ratio.plot(label='pden ratio', color='#FFAA00', logx=logx)
    s_pct_area.plot(label='pct area', color='g', logx=logx)
    f.legend(f.get_axes()[0].get_lines(), [l.get_label() for l in f.get_axes()[0].get_lines()], loc='upper right')
    

def pct_plot_prefs(df, attr, by, radius=800, name='', logx=False, type='pden', buf=None, inv=False):
    f=plt.figure()
    title = name + ' ' + attr + ' ' + type + ' by ' + str(by)
    if buf != None:
        title = title + ' buf ' + str(buf)
    plt.title(title)
    plt.xlabel("Percentile Basis")
    plt.ylabel("PDEN")

    r2 = min(radius, 800)
    for val in distinct(df, by):
        if val!=val:
            continue
        p=prefs(eq(df, by, [val]), attr + '_' + str(r2) + '_pct', attr + '_pct_' + str(radius), type=type, buf=buf, inv=inv, min=0.05, max=1)
        if buf != None:
            p=to_bar_plot(p)
        if val in colors:
            p.plot(label=str(val), color=colors[val])
        else:
            p.plot(label=str(val), logx=logx)
    f.legend(f.get_axes()[0].get_lines(), [l.get_label() for l in f.get_axes()[0].get_lines()], loc='upper right')


def pct_plot_pden(df, attr, radius=800, name='', logx=False, buf=None, inv=False):
    f=plt.figure()
    title = name + ' ' + attr + ' pden '
    if buf != None:
        title = title + ' buf ' + str(buf)
    plt.title(title)
    plt.xlabel("Percentile Basis")

    col1 = attr + '_' + str(min(radius, 800)) + '_pct'
    col2 = attr + '_pct_' + str(radius)

    def div(a, b):
        if b == 0:
            return 0
        return a / b
        
    s_pden = prefs(df, col1, col2, type='pden', buf=buf, inv=inv, min=0.05, max=1)
    s_pden_prime = prefs(df, col1, col2, type='pden_prime', buf=buf, inv=inv, min=0.05, max=1)
    s_pden_ratio = prefs(df, col1, col2, type='ratio', buf=buf, inv=inv, min=0.05, max=1)
    s_pct_area = prefs(df, col1, col2, type='pct_area', buf=buf, inv=inv, min=0.05, max=1)

    if buf != None:
        s_pden=to_bar_plot(s_pden)
        s_pden_prime=to_bar_plot(s_pden_prime)
        s_pden_ratio=to_bar_plot(s_pden_ratio)
        s_pct_area=to_bar_plot(s_pct_area)
    
    s_pden.plot(label='pden', color='r', logx=logx)
    s_pden_prime.plot(label='pden prime', color='b', logx=logx)
    s_pden_ratio.plot(label='pden ratio', color='#FFAA00', logx=logx)
    s_pct_area.plot(label='pct area', color='g', logx=logx)
    f.legend(f.get_axes()[0].get_lines(), [l.get_label() for l in f.get_axes()[0].get_lines()], loc='upper right')


def bc(df):
    return df[df['developed_df_800'] < 0.25]
    
def comp(df, subset):
    return df[-df['id'].isin(subset['id'])]

def cat(category):
    if category == 'Hiker' or category == 'Hunter' or category == 'Gatherer' or category == 'Runner':
        return 'foot'
    
    if category == 'Vehicle' or category == 'ATV' or category == 'Motorcycle':
        return 'vehicle'
    
    if category == 'Despondent' or category == 'Despondant':
        return 'despondent'
    
    if category == 'Child 1-3' or category == 'Child 4-6' or category == 'Child 7-9' or category == 'Child 10-12' or category == 'Child 13-15':
        return 'child'
    
    if category == 'Snowboarder' or category == 'Skier - Alpine' or category == 'Skier - Nordic':
        return 'ski'
    
    return 'other'

def well(df):
    return eq(df, 'status', ['Well'])

def unwell(df):
    return eq(df, 'status', ['Injured','DOA'])

def search(df):
    return eq(df, 'type', ['Search'])

def rescue(df):
    return eq(df, 'type', ['Rescue'])

def percentile(list, item, sort=True):
    if math.isnan(item): return item
    if sort:
        list = list[:]
        list.sort()
    i = 0
    start = -1
    while item >= list[i] and i < len(list) - 1:
        if item == list[i] and start < 0:
            start = i
        i += 1
    
    if start == -1:
        return float(i) / float(len(list))

    if float(i - start) / len(list) > 0.5:
        return None
    return (float(i) + float(start)) / (2*len(list))
    
def stats(df, prefix, val, samples):
    df[prefix + '_pct'] = map(percentile, samples, val)
    df[prefix + '_min'] = map(min, samples)
    df[prefix + '_max'] = map(max, samples)
    df[prefix + '_avg'] = map(lambda(s): float(sum(s)) / len(s), samples)
    df[prefix + '_med'] = map(lambda(s): s[len(s)/2], samples)
    df[prefix + '_delta'] = map(lambda(s): max(s) - min(s), samples)
    
def least(a1, a2):
    a3 = []
    for i in range(0, len(a1)):
        a3.append(min(a1[i], a2[i]))
    return a3
        
def generate_pct_fn(check):
    def fn(df):
        try:
            return float(len(df[check(df)])) / len(df)
        except:
            return 0
    return fn

def generate_sum_fn(check):
    def fn(df):
        try:
            return len(df[check(df)])
        except:
            return 0
    return fn
    
def pct_developed(data):
    t = 0
    for d in data:
        if d in [21,22,23,24]:
            t += 1
    return float(t) / len(data)
    
def pct_match(fn, d1, d2=None, d3=None):
    try:
        d1=coalesce(d1)
        d2=coalesce(d2)
        d3=coalesce(d3)
    except:
        pass
    if d3 is not None:
        tmp = map(fn, d1, d2, d3)
    elif d2 is not None:
        tmp = map(fn, d1, d2)
    else:
        tmp = map(fn, d1)    
    return sum(tmp), sum(tmp)/float(len(d1))

def table(df, row, col, fn, rows=None, cols=None):
    if rows == None:
        rows = distinct(df, row)
    if cols == None:
        cols = distinct(df, col)
    print '\t' + '\t'.join(str(x) for x in cols)
    for row_val in rows:
        row_str = str(row_val)
        for col_val in cols:
            row_str += '\t' + '%.2f' % fn(eq(eq(df, row, row_val), col, col_val))
        print row_str
   
def dist_str(a, b):
    if not (a == a):
        return a
    
    to = [float(a.split(',')[0].strip()), float(a.split(',')[1].strip())]
    fr = [float(b.split(',')[0].strip()), float(b.split(',')[1].strip())]
    dLat = (to[0]-fr[0])/180*math.pi
    dLon = (to[1]-fr[1])/180*math.pi
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(fr[0]/180*math.pi) * math.cos(to[0]/180*math.pi) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return c*6378137
    
def eval_safe(col):
    if col != col:
        return col
    return eval(col)
     
def load(filename):
    raster_ranges = [250,800,2000]
    sql_ranges = [800,2000]
    ranges = raster_ranges
    for r in sql_ranges:
        if not r in ranges:
            ranges.append(r)
    
    df=pd.read_csv(filename, delimiter='\t', na_values=[-999])
    for range in ranges:
        df['m_' + str(range)] = map(eval, df['m_' + str(range)])
        
    def sub(a, b):
       return a-b

    def dg(d):
        if d != d:
            return d
        if d < 800:
            return '< 0.8km'
        if d < 2000:
            return '< 2km'
        return '> 2km'
 
    def eg(de):                                                  
        if de < 400:
            return '< 0400ft'
        if de < 1000:
            return '< 1000ft'
        return '> 1000 ft'
       
    df['dist'] = map(dist_str, df['ipp'], df['find'])
    df['log_dist'] = map(lambda(x): math.log(x), df['dist'])
    df['trunc_dist'] = map(lambda(x): min(x, 5000), df['dist'])
    df['dg'] = map(dg, df['dist'])
    
    for range in raster_ranges:
        key = str(range)
        key2 = str(min(range, 800))
        df['elevation_' + key] = map(eval, df['elevation_' + key])
        df['slope_' + key] = map(eval, df['slope_' + key])
        df['aspect_' + key] = map(eval, df['aspect_' + key])
        df['tpi_' + key] = map(eval, df['tpi_' + key])
        df['conv_' + key] = map(eval, df['conv_' + key])
        df['elevation_pct_' + key] = map(eval, df['elevation_pct_' + key])
        df['tpi_pct_' + key] = map(eval, df['tpi_pct_' + key])
        df['conv_pct_' + key] = map(eval, df['conv_pct_' + key])
        df['canopy_' + key] = map(eval, df['canopy_' + key])
        df['landcover_' + key] = map(eval, df['landcover_' + key])
        df['developed_df_' + key] = map(pct_developed, df['landcover_' + key])
        
        stats(df, 'elevation_' + key, df['elevation'], df['elevation_' + key])
        stats(df, 'slope_' + key, df['slope'], df['slope_' + key])
        stats(df, 'aspect_' + key, df['aspect'], df['aspect_' + key])
        stats(df, 'tpi_' + key, df['tpi'], df['tpi_' + key])
        stats(df, 'conv_' + key, df['conv'], df['conv_' + key])
        stats(df, 'canopy_' + key, df['canopy'], df['canopy_' + key])
        stats(df, 'landcover_' + key, df['landcover'], df['landcover_' + key])
        
        df['elevation_pct_' + key + '_pct'] = map(percentile, df['elevation_pct_' + key], df['elevation_' + key2 + '_pct'])
        df['tpi_pct_' + key + '_pct'] = map(percentile, df['tpi_pct_' + key], df['tpi_' + key2 + '_pct'])
        df['conv_pct_' + key + '_pct'] = map(percentile, df['conv_pct_' + key], df['conv_' + key2 + '_pct'])
        
        if range == 800 and False:
            df['ipp_elevation_' + key] = map(eval_safe, df['ipp_elevation_' + key])
            df['ipp_tpi_' + key] = map(eval_safe, df['ipp_tpi_' + key])
            df['ipp_conv_' + key] = map(eval_safe, df['ipp_conv_' + key])
            df['ipp_elevation_pct_' + key] = map(eval_safe, df['ipp_elevation_pct_' + key])
            df['ipp_tpi_pct_' + key] = map(eval_safe, df['ipp_tpi_pct_' + key])
            df['ipp_conv_pct_' + key] = map(eval_safe, df['ipp_conv_pct_' + key])

            df['ipp_elevation_' + key + '_pct'] = map(percentile, df['ipp_elevation_' + key], df['ipp_elevation'])
            df['ipp_tpi_' + key + '_pct'] = map(percentile, df['ipp_tpi_' + key], df['ipp_tpi'])
            df['ipp_conv_' + key + '_pct'] = map(percentile, df['ipp_conv_' + key], df['ipp_conv'])

            df['ipp_elevation_pct_' + key + '_pct'] = map(percentile, df['ipp_elevation_pct_' + key], df['ipp_elevation_' + key2 + '_pct'])
            df['ipp_tpi_pct_' + key + '_pct'] = map(percentile, df['ipp_tpi_pct_' + key], df['ipp_tpi_' + key2 + '_pct'])
            df['ipp_conv_pct_' + key + '_pct'] = map(percentile, df['ipp_conv_pct_' + key], df['ipp_conv_' + key2 + '_pct'])
        
        
    df['offset_manmade'] = map(min, df['offset_road'], df['offset_trail'])
    df['offset_water'] = map(min, df['offset_stream'], df['offset_river'], df['offset_lake'])
    def offset75(offset):
        return offset < 75
    df['mm75'] = map(offset75, df['offset_manmade'])
    df['eg'] = map(eg, df['elevation_800_delta'])

    df['ipp_offset_manmade'] = map(min, df['ipp_offset_road'], df['ipp_offset_trail'])
    df['ipp_offset_water'] = map(min, df['ipp_offset_stream'], df['ipp_offset_river'], df['ipp_offset_lake'])

    for range in sql_ranges:
        key = str(range)
        key2 = str(min(range, 800))
        df['offset_road_' + key] = map(eval, df['offset_road_' + key])
        df['offset_trail_' + key] = map(eval, df['offset_trail_' + key])
        df['offset_lake_' + key] = map(eval, df['offset_lake_' + key])
        df['offset_river_' + key] = map(eval, df['offset_river_' + key])
        df['offset_stream_' + key] = map(eval, df['offset_stream_' + key])
        df['offset_bigstream_' + key] = map(eval, df['offset_bigstream_' + key])
        df['offset_drainage_' + key] = map(eval, df['offset_drainage_' + key])
        df['offset_manmade_' + key] = map(least, df['offset_road_' + key], df['offset_trail_' + key])
        df['offset_water_' + key] = map(least, df['offset_stream_' + key], least(df['offset_river_' + key], df['offset_lake_' + key]))
        df['offset_stream_pct_' + key] = map(eval, df['offset_stream_pct_' + key])
        
        stats(df, 'offset_road_' + key, df['offset_road'], df['offset_road_' + key])
        stats(df, 'offset_trail_' + key, df['offset_trail'], df['offset_trail_' + key])
        stats(df, 'offset_lake_' + key, df['offset_lake'], df['offset_lake_' + key])
        stats(df, 'offset_river_' + key, df['offset_river'], df['offset_river_' + key])
        stats(df, 'offset_stream_' + key, df['offset_stream'], df['offset_stream_' + key])
        stats(df, 'offset_bigstream_' + key, df['offset_bigstream'], df['offset_bigstream_' + key])
        stats(df, 'offset_drainage_' + key, df['offset_drainage'], df['offset_drainage_' + key])
        stats(df, 'offset_manmade_' + key, df['offset_manmade'], df['offset_manmade_' + key])
        stats(df, 'offset_water_' + key, df['offset_water'], df['offset_water_' + key])

        df['offset_stream_pct_' + key + '_pct'] = map(percentile, df['offset_stream_pct_' + key], df['offset_stream_' + key2 + '_pct'])
        
    df['well'] = map(lambda(s): s=='Well', df['status'])
    df['unwell'] = map(lambda(s): s!='Well', df['status'])
    df['cell'] = map(lambda(s): s=='Cell Phone', df['signalling'])
    df['bc'] = map(lambda(s): s < 0.25, df['developed_df_800'])
    df['mm40'] = map(lambda(x): x < 40, df['offset_manmade'])
    df['ipp_mm40'] = map(lambda(x): x < 40, df['ipp_offset_manmade'])

    
    df['cat'] = map(cat, df['category'])    
    
    sr=df[((df['type']=='Search') | (df['type']=='Rescue')) & ((df['status']=='Well') | (df['status']=='Injured') | (df['status']=='DOA'))] # exclude water rescues, aircraft, evidence, etc
    
    print "Excluding " + str(len(df) - len(sr)) + " water rescues, no traces, etc . . ."
    return sr
    
def breakdown(sr):
    data = {}
    print str(len(sr)) + " land SAR cases"
    data['bc'] = sr[sr['bc']==True]
    data['fc'] = sr[sr['bc']==False]
    print str(len(data['bc'])) + " backcountry SAR cases"

    data['nomanmade'] = gt(data['bc'], 'offset_manmade', 75)
    print str(len(data['nomanmade'])) + " cases away from linear manmade features"
    
    data['nolake'] = gt(data['nomanmade'], 'offset_lake', 75)
    data['noriver'] = gt(data['nolake'], 'offset_river', 75)
    print str(len(data['noriver'])) + " cases away from rivers and lakes"
    
    data['nostream'] = gt(data['noriver'], 'offset_stream', 75)
    data['nodrainage'] = gt(data['nostream'], 'offset_drainage', 75)

    print str(len(data['nostream'])) + " cases away from streams"
	    
    return data
    
def createimages(sr, d=None):
    rcParams['figure.figsize']=4.5,4.5

    if d==None:
        d = breakdown(sr)

    test=eq(d['bc'], 'cat', 'foot')
    hist_dual(test['slope'].values, coalesce(test['slope_800'].values), s1='Find Location', s2='Sample Points')
    plt.savefig('graphs/sample-slope-hist-all.gif')
    plt.close()

    plot_pden(test, 'offset_manmade')
    plt.savefig('graphs/sample-offset-manmade-pden.gif')
    plt.close()

    plot_pden(test, 'offset_manmade', buf=10)
    plt.savefig('graphs/sample-offset-manmade-pden-buf10.gif')
    plt.close()

    test.hist('elevation_800_pct', bins=14)
    plt.savefig('graphs/sample-elevation-pct.gif')
    plt.close()

    hist_dual(test['elevation_800_pct'].values, coalesce(test['elevation_pct_800']), bins=15)
    plt.savefig('graphs/sample-elevation-pct-dual.gif')
    plt.close()
    
    test.hist('elevation_pct_800_pct', bins=14)
    plt.savefig('graphs/sample-elevation-pct-basis.gif')
    plt.close()

    pct_plot_pden(test, 'elevation', radius=800, buf=0.1)
    plt.savefig('graphs/sample-elevation-pden.gif')
    plt.close()

    print "\n\nBackcountry Search Breakdown by Category"
    piechart(eq(d['bc'], 'type', 'Search'), 'cat', table=True)
    print "\n\nBackcountry Rescue Breakdown by Category"
    piechart(eq(d['bc'], 'type', 'Rescue'), 'cat', table=True)
        
    print "\n\nBackcountry Search Breakdown by Status"
    piechart(eq(d['bc'], 'type', 'Search'), 'status', table=True)
    print "\n\nBackcountry Rescue Breakdown by Status"
    piechart(eq(d['bc'], 'type', 'Rescue'), 'status', table=True)
    
    
    plot_prefs(d['bc'], 'offset_manmade', 'cat', radius=2000)
    plt.savefig('graphs/backcountry-offset-manmade-pden-bycat.gif')
    plt.close()
    
    plot_prefs(d['bc'], 'offset_manmade', 'well', radius=2000)
    plt.savefig('graphs/backcountry-offset-manmade-pden-bywell.gif')
    plt.close()

    plot_pden(d['bc'], 'offset_manmade', buf=10, radius=2000)
    plt.savefig('graphs/backcountry-offset-manmade-pden-buf10.gif')
    plt.close()
    
    plot_prefs(d['bc'], 'offset_manmade', 'well', buf=10, radius=2000)
    plt.savefig('graphs/backcountry-offset-manmade-pden-buf10-bywell.gif')
    plt.close()

    hist_by(d['bc'], 'log_dist', 'well')    
    plt.savefig('graphs/backcountry-log-dist-bywell.gif')
    plt.close()

    plot_prefs(d['bc'], 'offset_manmade', 'dg', radius=2000)
    plt.savefig('graphs/backcountry-offset-manmade-pden-bydg.gif')
    plt.close()

    plot_prefs(eq(d['bc'], 'mm40', False), 'offset_stream', 'dg', radius=2000)
    plt.savefig('graphs/backcountry-nm-stream-pden-bydg.gif')
    plt.close()

    pct_plot_prefs(eq(d['bc'], 'mm40', False), 'elevation', 'dg', radius=2000)
    plt.savefig('graphs/backcountry-nm-elevation-pden-bydg.gif')
    plt.close()

    pct_plot_prefs(eq(d['bc'], 'mm40', False), 'conv', 'dg', radius=2000, inv=True)
    plt.savefig('graphs/backcountry-nm-conv-pden-bydg.gif')
    plt.close()

    print "\n\nBackcountry On-Trail Percentage by Category"
    table(d['bc'], 'cat', 'bc', generate_pct_fn(lambda(x): x['mm40'] == True ))    

    print "\n\nBackcountry Injury/DOA rate by category and on-trail"
    table(d['bc'], 'cat', 'mm40', generate_pct_fn(lambda(x): x['well'] == False ))    

    hist_by(d['bc'], 'log_dist', 'mm40')
    plt.savefig('graphs/backcountry-log-dist-bymm.gif')
    plt.close()
    
    hist_by(d['bc'], 'canopy_2000_pct', 'mm40')
    plt.savefig('graphs/backcountry-canopy-bymm.gif')
    plt.close()
    

    f=eq(d['bc'], 'cat', 'foot')
    
    plot_prefs(f, 'offset_road', 'well')
    plt.savefig('graphs/foot-offset-road-pden-bywell.gif')
    plt.close()

    plot_prefs(f, 'offset_road', 'well', buf=10)
    plt.savefig('graphs/foot-offset-road-pden-bywell-buf10.gif')
    plt.close()
    
    fnr = gt(f, 'offset_road', 40)
        
    plot_prefs(fnr, 'offset_trail', 'well')
    plt.savefig('graphs/foot-offset-trail-pden-bywell.gif')
    plt.close()

    plot_prefs(fnr, 'offset_trail', 'well', buf=10)
    plt.savefig('graphs/foot-offset-trail-pden-bywell-buf10.gif')
    plt.close()

    fm = lt(f, 'offset_manmade', 40)
    fnm = gt(f, 'offset_manmade', 40)
    fnl = gt(fnm, 'offset_lake', 40)
    
    plot_prefs(fnl, 'offset_stream', 'well')
    plt.savefig('graphs/foot-nl-offset-stream-pden-bywell.gif')
    plt.close()

    plot_prefs(fnm, 'offset_stream', 'well', buf=20)
    plt.savefig('graphs/foot-nl-offset-stream-pden-bywell-buf20.gif')
    plt.close()

    plot_prefs(fnl, 'offset_stream', 'category')
    plt.savefig('graphs/foot-nl-offset-stream-pden-bycat.gif')
    plt.close()

    pct_plot_prefs(fnm, 'elevation', 'well', radius=2000, buf=0.1)
    plt.savefig('graphs/foot-nm-elevation-pden-bywell-buf.gif')
    plt.close()

    pct_plot_prefs(fnm, 'conv', 'well', radius=2000, buf=0.1)
    plt.savefig('graphs/foot-nm-conv-pden-bywell-buf.gif')
    plt.close()

    hist_dual(f['log_dist'].values, d['bc']['log_dist'].values)
    plt.savefig('graphs/foot-dist.gif')
    plt.close()

    hist_by(f, 'log_dist', 'ipp_mm40')
    plt.savefig('graphs/foot-dist-by-ipp.gif')
    plt.close()

    plot_prefs(f, 'offset_manmade', 'dg', radius=2000)
    plt.savefig('graphs/foot-manmade-pden-bydg.gif')
    plt.close()

    plot_prefs(f, 'offset_manmade', 'ipp_mm40', radius=2000)
    plt.savefig('graphs/foot-manmade-pden-byipp.gif')
    plt.close()

    plot_prefs(fnl, 'offset_stream', 'dg', radius=2000)
    plt.savefig('graphs/foot-nl-stream-pden-bydg.gif')
    plt.close()

    pct_plot_prefs(fnl, 'elevation', 'dg', radius=2000)
    plt.savefig('graphs/foot-nl-elevation-pden-bydg.gif')
    plt.close()

    unwell(fnl).plot(kind='scatter', x='log_dist', y='elevation_2000_pct')
    plt.savefig('graphs/foot-nl-unwell-elevation-v-distance.gif')
    plt.close()

    
    
    c=eq(d['bc'], 'cat', 'child')
    
    hist_dual(c['log_dist'].values, d['bc']['log_dist'].values)
    plt.savefig('graphs/child-dist.gif')
    plt.close()

    hist_by(c, 'log_dist', 'ipp_mm40')
    plt.savefig('graphs/child-dist-by-ipp.gif')
    plt.close()

    plot_pden(c, 'offset_road', buf=10)
    plt.savefig('graphs/child-offset-road-pden-buf10.gif')
    plt.close()
    
    cnr = gt(c, 'offset_road', 30)

    plot_pden(cnr, 'offset_trail', buf=10)
    plt.savefig('graphs/child-offset-trail-pden-buf10.gif')
    plt.close()
    
    cnm = gt(cnr, 'offset_trail', 70)    
    cnw = gt(cnm, 'offset_water', 40)
    
    cnw.hist('slope_800_pct')
    plt.savefig('graphs/child-nw-slope-800-pct.gif')
    plt.close()
    
    cnw.plot(x='age', y='slope_800_pct', kind='scatter')
    plt.savefig('graphs/child-nw-slope-800-pct-byage.gif')
    plt.close()

    plot_prefs(c, 'offset_manmade', 'dg', radius=2000)    
    plt.savefig('graphs/child-manmade-pden-bydg.gif')
    plt.close()

    plot_prefs(cnm, 'offset_water', 'dg', radius=2000)    
    plt.savefig('graphs/child-nm-water-pden-bydg.gif')
    plt.close()



    s=eq(d['bc'], 'cat', 'despondent')
    snm = gt(s, 'offset_manmade', 40)
    snw = gt(snm, 'offset_water', 40)
    
    plot_pden(s, 'offset_manmade')
    plt.savefig('graphs/despondent-manmade-pden.gif')
    plt.close()
    
    pct_plot_pden(snw, 'elevation', inv=True, radius=800)
    plt.savefig('graphs/despondent-nw-elevation-pden.gif')
    plt.close()

    snw.hist('canopy_800_pct')
    plt.savefig('graphs/despondent-nw-canopy.gif')
    plt.close()
    
    hist_dual(s['log_dist'].values, d['bc']['log_dist'].values)    
    
    v=eq(d['bc'], 'cat', 'vehicle')
    plot_prefs(v, 'offset_manmade', 'category')
    plt.savefig('graphs/vehicle-offset-manmade-bycat.gif')
    plt.close()

    plot_pden(v, 'offset_manmade', buf=10)
    plt.savefig('graphs/vehicle-offset-manmade-buf10.gif')
    plt.close()

    hist_dual(v['log_dist'].values, d['bc']['log_dist'].values)
    plt.savefig('graphs/vehicle-distance.gif')
    plt.close()

    plot_prefs(v, 'offset_manmade', 'dg')
    plt.savefig('graphs/vehicle-manmade-pden-bydg.gif')
    plt.close()
    
    print "\n\nVehicle numbers by category and on-trail"
    table(v, 'category', 'mm40', generate_sum_fn(lambda(x): x['bc']==True), cols=[True, False])
    
    print "\n\nVehicle Injury/DOA rate by category and on-trail"
    table(v, 'category', 'mm40', generate_pct_fn(lambda(x): x['well']==False), cols=[True, False])
    
    vnm = gt(v, 'offset_manmade', 75)
    
    plot_prefs(well(vnm), 'offset_water', 'category')
    plt.savefig('graphs/vehicle-water-well-pref-bycategory.gif')
    plt.close()
    
    plot_prefs(unwell(vnm), 'offset_water', 'category')
    plt.savefig('graphs/vehicle-water-unwell-pref-bycategory.gif')
    plt.close()
    
    
    
    s=eq(d['bc'], 'cat', 'ski')
    
    plot_prefs(s, 'offset_manmade', 'category', radius=2000)
    plt.savefig('graphs/ski-offset-manmade-pden-bycat.gif')
    plt.close()

    plot_pden(s, 'offset_manmade', buf=10, radius=2000)
    plt.savefig('graphs/ski-offset-manmade-pden-buf10.gif')
    plt.close()

    plot_prefs(s, 'offset_manmade', 'dg', radius=2000)
    plt.savefig('graphs/ski-manmade-pden-bydg.gif')
    plt.close()

    snm=gt(s, 'offset_manmade', 40)
    
    pct_plot_prefs(snm, 'elevation', 'category', radius=2000)
    plt.savefig('graphs/ski-nm-elevation-pden-bycat.gif')
    plt.close()

    pct_plot_pden(s, 'elevation', radius=2000)
    plt.savefig('graphs/ski-elevation-pden.gif')
    plt.close()

    snm.hist('elevation_pct_2000_pct')
    plt.savefig('graphs/ski-nm-elevation-2000-pct.gif')
    plt.close()

    snm.hist('elevation_pct_250_pct')
    plt.savefig('graphs/ski-nm-elevation-250-pct.gif')
    plt.close()

    pct_plot_pden(snm, 'elevation', radius=800)
    plt.savefig('graphs/ski-nm-elevation-pden-800.gif')
    plt.close()

    plot_pden(snm, 'offset_stream')
    plt.savefig('graphs/ski-offset-stream-pden.gif')
    plt.close()
    
