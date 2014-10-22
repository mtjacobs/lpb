import math
import numpy
import pandas as pd
import matplotlib.pyplot as plt

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
            values.append(val)
    return values
    
def hist_dual(series1, series2, normal=True, histtype='stepfilled', alpha=0.4, name='', s1='Series 1', s2='Series 2'):
    f=plt.figure()
    plt.title(name)
    bins = numpy.linspace(min( min(nonan(series1)), min(nonan(series1))), max( max(nonan(series1)), max(nonan(series2))), 20)
    plt.hist(series1, bins, alpha=alpha, label=s1,normed=normal,histtype=histtype)
    plt.hist(series2, bins, alpha=alpha, label=s2,normed=normal,histtype=histtype)
    plt.legend(loc='upper right')
    plt.show()
    
    
def hist_by(df, col, by, normal=True, histtype='stepfilled', alpha=0.4, name=''):
    f=plt.figure()
    plt.title(name + ' ' + col + ' by ' + str(by))
    bins = numpy.linspace(min(nonan(df[col].values)), max(nonan(df[col].values)), 20)
    for val in distinct(df, by):
        plt.hist(df[df[by]==val][col].values, bins, alpha=alpha, label=str(val),normed=normal,histtype=histtype)
    plt.legend(loc='upper right')
    plt.show()
    
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
    
def pref(df, col, range, offset, buf=None):
    pct_area = sum(within(df[col + '_' + str(range)], offset))
    pct_finds = len(lt(df, col, offset))
    if buf != None:
        pct_area = pct_area - sum(within(df[col + '_' + str(range)], offset-buf))
        pct_finds = pct_finds - len(lt(df, col, offset-buf))
    return pct_finds / float(pct_area)

def pref_prime(df, col, range, offset, buf=None):
    pct_area = sum(within(df[col + '_' + str(range)], offset))
    pct_finds = len(lt(df, col, offset))
    if buf != None:
        pct_area = pct_area - sum(within(df[col + '_' + str(range)], offset-buf))
        pct_finds = pct_finds - len(lt(df, col, offset-buf))
    return (len(df) - pct_finds) / float(len(df) - pct_area)

def prefs(df, col, r, type='pden', buf=None):
    data = []
    offsets = range(5, 200, 5)
    if buf != None:
        offsets = range(buf, 200, buf)
    for offset in offsets:
        try:
            if type == 'pden':
                data.append(pref(df, col, r, offset, buf=buf))
            elif type == 'pct_area':
                pct_area = sum(within(df[col + '_' + str(r)], offset))
                if buf != None:
                        pct_area = pct_area - sum(within(df[col + '_' + str(range)], offset-buf))
                data.append(pct_area / float(len(df)))
            elif type == 'pden_prime':
                data.append(pref_prime(df, col, r, offset, buf=buf))
            elif type == 'ratio':
                data.append(pref(df, col, r, offset, buf=buf) / pref_prime(df, col, r, offset, buf=buf))
        except:
            data.append(0)
    
    return pd.Series(data, index=offsets)
    
def plot_prefs(df, attr, by, radius=800, name='', logx=False, type='pden', buf=None):
    f=plt.figure()
    title = name + ' ' + attr + ' ' + type + ' by ' + str(by)
    if buf != None:
        title = title + ' buf ' + str(buf)
    plt.title(title)
    for val in distinct(df, by):
        if val in colors:
            prefs(eq(df, by, [val]), attr, radius, type=type, buf=buf).plot(label=str(val), color=colors[val], logx=logx)
        else:
            prefs(eq(df, by, [val]), attr, radius, type=type, buf=buf).plot(label=str(val), logx=logx)
    f.legend(f.get_axes()[0].get_lines(), [l.get_label() for l in f.get_axes()[0].get_lines()], loc='upper right')


def plot_pden(df, attr, radius=800, name='', logx=False, buf=None):
    f=plt.figure()
    title = name + ' ' + attr + ' pden '
    if buf != None:
        title = title + ' buf ' + str(buf)
    plt.title(title)

    def div(a, b):
        if b == 0:
            return 0
        return a / b
        
    s_pden = prefs(df, attr, radius, type='pden', buf=buf)
    s_pden_prime = prefs(df, attr, radius, type='pden_prime', buf=buf)
    s_pden_ratio = prefs(df, attr, radius, type='ratio', buf=buf)
    s_pct_area = prefs(df, attr, radius, type='pct_area', buf=buf)

    
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
    
    if category == 'Dementia' or category == 'Mental Illness' or category == 'Mental Illnes':
        return 'mental'
        
    if category == 'Snowboarder' or category == 'Skier - Alpine' or category == 'Skier - Nordic':
        return 'ski'
    
    return 'other'

def well(df):
    return eq(df, 'status', ['Well'])

def unwell(df):
    return eq(df, 'status', ['Injured','DOA'])

def percentile(list, item):
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
    
def load(filename):
    raster_ranges = [250,800]
    sql_ranges = [800]
    ranges = raster_ranges
    for r in sql_ranges:
        if not r in ranges:
            ranges.append(r)
    
    df=pd.read_csv(filename, delimiter='\t', na_values=[-999])
    for range in ranges:
        df['m_' + str(range)] = map(eval, df['m_' + str(range)])
        
    def sub(a, b):
       return a-b

    for range in raster_ranges:
        key = str(range)
        df['elevation_' + key] = map(eval, df['elevation_' + key])
        df['slope_' + key] = map(eval, df['slope_' + key])
#        df['aspect_' + key] = map(eval, df['aspect_' + key])
        df['canopy_' + key] = map(eval, df['canopy_' + key])
        df['landcover_' + key] = map(eval, df['landcover_' + key])
        df['developed_df_' + key] = map(pct_developed, df['landcover_' + key])
        
        stats(df, 'elevation_' + key, df['elevation'], df['elevation_' + key])
        stats(df, 'slope_' + key, df['slope'], df['slope_' + key])
#        stats(df, 'aspect_' + key, df['aspect'], df['aspect_' + key])
        stats(df, 'canopy_' + key, df['canopy'], df['canopy_' + key])
        stats(df, 'landcover_' + key, df['landcover'], df['landcover_' + key])
        
    df['offset_manmade'] = map(min, df['offset_road'], df['offset_trail'], df['offset_fsline'])
    df['offset_water'] = map(min, df['offset_stream'], df['offset_river'], df['offset_lake'])
    def offset75(offset):
        return offset < 75
    df['mm75'] = map(offset75, df['offset_manmade'])

    for range in sql_ranges:
        key = str(range)
        df['offset_road_' + key] = map(eval, df['offset_road_' + key])
        df['offset_trail_' + key] = map(eval, df['offset_trail_' + key])
        df['offset_fsline_' + key] = map(eval, df['offset_fsline_' + key])
        df['offset_lake_' + key] = map(eval, df['offset_lake_' + key])
        df['offset_river_' + key] = map(eval, df['offset_river_' + key])
        df['offset_stream_' + key] = map(eval, df['offset_stream_' + key])
        df['offset_drainage_' + key] = map(eval, df['offset_drainage_' + key])
        df['offset_manmade_' + key] = map(least, map(least, df['offset_road_' + key], df['offset_trail_' + key]), df['offset_fsline_' + key])
        df['offset_water_' + key] = map(least, df['offset_stream_' + key], least(df['offset_river_' + key], df['offset_lake_' + key]))

        stats(df, 'offset_road_' + key, df['offset_road'], df['offset_road_' + key])
        stats(df, 'offset_trail_' + key, df['offset_trail'], df['offset_trail_' + key])
        stats(df, 'offset_fsline_' + key, df['offset_fsline'], df['offset_fsline_' + key])
        stats(df, 'offset_lake_' + key, df['offset_lake'], df['offset_lake_' + key])
        stats(df, 'offset_river_' + key, df['offset_river'], df['offset_river_' + key])
        stats(df, 'offset_stream_' + key, df['offset_stream'], df['offset_stream_' + key])
        stats(df, 'offset_drainage_' + key, df['offset_drainage'], df['offset_drainage_' + key])
        stats(df, 'offset_manmade_' + key, df['offset_manmade'], df['offset_manmade_' + key])
        stats(df, 'offset_water_' + key, df['offset_water'], df['offset_water_' + key])
        
    df['well'] = map(lambda(s): s=='Well', df['status'])
    df['unwell'] = map(lambda(s): s!='Well', df['status'])
    df['cell'] = map(lambda(s): s=='Cell Phone', df['signalling'])
    df['bc'] = map(lambda(s): s < 0.25, df['developed_df_800'])
    
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
    if d==None:
        d = breakdown(sr)

    test=eq(d['bc'], 'cat', 'foot')
    plot_prefs(test, 'offset_manmade', 'category')
    plt.savefig('graphs/sample-pref-bycat.gif')
    plt.close()
    
    hist_dual(test['slope'].values, coalesce(test['slope_800'].values), s1='Find Location', s2='Sample Points')
    plt.savefig('graphs/sample-slope-hist-all.gif')
    plt.close()

    plot_pden(test, 'offset_manmade')
    plt.savefig('graphs/sample-offset-manmade-pden.gif')
    plt.close()

    plot_pden(test, 'offset_manmade', buf=10)
    plt.savefig('graphs/sample-offset-manmade-pden-buf10.gif')
    plt.close()

    plot_pden(test, 'offset_manmade', logx=True)
    plt.savefig('graphs/sample-offset-manmade-pden-logx.gif')
    plt.close()

    test.boxplot(column='offset_water', by='well')
    plt.savefig('graphs/sample-nm-offset-water-800-pct-bywell.gif')
    plt.close()

    piechart(sr, 'cat')
    plt.savefig('graphs/search-rescue.gif')
    plt.close()

    print "\n\nBackcountry Breakdown by Category"
    piechart(d['bc'], 'cat', table=True)

    piechart(d['bc'], 'cat')
    plt.savefig('graphs/backcountry.gif')
    plt.close()
    
    piechart(d['fc'], 'cat')
    plt.savefig('graphs/frontcountry.gif')
    plt.close()
    
    plot_prefs(d['bc'], 'offset_manmade', 'cat')
    plt.savefig('graphs/backcountry-offset-manmade-pden-bycat.gif')
    plt.close()

    plot_pden(d['bc'], 'offset_manmade', buf=10)
    plt.savefig('graphs/backcountry-offset-manmade-pden-buf10.gif')
    plt.close()
    
    d['bc']['mm40'] = map(lambda(x): x < 40, d['bc']['offset_manmade'])

    print "\n\nBackcountry On-Trail Percentage by Category"
    table(d['bc'], 'cat', 'bc', generate_pct_fn(lambda(x): x['mm40'] == True ))    

    print "\n\nBackcountry Injury/DOA rate by category and on-trail"
    table(d['bc'], 'cat', 'mm40', generate_pct_fn(lambda(x): x['well'] == False ))    



    f=eq(d['bc'], 'cat', 'foot')
    
    plot_prefs(f, 'offset_manmade', 'category')
    plt.savefig('graphs/foot-offset-manmade-pden-bycat.gif')
    plt.close()

    plot_pden(f, 'offset_manmade', buf=10)
    plt.savefig('graphs/foot-offset-manmade-pden-buf10.gif')
    plt.close()
        
    fm = lt(f, 'offset_manmade', 40)
    fnm = gt(f, 'offset_manmade', 40)
    
    print "\n\nFoot Status for On-Trail"
    piechart(fm, 'status', table=True)
    
    print "\n\nFoot STatus for Off-Trail"
    piechart(fnm, 'status', table=True)
    
    
    plot_prefs(fnm, 'offset_water', 'well')
    plt.savefig('graphs/foot-nm-offset-water-pden-bywell.gif')
    plt.close()

    plot_prefs(fnm, 'offset_stream', 'well')
    plt.savefig('graphs/foot-nm-offset-stream-pden-bywell.gif')
    plt.close()

    plot_prefs(fnm, 'offset_stream', 'well', buf=20)
    plt.savefig('graphs/foot-nm-offset-stream-pden-bywell-buf20.gif')
    plt.close()

    plot_prefs(fnm, 'offset_stream', 'category')
    plt.savefig('graphs/foot-nm-offset-stream-pden-bycategory.gif')
    plt.close()
    
    plot_prefs(gt(fnm, 'elevation_800_pct', 0.2), 'offset_stream', 'well')
    plt.savefig('graphs/foot-nm-offset-stream-pden-high-bycategory.gif')
    plt.close()

    fnm.boxplot(column='offset_stream_800_pct', by='well')
    plt.savefig('graphs/foot-nm-offset-stream-800-pct-bywell.gif')
    plt.close()
    
    hist_by(fnm, 'elevation_800_pct', 'well')
    plt.savefig('graphs/foot-nm-elevation-800-pct-bywell.gif')
    plt.close()
    
    print "\n\nFoot, No-Manmade percentage within 100m of a stream by category and well"
    table(fnm, 'category', 'well', generate_pct_fn(lambda(x): x['offset_water'] <  100 ))
    
    print "\n\nFoot, No-Manmade Injury/DOA rate by category and near-water"
    fnm['w100'] = map(lambda(x): x < 100, fnm['offset_water'])
    fnm['s80'] = map(lambda(x): x < 80, fnm['offset_stream'])
    table(fnm, 'category', 'w100', generate_pct_fn(lambda(x): x['well'] == False ))

    fnw=gt(fnm, 'offset_water', 75)
    
    hist_by(fnw, 'elevation_800_pct', 'well')
    plt.savefig('graphs/foot-nw-elevation-800-pct-bywell.gif')
    plt.close()
    
    plot_prefs(fnw, 'offset_drainage', 'well')
    plt.savefig('graphs/foot-nw-offset-drainage-pden-bywell.gif')
    plt.close()
    
    
    c=eq(d['bc'], 'cat', 'child')
    
    plot_pden(c, 'offset_manmade')
    plt.savefig('graphs/child-offset-manmade-pden.gif')
    plt.close()
        
    piechart(c, 'mm75')
    plt.savefig('graphs/child-manmade75.gif')
    plt.close()
    
    cnm = gt(c, 'offset_manmade', 45)
    cnw = gt(cnm, 'offset_stream', 40)
    
    cnw.hist('slope')
    plt.savefig('graphs/child-nw-slope.gif')
    plt.close()

    cnw.hist('slope_800_pct')
    plt.savefig('graphs/child-nw-slope-800-pct.gif')
    plt.close()
    
    cnw.plot(x='age', y='slope_800_pct', kind='scatter')
    plt.savefig('graphs/child-nw-slope-800-pct-byage.gif')
    plt.close()
    
    
    
    s=eq(d['bc'], 'cat', 'despondent')
    
    plot_pden(s, 'offset_manmade')
    plt.savefig('graphs/despondent-offset-manmade-pden.gif')
    plt.close()
    
    snm = gt(s, 'offset_manmade', 60)
    snw = gt(snm, 'offset_water', 40)
    
    snw.hist('elevation_800_pct')
    plt.savefig('graphs/despondent-nw-elevation-800-pct.gif')
    plt.close()

    snw.hist('slope_800_pct')
    plt.savefig('graphs/despondent-nw-slope-800-pct.gif')
    plt.close()

    snw.hist('canopy_800_pct')
    plt.savefig('graphs/despondent-nw-canopy-800-pct.gif')
    plt.close()
    
    v=eq(d['bc'], 'cat', 'vehicle')
    plot_prefs(v, 'offset_manmade', 'category')
    plt.savefig('graphs/vehicle-offset-manmade-bycat.gif')
    plt.close()

    plot_pden(v, 'offset_manmade', buf=10)
    plt.savefig('graphs/vehicle-offset-manmade-buf10.gif')
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
    
    plot_prefs(s, 'offset_manmade', 'category')
    plt.savefig('graphs/ski-offset-manmade-pden-bycat.gif')
    plt.close()

    plot_pden(s, 'offset_manmade', buf=10)
    plt.savefig('graphs/ski-offset-manmade-pden-buf10.gif')
    plt.close()
    
    snm=gt(s, 'offset_manmade', 40)

    plot_pden(snm, 'offset_stream')
    plt.savefig('graphs/ski-offset-stream-pden.gif')
    plt.close()
    
    snw = gt(s, 'offset_water', 40)
    
    plot_pden(snw, 'offset_drainage')
    plt.savefig('graphs/ski-offset-drainage-pden.gif')
    plt.close()
    