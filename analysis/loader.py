import csv

def load_isrid(filename):
    data = []
    header = None
    fields = {'id': 'Incident #', 'type': 'Incident Type', 'terrain': 'Terrain', 'category': 'Subject Category', 'age': 'Age', 'sex': 'Sex', 'status': 'Subject Status', 'time': 'Total Time Lost', 'ipp': 'IPP Coord.', 'find': 'Find Coord', 'delta_e': 'Elevation Change (ft)', 'distance': 'Distance IPP (km)', 'weather': 'Weather', 'comments': 'Comments', 'ecoregion': 'EcoRegion Domain', 'manhours': 'Total Man Hours', 'signalling': 'Signalling'}
    reader = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='"')
    for row in reader:
        if header == None:
            header = {}
            for idx in range(0, len(row)):
                header[row[idx]] = idx
            continue
        
        obj = {}
        for field in fields:
            try:
                obj[field] = row[header[fields[field]]]
                if obj[field] == '':
                    obj[field] = None
            except:
                obj[field] = None
        
        try:
            if 'ipp' in obj and obj['ipp'] != None:
                obj['ipp'] = [float(obj['ipp'].split(',')[0].strip()), float(obj['ipp'].split(',')[1].strip())]

            if obj['find'] != None:
                obj['find'] = [float(obj['find'].split(',')[0].strip()), float(obj['find'].split(',')[1].strip())]
                data.append(obj)
        except:
            pass
            
    return data

def load_processed(filename):
    reader = csv.DictReader(open(filename, 'rb'), delimiter='\t', quotechar='"')
    return reader
