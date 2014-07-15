import csv

def load(filename):
    data = []
    header = None
    fields = {'id': 'Incident #', 'type': 'Incident Type', 'terrain': 'Terrain', 'category': 'Subject Category', 'ipp': 'IPP Coord.', 'find': 'Find Coord', 'delta_e': 'Elevation Change (ft)', 'distance': 'Distance IPP (km)' }
    reader = csv.reader(open(filename, 'rb'), delimiter=',', quotechar='"')
    for row in reader:
        if header == None:
            header = {}
            for idx in range(0, len(row)):
                header[row[idx]] = idx
            continue
        
        obj = {}
        for field in fields:
            obj[field] = row[header[fields[field]]]
            if obj[field] == '':
                obj[field] = None
                
        if obj['ipp'] != None:
            obj['ipp'] = [float(obj['ipp'].split(',')[0].strip()), float(obj['ipp'].split(',')[1].strip())]

        if obj['find'] != None:
            obj['find'] = [float(obj['find'].split(',')[0].strip()), float(obj['find'].split(',')[1].strip())]
            data.append(obj)
    return data