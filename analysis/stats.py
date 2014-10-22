
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
        return -999
    return (float(i) + float(start)) / (2*len(list))

def generate_stats(list):
    total = 0
    range = [99999, 0]
    
    for val in list:
        total += val
        range[0] = min(range[0], val)
        range[1] = max(range[1], val)
    
    avg = float(total) / len(list)
    return (avg, range, list)