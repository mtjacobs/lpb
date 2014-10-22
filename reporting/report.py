import pandas as pd

df=pd.read_csv('results2.csv', delimiter='\t')
h=df['distance'].hist()
h.figure.savefig('graphs/distance-histogram.jpg')
