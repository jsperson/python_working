import pandas as pdtotal = 0
for chunk in pd.read_csv('data.csv', chunksize=1000):    
    total += sum(chunk['x'])
    print(total)
    
