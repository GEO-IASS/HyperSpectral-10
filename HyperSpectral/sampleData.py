import pandas as pd
import seaborn as sns


data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

print(data.head())