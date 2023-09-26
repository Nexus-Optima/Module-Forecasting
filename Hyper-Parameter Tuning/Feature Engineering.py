import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns

df = pd.read_csv('../Data/ICAC multiple variables.csv')
print(df.columns)
df=df[:2500]
df = df.drop(['Date'], axis = 1)
df['Gap']=df['PROD']-df['CONS']
df['lag_gap']=df['Gap'].shift(365)
df['diffSMUYearly']=df['S/MU'].diff(365)
df['lagDaily']=df['Output'].shift(1)
df['lagMonthly']=df['Output'].shift(30)
df['lagQuarterly']=df['Output'].shift(90)
df['diffYearly']=df['Output'].diff(365)
df['rolling_mean'] = df['Output'].rolling(window=30).mean()
df = df.drop(['AREA','YIELD','IMPTS','EXPTS','Season','Gap','PROD','CONS','S/MU','BSTK','ENDSTK','Month','Day','Year'], axis = 1)

cor = df.corr()
print(cor)
plt.figure(figsize=(10,6))
sns.heatmap(cor,annot=True)
plt.show()
