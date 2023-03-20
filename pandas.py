import pandas as pd


data = pd.read_csv(r'D:\Users\Tempis\Desktop\info.csv')
rows = len(data.axes[0])
cols = len(data.axes[1])

print("number of rows = ",rows)

print("number of columns = ",cols)

df = pd.DataFrame(data)

result = df.loc[["Dima", "James"],["Name", "Score"]]

df.to_csv(r'D:\Users\Tempis\Desktop\subset.csv', index=False)