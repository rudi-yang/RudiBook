import pandas as pd

from sqlalchemy import create_engine

conn = create_engine('mysql+mysqldb://rudi:password@localhost:3306/test?charset=utf8')

data = pd.read_csv("debug_case.csv", sep="\t")
data.fillna("None", inplace=True)
data = data.drop(['Unnamed: 0', 'Unnamed: 0.1', "index"], axis=1)
data["comment"] = ""

print(data.dtypes)
data.to_sql("table", con=conn, if_exists="append", index=False)
