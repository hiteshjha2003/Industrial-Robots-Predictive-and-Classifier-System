import pandas as pd 

path = r"D:\COACHXLIVE\RESUME OF HITESH\Wifey Docs\projects\end_to_end_ml_pipeline\industrial-robot-predictive-mtce\data\01_raw\sample.csv.gz"

df = pd.read_csv(path, compression='gzip')

# Keep only 604,800 rows
# df = df.iloc[:604800]

print(df.info())
print("Rows:", len(df))
print('Failure modes:', df['failure_mode'].value_counts().to_dict())


# path2 = r"D:\COACHXLIVE\RESUME OF HITESH\Wifey Docs\projects\end_to_end_ml_pipeline\industrial-robot-predictive-mtce\data\01_raw\sample.csv.gz"
# # OPTIONAL: Save the trimmed dataset
# df.to_csv(path2, 
#           index=False, compression='gzip')
