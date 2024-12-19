# Dealing with Missing Data

## Simple/Mean/Median/Mode Imputation
Mean/Median/Mode Imputation: Replacing missing values with the mean, median, or mode of the column.
Python:
* Mean: df.fillna(df.mean())
* Median: df.fillna(df.median())
* Mode: df.fillna(df.mode().iloc[0])
