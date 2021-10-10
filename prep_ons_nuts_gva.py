import pandas as pd
import os
fname = "regionalgrossvalueaddedbalancedbyindustryallnutslevelregions.xlsx"

df = pd.read_excel(os.path.join('scratch', fname),
                   sheet_name='Table1b',
                   skiprows=0)

df.columns = df.iloc[0, :]
df = df.iloc[1:, :]
df = df[df['SIC07 code'] == 'Total']
df = df.rename(columns={df.columns[0]: 'NUTS1',
                        df.columns[-1]: str(df.columns[-1])[:4]})
df.to_csv(os.path.join('data', 'nuts1_gva_2016.csv'))
