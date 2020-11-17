import pandas as pd
import numpy as np

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

total_company = list(set(list(train_df['stock']) + list(test_df['stock'])))
print(len(total_company))
column_name = ['Company']

distinct_company = pd.DataFrame(columns=column_name, data=sorted(total_company))

distinct_company.to_csv('./dis_comp.csv', index=False)