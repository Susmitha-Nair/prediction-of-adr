import pandas as pd
import numpy as np
import ast

file = pd.read_csv("ADR_list.csv")
file_1 = file[["adr"]]
len(file)
len(file_1)
len(file.column)
len(file_1.column)
file_1['adr'] = file_1['adr'].apply(ast.literal_eval)
file_2 = file_1['adr'].str.join('|').str.get_dummies()
len(file_2)
len(file_2.column)
print(file_2)
file_4 = file[['combined_medication_ADR_for_label$combined_pert']]
file_3 = pd.concat([file_4.reset_index(drop=True), file_2], axis=1)
file_3.to_csv("ADR_combine_hot_encoded.csv")



