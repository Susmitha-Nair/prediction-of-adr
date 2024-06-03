import pandas as pd
import os

if __name__ == "__main__":
	base_path = "/sas/vidhya/susmitha/ADR_Prediction/final_results/extraTree/pca/subset_1/"
	performance = pd.DataFrame([])
	for file in os.listdir(base_path):
		if file.endswith("_acc.csv"):
			file_2 = pd.read_csv(file)
			performance = performance.append(file_2)
	
	print(performance)
	performance.to_csv("performance_subset_1.csv")
