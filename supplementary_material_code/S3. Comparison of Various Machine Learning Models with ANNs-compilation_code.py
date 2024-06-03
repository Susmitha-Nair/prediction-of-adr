import pandas as pd
import os
import glob
if __name__ == "__main__":
	base_path = "/sas/vidhya/susmitha/ADR_Prediction/final_results/extraTree/pca/subset_1/"
#	os.chdir(base_path)	
	performance = pd.DataFrame()

#	all_files =[i for i in glob.glob('*.{}'.format('csv'))] # os.listdir(base_path)

#	file_content = pd.concat([pd.read_csv(files) for files in all_files], sort=False, ignore_index=True)
#	file_content.to_csv('./dataset/combined.csv')
#	print(file_content)
	i=0

	for filecontent in os.listdir(base_path):
		print('Filename:', filecontent)
		i = i+1
		print(i)
		if filecontent.endswith(".csv"):
			file_2 = pd.read_csv(filecontent)
			file_2 = file_2.loc[:,['name','pred']]
			filename = filecontent.replace('.csv','')
			file_2.columns = ['name', filename]
			if performance.empty == True:
				performance = file_2
			else:
				performance = performance.merge(file_2, on ='name')
	
	print(performance)
	performance.to_csv("./dataset/compiled_subset_1_extraTree.csv")

