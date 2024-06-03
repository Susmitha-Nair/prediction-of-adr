Readme
Prediction of Adverse Drug Effect for drugs in combination using transcriptomic data.

Computational analysis of drugs in combination to predict Adverse drug reaction is a promising solution to reduce the fatality and economic loss reported in post-marketing surveillance. Recent works suggest that transcriptomic data are potentially informative about the phenotypic properties of drugs, including Adverse Drug Reaction by a single drug. In this study, we aim to build predictive models for Adverse Drug Reactions caused by drugs in combinations using transcriptomic data. Our model provides a scalable approach to derive up to 243 Adverse Drug Reactions with an average accuracy of 84%. We suggest 11 Adverse Drug Reactions that can be predicted without any inclusion of negative samples using our model. Our model can be used to fill in the gaps that exist when it comes to extracting drug-drug-ADR association. 

Getting stated:
These instructions will get to you a copy of the research work up on your local machine for testing purpose.

Prerequisite:
R environment – version: 3.6.3
•	Link to install: Follow the steps in  https://www.datacamp.com/community/tutorials/installing-R-windows-mac-ubuntu
•	Packages: tidyr. For installing, use command: install.packages(“tidyr”)
Python – version: 3.7.3
Jupyter notebook
•	Link to install: https://jupyter.org/install
•	Packages: pandas, numpy, ast, sklearn
Description of data
•	Input Data
o	LINCS_Gene_Experssion_signatures_CD.csv: Contains differential Gene Expression for 978 landmark genes for 20.338 drug. Dimension: 20,338 x 978
o	GO_transformed_signatures_log.csv: Expression values for the 20,338 drug for 4,438 Gene ontology terms. Dimension: 20,338 x 4,438
o	MACCS_bitmatrix.csv: Chemical Fingerprints representing drugs. Dimension: 20,338 x 166
o	Label_two_side.csv: Dataset for Adverse Drug Reaction caused by drugs in combination as in PharmGKB dataset
•	Processed File
o	represesntation_of_single_drug.csv : Drug represented by their DGE, expression for Gene Ontology and MACCS bit
o	ADR_list.csv: list of Adverse Drug Reaction in categorical text format corresponding to the drugs in combination as extracted from PharmGKB TwoSide table
o	dataset_subset.csv: drugs in combination represented as concatenation of each of the drug vectors
o	ADR_combine_hot_encoded.csv: One hot encoded vector of Adverse Drug Reaction corresponding to each drug combination
o	required_ADR.csv: list of 243 ADRs after processing as per the literature
o	ADR_combined.csv: one hot encoded vector of 243 ADRs after processing as per the literature

•	Output
o	pca_dataset_for_5000_features.csv: dataset after PCA reduction
o	pca_loading_for_model.csv: Eigen vectors after PCA reduction
o	pca_center_for_model.csv: moved centers after PCA reduction

		
Process for deployment:
1.	Download the input files for the research work:
a.	Folder: Input Dataset
b.	Files:
i.	label_two_side.csv: Dataset for Adverse Drug Reactions caused by drugs in combination as in PharmGKB dataset
ii.	GO_transformed_signatures_log.csv : Expression value of Gene Ontology terms for 20k drugs. 
iii.	LINCS_Gene_Experssion_signatures_CD.csv : Differential Gene Expression (DGE) of 978 landmark genes for 20k drugs.
iv.	MACCS_bitmatrix.csv: Chemical structure of drugs in MACCS format.
The datasets - GO_transformed_signatures_log.csv, LINCS_Gene_Experssion_signatures_CD.csv and  MACCS_bitmatrix.csv can be downloaded from  http://maayanlab.net/SEP-L1000/ . For further information about the dataset, please refer to the publication.

2.	Preprocessing of the input data
a.	Code name: data_preparation_for_model.R
b.	Environment: R
c.	Input: files downloaded in Step 1 
d.	Outputs: (Folder: Processed_Files)
i.	represesntation_of_single_drug.csv : Drug represented by their DGE, expression for Gene Ontology and MACCS bit
ii.	ADR_list.csv: list of Adverse Drug Reaction in categorical text format corresponding to the drugs in combination as extracted from PharmGKB TwoSide table
iii.	dataset_subset.csv: drugs in combination represented as concatenation of each of the drug vectors

3.	Convert the ADR list from Step 2 to one hot encoded format
a.	Code name: one_hot_code.py
b.	Environment: python
c.	Input: ADR_list.csv
d.	Output: (Folder: Processed_Files)
i.	ADR_combine_hot_encoded.csv: One hot encoded vector of Adverse Drug Reaction corresponding to each drug combination

4.	Preprocess the one hot encoded Adverse drug Reaction list
a.	Code name: code_for_required_ADR.R
b.	Environment: R
c.	Input: ADR_combine_hot_encoded.csv
d.	Output: (Folder: Processed Files)
i.	required_ADR.csv: list of 243 ADRs after processing as per the literature
ii.	ADR_combined.csv: one hot encoded vector of 243 ADRs after processing as per the literature
Write up below has been explained for one subset. Similar steps can lead to results for all 5 subsets. However, final output datasets for all the subsets are present in the folder 

5.	Prepare the division of sets
a.	Code name: subset_formation.R
i.	Environment: R
ii.	Input: dataset_subset.csv,ADR_combined.csv
iii.	Output: (Folder: Processed Files)
1.	ADR_subset_1.csv

b.	Code_name:code_for_validation_and_pca.R
i.	Environment: R
ii.	Input: ADR_subset_1.csv, dataset_subset.csv
iii.	Output: (Folder: Processed Files)
1.	dataset_subset_1.csv	
2.	dataset_validation_1.csv
3.	dataset_for_validation_after_pca.csv
4.	dataset_for_validation_after_pca_2000_features.csv
5.	dataset_for_validation_after_pca_3000_features.csv
6.	dataset_for_validation_after_pca_2250_features.csv
7.	dataset_for_validation_after_pca_2500_features.csv
8.	dataset_for_validation_after_pca_2750_features.csv
9.	dataset_for_validation_after_pca_500_features.csv
10.	dataset_for_validation_after_pca_750_features.csv
11.	dataset_for_validation_after_pca_1000_features.csv
12.	dataset_for_validation_after_pca_1250_features.csv
13.	dataset_for_validation_after_pca_1500_features.csv
14.	dataset_for_validation_after_pca_1750_features.csv
15.	dataset_after_pca.csv
16.	dataset_after_pca_2000_features.csv
17.	dataset_after_pca_3000_features.csv
18.	dataset_after_pca_2250_features.csv
19.	dataset_after_pca_2500_features.csv
20.	dataset_after_pca_2750_features.csv
21.	dataset_after_pca_500_features.csv
22.	dataset_after_pca_750_features.csv
23.	dataset_after_pca_1000_features.csv
24.	dataset_after_pca_1250_features.csv
25.	dataset_after_pca_1500_features.csv
26.	dataset_after_pca_1750_features.csv
27.	pca_for_subset_1_for_all_features.csv
28.	center_for_pca_subset_1_all_features.csv
29.	rotation_for_pca_subset_1_all_features.csv
30.	ADR_dataset_for_training_subset_1.csv
31.	ADR_validation_for_validation_subset_1.csv


6.	Run ANN for complete dataset
a.	Code name: subset_formation.R
i.	Environment:Python
ii.	Input: dataset_subset.csv, ADR_dataset_for_training_subset_1.csv, ADR_validation_for_validation_subset_1.csv
iii.	Output: (Folder: output_files)
1.	model-1.json
2.	model-1.h5
3.	predict-1.py


7.	Run machine learning algorithms on the PCA reduced data
a.	Code name: extraTree.py,KNN.py,Logistic_Regression.py,Naïve_Bayes.py,ANN.py
i.	Environment: python (preferably jupyter)
ii.	Input:
1.	dataset_after_pca_1000_features.csv
2.	ADR_dataset_for_training_subset_1.csv
3.	dataset_for_validation_after_pca_1000_features.csv
4.	ADR_validation_for_validation_subset_1.csv
iii.	Output (Folder: output): 486 files – 243 for ADRs and 243 for evaluation metrics of ADR
b.	Code name: compilation.py
i.	Environment: python (preferably jupyter)
ii.	Input:
1.	243 files with listed evaluation metrics for ADRs. Files end with “_acc.csv”
iii.	Output (Folder: output): 
1.	Performance_subset_1.csv
Changing the value of variable “sample” would result with different number of pca components. For example, if sample =1000, number of PCA components taken is 1000.

All the codes are present in the Codes folder

Data Selection
1.	Input Data: 
a.	dataset_subset_1: Training dataset for subset 1
b.	dataset_validation_1.csv : Testing data for subset 1
c.	dataset_subset_2: Training dataset for subset 2
d.	dataset_validation_2.csv : Testing data for subset 2
e.	dataset_subset3: Training dataset for subset 3
f.	dataset_validation_3.csv : Testing data for subset 3
g.	dataset_subset_4: Training dataset for subset 4
h.	dataset_validation_4.csv : Testing data for subset 4
i.	thirty_percent.csv: ADR’s for 30% cut-off
j.	fifty_percent.csv: ADR’s for 50% cut-off
k.	seventy_percent.csv: ADR’s for 70% cut-off
l.	ninety_percent.csv: ADR’s for 90% cut-off

2.	Output Data
a.	ten.csv : ANN Results for 10% cut-off
b.	thirty.csv : ANN Results for 30% cut-off
c.	fifty.csv : ANN Results for 50% cut-off
d.	seventy.csv : ANN Results for 70% cut-off
e.	ninety.csv : ANN Results for 90% cut-off
f.	Training Predictions: Predictions for 4 training subsets for each of the 5 cut offs
g.	Testing Predictions: Predictions for 4 validation subsets for each of the 5 cut offs


3.	Training the model:
a.	Code name:  training.ipynb, comparison.ipynb
b.	Environment: Python

Model Tuning
1.	Input Data: 
a.	dataset_subset_1: Training dataset for subset 1
b.	dataset_validation_1.csv : Testing data for subset 1
c.	dataset_subset_2: Training dataset for subset 2
d.	dataset_validation_2.csv : Testing data for subset 2
e.	dataset_subset3: Training dataset for subset 3
f.	dataset_validation_3.csv : Testing data for subset 3
g.	dataset_subset_4: Training dataset for subset 4
h.	dataset_validation_4.csv : Testing data for subset 4

2.	Output Data
a.	Training Predictions: Predictions for 4 training subsets for different parameters
b.	Testing Predictions: Predictions for 4 validation subsets for different parameters

3.	Training the model:
a.	Code name:  training.ipynb, tuning.ipynb
b.	Environment: Python

Selecting Thresholds
1.	Input Data: 
a.	Training Predictions: Predictions for 4 training subsets for different parameters
b.	Testing Predictions: Predictions for 4 validation subsets for different parameters 

2.	Output Data
a.	selectedThresholds.csv

3.	Training the model:
a.	Code name:  thresholdSelection.ipynb, comparision.ipynb
b.	Environment: Python


Authors:
Susmitha Shankar, Ishita Bhandari, David Okou, Gowri Srinivasa, Prashanth Athri
License
This project is licensed under the MIT License - see the LICENSE.md file for details



