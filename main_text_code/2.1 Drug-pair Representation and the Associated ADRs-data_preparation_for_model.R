setwd("~/susmitha/ADR/ADR_related")

#read the datasets
Gene_Epression<-read.csv("LINCS_Gene_Experssion_signatures_CD.csv")
Enriched_Pathways<-read.csv("GO_transformed_signatures_log.csv")
Chemical_Descriptor<-read.csv("MACCS_bitmatrix.csv")
combined_medication_ADR<-read.csv("label_two_side.csv")

#-------------------------analysis of two sided----------------------------------------------------------------------#

#library(dplyr)
#library(plyr)
library(tidyr)
#library(tidyverse)
#count(unique(combined_medication_ADR$event_name))
#total<-combined_medication_ADR
#names(total)
#total$X<-NULL
#total$event_name<-NULL
#total_unique<-unique(total)
#total_non_na<-total[complete.cases(total),]
#total_non_na_unique<-unique(total_non_na)

#-----------------------------------------------------------------------------------------------------------------------#

#representation_of_single_drug

dataset_for_combined_single_drug<-merge(Gene_Epression, Enriched_Pathways, by = "pert_id")
dataset_for_combined_single_drug<-merge(dataset_for_combined_single_drug,Chemical_Descriptor, by = "pert_id")
write.csv(dataset_for_combined_single_drug,"represesntation_of_single_drug.csv")

# formation of complete dataset

combined_medication_ADR<-as.data.frame(combined_medication_ADR[,c(3,4,2)])
combined_medication_ADR<-combined_medication_ADR[complete.cases(combined_medication_ADR),]
combined_medication_ADR_pert<-combined_medication_ADR[,c(1,2)]
combined_medication_ADR_pert_unique<-unique(combined_medication_ADR_pert)
names(combined_medication_ADR_pert_unique)[names(combined_medication_ADR_pert) == "pert_id_1"]<-"pert_id"
dataset_for_combined_medication<-merge(combined_medication_ADR_pert_unique,dataset_for_combined_single_drug, by = "pert_id")
colnames(dataset_for_combined_medication) <- paste("pert_1", colnames(dataset_for_combined_medication), sep = "_")
names(dataset_for_combined_medication)[names(dataset_for_combined_medication) == "pert_1_pert_id_2"]<-"pert_id"
dataset_for_combined_medication<-merge(dataset_for_combined_medication,dataset_for_combined_single_drug, by = "pert_id")
names(dataset_for_combined_medication)[names(dataset_for_combined_medication) == "pert_id"]<-"pert_id_2"
names(dataset_for_combined_medication)[names(dataset_for_combined_medication) == "pert_1_pert_id"]<-"pert_id_1"
View(dataset_for_combined_medication[1:10,1:10])

#ormation of combined columns - names of combined drugs
perts_of_combined_med<-dataset_for_combined_medication[,c(2,1)]
View(perts_of_combined_med[1:10,])
perts_of_combined_med<-unite(perts_of_combined_med,col = "combined_pert")
dataset_for_combined_medication<-cbind(perts_of_combined_med,dataset_for_combined_medication)

# formation of single row entry for combined drugs to get one hot encoding
combined_medication_ADR_for_label<-combined_medication_ADR
event_name <- combined_medication_ADR_for_label$event_name
combined_medication_ADR_for_label$event_name<-NULL
View(combined_medication_ADR_for_label[1:10,])
combined_medication_ADR_for_label<-unite(combined_medication_ADR_for_label, col = "combined_pert")
combined_medication_ADR_for_label<-cbind(combined_medication_ADR_for_label,event_name)
combined_medication_ADR_for_label$event_name<-paste0("'",event_name,"'")
names(combined_medication_ADR_for_label)
View(combined_medication_ADR_for_label[1:10,])
combined_medication_ADR_for_label<-as.data.frame(combined_medication_ADR_for_label,stringsAsFactors = FALSE)
combined_medication_ADR_for_label_grouped<-combined_medication_ADR_for_label %>% group_by(combined_medication_ADR_for_label$combined_pert) %>% summarise(adr = paste(event_name, collapse = ","))
combined_medication_ADR_for_label_grouped$adr<-paste0("{",combined_medication_ADR_for_label_grouped$adr,"}")
#View(combined_medication_ADR_for_label_grouped[1:10,])
write.csv(combined_medication_ADR_for_label_grouped$adr,file = "ADR_list.csv")
# the ADR_list is then sent to python script to be one hot encoded
one_hot_encoded<-read.csv("one_hot_encoded_combined_medication.csv")
one_hot_encoded$X<-NULL
View(one_hot_encoded[1:10,1:10])
#-------------------------analysis------------------------------------------------------#

one_hot_encoded_row_sum<-one_hot_encoded
one_hot_encoded_col_sum<-as.data.frame(colSums(one_hot_encoded_row_sum))
subset_based_on_column<- subset(one_hot_encoded_col_sum, one_hot_encoded_col_sum$`colSums(one_hot_encoded_row_sum)` > 3739)
subset_based_on_column<-t(subset_based_on_column)
subset_label<-subset(one_hot_encoded, select = (names(one_hot_encoded) %in% row.names(subset_based_on_column)))

#-------------------------------------------------------------------------------------------#
# form complete dataset with ADR included 

label_dataset<-cbind(combined_medication_ADR_for_label_grouped$`combined_medication_ADR_for_label$combined_pert`,one_hot_encoded)
label_dataset_subset<-cbind(combined_medication_ADR_for_label_grouped$`combined_medication_ADR_for_label$combined_pert`, subset_label)
names(label_dataset)[names(label_dataset) == "combined_medication_ADR_for_label_grouped$`combined_medication_ADR_for_label$combined_pert`"]<-"combined_pert"
names(label_dataset_subset)[names(label_dataset_subset) == "combined_medication_ADR_for_label_grouped$`combined_medication_ADR_for_label$combined_pert`"]<-"combined_pert"
View(label_dataset[1:10,1:10])

dataset<-merge(dataset_for_combined_medication,label_dataset, by = "combined_pert")
dataset_subset<-merge(dataset_for_combined_medication,label_dataset_subset, by = "combined_pert")
dataset_subset_non_na<-dataset_subset[complete.cases(dataset_subset),]
write.csv(dataset, file = "dataset_for_combined_medication.csv")
write.csv(dataset_subset, file = "dataset_subset.csv")


