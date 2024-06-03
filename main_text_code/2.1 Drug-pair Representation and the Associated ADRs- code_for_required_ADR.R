#code to get ADR combined

one_hot_data<-read.csv("ADR_combine_hot_encoded.csv")
one_hot_data$X<-NULL
row.names(one_hot_data)<-one_hot_data$combined_medication_ADR_for_label.combined_pert
one_hot_data$combined_medication_ADR_for_label.combined_pert<-NULL
columnSum<-as.data.frame(colSums(one_hot_data))
required_ADR<-subset(columnSum,colSums(one_hot_data) > 3739)
write.csv(required_ADR,"required_ADR.csv")
ADR_combined<-subset(one_hot_data, select = (names(one_hot_data) %in% row.names(required_ADR)))
write.csv(ADR_combined,"ADR_combined.csv")
