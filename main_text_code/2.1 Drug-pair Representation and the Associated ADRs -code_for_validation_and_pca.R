ADR_subset_1<-read.csv("ADR_subset_1.csv")
dataset_subset<-read.csv("dataset_subset.csv")
'%nin%' = Negate('%in%')
ADR_subset_1$X.1<-NULL
dataset_subset$pert_id_1<-NULL

dataset_subset$pert_id_2<-NULL

dataset_subset$X<-NULL

dataset_subset_1<-subset(dataset_subset,dataset_subset$combined_pert %in% ADR_subset_1$X)
dataset_validation_1<-subset(dataset_subset,dataset_subset$combined_pert %nin% ADR_subset_1$X)
write.csv(dataset_subset_1,"dataset_subset_1.csv")
write.csv(dataset_validation_1,"dataset_validation_1.csv")

row.names(dataset_subset_1)<-dataset_subset_1$combined_pert
row.names(dataset_validation_1)<-dataset_validation_1$combined_pert
dataset_subset_1$combined_pert<-NULL
dataset_validation_2$combined_pert<-NULL

dataset_for_model<-dataset_subset_1[,1:11164]
dataset_for_model_ADR<-dataset_subset_1[,11165:11407]
row.names(dataset_for_model_ADR)<-row.names(dataset_for_model)

dataset_for_validation<-dataset_validation_1[,1:11164]
dataset_for_validation_ADR<-dataset_validation_1[,11165:11407]
row.names(dataset_for_validation_ADR)<-row.names(dataset_for_validation)

pca_for_subset_1<-prcomp(dataset_for_model)
#write.csv(pca_for_subset_1$x,"pca_for_subset_1_for_all_features.csv")
#write.csv(pca_for_subset_1$center,"center_for_pca_subset_1_all_features.csv")
#write.csv(pca_for_subset_1$rotation,"rotation_for_pca_subset_1_all_features.csv")

center<-as.data.frame(t(center))
center<-as.data.frame(t(pca_for_subset_1$center))
rotation<-pca_for_subset_1$rotation

center_new<-center[rep(seq_len(nrow(center)), each = 2000),]
dataset_center<-dataset_for_validation - center_new
dataset_prediction<-as.matrix(dataset_center) %*% as.matrix(pca_for_subset_1$rotation)

write.csv(dataset_prediction,"dataset_for_validation_after_pca.csv")
write.csv(dataset_prediction[,1:2000],"dataset_for_validation_after_pca_2000_features.csv")
write.csv(dataset_prediction[,1:3000],"dataset_for_validation_after_pca_3000_features.csv")
write.csv(dataset_prediction[,1:2250],"dataset_for_validation_after_pca_2250_features.csv")
write.csv(dataset_prediction[,1:2500],"dataset_for_validation_after_pca_2500_features.csv")
write.csv(dataset_prediction[,1:2750],"dataset_for_validation_after_pca_2750_features.csv")
write.csv(dataset_prediction[,1:500],"dataset_for_validation_after_pca_500_features.csv")
write.csv(dataset_prediction[,1:750],"dataset_for_validation_after_pca_750_features.csv")
write.csv(dataset_prediction[,1:1000],"dataset_for_validation_after_pca_1000_features.csv")
write.csv(dataset_prediction[,1:1250],"dataset_for_validation_after_pca_1250_features.csv")
write.csv(dataset_prediction[,1:1500],"dataset_for_validation_after_pca_1500_features.csv")
write.csv(dataset_prediction[,1:1750],"dataset_for_validation_after_pca_1750_features.csv")

write.csv(pca_for_subset_1$x,"dataset_after_pca.csv")
write.csv(pca_for_subset_1$x[,1:2000],"dataset_after_pca_2000_features.csv")
write.csv(pca_for_subset_1$x[,1:3000],"dataset_after_pca_3000_features.csv")
write.csv(pca_for_subset_1$x[,1:2250],"dataset_after_pca_2250_features.csv")
write.csv(pca_for_subset_1$x[,1:2500],"dataset_after_pca_2500_features.csv")
write.csv(pca_for_subset_1$x[,1:2750],"dataset_after_pca_2750_features.csv")
write.csv(pca_for_subset_1$x[,1:500],"dataset_after_pca_500_features.csv")
write.csv(pca_for_subset_1$x[,1:750],"dataset_after_pca_750_features.csv")
write.csv(pca_for_subset_1$x[,1:1000],"dataset_after_pca_1000_features.csv")
write.csv(pca_for_subset_1$x[,1:1250],"dataset_after_pca_1250_features.csv")
write.csv(pca_for_subset_1$x[,1:1500],"dataset_after_pca_1500_features.csv")
write.csv(pca_for_subset_1$x[,1:1750],"dataset_after_pca_1750_features.csv")


write.csv(pca_for_subset_1$x,"pca_for_subset_1_for_all_features.csv")
write.csv(pca_for_subset_1$center,"center_for_pca_subset_1_all_features.csv")
write.csv(pca_for_subset_1$rotation,"rotation_for_pca_subset_1_all_features.csv")

write.csv(dataset_for_model_ADR,"ADR_dataset_for_training_subset_1.csv")
write.csv(dataset_for_validation_ADR,"ADR_validation_for_validation_subset_1.csv")
