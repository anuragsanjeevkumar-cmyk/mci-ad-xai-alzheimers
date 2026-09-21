
load("C:/Users/anura/Desktop/MCI_AD_XAI_Project/Data/ADNIMERGE2/ADNIMERGE2/data/UCSFFSX7.rda")
obj <- get("UCSFFSX7")
write.csv(obj, "C:/Users/anura/Desktop/MCI_AD_XAI_Project/Data/UCSFFSX7_from_R.csv", row.names=FALSE)
