library(e1071)
library(nnet)
library(zoo)

library(randomForest)

data_phonemes <- read.csv("phoneme_train.txt",header = TRUE,sep=" ",row.names = NULL)
data_phonemes <- data_phonemes[2:length(data_phonemes)]
#random-foret
fit.RF<-randomForest(as.factor(y) ~ .,data=data_phonemes,importance=TRUE,mtry=16)
model.phoneme <- fit.RF

# NN with SVM
data.robotics <- read.table("robotics_train.txt", header = TRUE, sep = " ")
model.svm <- svm(x = subset(data.robotics, select = -y), y = data.robotics$y, kernel = "radial")
model.robotics <- nnet(data.robotics$y ~ ., data = subset(data.robotics, select = -y), size = 80,trace=TRUE,
                       MaxNWts=100000, maxit = 1000)

library(nnet)
data<-read.csv("communities_train.csv")
X <- subset(data, select = -c(county,community, fold, communityname, ViolentCrimesPerPop))
y <- data$ViolentCrimesPerPop
df_modified <- X
pct_na <- colSums(is.na(df_modified)) / nrow(df_modified)
df_modified <- df_modified[, pct_na < 0.8]
df_modified[is.na(df_modified)] <- 0.545
model.communities <- nnet(y ~ ., data = df_modified, size = 100, MaxNWts = 10350)

prediction_phoneme <- function(dataset) {
  library(randomForest)
  predict(model.phoneme,newdata=dataset,type="class")
}

prediction_robotics <- function(dataset) {
  library("nnet")
  library("e1071")
  
  # Prédictions avec le modèle de réseau de neurones
  predictions_nn <- predict(model.robotics, newdata=dataset)
  
  # Indices des exemples pour lesquels la probabilité de la prédiction du modèle de réseau de neurones est supérieure à 0,92
  idx <- which(predictions_nn > 0.95)
  
  # Si aucun exemple n'a une probabilité supérieure à 0,92, renvoyer directement les prédictions du modèle de réseau de neurones
  if (length(idx) == 0) {
    return(predictions_nn)
  }
  
  # Sinon, faire la prédiction avec le modèle SVM uniquement pour les exemples dont la probabilité de la prédiction du modèle de réseau de neurones est supérieure à 0,92
  predictions_SVM <- predict(model.svm, newdata = dataset[idx,])
  
  # Remplacement des prédictions du modèle de réseau de neurones par celles du modèle SVM pour les exemples dont la probabilité de la prédiction du modèle de réseau de neurones est supérieure à 0,92
  predictions_nn[idx] <- predictions_SVM
  
  predictions_nn
}



prediction_communities <- function(dataset) {
  library(nnet)
  df_modified <- subset(dataset, select = -c(county,community, fold, communityname))
  df_modified[is.na(df_modified)] <- 0.2
  prediction_linear <- predict(model.communities, df_modified)
  prediction_linear  
}


save(
  "model.phoneme",
  "model.svm",
  "model.robotics",
  "model.communities",
  "prediction_phoneme",
  "prediction_robotics",
  "prediction_communities",
  file = "env.Rdata"
)
