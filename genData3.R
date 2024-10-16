library(e1071)
library(nnet)
library(zoo)
library(randomForest)
library(kernlab)


data_phonemes <- read.csv("phoneme_train.txt",header = TRUE,sep=" ",row.names = NULL)
data_phonemes <- data_phonemes[2:length(data_phonemes)]
load("nn.best.rdata")
model.phoneme <- nn.best
  
# KSVM
data.robotics <- read.table("robotics_train.txt", header = TRUE, sep = " ")
model.robotics <- ksvm(y ~ ., data = data.robotics, kernel = "rbfdot", type = "eps-bsvr")
  
  
model.communities <- 
  
prediction_phoneme <- function(dataset) {
  library("nnet")
  predict(model.phoneme, newdata=dataset,type="class")
}
prediction_robotics <- function(dataset) {
  library(kernlab)
  predict(model.robotics, dataset)
  
}
prediction_communities <- function(dataset) {
  
}


save(
  "model.phoneme",
  "model.robotics",
  "model.communities",
  "prediction_phoneme",
  "prediction_robotics",
  "prediction_communities",
  file = "env.Rdata"
)

