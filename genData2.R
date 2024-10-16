
model.phoneme <-
  
# SVM
library("e1071")
data.robotics <- read.table("robotics_train.txt", header = TRUE, sep = " ")
model.robotics <- svm(x = subset(data.robotics, select = -y), y = data.robotics$y, kernel = "radial")
library(nnet)
data<-read.csv("communities_train.csv")
X <- subset(data, select = -c(county,community, fold, communityname, ViolentCrimesPerPop))
y <- data$ViolentCrimesPerPop
df_modified <- X
pct_na <- colSums(is.na(df_modified)) / nrow(df_modified)
df_modified <- df_modified[, pct_na < 0.8]
df_modified[is.na(df_modified)] <- 0.545
model.communities <- nnet(y ~ ., data = df_modified, size = 2)

  
prediction_phoneme <- function(dataset) {
    # Ne pas oublier de charger **à l’intérieur de la fonction** les
    # bibliothèques utilisées.
    #library(...)
    # Attention à ce que retourne un modèle en prédiction. Par exemple,
    # la lda retourne une liste nommée. On sélectionne alors les
    # classes.
  }
prediction_robotics <- function(dataset) {
  library("e1071")
  predict(model.robotics, newdata=dataset)
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
  "model.robotics",
  "model.communities",
  "prediction_phoneme",
  "prediction_robotics",
  "prediction_communities",
  file = "env.Rdata"
)
