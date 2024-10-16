# Charger les bibliothèques nécessaires
library(tidyverse)
library(caret)
library(e1071)
library(splines)
#library(regmix)
library(mixtools)
library(nnet)

# Charger le jeu de données
df <- read.table("robotics_train.txt", header = TRUE, sep = " ")

# Séparer les variables explicatives des variables cibles
X <- df[, c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8")]
y <- df$y

# Diviser le jeu de données en un ensemble d'entraînement et un ensemble de test
set.seed(123)
train_index <- sample(1:nrow(df), 0.8 * nrow(df))
data.train <- df[train_index, ]
data.test <- df[-train_index, ]
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Entraîner un modèle de régression linéaire simple
model_lm <- lm(y_train ~ ., data = X_train)

# Entraîner un modèle de régression SVM en utilisant une fonction à noyau linéaire
model_svm <- svm(X_train, y_train, kernel = "linear")

# Entraîner un modèle de régression spline en utilisant la fonction bs() de la bibliothèque splines
model_spline <- lm(y_train ~ bs(X_train, knots = c(2, 5, 7)), data = X_train)

# Entraîner un modèle de mélange de régressions en utilisant la fonction regmix() de la bibliothèque regmix
model_regmix <- regmixEM(X_train, y_train)

# Entraîner un modèle de régression avec réseau de neurones en utilisant la fonction nnet() de la bibliothèque nnet
model_nnet <- nnet(X_train, y_train, size = 1)

# Calculer les prédictions pour chaque modèle sur l'ensemble de test
predictions_lm <- predict(model_lm, newdata = X_test)
predictions_svm <- predict(model_svm, newdata = X_test)
predictions_spline <- predict(model_spline, newdata = X_test)
predictions_regmix <- predict(model_regmix, newdata = X_test)
predictions_nnet <- predict(model_nnet, newdata = X_test)

# Calculer le MSE pour chaque modèle sur l'ensemble de test
mse_lm <- mean((y_test - predictions_lm)^2)
mse_svm <- mean((y_test - predictions_svm)^2)
mse_spline <- mean((y_test - predictions_spline)^2)
mse_regmix <- mean((y_test - predictions_regmix)^2)
mse_nnet <- mean((y_test - predictions_nnet)^2)

# Afficher les résultats
cat("MSE pour le modèle de régression linéaire simple :", mse_lm, "\n")
cat("MSE pour le modèle de régression SVM :", mse_svm, "\n")
cat("MSE pour le modèle de régression spline :", mse_spline, "\n")
cat("MSE pour le modèle de mélange de régressions :", mse_regmix, "\n")
cat("MSE pour le modèle de régression avec réseau de neurones :", mse_nnet, "\n")
