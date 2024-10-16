# Charger le fichier en tant que data frame
library(unix)
library(ggplot2)
library(e1071)
library(splines)
library(kernlab)
library(nnet)
library(dplyr)
data <- read.table("robotics_train.txt", header = TRUE, sep = " ")

# Sélectionner les variables explicatives (X) et la variable à prédire (y)
X <- subset(data, select = -y)
y <- data$y

# Diviser les données en un ensemble d'entraînement et un ensemble de test
train_index <- sample(1:nrow(data), 0.8 * nrow(data))
data.train <- data[train_index, ]
data.test <- data[-train_index, ]
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# Créer un modèle de régression linéaire
model_linear <- lm(y_train ~ ., data = X_train)

# Créer un modèle de régression généralisée
model_glm <- glm(y_train ~ ., data = X_train, family = gaussian)
# Créer un nouvel enregistrement
#new_record <- data.frame(X2 = 0, X3 = 1, X4 = 0, X5 = -1, X6 = 1, X7 = 0, X8 = -1)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle de régression linéaire
prediction_linear <- predict(model_linear, X_test)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle de régression généralisée
prediction_glm <- predict(model_glm, X_test, type = "response")

# Calculer le coefficient de détermination (R^2) pour le modèle de régression linéaire
r2_linear <- summary(model_linear)$r.squared

# Calculer l'erreur quadratique moyenne (MSE) pour le modèle de régression linéaire
predictions_linear <- predict(model_linear, X_test)
mse_linear <- mean((predictions_linear - y_test)^2)

# Tracer les prédictions et les valeurs réelles du modèle de régression linéaire sur un graphique
ggplot(data = data.frame(y_test, predictions_linear), aes(x = y_test, y = predictions_linear)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Calculer le coefficient de détermination (R^2) pour le modèle de régression généralisée
r2_glm <- summary(model_glm)$r.squared

# Calculer l'erreur quadratique moyenne (MSE) pour le modèle de régression généralisée
predictions_glm <- predict(model_glm, X_test, type = "response")

mse_glm <- mean((predictions_glm - y_test)^2)

# Tracer les prédictions et les valeurs réelles du modèle de régression linéaire sur un graphique
ggplot(data = data.frame(y_test, predictions_glm), aes(x = y_test, y = predictions_glm)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Créer un modèle SVM
model_svm <- svm(y_train ~ ., data = X_train, kernel = "radial")

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle SVM
predictions_svm <- predict(model_svm, X_test)

mse_svm <- mean((predictions_svm - y_test)^2)

ggplot(data = data.frame(y_test, predictions_svm), aes(x = y_test, y = predictions_svm)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Charger la bibliothèque splines

# Créer un modèle de régression avec splines
model_spline <- lm(y ~ bs(X1,df=5) + bs(X2,df=5) + bs(X3,df=5) + bs(X4,df=5) + bs(X5,df=5) + bs(X6,df=5) + bs(X7,df=5) + bs(X8,df=5), data = data.train)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle de spline
predictions_spline <- predict(model_spline, X_test)

mse_spline <- mean((predictions_spline - y_test)^2)

ggplot(data = data.frame(y_test, predictions_spline), aes(x = y_test, y = predictions_spline)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Créer un modèle à noyaux
model_kernel <- ksvm(y_train ~ ., data = X_train, kernel = "rbfdot")

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle à noyaux
predictions_kernel <- predict(model_kernel, X_test)

mse_kernel <- mean((predictions_kernel - y_test)^2)

ggplot(data = data.frame(y_test, predictions_kernel), aes(x = y_test, y = predictions_kernel)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Créer un modèle de réseau de neurones simple
model_simple <- nnet(y_train ~ ., data = X_train, size = 5)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle simple
predictions_simple <- predict(model_simple, X_test)

mse_simple <- mean((predictions_simple - y_test)^2)

ggplot(data = data.frame(y_test, predictions_simple), aes(x = y_test, y = predictions_simple)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Créer un modèle de réseau de neurones avec des neurones cachés
model_10 <- nnet(y_train ~ ., data = X_train, size = 10)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle avec des neurones cachés
predictions_10 <- predict(model_10, X_test)

mse_10 <- mean((predictions_10 - y_test)^2)

ggplot(data = data.frame(y_test, predictions_10), aes(x = y_test, y = predictions_10)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Créer un modèle de réseau de neurones avec des neurones cachés
model_15 <- nnet(y_train ~ ., data = X_train, size = 15)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle avec des neurones cachés
predictions_15 <- predict(model_15, X_test)

mse_15 <- mean((predictions_15 - y_test)^2)

ggplot(data = data.frame(y_test, predictions_15), aes(x = y_test, y = predictions_15)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Créer un modèle de réseau de neurones avec des neurones cachés
model_20 <- nnet(y_train ~ ., data = X_train, size = 20)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle avec des neurones cachés
predictions_20 <- predict(model_20, X_test)

mse_20 <- mean((predictions_20 - y_test)^2)

ggplot(data = data.frame(y_test, predictions_20), aes(x = y_test, y = predictions_20)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

# Créer un modèle de réseau de neurones avec des neurones cachés
model_40 <- nnet(y_train ~ ., data = X_train, size = 40)

# Prédire la valeur de y pour le nouvel enregistrement avec le modèle avec des neurones cachés
predictions_40 <- predict(model_20, X_test)

mse_40 <- mean((predictions_40 - y_test)^2)

ggplot(data = data.frame(y_test, predictions_40), aes(x = y_test, y = predictions_40)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")




#### CV du svm
library(caret)
# Séparer les données en variables d'entrée (X) et cible (y)
X <- data[-9]
y <- data$y
k <- 10
control <- trainControl(method = "cv", number = k)
model_svm <- train(x = X, y = y, method = "svmRadial", trControl = control)
plot(model_svm)
predictions_svm2 <- predict(model_svm, X_test)
mse_svm2 <- mean((predictions - y_test)^2)
ggplot(data = data.frame(y_test, predictions_svm2), aes(x = y_test, y = predictions_svm2)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")

ggplot(data = data.frame(y_test, predictions_svm), aes(x = y_test, y = predictions_svm)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")



# tentative de CV SVM
# Séparer les données en variables d'entrée (X) et cible (y)
X <- select(data, -y)
y <- data$y
k <- 20
folds <- createFolds(y, k = k)
results <- list()
for (i in 1:k) {
  # Sélectionner le pli de données de test
  test_index <- folds[[i]]
  X_test <- X[test_index, ]
  y_test <- y[test_index]
  
  # Sélectionner les données d'entraînement
  train_index <- unlist(folds[-i])
  X_train <- X[train_index, ]
  y_train <- y[train_index]
  
  # Entraîner le modèle SVM sur les données d'entraînement
  model_svm <- svm(x = X_train, y = y_train)
  
  # Prédire les cibles sur les données de test
  y_pred <- predict(model_svm, newdata = X_test)
  
  # Calculer l'erreur quadratique moyenne (MSE) sur les données de test
  mse <- mean((y_test - y_pred) ^ 2)
  
  # Ajouter les résultats à la liste
  results[[i]] <- mse
}
mean_mse <- mean(unlist(results))
boxplot(results)


plot(unlist(results))

# 0.009119795
# 0.008535029
