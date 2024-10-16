library("corrplot")
library(kernlab)
library(MASS)

# Récupération des données
data <- read.table("robotics_train.txt",row.names = NULL, header = T)
data <- data[2:length(data)]

# Premier tests
fit <- lm(y ~ ., data=data)
summary(fit)

# Matrice de corrélation
M <- cor(data)
corrplot(M, method="circle")

# Base d'apprentissage / Base de tests
n <- nrow(data)
ntrain <- round(2*n/3)
ntest <- n - ntrain
idx.train <- sample(n, ntrain)
data.train <- data[idx.train,]
data.test <- data[-idx.train,]

#Regression linéaire
fit <- lm(y ~ ., data=data.train)
pred <- predict(fit, newdata = data.test)
pred
plot(pred, data.test$y, xlim = c(0,1.4), ylim = c(0,1.4))
abline(0,1)
mse<- mean((data.test$y-pred)^2)
mse


# Spline
fit1<-lm(y ~ ns(x,df=5), data.train)
fit2<-lm(y ~ bs(x,df=5), data.train)

#SVM
svmfit<-ksvm(y ~ .,data=data.train,scaled=F,type="eps-svr",
             kernel="rbfdot",C=100,epsilon=0.0001,kpar=list(sigma=1))
yhat<-predict(svmfit,newdata=data.test)
plot(yhat, data.test$y, xlim = c(0,1.4), ylim = c(0,1.4))
abline(0,1)
mse<- mean((data.test$y-yhat)^2)
mse


# -----------------------------
# Charger les libraries nécessaires
# Chargement des packages nécessaires
library(lm)
library(splines)
library(rpart)
library(nnet)

# Séparation du jeu de données en données d'entraînement et de test
set.seed(123)  # Fixation de la graine pour obtenir des résultats reproductibles
indices <- sample(1:nrow(data), size = 0.8*nrow(data))  # Sélection aléatoire de 80% des observations
train <- data[indices, ]  # Donnees d'entrainement
test <- data[-indices, ]  # Donnees de test

# Régression linéaire
model <- lm(y ~ ., data = train)  # Construction du modèle
predictions <- predict(model, newdata = test)  # Prédiction sur les données de test
mse <- mean((predictions - test$y)^2)
print(mse)
plot(predictions, test$y, xlim = c(0,1.4), ylim = c(0,1.4))
abline(0,1)

# Splines
model <- lm(y ~ bs(X1, df = 5) + bs(X2, df = 5) + bs(X3, df = 5) + bs(X4, df = 5) + bs(X5, df = 5) +
              bs(X6, df = 5) + bs(X7, df = 5) + bs(X8, df = 5),
            data = train)  # Construction du modèle
predictions <- predict(model, newdata = test)  # Prédiction sur les données de test
mse <- mean((predictions - test$y)^2)
print(mse)
plot(predictions, test$y, xlim = c(0,1.4), ylim = c(0,1.4))
abline(0,1)

# Arbre de décision
model <- rpart(y ~ ., data = train)  # Construction du modèle
predictions <- predict(model, newdata = test)  # Prédiction sur les données de test
mse <- mean((predictions - test$y)^2)
print(mse)
plot(predictions, test$y, xlim = c(0,1.4), ylim = c(0,1.4))
abline(0,1)

indices <- sample(1:nrow(data), size = 0.8*nrow(data))  # Sélection aléatoire de 80% des observations
train <- data[indices, ]  # Donnees d'entrainement
test <- data[-indices, ]  # Donnees de test

# Réseau neuronal
model <- nnet(y ~ ., data = train, size = 15)  # Construction du modèle avec 10 neurones cachés
predictions <- predict(model, newdata = test)  # Prédiction sur les données de test
mse <- mean((predictions - test$y)^2)  # Calcul de l'MSE
print(mse)
plot(predictions, test$y, xlim = c(0,1.4), ylim = c(0,1.4))
abline(0,1)


model <- train(y ~ ., data = train, method = "bag", trControl = trainControl(method = "none"), tuneLength = 10)  # Entraînement du modèle avec bagging
predictions <- predict(model, newdata = test)  # Prédiction sur les données de test

print(mse)
plot(predictions, test$y, xlim = c(0,1.4), ylim = c(0,1.4))
abline(0,1)

