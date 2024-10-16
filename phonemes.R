data_phonemes <- read.csv("phoneme_train.txt",header = TRUE,sep=" ",row.names = NULL)
data_phonemes <- data_phonemes[2:length(data_phonemes)]


# classification : knn?? random-foret?? NaiveBayes?? lda?? qda?? mda?? nn

# g??n??rer 2 base de donn??e pour train
x<-data_phonemes
pca <- prcomp(x[,-257])
lambda <- pca$sdev^2
plot(cumsum(lambda)/sum(lambda),type="l",xlab="q",ylab="proportion of explained")
q<-100
X<-pca$x[,1:100]
data_pca <- data.frame(X)
data_pca['y']<-data_phonemes['y']
#scale
data_phonemes[,-257]<-apply(data_phonemes[,-257], 2, scale)
data_pca[,-101]<-apply(data_pca[,-101], 2, scale)

n <- nrow(data_phonemes)
train<-sample(1:n,round(2*n/3))
data_train <- data_phonemes[train,]
data_test <- data_phonemes[-train,]
data_pca_train <- data_pca[train,]
data_pca_test <- data_pca[-train,]
ntest<-nrow(data_test)


#knn
library(FNN)
ERR.knn<-matrix(0,20,2)
for(k in 1:20){
  knn.class<-knn(data_train[,-257],
                 data_test[,-257],data_train$y,k=k)
  knn.class_1<-knn(data_pca_train[,-101],
                   data_pca_test[,-101],data_pca_train$y,k=k)
  ERR.knn[k,1]<-mean(data_test$y != knn.class)
  ERR.knn[k,2]<-mean(data_pca_test$y != knn.class_1)
}

plot(1:20,ERR.knn[,1],type="b")
lines(1:20,ERR.knn[,2],type="b",col = "red")
err.knn.min=min(ERR.knn)
print(err.knn.min)

k_min=which(ERR.knn[,1]==min(ERR.knn[,1]),arr.ind=TRUE)

# Naive Bayes
library(naivebayes)
fit.nb<- naive_bayes(as.factor(y)~.,data=data_train)
pred.nb<-predict(fit.nb,newdata=data_test,type="class")
perf <-table(data_test$y,pred.nb)
print(perf)
err.nb <-1-sum(diag(perf))/ntest
print(err.nb)

# LDA
library(MASS)
fit.lda<- lda(y~.,data=data_train)
pred.lda<-predict(fit.lda,newdata=data_test)
perf <-table(data_test$y,pred.lda$class)
print(perf)
err.lda <- 1-sum(diag(perf))/ntest
print(err.lda)

# LDA_pca
fit.lda_pca<- lda(y~.,data=data_pca_train)
pred.lda_pca<-predict(fit.lda_pca,newdata=data_pca_test)
perf_pca <-table(data_pca_test$y,pred.lda_pca$class)
print(perf_pca)
err.lda_pca <- 1-sum(diag(perf_pca))/ntest
print(err.lda_pca)

# QDA
fit.qda<- qda(y~.,data=data_pca_train)
pred.qda<-predict(fit.qda,newdata=data_pca_test)
perf <-table(data_pca_test$y,pred.qda$class)
print(perf)
err.qda <-1-sum(diag(perf))/ntest
print(err.qda)
#performance tr??s mal ???


# Random forests
library(randomForest)
fit.RF<-randomForest(as.factor(y) ~ .,data=data_train,importance=TRUE,mtry=16)
pred.RF<-predict(fit.RF,newdata=data_test,type="class")
err.RF<-1-mean(data_test$y==pred.RF)
print(err.RF)
varImpPlot(fit.RF)

#Random forests en pca (performance mieux)
fit.RF.pca<-randomForest(as.factor(y) ~ .,data=data_pca_train,importance=TRUE,mtry=10)
pred.RF.pca<-predict(fit.RF.pca,newdata=data_pca_test,type="class")
err.RF.pca<-1-mean(data_pca_test$y==pred.RF.pca)
print(err.RF.pca)
varImpPlot(fit.RF.pca) 


#mda
library(mclust)
fit.mda <-  MclustDA(data_train[,-257],as.factor(data_train[,257]))
summary(fit.mda, newdata = data_test[,-257], newclass = data_test[,257])

library(mda)
fit.mda<- mda(y~.,data=data_train)
pred.mda<-predict(fit.mda,newdata=data_test)
perf <-table(data_test$y,pred.mda)
print(perf)
pred.mda <-1-sum(diag(perf))/ntest
print(pred.mda)

#nn
#library(nnet)

#nn<- nnet(as.factor(y) ~ ., data=data_train,size=30, 
#          decay=0.1,maxit = 200,MaxNWts=100000)

#nn<- nnet(as.factor(y) ~ ., data=data_train,size=100, 
#          decay=0.05,maxit = 200,MaxNWts=100000)

#pred<-predict(nn,newdata=data_test,type="class")
#perf <-table(data_test$y,pred)
#print(perf)
#pred.nn <-1-sum(diag(perf))/ntest
#print(pred.nn)

# Selection of the optimal weight decay coefficient by 5-fold cross-validation
library(nnet)
library("caret")
K<-5
ntrain<-nrow(data_phonemes)
lambda<-c(0,0.01,0.1,1,5,10,100)
N<-length(lambda)
err<-matrix(0,N)
folds<-createFolds(y=data_phonemes[,257],k=K)
for(i in (1:N)){
  for(k in (1:K)){
    nn<- nnet(as.factor(y) ~ ., data=data_phonemes[-folds[[k]],],size=5, 
              decay=lambda[i],trace=TRUE,MaxNWts = 100000,maxit = 1000)
    pred<-predict(nn,newdata=data_phonemes[folds[[k]],],type="class")
    err[i]<-err[i]+ (1-mean(data_phonemes[folds[[k]],]$y==pred))
  }
  err[i]<-err[i]/K
}
plot(lambda,err,type='l')
lambda.opt<-lambda[which.min(ERRmean)] # Best decay coefficient

# Re-training on the whole training set using the optimal lambda
loss<-Inf
for(i in 1:1){
  nn<- nnet(as.factor(y)~., data=data_train, size=50,MaxNWts=100000,
            trace=TRUE,decay=lambda.opt, maxit = 300)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn.best<-nn
  }
}
pred.nn<-predict(nn.best,newdata=data_test,type="class")
err.nn=1-mean(data_test$y==pred.nn)
print(err.nn) 

# Effectuez une validation crois??e k-fold sur tous les mod??les pour s??lectionner le mod??le final
# knn lda lda_pca mda random-foret nn
K<-5
ntrain<-nrow(data_phonemes)
err<-matrix(0,6,K)
folds<-createFolds(y=data_phonemes[,257],k=K)
for(k in (1:K)){
  #knn
  knn.class<-knn(data_phonemes[-folds[[k]],-257],
                 data_phonemes[folds[[k]],-257],data_phonemes[-folds[[k]],257],k=k_min)
  err[1,k]<-mean(data_phonemes[folds[[k]],257] != knn.class)
  
  # LDA
  fit.lda<- lda(y~.,data=data_phonemes[-folds[[k]],])
  pred.lda<-predict(fit.lda,newdata=data_phonemes[folds[[k]],])
  err[2,k]<-mean(data_phonemes[folds[[k]],257] != pred.lda$class)
  
  # LDA_pca
  fit.lda.pca<- lda(y~.,data=data_pca[-folds[[k]],])
  pred.lda.pca<-predict(fit.lda.pca,newdata=data_pca[folds[[k]],])
  err[3,k]<-mean(data_pca[folds[[k]],101] != pred.lda.pca$class)
  
  #mda
  fit.mda<- mda(y~.,data=data_phonemes[-folds[[k]],])
  pred.mda<-predict(fit.mda,newdata=data_phonemes[folds[[k]],])
  err[4,k]<-mean(data_phonemes[folds[[k]],257] != pred.mda)
  
  #random-foret
  fit.RF<-randomForest(as.factor(y) ~ .,data=data_phonemes[-folds[[k]],],importance=TRUE,mtry=16)
  pred.RF<-predict(fit.RF,newdata=data_phonemes[folds[[k]],],type="class")
  err[5,k]<-mean(data_phonemes[folds[[k]],257] != pred.RF)
  
  #nn
  nn<- nnet(as.factor(y) ~ ., data=data_phonemes[-folds[[k]],],size=50, 
            decay=lambda.opt,trace=TRUE,MaxNWts = 100000,maxit = 300)
  pred<-predict(nn,newdata=data_phonemes[folds[[k]],],type="class")
  err[6,k]<-mean(data_phonemes[folds[[k]],257] != pred)
}

ERR <- data.frame(t(err))
names(ERR) <- c("KNN","LDA","LDA_PCA","MDA","RF","NN")
boxplot(ERR)


loss<-Inf
for(i in 1:5){
  nn<- nnet(as.factor(y)~., data=data_phonemes, size=80,MaxNWts=100000,
            trace=TRUE,decay=lambda.opt, maxit = 300)
  print(c(i,nn$value))
  if(nn$value<loss){
    loss<-nn$value
    nn.best<-nn
  }
}
model.phoneme<-nn.best
prediction_phoneme <- function(dataset){
  predict(model.phoneme,newdata=dataset,type="class")
}
save("model.phoneme","prediction_phoneme", file = "env.Rdata")