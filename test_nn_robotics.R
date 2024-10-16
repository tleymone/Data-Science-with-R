# Install and load the necessary packages
install.packages(c("e1071", "kernlab", "neuralnet", "caret"))
library(e1071)
library(kernlab)
library(neuralnet)
library(caret)

# Load the dataset
data <- read.table("robotics_train.txt", header = TRUE, sep = " ", quote = "\"", row.names = NULL)
data <- data[2:length(data)]

# Split the data into input features and output labels
X <- data[, 1:8]
y <- data[, 9]

# Split the data into a train and test set
set.seed(123)
train_ind <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_ind, ]
y_train <- y[train_ind]
X_test <- X[-train_ind, ]
y_test <- y[-train_ind]

# Define a set of models to evaluate
models <- list(
  svm = svm(y ~ ., data = data, kernel = "linear"),
  svm_poly = svm(y ~ ., data = data, kernel = "polynomial"),
  svm_radial = svm(y ~ ., data = data, kernel = "radial"),
  ksvm = kernlab::ksvm(y ~ ., data = data, kernel = "rbfdot")
 # nn = neuralnet(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8, data = data, hidden = 5)
)

# Train and evaluate each model on the train set
results <- lapply(models, function(model) {
  pred <- predict(model, X_train)
  acc <- caret::confusionMatrix(pred, y_train)$overall["Accuracy"]
  return(acc)
})
names(results) <- names(models)

# Select the best model based on the results from the train set
best_model <- names(which.max(results))

# Fine-tune the selected model using hyperparameter optimization and feature selection
tuned_model <- caret::train(
  y ~ ., data = data, method = best_model,
  tuneGrid = caret::expand.grid(. . .),  # specify hyperparameter values to try
  trControl = caret::trainControl(method = "cv")
)

# Evaluate the final model on the test set
predictions <- predict(tuned_model, X_test)
accuracy <- caret::confusionMatrix(predictions, y_test
                                   