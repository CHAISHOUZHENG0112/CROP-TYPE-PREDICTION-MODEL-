# --------------------------------------------------
# R script: CROP TYPE PREDICTION
# Project: CROP TYPE PREDICTION
#
# Date: 17/5/2025
# Time: 3:54AM
# Author: CHAI SHOU ZHENG
# Dataset: 'WinnData.csv'
# --------------------------------------------------
# clean up the environment before starting
rm(list = ls())

# Load libraries
library(tree)
library(e1071)
library(ROCR)
library(randomForest)
library(adabag)
library(rpart)
library(neuralnet)
library(car)
library(future)   
library(caret)    
library(adabag)   

# Set the directory 
setwd("FIT3152/Assignment2")

# Create individual data set
rm(list = ls())
set.seed(34035958) 
WD = read.csv("WinnData.csv")
WD = WD[sample(nrow(WD), 5000, replace = FALSE),]
WD = WD[,c(sort(sample(1:30,20, replace = FALSE)), 31)]

# Save the extracted data set as a new CSV file
write.csv(WD, "Extracted_WinnData_34035958.csv", row.names = FALSE)

# --------------------------------------------------
# Q1. Data Exploration
# --------------------------------------------------

dim(WD) # View the dimensions of the dataset
str(WD) # View the structure of the dataset
colSums(is.na(WD)) # Check for missing values in each column

# Proportion of “Oats” (class = 1) vs “Other” (class = 2)
table(WD$Class) # View counts of each class

# View proportions of each class as percentages
round(prop.table(table(WD$Class))* 100, 2)

########################
# Desricptive Anaylsis 
########################

# Calculate means of all 20 predictors
colMeans(WD[, -ncol(WD)])  # exclude the last column 'Class'(dependent variable)

# Plot histograms of mean for all predictors
par(mfrow = c(4, 5))  # 4x5 layout

for (col in colnames(WD[, -ncol(WD)])) {
  h <- hist(WD[[col]], 
            main = paste(col),
            xlab = paste(col, "Value"),
            ylab = "Frequency",
            col = "skyblue",
            breaks = 30)
  
  # Draw mean line
  mean_val <- mean(WD[[col]])
  abline(v = mean_val, col = "red", lwd = 2)
  
  # Place label just below top bar height
  text(
    x = mean_val,
    y = max(h$counts) * 0.9,  # slightly below top
    labels = paste("Mean =", round(mean_val, 2)),
    pos = 4,
    col = "red",
    cex = 0.75
  )
}

par(mfrow = c(1, 1))  # Reset layout



# Standard deviation of all 20 predictors (excluding 'Class')
sapply(WD[, -ncol(WD)], sd)

# Calculate standard deviations
std_devs <- sapply(WD[, -ncol(WD)], sd)

# Set Y-axis limit slightly above max
y_max <- max(std_devs) + 2  # Add padding for labels

# Create bar plot of standard deviation for all predictors
bp <- barplot(
  std_devs,
  main = "Standard Deviation of Predictor Variables",
  ylab = "Standard Deviation",
  xlab = "",
  col = "skyblue",
  las = 2,
  cex.names = 0.8
)

# Add numeric values on top of each bar
text(
  x = bp,
  y = std_devs + 0.5,
  labels = round(std_devs, 2),
  pos = 3,           # Position above the bar
  cex = 0.7,         # Font size
  col = "black"
)

# Add x-axis label
mtext("Predictor Variables", side = 1, line = 5)

# --------------------------------------------------
# Q2. Data Preprocessing
# --------------------------------------------------
# Convert the target variable 'Class' to a factor
WD$Class <- as.factor(WD$Class)

# Calculate proportion of zeros for each predictor
zero_props <- colMeans(WD[, -ncol(WD)] == 0)

# Convert the proportions into a tidy data frame for easier inspection or plotting
zero_df <- data.frame(
  Feature = names(zero_props),
  Proportion_Zero = as.numeric(zero_props)
)

print(zero_df)


# --------------------------------------------------
# Q3. Train-Test Split
# --------------------------------------------------
set.seed(34035958)  # set seed 

# Create a 70-30 train-test split
train.row <- sample(1:nrow(WD), 0.7 * nrow(WD))

# Subset the dataset into training and testing sets
WD.train <- WD[train.row, ] # 70% training data
WD.test  <- WD[-train.row, ] # 30% testing data

# --------------------------------------------------
# Q4. Classification Model Implementation
# --------------------------------------------------

#################
#  Decision Tree
#################

# Train a Decision Tree model using the training dataset
WD.tree <- tree(Class ~ ., data = WD.train)
print(summary(WD.tree)) # Display summary 

# Visualize the structure of the decision tree
plot(WD.tree)
text(WD.tree, pretty = 0)


#################
#  Naïve Bayes
#################

# Train a Naïve Bayes classifier using default settings
WD.bayes <- naiveBayes(Class ~ ., data = WD.train)


#################
#  Bagging
#################
set.seed(34035958) 
# Train a Bagging ensemble model using 5 bootstrap samples
# Each sample is used to train a separate decision tree
WD.bag <- bagging(Class ~ ., data = WD.train, mfinal = 5)


#################
#  Boosting
#################
set.seed(34035958) 
# Train a Boosting ensemble model with 10 trees
WD.boost <- boosting(Class ~ ., data = WD.train, mfinal = 10)


##################
#  Random Forest
##################
set.seed(34035958)
# Train a Random Forest classifier using default parameters
WD.rf <- randomForest(Class ~ ., data = WD.train, na.action = na.exclude)


# --------------------------------------------------
# Q5.Classification Performance Evaluation
# --------------------------------------------------

# function of evaluate confusion metrics
evaluate_confusion_metrics <- function(model_name, cm) {
  # Assumes cm = table(Predicted, Actual) or similar 2x2 confusion matrix
  TP <- cm["1", "1"]
  TN <- cm["0", "0"]
  FP <- cm["1", "0"]
  FN <- cm["0", "1"]
  
  accuracy  <- (TP + TN) / sum(cm)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  f1_score  <- 2 * precision * recall / (precision + recall)
  
  # Print evaluation results
  cat("\n-------------------------")
  cat(sprintf("\nModel: %s", model_name))
  cat(sprintf("\nAccuracy:  %.4f", accuracy))
  cat(sprintf("\nPrecision: %.4f", precision))
  cat(sprintf("\nRecall:    %.4f", recall))
  cat(sprintf("\nF1 Score:  %.4f", f1_score))
  cat("\n-------------------------\n")
}


#################
# Decision Tree
#################

# Predict classes on test data
WD.predtree.class <- predict(WD.tree, WD.test, type = "class")

# Confusion matrix
cat("\n# Decision Tree Confusion Matrix\n")
conf.tree <- table(Predicted_Class = WD.predtree.class, Actual_Class = WD.test$Class)
print(conf.tree)

evaluate_confusion_metrics("Decision Tree", conf.tree)


#################
#  Naïve Bayes
#################

# Predict class labels
WD.predbayes.class <- predict(WD.bayes, WD.test)

# Confusion matrix
cat("\n# Naïve Bayes Confusion Matrix\n")
conf.nb <- table(Predicted_Class = WD.predbayes.class, Actual_Class = WD.test$Class)
print(conf.nb)

evaluate_confusion_metrics("Naive Bayes", conf.nb)


#################
#  Bagging
#################

# Predict using Bagging model
WD.predbag <- predict.bagging(WD.bag, WD.test)

# Confusion matrix
cat("\n# Bagging Confusion Matrix\n")
print(WD.predbag$confusion)

evaluate_confusion_metrics("Bagging", WD.predbag$confusion)


#################
#  Boosting
#################

# Predict using Boosting model
WD.predboost <- predict.boosting(WD.boost, newdata = WD.test)

# Confusion matrix
cat("\n# Boosting Confusion Matrix\n")
print(WD.predboost$confusion)

evaluate_confusion_metrics("Boosting", WD.predboost$confusion)


##################
#  Random Forest
##################

# Predict class labels
WD.predrf.class <- predict(WD.rf, WD.test)

# Confusion matrix
cat("\n# Random Forest Confusion Matrix\n")
conf.rf <- table(Predicted_Class = WD.predrf.class, Actual_Class = WD.test$Class)
print(conf.rf)

evaluate_confusion_metrics("Random Forest", conf.rf)


# --------------------------------------------------
# Q6. ROC Curve and AUC Comparison
# --------------------------------------------------

# Function which returns prediction and performance objects for ROC curve based on predicted probabilities and true labels
get_roc_perf <- function(prob_vector, true_labels) {
  pred <- ROCR::prediction(prob_vector, true_labels)
  perf <- ROCR::performance(pred, "tpr", "fpr")
  return(list(pred = pred, perf = perf))
}

# Function which prints AUC given a model name and prediction object
print_auc <- function(model_name, pred_obj) {
  auc <- ROCR::performance(pred_obj, "auc")@y.values[[1]]
  cat(sprintf("AUC (%s): %.4f\n", model_name, auc))
}


##################
#  Decision Tree
##################

# Predict class probabilities on test data
WD.predtree.prob <- predict(WD.tree, WD.test, type = "vector")

# Get ROC prediction and performance
res.tree  <- get_roc_perf(WD.predtree.prob[, 2], WD.test$Class)
WD.tree.pred <- res.tree$pred
WD.tree.perf <- res.tree$perf

# Plot ROC curve
plot(WD.tree.perf, main = "ROC Curve", col = "blue", lwd = 2)
abline(0, 1, lty = 2, col = "gray")  # Random baseline

# AUC
print_auc("Decision Tree", WD.tree.pred)


#################
#  Naïve Bayes
#################

# Predict class probabilities for ROC
WD.predbayes.prob <- predict(WD.bayes, WD.test, type = "raw")

# Get ROC prediction and performance
res.nb  <- get_roc_perf(WD.predbayes.prob[, 2], WD.test$Class)
WD.nb.pred <- res.nb$pred
WD.nb.perf <- res.nb$perf

# Add Naïve Bayes ROC to the existing ROC plot
plot(WD.nb.perf, add = TRUE, col = "chocolate1")

# AUC
print_auc("Naive Bayes", WD.nb.pred)


#################
#  Bagging
#################

# Get ROC prediction and performance
res.bag  <- get_roc_perf(WD.predbag$prob[, 2], WD.test$Class)
WD.bag.pred <- res.bag$pred
WD.bag.perf <- res.bag$perf

# Add to existing ROC plot
plot(WD.bag.perf, add = TRUE, col = "green")

# AUC
print_auc("Bagging", WD.bag.pred)


#################
#  Boosting
#################

# Get ROC prediction and performance
res.boost <- get_roc_perf(WD.predboost$prob[, 2], WD.test$Class)
WD.boost.pred <- res.boost$pred
WD.boost.perf <- res.boost$perf

# Add to existing ROC plot
plot(WD.boost.perf, add = TRUE, col = "deeppink")

# AUC
print_auc("Boosting", WD.boost.pred)


###################
#  Random Forest
###################

# Predict class probabilities for ROC
WD.predrf.prob <- predict(WD.rf, WD.test, type = "prob")

# Get ROC prediction and performance
res.rf <- get_roc_perf(WD.predrf.prob[, 2], WD.test$Class)
WD.rf.pred <- res.rf$pred
WD.rf.perf <- res.rf$perf

# Add to existing ROC plot
plot(WD.rf.perf, add = TRUE, col = "burlywood4", lwd = 2) 

# AUC
print_auc("Random Forest", WD.rf.pred)

# add a legend
legend("bottomright",
       legend = c("Decision Tree", "Naive Bayes", "Bagging", "Boosting", "Random Forest"),
       col = c("blue", "chocolate1", "green", "deeppink", "burlywood4"),  # match your current colors
       lty = 1,
       lwd = 2,
       cex = 0.8,
       box.lty = 0)


# --------------------------------------------------
# Q7. Summary of Classifier Performance
# --------------------------------------------------

# Create a dataframe with all the evaluation metrics
results_q7 <- data.frame(
  Model = c("Decision Tree", "Naive Bayes", "Bagging", "Boosting", "Random Forest"),
  Accuracy = c(0.8487, 0.8147, 0.8733, 0.8767, 0.8773),
  Precision = c(0.3304, 0.2554, 0.4737, 0.5217, 0.5500),
  Recall = c(0.2021, 0.2500, 0.0957, 0.1915, 0.1170),
  F1_Score = c(0.2508, 0.2527, 0.1593, 0.2802, 0.1930)
)

# Print the table
print(results_q7)


cat("\n# AUC Scores\n")
print_auc("Decision Tree", res.tree$pred)
print_auc("Naive Bayes", res.nb$pred)
print_auc("Bagging", res.bag$pred)
print_auc("Boosting", res.boost$pred)
print_auc("Random Forest", res.rf$pred)

# --------------------------------------------------
# Investigation Tasks
# --------------------------------------------------

# --------------------------------------------------
# Q8. Attribute Importance
# --------------------------------------------------

# Decision Tree model
cat("\n# Decision Tree Attribute Importance\n")
print(summary(WD.tree))  # Shows which attributes were used and how many times they appear in splits

# Bagging model
cat("\n# Bagging Attribute Importance\n")
print(WD.bag$importance)  # Higher values indicate greater contribution to classification in bagged trees

# Boosting model
cat("\n# Boosting Attribute Importance\n")
print(WD.boost$importance)  # Importance is based on frequency and quality of splits in boosting iterations

# Random Forest model
cat("\n# Random Forest Attribute Importance\n")
print(WD.rf$importance)  # Measures decrease in node impurity (e.g. Gini) or accuracy when attribute is permuted


# Create bar graph of attribute important for RF
# Extract importance values
importance_vals <- WD.rf$importance[, 1]  # Select the MeanDecreaseGini column
attribute_names <- rownames(WD.rf$importance)
importance_vals


# Create barplot to visualize Random Forest Attribute Importance
barplot(
  importance_vals,
  names.arg = attribute_names,
  las = 2,  # Rotate labels vertically
  col = "skyblue",
  main = "Random Forest - Attribute Importance",
  ylab = "Mean Decrease in Gini",
  cex.names = 0.8
)


#####################################
# Identify Top and Bottom Attributes
#####################################

# Define a helper function to find top and bottom 3 attributes based on importance
get_top_bottom_attributes <- function(importance_vector, model_name) {
  sorted <- sort(importance_vector, decreasing = TRUE)
  top3 <- head(sorted, 3)
  
  # Get the full sorted list
  full_sorted <- sort(importance_vector, decreasing = FALSE)
  bottom3 <- head(full_sorted, 3)
  
  cat("\n==============================\n")
  cat(sprintf("Model: %s\n", model_name))
  cat("------------------------------\n")
  cat("Top 3 Most Important:\n")
  print(top3)
  cat("Bottom 3 Least Important:\n")
  print(bottom3)
  cat("==============================\n")
}

# Bagging
get_top_bottom_attributes(WD.bag$importance, "Bagging")

# Boosting
get_top_bottom_attributes(WD.boost$importance, "Boosting")

# Random Forest (extract from MeanDecreaseGini column)
get_top_bottom_attributes(WD.rf$importance[, 1], "Random Forest")

# --------------------------------------------------
# Q10. Simplified Decision Tree
# --------------------------------------------------
# Re-balance the class 
# Identify indices by class
class0_other_idx <- which(WD$Class == 0)  # "Other"
class1_oats_idx  <- which(WD$Class == 1)  # "Oats"

# Sample equal size from both classes 
set.seed(34035958)  
sample_other_q10 <- sample(class0_other_idx, 600)
sample_oats_q10  <- sample(class1_oats_idx, 600)

# Combine to form balanced training set
balanced_idx_q10 <- c(sample_other_q10, sample_oats_q10)

# Assign training and test sets for Q10
WD.train_q10 <- WD[balanced_idx_q10, ]
WD.test_q10  <- WD[-balanced_idx_q10, ]

# Check the class distribution
table(WD.train_q10$Class)

# Fit decision tree using only A26, A09, and A25
WD.tree_selected <- tree(Class ~ A26 + A09 + A25, data = WD.train_q10)

# 2. Cross-validation to find the optimal size using misclassification rate
set.seed(34035958)
cv_result_selected <- cv.tree(WD.tree_selected, FUN = prune.misclass)

# 3. Find the best tree size (lowest deviance)
best_size_selected <- cv_result_selected$size[which.min(cv_result_selected$dev)]

# 4. Prune the tree to that size
WD.tree_selected_pruned <- prune.tree(WD.tree_selected, best = best_size_selected)

# 5. Optional: view summary of the pruned tree
summary(WD.tree_selected_pruned)

# Plot the tree
plot(WD.tree_selected_pruned)
text(WD.tree_selected_pruned, pretty = 0)

# Predict using the pruned tree from Q10 on Q3 test data
WD.predtree_q3 <- predict(WD.tree_selected_pruned, WD.test, type = "class")

# Confusion matrix for Q3 test set
conf.tree_q3 <- table(Predicted = WD.predtree_q3, Actual = WD.test$Class)
cat("\n# Simplified Decision Tree on Q3 Test Set\n")
print(conf.tree_q3)

# Evaluate confusion matrix metrics
evaluate_confusion_metrics("Simplified Decision Tree on Q3 Test", conf.tree_q3)

# 4. Predict probabilities (for AUC)
WD.tree_q3.prob <- predict(WD.tree_selected_pruned, WD.test, type = "vector")

# 5. Create prediction object using probabilities for class 1
WD.tree_q3.pred <- ROCR::prediction(WD.tree_q3.prob[, 2], WD.test$Class)

# 6. Print AUC
print_auc("Simplified Decision Tree on Q3", WD.tree_q3.pred)

# --------------------------------------------------
# Q11. Improved Random Forest
# --------------------------------------------------
# Re-balance the class 

# Identify indices by class
class0_other_idx <- which(WD$Class == 0)  # "Other"
class1_oats_idx  <- which(WD$Class == 1)  # "Oats"

# Sample equal size from both classes (e.g., 500 each)
set.seed(34035958)
sample_other <- sample(class0_other_idx, 440)
sample_oats  <- sample(class1_oats_idx, 440)

# Combine to form balanced training set
balanced_idx <- c(sample_other, sample_oats)

# Assign new training and test sets
WD.train_sampled <- WD[balanced_idx, ]
WD.test_sampled  <- WD[-balanced_idx, ]

# Remove Least Important Predictors
cols_to_remove <- c("A16", "A19")
WD.train_sampled <- WD.train_sampled[, !(names(WD.train_sampled) %in% cols_to_remove)]
WD.test_sampled  <- WD.test_sampled[, !(names(WD.test_sampled) %in% cols_to_remove)]

set.seed(34035958)
WD.rf.best <- randomForest(Class ~ ., 
                           data = WD.train_sampled, 
                           mtry = 2,          
                           ntree = 300,       
                           importance = TRUE)

# Remove the same least important predictors from the Q3 test set
WD.test_q3 <- WD.test[, !(names(WD.test) %in% c("A16", "A19"))]

# 2. Make predictions using the improved model
pred.rf.q3 <- predict(WD.rf.best, WD.test_q3, type = "class")

# Confusion matrix
conf.rf.q3 <- table(Predicted = pred.rf.q3, Actual = WD.test_q3$Class)
cat("\n# Confusion Matrix: Improved RF on Q3 Unbalanced Test Data\n")
print(conf.rf.q3)

# Evaluate confusion matrix metrics
evaluate_confusion_metrics("Improved RF on Q3 Test", conf.rf.q3)

# Predict class probabilities for AUC
prob.rf.q3 <- predict(WD.rf.best, WD.test_q3, type = "prob")[, 2]

# Get ROC/AUC performance
res.rf.q3 <- get_roc_perf(prob.rf.q3, WD.test_q3$Class)
print_auc("Improved RF on Q3 Test", res.rf.q3$pred)



# Cross-validation using AUC (ROC) to tune 'mtry'
set.seed(34035958)

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Rename class for caret (caret expects factor levels to be character)
WD.train_sampled$Class <- factor(ifelse(WD.train_sampled$Class == 1, "Oats", "Other"))

# Train with cross-validation
rf_cv <- train(Class ~ ., data = WD.train_sampled,
               method = "rf",
               metric = "ROC",        # AUC is used for tuning
               trControl = ctrl,
               tuneLength = 5)        # Tries 5 different mtry values

# View best parameters of mtry
print(rf_cv$bestTune)

# --------------------------------------------------
# Q12. ANN
# --------------------------------------------------

# Convert Class from factor to numeric
WD$Class <- as.numeric(as.character(WD$Class))

# Sample 500 from each class to balance
set.seed(34035958)
class0_idx <- which(WD$Class == 0)
class1_idx <- which(WD$Class == 1)

sample0 <- sample(class0_idx, 380)
sample1 <- sample(class1_idx, 380)

balanced_idx <- c(sample0, sample1)
WD_balanced <- WD[balanced_idx, ]

# Shuffle the balanced dataset
WD_balanced <- WD_balanced[sample(1:nrow(WD_balanced)), ]

# 80/20 train-test split
set.seed(34035958)
train_idx <- sample(1:nrow(WD_balanced), 0.8 * nrow(WD_balanced))
WD.train_ann <- WD_balanced[train_idx, ]
WD.test_ann  <- WD_balanced[-train_idx, ]

# Choose top predictors from Q4
selected_vars <- c("A26", "A09", "A25")

# Ensure selected predictors are numeric
WD.train_ann[, selected_vars] <- lapply(WD.train_ann[, selected_vars], as.numeric)
WD.test_ann[, selected_vars]  <- lapply(WD.test_ann[, selected_vars], as.numeric)

# Train neural network 
WD.nn <- neuralnet(Class == 1 ~ A26 + A09 + A25, 
                   data = WD.train_ann,
                   hidden = 3,
                   linear.output = FALSE)

# Ensure the same predictors are numeric in the Q3 test set
WD.test[, selected_vars] <- lapply(WD.test[, selected_vars], as.numeric)

# Predict using the neural network on original test set from Q3
WD.nn.pred_q3 <- compute(WD.nn, WD.test[, selected_vars])
prob_q3 <- WD.nn.pred_q3$net.result
pred_q3 <- ifelse(prob_q3 > 0.5, 1, 0)

# Confusion matrix using original test set (unbalanced)
conf.mat_q3 <- table(observed = WD.test$Class, predicted = pred_q3)
print(conf.mat_q3)

# Evaluate performance on original test set
evaluate_confusion_metrics("Neural Network", conf.mat_q3)

# AUC
pred_nn <- ROCR::prediction(prob_q3, WD.test$Class)
print_auc("Neural Network", pred_nn)


# --------------------------------------------------
# Q13. SVM
# --------------------------------------------------

# Rebalance the data (specific to SVM)
class0_idx_svm <- which(WD$Class == 0)
class1_idx_svm <- which(WD$Class == 1)

set.seed(34035958)
sample0_svm <- sample(class0_idx_svm, 380)
sample1_svm <- sample(class1_idx_svm, 380)
balanced_idx_svm <- c(sample0_svm, sample1_svm)
WD_svm_balanced <- WD[balanced_idx_svm, ]

# Remove least important predictors for SVM
cols_to_remove <- c("A16", "A19")
WD_svm_balanced <- WD_svm_balanced[, !(names(WD_svm_balanced) %in% cols_to_remove)]
WD_test_q3_svm <- WD.test[, !(names(WD.test) %in% cols_to_remove)]

# Convert target to factor for classification
WD_svm_balanced$Class <- as.factor(WD_svm_balanced$Class)

# Train SVM model
svm_model <- svm(Class ~ ., data = WD_svm_balanced, kernel = "radial", probability = TRUE)

# Predict on Q3 test set
svm_pred_class <- predict(svm_model, WD_test_q3_svm)
svm_pred_prob <- attr(predict(svm_model, WD_test_q3_svm, probability = TRUE), "probabilities")[, "1"]

# Confusion matrix
conf.svm <- table(Predicted = svm_pred_class, Actual = WD_test_q3_svm$Class)
cat("\n# SVM Confusion Matrix\n")
print(conf.svm)

# Evaluate performance
evaluate_confusion_metrics("SVM", conf.svm)

# AUC
svm_roc <- ROCR::prediction(svm_pred_prob, WD_test_q3_svm$Class)
print_auc("SVM", svm_roc)

