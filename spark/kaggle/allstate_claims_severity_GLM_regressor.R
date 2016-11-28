#
# Simple and silly solution for the "Allstate Claims Severity" competition on Kaggle
# Competition page: https://www.kaggle.com/c/allstate-claims-severity
#
if (nchar(Sys.getenv("SPARK_HOME")) < 1) {
  Sys.setenv(SPARK_HOME = "/PATH_TO_YOUR_SPARK")
}
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))

library(dplyr)

params.trainInput <- "/path/to/train.csv"
params.testInput  <- "/path/to/test.csv"

params.trainSample <- 1.0
params.testSample  <- 1.0

params.outputFile  <- "/path/to/submission.csv"


#****************************************
print("Reading data from train.csv file")
#****************************************

trainInput <- read.df(params.trainInput, "csv", header = "true", inferSchema = "true") %>% cache
testInput  <- read.df(params.testInput,  "csv", header = "true", inferSchema = "true") %>% cache


#*****************************************
print("Preparing data for training model")
#*****************************************

data <- trainInput %>% withColumnRenamed("loss", "label") %>% sample(F, params.trainSample)

split <- data %>% randomSplit(c(0.7, 0.3))

trainingData   <- split[[1]] %>% cache
validationData <- split[[2]] %>% cache

testData <- testInput %>% sample(F, params.testSample) %>% cache

#*******************************************************
print("Training algorithm")
#*******************************************************

# Only cont* fields and label
label_and_features <- seq.int(118,132)
only_features      <- seq.int(118,131)

#Training the model
model <- trainingData[,label_and_features] %>% 
         spark.glm(label ~ ., family="gaussian")

#********************************************************************
print("Evaluating model on train and test data and calculating RMSE")
#********************************************************************

trainPredictionsAndLabels <- model %>% predict(trainingData) %>% 
                                       subset(select=(c("label","prediction")))

validPredictionsAndLabels <- model %>% predict(validationData) %>% 
                                      subset(select=(c("label","prediction")))

mse <- function(df){
  df$se <- (df$label - df$prediction)^2
  mean(as.data.frame(trainPredictionsAndLabels)$se)
}

rmse <- function(df){
  sqrt(mse(df))
}

mae <- function(df){
  df$se <- abs(df$label - df$prediction)
  mean(as.data.frame(trainPredictionsAndLabels)$se)
}

output = c("=====================================================================",
  paste0("Param trainSample: ", params.trainSample),
  paste0("Param testSample: ", params.testSample),
  paste0("TrainingData count: ", nrow(trainingData)),
  paste0("ValidationData count: ", nrow(validationData)),
  paste0("TestData count: ", nrow(testData)),
  "=====================================================================",
  paste0("Training data MSE = ", mse(trainPredictionsAndLabels)),
  paste0("Training data RMSE = ", rmse(trainPredictionsAndLabels)),
  paste0("Training data MAE =  ", mae(trainPredictionsAndLabels)),
  "=====================================================================",
  paste0("Validation data MSE = ", mse(validPredictionsAndLabels)),
  paste0("Validation data RMSE = ", rmse(validPredictionsAndLabels)),
  paste0("Validation data MAE =  ", mae(validPredictionsAndLabels)),
  "=====================================================================")
  
print(output)

#*****************************************
print("Run prediction over test dataset")
#*****************************************
 
#Predicts and saves file ready for Kaggle!
outputData <- model %>% predict(testData[,only_features]) %>% 
                        subset(select=(c("id","prediction"))) %>% 
                        withColumnRenamed("prediction", "loss")

write.df(outputData, param.outputFile)
