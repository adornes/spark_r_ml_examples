Spark R Machine Learning Examples
=====================================

This repository is part of a series on Apache Spark examples, aimed at demonstrating the implementation of Machine Learning solutions in  different programming languages supported by Spark. Java is the only language not covered, due to its many disadvantages (and not a single advantage) compared to the other languages. Check the other repositories:

* **Scala**  - [github.com/adornes/spark_scala_ml_examples](https://github.com/adornes/spark_scala_ml_examples)
* **Python** - [github.com/adornes/spark_python_ml_examples](https://github.com/adornes/spark_python_ml_examples)
* **R**      - You are here!

This repository aims at demonstrating how to build a [Spark 2.0](https://spark.apache.org/releases/spark-release-2-0-0.html) application with [R](https://www.r-project.org/) for solving Machine Learning problems, ready to be run locally or on any cloud platform such as [AWS Elastic MapReduce (EMR)](https://aws.amazon.com/emr/).

Each R script in the package can be run as an individual application, as described in the next sections.  

### Why Spark?

Since almost all personal computers nowadays have many Gigabytes of RAM (and it is in an accelerated growing) and powerful CPUs and GPUs, many real-world machine learning problems can be solved with a single computer and frameworks such as [ScikitLearn](http://scikit-learn.org/), with no need of a distributed system, this is, a cluster of many computers. Sometimes, though, data grows and keeps growing. Who never heard the term "Big Data"? When it happens, a non-distributed/scalable solution may solve for a short time, but afterwards such solution will need to be reviewed and maybe significantly changed.

Spark started as a research project at [UC Berkeley](http://www.berkeley.edu/) in the [AMPLab](https://amplab.cs.berkeley.edu/), a research group that focuses on big data analytics. Since then, it became an [Apache](https://www.apache.org/) project and has delivered many new releases, reaching a consistent maturity with a wide range of functionalities. Most of all, Spark can perform data processing over some Gigabytes or hundreds of Petabytes with basically the same programming code, only requiring a proper cluster of machines in the background (check [this link](https://databricks.com/blog/2014/10/10/spark-petabyte-sort.html)). In some very specific cases the developer may need to tune the process by changing granularity of data distribution and other related aspects, but in general there are plenty of providers that automate all this cluster configuration for the developer. For instance, the scripts in this repository could be run with [AWS Elastic MapReduce (EMR)](https://aws.amazon.com/emr/), as described [here](https://aws.amazon.com/blogs/big-data/running-r-on-aws/) and [here](https://aws.amazon.com/blogs/big-data/statistical-analysis-with-open-source-r-and-rstudio-on-amazon-emr/). 


### Why R?

[R](https://www.r-project.org/) is one of the best (or maybe the best) language in terms of libraries for statistical methods, models and graphs. The obvious reason is that it was created (and is maintained) with Statisticians in mind. Unfortunately, such distinction doesn't hold when it comes to Spark. 

[SparkR](https://spark.apache.org/docs/2.0.2/sparkr.html), an R package that provides a programming interface for using Spark from R, supports only very few Machine Learning algorithms (check [the API documentation for version 2.0.2](https://spark.apache.org/docs/2.0.2/api/R/)). Besides that, it also doesn't provide any *wrapper* for other important components of the Spark platform, such as *Cross Validation*, *Pipelines* and *ParamGridBuilder*, explored in the other repositories [for Scala](https://github.com/adornes/spark_scala_ml_examples) and [for Python](https://github.com/adornes/spark_python_ml_examples).

SparkR ends up being an important package for introducing the public of R users to the distributed processing of large scale datasets, or just *Big Data*.

### Script: allstate_claims_severity_GLM_regressor

[Allstate Corporation](https://www.allstate.com), the second largest insurance company in United States, founded in 1931, recently launched a Machine Learning recruitment challenge in partnership with [Kaggle](https://www.kaggle.com/c/allstate-claims-severity) asking for competitors, Data Science professionals and enthusiasts, to predict the cost, and hence the severity, of claims.
 
The competition organizers provide the competitors with more than 300.000 examples with masked and anonymous data consisting of more than 100 categorical and numerical attributes, thus being compliant with confidentiality constraints and still more than enough for building and evaluating a variety of Machine Learning techniques. 

This script in R obtain the training and test input datasets and trains a [Generalized Linear Model](https://en.wikipedia.org/wiki/Generalized_linear_model) over it.
The objective is to demonstrate the use of [Spark 2.0](https://spark.apache.org/releases/spark-release-2-0-0.html) Machine Learning models with [R](https://www.r-project.org/). In order to keep this main objective, more sophisticated techniques (such as a thorough exploratory data analysis and feature engineering) are intentionally omitted.


#### Flow of Execution and Overall Learnings

* *SparkR.session* is used for building a *Spark session*.
    
    ```r
    if (nchar(Sys.getenv("SPARK_HOME")) < 1) {
      Sys.setenv(SPARK_HOME = "/PATH_TO_YOUR_SPARK")
    }
    library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
    sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))
    ```

* The *dplyr* package is used in order to chain function calls, which is more intuitive and easy to understand, besides its ugly syntax, in my humble opinion. 

    ```r
    library(dplyr)
    ```

* Some parameters that are used later in the code are also set here at the beginning of the script:

    ```r
    params.trainInput <- "/path/to/train.csv"
    params.testInput  <- "/path/to/test.csv"

    params.trainSample <- 1.0
    params.testSample  <- 1.0

    params.outputFile  <- "/path/to/submission.csv"
    ```

* The reading process includes important settings: It is set to read the header of the CSV file, which is directly applied to the columns' names of the dataframe created; and **inferSchema** property is set to *true*. Finally, both raw dataframes are *cached* since they are again used later in the code for *fitting* the **StringIndexer** transformations and it wouldn't be good to read the CSV files from the filesystem again. 


    ```r
    trainInput <- read.df(params.trainInput, "csv", header = "true", inferSchema = "true") %>% cache
    testInput  <- read.df(params.testInput,  "csv", header = "true", inferSchema = "true") %>% cache
    ```

* The column "loss" is renamed to "label". For some reason, in [the Python version](https://github.com/adornes/spark_python_ml_examples), even after using the *setLabelCol* on the regression model, it still looks for a column called "label", raising an ugly error: `pyspark.sql.utils.IllegalArgumentException: u'Field "label" does not exist.'`. It may be hardcoded somewhere in Spark's source code.
 
* The content of *train.csv* is split into *training* and *validation* data, 70% and 30%, respectively. The content of "test.csv" is reserved for building the final CSV file for submission on Kaggle. Both original dataframes are sampled according to parameters provided in the beginning of the script, which is particularly useful for running fast executions in your local machine;
  
    ```r
    data <- trainInput %>% withColumnRenamed("loss", "label") %>% sample(F, params.trainSample)

    split <- data %>% randomSplit(c(0.7, 0.3))

    trainingData   <- split[[1]] %>% cache
    validationData <- split[[2]] %>% cache

    testData <- testInput %>% sample(F, params.testSample) %>% cache
    ```
  
* In the [Scala](https://github.com/adornes/spark_scala_ml_examples) and [Python](https://github.com/adornes/spark_python_ml_examples) versions I used a [StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#stringindexer) transformation for creating a numeric representation for the categorical values, although a best choice would be an [OneHotEncoder](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder), which yields a different new column for each category holding a boolean (0/1) value. SparkR still doesn't provide a RandomForest model (or any other model based on decision trees), but only Generalized Linear Models. Once it is based on linear regression, numerical fields are supposed to always represent ordinal values (where one is greater/less than the other), which is a bad assumption for categorical values, even if represented by numbers. So, the only good choice here for transforming the categorical columns would be the [OneHotEncoder](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder), which would generate more than one thousand of new columns. For keeping it simples, the decision was to simply ignore the categorical values.

    ```r
    label_and_features <- seq.int(118,132)
    only_features      <- seq.int(118,131)
    ```
  
* Then, the GLM model is trained over the *training* dataset: 

    ```r
    model <- trainingData[,label_and_features] %>% 
             spark.glm(label ~ ., family="gaussian")
    ```
  
* As aforementioned, SparkR still lacks support for *Cross Validation*, *Pipeline* and *ParamGridBuilder* as used in the other version [for Scala](https://github.com/adornes/spark_scala_ml_examples) and [for Python](https://github.com/adornes/spark_python_ml_examples).
  
* The trained model can be used to obtain predictions for the *training* and *validation* datasets.

    ```r
    trainPredictionsAndLabels <- model %>% predict(trainingData) %>% 
                                           subset(select=(c("label","prediction")))

    validPredictionsAndLabels <- model %>% predict(validationData) %>% 
                                          subset(select=(c("label","prediction")))
    ```
  
* Some manually created functions are then used to calculate [MSE](https://en.wikipedia.org/wiki/Mean_squared_error), [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) over the predictions.

    ```r
    mse <- function(df){
      df$se <- (df$label - df$prediction)^2
      mean(as.data.frame(trainPredictionsAndLabels)$se)
    }

    rmse <- function(df){
      sqrt(mse(df))
    }

    mae <- function(df){
      df$ae <- abs(df$label - df$prediction)
      mean(as.data.frame(trainPredictionsAndLabels)$ae)
    }
    ```
      
* Finally, the prediction over the *test* dataset can be saved and submitted on Kaggle!
 
    ```r
    outputData <- model %>% predict(testData[,only_features]) %>% 
                            subset(select=(c("id","prediction"))) %>% 
                            withColumnRenamed("prediction", "loss")

    write.df(outputData, param.outputFile)
    ```
  

#### Submission on Kaggle

As mentioned along the explanations, many improvements could/should be done in terms of exploratory data analysis, feature engineering, evaluating other models (starting by the simplest ones, as Linear Regression) and then decreasing the predictions error.
 
For being over-simplistic, this model achieved a Mean Absolute Error (MAE) of 1286 in the [public leaderboard](https://www.kaggle.com/c/allstate-claims-severity/leaderboard), far from the top positions.

The submission file and the detailed metrics of the model evaluation can be found under the `output` directory.


### Corrections/Suggestions or just a Hello!

Don't hesitate to contact me directly or create *pull requests* here if you have any correction or suggestion for the code or for this documentation! Thanks! 

* [Github](https://www.github.com/adornes)
* [Twitter](https://twitter.com/daniel_adornes)
* [LinkedIn](https://www.linkedin.com/in/adornes)