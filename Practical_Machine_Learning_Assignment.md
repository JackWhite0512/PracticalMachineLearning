## Practical Machine Learning Assignment

### By Gianluca Kikidis

# Prediction models

The goal in this assignment project is to test different prediction
models and see which one fits best in order to predict the manner in
which people did certain exercises. This is the “classe” variable in the
training set.

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.

First, I load our packages needed:

    library(caret)

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    library(randomForest)

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(rpart)
    library(rpart.plot)
    library(dplyr)

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     combine

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

    library(corrplot)

    ## corrplot 0.84 loaded

    library(gbm)

    ## Loaded gbm 2.1.8

    library(rattle)

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    ## 
    ## Attaching package: 'rattle'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     importance

Then, I load the testing and training set (i load them locally because
of computer related problems)

    training <- read.csv("pml-training.csv", stringsAsFactors = F,na.strings = c("","NA","#DIV/0!"))
    testing <- read.csv("pml-testing.csv", stringsAsFactors = F,na.strings = c("","NA","#DIV/0!"))
    dim(training);dim(testing)

    ## [1] 19622   160

    ## [1]  20 160

Next, I create the data partition with the training set and testing set,
i chose a p of 0.75, making 25% of the data set a testing set.

    inTrain <- createDataPartition(training$classe, p = 0.75, list = F)
    training <- training[inTrain,]
    testing <- training[-inTrain,]
    dim(training); dim(testing)

    ## [1] 14718   160

    ## [1] 3700  160

## Data pre processing

I remove near zero variables that won’t be needed for further analysis:

    zero <- nearZeroVar(training)
    training <- training[, -zero]
    testing  <- testing[, -zero]

I also remove variables that are over 95% NA

    na    <- sapply(training, function(x) mean(is.na(x))) > 0.95
    training <- training[, na==FALSE]
    testing  <- testing[, na==FALSE]

And finally i remove identification correlated variables that i do not
need:

    training <- training[, -(1:5)]
    testing  <- testing[, -(1:5)]
    dim(training); dim(testing)

    ## [1] 14718    54

    ## [1] 3700   54

## Prediction models

Now, with the Pre process done i start the first prediction model (my
favorite), which is random forest:

    set.seed(1234)
    control <- trainControl(method="cv", number=3, verboseIter=FALSE)
    RandomForest <- train(classe ~ ., data=training, method="rf",
                              trControl=control)
    RandomForest$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x))) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.22%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 4183    1    0    0    1 0.0004778973
    ## B    8 2838    2    0    0 0.0035112360
    ## C    0    4 2563    0    0 0.0015582392
    ## D    0    0   10 2401    1 0.0045605307
    ## E    0    0    0    6 2700 0.0022172949

I create a confusion matrix for the model and make a prediction:

    testing$classe <- as.factor(testing$classe)
    RForestPrediction <- predict(RandomForest, testing)
    RForestconfmat <- confusionMatrix(RForestPrediction, testing$classe)
    RForestconfmat

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1032    0    0    0    0
    ##          B    0  763    0    0    0
    ##          C    0    0  613    0    0
    ##          D    0    0    0  591    0
    ##          E    0    0    0    0  701
    ## 
    ## Overall Statistics
    ##                                     
    ##                Accuracy : 1         
    ##                  95% CI : (0.999, 1)
    ##     No Information Rate : 0.2789    
    ##     P-Value [Acc > NIR] : < 2.2e-16 
    ##                                     
    ##                   Kappa : 1         
    ##                                     
    ##  Mcnemar's Test P-Value : NA        
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2789   0.2062   0.1657   0.1597   0.1895
    ## Detection Rate         0.2789   0.2062   0.1657   0.1597   0.1895
    ## Detection Prevalence   0.2789   0.2062   0.1657   0.1597   0.1895
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Here is the plot also:

    plot(RForestconfmat$table, col = RForestconfmat$byClass, 
               main = paste("Random Forest - Accuracy"))

![](Practical_Machine_Learning_Assignment_files/figure-markdown_strict/unnamed-chunk-9-1.png)

    RForestconfmat$overall['Accuracy']

    ## Accuracy 
    ##        1

And the accuracy, which is: 100%

## Generalized Boosted Model

I tried generalized boosted models, but sadly my pc could not handle it.
I tried it on another PC too, but it might be to heavy and ram eating…
therefore i just post the code i would have used for this model:

    # set.seed(1234)
    # gbm<- train(classe~., data=training, method="gbm", verbose= FALSE)
    # gbm$finalmodel

    # gbm$overall
    # plot(gbm$table, col = gbm$byClass, 
    #     main = paste("GBM - Accuracy =", round(gbm$overall['Accuracy'], 4)))

## Decision tree

Next step is the decision tree model:

    DecTree<- train(classe ~. , data=training, method= "rpart")

Here is the plot:

    fancyRpartPlot(DecTree$finalModel)

![](Practical_Machine_Learning_Assignment_files/figure-markdown_strict/unnamed-chunk-13-1.png)
And the results:

    DecTree$results

    ##           cp  Accuracy      Kappa AccuracySD    KappaSD
    ## 1 0.03878287 0.5397078 0.41006998 0.03220764 0.04690090
    ## 2 0.06095130 0.3964866 0.17678455 0.05575913 0.09365580
    ## 3 0.11563657 0.3382722 0.08412388 0.04022860 0.05900409

Next, i once again make the prediction and create a confusion matrix:

    set.seed(1234)
    DecTree_pred<- predict(DecTree, testing)
    confusionMatrix(DecTree_pred, testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   A   B   C   D   E
    ##          A 945 317 282 262  97
    ##          B  14 272  15 103  90
    ##          C  72 174 316 226 182
    ##          D   0   0   0   0   0
    ##          E   1   0   0   0 332
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5041          
    ##                  95% CI : (0.4878, 0.5203)
    ##     No Information Rate : 0.2789          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3547          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9157  0.35649  0.51550   0.0000  0.47361
    ## Specificity            0.6409  0.92441  0.78814   1.0000  0.99967
    ## Pos Pred Value         0.4966  0.55061  0.32577      NaN  0.99700
    ## Neg Pred Value         0.9516  0.84685  0.89121   0.8403  0.89041
    ## Prevalence             0.2789  0.20622  0.16568   0.1597  0.18946
    ## Detection Rate         0.2554  0.07351  0.08541   0.0000  0.08973
    ## Detection Prevalence   0.5143  0.13351  0.26216   0.0000  0.09000
    ## Balanced Accuracy      0.7783  0.64045  0.65182   0.5000  0.73664

The accuracy here is only 49%.

## Conclusion

In conclusion, Random Forest has a higher accuracy compared to Decision
Trees, and my guess would be: slightly higher than GBM also, with a 100%
accuracy.

## Final Prediction

This is hence the final prediction with Random Forest:

    Prediction = predict(RandomForest, testing)
    head(Prediction)

    ## [1] A A A A A A
    ## Levels: A B C D E
