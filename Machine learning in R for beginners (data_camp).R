# Machine learning in R for beginners (data_camp)
  # with KNN algorithm (k-nearist neighbors)
    # with packages : 'ggvis', 'class', 'gmodels', 'caret'(lattice, ggplot2)
      # with "iris" data

# iris scatter plot
> iris %>% ggvis(~Sepal.Length, ~Sepal.Width, fill = ~Species) %>%
  + layer_points()

# overall correlation 'Petal.Length'~'Petal.Width'
cor(iris$Petal.Length, iris$Petal.Width) # 0.9628654
    # Versicolor~0.79, Setosa~0.31, Virginica~0.32


# Data preperation - normalization
  # normalize ~ do this when you suspect that the data is not consistent.
    # distances between variables with larger ranges will not be over-emphasised.
x=levels(iris$Species)

normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

iris_norm <- as.data.frame(lapply(iris[1:4], normalize))

summary(iris_norm)

"""
Sepal.Length     Sepal.Width      Petal.Length   
Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
1st Qu.:0.2222   1st Qu.:0.3333   1st Qu.:0.1017  
Median :0.4167   Median :0.4167   Median :0.5678  
Mean   :0.4287   Mean   :0.4406   Mean   :0.4675  
3rd Qu.:0.5833   3rd Qu.:0.5417   3rd Qu.:0.6949  
Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
Petal.Width     
Min.   :0.00000  
1st Qu.:0.08333  
Median :0.50000  
Mean   :0.45806  
3rd Qu.:0.70833  
Max.   :1.00000
"""

# training set

ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))

iris.training = iris[ind==1, 1:4]
head(iris.training)
"""
   Sepal.Length Sepal.Width Petal.Length Petal.Width
3           4.7         3.2          1.3         0.2
4           4.6         3.1          1.5         0.2
6           5.4         3.9          1.7         0.4
8           5.0         3.4          1.5         0.2
10          4.9         3.1          1.5         0.1
12          4.8         3.4          1.6         0.2
"""
iris.test = iris[ind==2, 1:4]
head(iris.test)
"""   Sepal.Length Sepal.Width Petal.Length Petal.Width
1           5.1         3.5          1.4         0.2
2           4.9         3.0          1.4         0.2
5           5.0         3.6          1.4         0.2
7           4.6         3.4          1.4         0.3
9           4.4         2.9          1.4         0.2
11          5.4         3.7          1.5         0.2
"""

iris.trainLabels = iris[ind==1,5]
print(iris.trainLabels)
"""

  [1] setosa     setosa     setosa     setosa    
  [5] setosa     setosa     setosa     setosa    
  [9] setosa     setosa     setosa     setosa    
 [13] setosa     setosa     setosa     setosa    
 [17] setosa     setosa     setosa     setosa    
 [21] setosa     setosa     setosa     setosa    
 [25] setosa     setosa     setosa     setosa    
 [29] setosa     setosa     setosa     setosa    
 [33] setosa     setosa     versicolor versicolor
 [37] versicolor versicolor versicolor versicolor
 [41] versicolor versicolor versicolor versicolor
 [45] versicolor versicolor versicolor versicolor
 .
 .
 .
  [93] virginica  virginica  virginica  virginica 
 [97] virginica  virginica  virginica  virginica 
[101] virginica  virginica  virginica  virginica 
Levels: setosa versicolor virginica
"""
iris.testLabels = iris[ind==2, 5]
print(iris.testLabels)
"""
 [1] setosa     setosa     setosa     setosa     setosa    
 [6] setosa     setosa     setosa     setosa     setosa    
[11] setosa     setosa     setosa     setosa     setosa    
[16] setosa     versicolor versicolor versicolor versicolor
[21] versicolor versicolor versicolor versicolor versicolor
[26] versicolor versicolor versicolor versicolor versicolor
[31] versicolor virginica  virginica  virginica  virginica 
[36] virginica  virginica  virginica  virginica  virginica 
[41] virginica  virginica  virginica  virginica  virginica 
[46] virginica 
Levels: setosa versicolor virginica
"""


# KNN modeling
library(class)
iris_pred = knn(train = iris.training, test = iris.test, cl = iris.trainLabels, k=3)
"""
 [1] setosa     setosa     setosa     setosa     setosa    
 [6] setosa     setosa     setosa     setosa     setosa    
[11] setosa     setosa     setosa     setosa     setosa    
[16] setosa     versicolor versicolor versicolor versicolor
[21] versicolor versicolor versicolor virginica  versicolor
[26] versicolor versicolor versicolor virginica  versicolor
[31] versicolor virginica  virginica  versicolor virginica 
[36] virginica  virginica  virginica  virginica  virginica 
[41] virginica  virginica  virginica  virginica  virginica 
[46] virginica 
Levels: setosa versicolor virginica
"""


# evaluating model
irisTestLabels = data.frame(iris.testLabels)
merge = data.frame(iris_pred, iris.testLabels)
names(merge) = c("Predicted Species", "Observed Species")
merge
"""
   Predicted Species Observed Species
1             setosa           setosa
2             setosa           setosa
3             setosa           setosa
4             setosa           setosa
5             setosa           setosa
6             setosa           setosa
7             setosa           setosa
8             setosa           setosa
9             setosa           setosa
10            setosa           setosa
11            setosa           setosa
12            setosa           setosa
13            setosa           setosa
14            setosa           setosa
15            setosa           setosa
16            setosa           setosa
17        versicolor       versicolor
18        versicolor       versicolor
19        versicolor       versicolor
20        versicolor       versicolor
21        versicolor       versicolor
22        versicolor       versicolor
23        versicolor       versicolor
24         virginica       versicolor
25        versicolor       versicolor
26        versicolor       versicolor
27        versicolor       versicolor
28        versicolor       versicolor
29         virginica       versicolor
30        versicolor       versicolor
31        versicolor       versicolor
32         virginica        virginica
33         virginica        virginica
34        versicolor        virginica
35         virginica        virginica
36         virginica        virginica
37         virginica        virginica
38         virginica        virginica
39         virginica        virginica
40         virginica        virginica
41         virginica        virginica
42         virginica        virginica
43         virginica        virginica
44         virginica        virginica
45         virginica        virginica
46         virginica        virginica
"""

library(gmodels)
CrossTable(x = iris.testLabels, y = iris_pred, prop.chisq=FALSE)

Cell Contents
|-------------------------|
  |                       N |
  |           N / Row Total |
  |           N / Col Total |
  |         N / Table Total |
  |-------------------------|
  
  
  Total Observations in Table:  46 


| iris_pred 
iris.testLabels |     setosa | versicolor |  virginica |  Row Total | 
  ----------------|------------|------------|------------|------------|
  setosa |         16 |          0 |          0 |         16 | 
  |      1.000 |      0.000 |      0.000 |      0.348 | 
  |      1.000 |      0.000 |      0.000 |            | 
  |      0.348 |      0.000 |      0.000 |            | 
  ----------------|------------|------------|------------|------------|
  versicolor |          0 |         13 |          2 |         15 | 
  |      0.000 |      0.867 |      0.133 |      0.326 | 
  |      0.000 |      0.929 |      0.125 |            | 
  |      0.000 |      0.283 |      0.043 |            | 
  ----------------|------------|------------|------------|------------|
  virginica |          0 |          1 |         14 |         15 | 
  |      0.000 |      0.067 |      0.933 |      0.326 | 
  |      0.000 |      0.071 |      0.875 |            | 
  |      0.000 |      0.022 |      0.304 |            | 
  ----------------|------------|------------|------------|------------|
  Column Total |         16 |         14 |         16 |         46 | 
  |      0.348 |      0.304 |      0.348 |            | 
  ----------------|------------|------------|------------|------------|


# Machine Learning in R with caret
library(caret)
index <- createDataPartition(iris$Species, p=0.75, list=FALSE)
iris.training <- iris[index,]
iris.test <- iris[-index,]

names(getModelInfo())
"""
  [1] "ada"                 "AdaBag"             
  [3] "AdaBoost.M1"         "adaboost"           
  [5] "amdai"               "ANFIS"              
  [7] "avNNet"              "awnb"               
  [9] "awtan"               "bag"                
 [11] "bagEarth"            "bagEarthGCV"        
 [13] "bagFDA"              "bagFDAGCV"          
 [15] "bam"                 "bartMachine"        
 [17] "bayesglm"            "binda"              
 [19] "blackboost"          "blasso"    
 .
 .
 .
 [233] "WM"                  "wsrf"               
[235] "xgbDART"             "xgbLinear"          
[237] "xgbTree"             "xyf"   
"""
model_knn <- train(iris.training[, 1:4], iris.training[, 5], method='knn')
 
model_cart <- train(iris.training[, 1:4], iris.training[, 5], method='rpart2')

predictions<-predict(object=model_knn,iris.test[,1:4])
table(predictions)
"""
predictions
    setosa versicolor  virginica 
        12         10         14
"""
confusionMatrix(predictions,iris.test[,5])
"""
predictions
    setosa versicolor  virginica 
        12         10         14 
> confusionMatrix(predictions,iris.test[,5])
Confusion Matrix and Statistics

            Reference
Prediction   setosa versicolor virginica
  setosa         12          0         0
  versicolor      0         10         0
  virginica       0          2        12

Overall Statistics
                                          
               Accuracy : 0.9444          
                 95% CI : (0.8134, 0.9932)
    No Information Rate : 0.3333          
    P-Value [Acc > NIR] : 1.728e-14       
                                          
                  Kappa : 0.9167          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: setosa Class: versicolor
Sensitivity                 1.0000            0.8333
Specificity                 1.0000            1.0000
Pos Pred Value              1.0000            1.0000
Neg Pred Value              1.0000            0.9231
Prevalence                  0.3333            0.3333
Detection Rate              0.3333            0.2778
Detection Prevalence        0.3333            0.2778
Balanced Accuracy           1.0000            0.9167
                     Class: virginica
Sensitivity                    1.0000
Specificity                    0.9167
Pos Pred Value                 0.8571
Neg Pred Value                 1.0000
Prevalence                     0.3333
Detection Rate                 0.3333
Detection Prevalence           0.3889
Balanced Accuracy              0.9583
"""

model_knn <- train(iris.training[, 1:4], iris.training[, 5], method='knn', preProcess=c("center", "scale"))
predictions<-predict.train(object=model_knn,iris.test[,1:4], type="raw")
confusionMatrix(predictions,iris.test[,5])
"""Confusion Matrix and Statistics

            Reference
Prediction   setosa versicolor virginica
  setosa         12          0         0
  versicolor      0         10         0
  virginica       0          2        12

Overall Statistics
                                          
               Accuracy : 0.9444          
                 95% CI : (0.8134, 0.9932)
    No Information Rate : 0.3333          
    P-Value [Acc > NIR] : 1.728e-14       
                                          
                  Kappa : 0.9167          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: setosa Class: versicolor
Sensitivity                 1.0000            0.8333
Specificity                 1.0000            1.0000
Pos Pred Value              1.0000            1.0000
Neg Pred Value              1.0000            0.9231
Prevalence                  0.3333            0.3333
Detection Rate              0.3333            0.2778
Detection Prevalence        0.3333            0.2778
Balanced Accuracy           1.0000            0.9167
                     Class: virginica
Sensitivity                    1.0000
Specificity                    0.9167
Pos Pred Value                 0.8571
Neg Pred Value                 1.0000
Prevalence                     0.3333
Detection Rate                 0.3333
Detection Prevalence           0.3889
Balanced Accuracy              0.9583
"""
