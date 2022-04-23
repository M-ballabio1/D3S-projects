# Supervised method: KNNclassifier
# Author: Matteo Ballabio

diabetes <- read.table(file.choose(), sep = ',' )
diabetes<- diabetes[-1,]; ## remove header out 768 rows

# Structure 
str(diabetes)

# convert data to numeric
data_numeric<- function(x){ (as.numeric( as.character(x)))}
diabetes.numeric <-  as.data.frame(lapply(diabetes, data_numeric))

#normilize data --> operazione molto utile soprattuto con Neural Net
data_norm<- function(x){ ((x-min(x))/(max(x)-min(x)) )}
diabetes.norm <-  as.data.frame(lapply(diabetes.numeric, data_norm)) 

# find train and test range
cols<- ncol(diabetes)
rows<- nrow(diabetes)
train.rows<- as.integer(rows*0.7) # 70% of data as training and 30% as testing

# split data to train and test
xtrain <- diabetes.norm[1:train.rows,1:(cols-1)]
ytrain <- diabetes.norm[1:train.rows, cols]
xtest <- diabetes.norm[(train.rows+1):rows,1:(cols-1)] 
ytest <- diabetes.norm[(train.rows+1):rows,cols]

#INFERENCE MODELS:

# Installing Packages
install.packages("xgboost")
library(xgboost)

#Tune and Run the model

xgb <- xgboost(data = data.matrix(xtrain[,-1]), 
               label = ytrain, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3
)

# predict values in test set
ytest_pred <- predict(xgb, data.matrix(xtest[,-1])
                  
#find confusion matrix --> permette di identificare TP-TN-FP-FN
confusion.matrix<-table(ytest_pred,ytest)
print(confusion.matrix)

# find f- meausre
fMeasure<- (confusion.matrix[1,1]+ confusion.matrix[2,2])/
  (confusion.matrix[1,1]+ confusion.matrix[1,2]+confusion.matrix[2,1]+ confusion.matrix[2,2])
print(fMeasure)
                       
                  
#################
#apply KNN
library(class)
ytest_pred <- knn(xtrain,xtest,ytrain, k=1,prob=T)

#find confusion matrix --> permette di identificare TP-TN-FP-FN
confusion.matrix<-table(ytest_pred,ytest)
print(confusion.matrix)

# find f-measure --> ACCURACY
fMeasure<- (confusion.matrix[1,1]+ confusion.matrix[2,2])/
  (confusion.matrix[1,1]+ confusion.matrix[1,2]+confusion.matrix[2,1]+ confusion.matrix[2,2])
print(fMeasure)

# Precision | Recall | F1-SCORE with one function
measurePrecisionRecall <- function(actual_labels, predict){
  conMatrix = table(ytest_pred,ytest)
  precision <- conMatrix['0','0'] / ifelse(sum(conMatrix[,'0'])== 0, 1, sum(conMatrix[,'0']))
  recall <- conMatrix['0','0'] / ifelse(sum(conMatrix['0',])== 0, 1, sum(conMatrix['0',]))
  fmeasure <- 2 * precision * recall / ifelse(precision + recall == 0, 1, precision + recall)
  
  cat('precision:  ')
  cat(precision * 100)
  cat('%')
  cat('\n')
  
  cat('recall:     ')
  cat(recall * 100)
  cat('%')
  cat('\n')
  
  cat('f-measure:  ')
  cat(fmeasure * 100)
  cat('%')
  cat('\n')
}

# Evaluation Accuracy
measurePrecisionRecall(ytest_pred,ytest)

# KNN
# precision:  76.97368%
# recall:     78.52349%
# f-measure:  77.74086%
