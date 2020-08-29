#SVR (default l2_regularization)
#Data Preprocessing
library(MASS)
dataset = Boston
n <- length(Boston$crim) # no of observations
#No Missing data
#No Categorical data to handle

#Splitting dataset into train & test
dat <- dataset
train_rows <- sample(1:n, .8*n)
train_set <- dat[train_rows,]
test_set <- dat[-train_rows,]

#No Feature Scaling 

#SVR execution with linear kernal with initial cost value 5
#install.packages('e1071')
library(e1071)
svrfit1 = svm(crim ~., data=train_set , kernel ="linear", cost =10,type = 'eps-regression',
              scale =TRUE, Cross = 10)
summary (svrfit1)

#SVR execution with less cost value .01
svrfit2 = svm(crim ~., data=train_set , kernel ="linear", cost =.01, type = 'eps-regression',
             scale =TRUE, Cross = 10)

summary (svrfit2)

#Tune parameter with less cost function
set.seed (41)
tune.out=tune(svm ,crim~.,data=train_set ,kernel ="linear", type = 'eps-regression',
              ranges =list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100)), Cross = 10)

summary (tune.out)


bestmod =tune.out$best.model
summary (bestmod)

#Run the best model on testing datset
ypred1=predict (svrfit1 ,test_set) # high cost value model
ypred2=predict (svrfit2 ,test_set) # low cost value model
ypred=predict (bestmod ,test_set) # best cost value model

#MSE, RMSE, R^2 calculation
rmse <- function(test.out,predict.out)
{
  errval=test.out-predict.out
  SSE = sum(errval^2)
  MSE = mean(errval^2)
  RMSE = sqrt(MSE)
  SST = sum((test.out-mean(test.out))^2)
  SSR = SST - SSE
  print(paste('SSE value is ',SSE))
  print(paste('MSE value is ',MSE))
  print(paste('RMSE value is',RMSE))
  print(paste('SSR value is',SSR))
  print(paste('SST value is',SST))
  print(paste('R-square value is',(1- (SSE/SST))))
}

rmse(test_set$crim,ypred1)
rmse(test_set$crim,ypred2)
rmse(test_set$crim,ypred)


