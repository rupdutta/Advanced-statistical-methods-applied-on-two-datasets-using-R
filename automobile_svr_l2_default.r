# SVR l2_regularization
#External Dataset Automobile_Data_Set.csv hold several car information
#######################################################################

#Data Preprocessing
dataset = read.csv('Automobile_Data_Set.csv')

#Handling Missing data
#converting relevant columns to numeric type
cols.num <- c("symboling","normalized.losses","wheel.base","length","width","height",
              "curb.weight","engine.size","bore","stroke","compression.ratio","horsepower",
              "peak.rpm","city.mpg","highway.mpg","price")
dataset[cols.num] <- sapply(dataset[cols.num],as.character)
dataset[cols.num] <- sapply(dataset[cols.num],as.numeric)

#Storing columns have missing values NA
col_missing_values = colnames(dataset)[colSums(is.na(dataset)) > 0]

#Replacing 41 missing values of column normalized.losses
dataset$normalized.losses = ifelse(is.na(dataset$normalized.losses),
                                   ave(dataset$normalized.losses, FUN = function(x) mean(x, na.rm = TRUE)),
                                   dataset$normalized.losses)

#Replacing 4 missing values of column bore
dataset$bore = ifelse(is.na(dataset$bore),
                      ave(dataset$bore, FUN = function(x) mean(x, na.rm = TRUE)),
                      dataset$bore)

#Replacing 4 missing values of column stroke
dataset$stroke = ifelse(is.na(dataset$stroke),
                      ave(dataset$stroke, FUN = function(x) mean(x, na.rm = TRUE)),
                      dataset$stroke)

#Replacing 2 missing values of column horsepower
dataset$horsepower = ifelse(is.na(dataset$horsepower),
                        ave(dataset$horsepower, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$horsepower)

#Replacing 2 missing values of column peak.rpm
dataset$peak.rpm = ifelse(is.na(dataset$peak.rpm),
                            ave(dataset$peak.rpm, FUN = function(x) mean(x, na.rm = TRUE)),
                            dataset$peak.rpm)

#Replacing 2 missing values of column price
dataset$price = ifelse(is.na(dataset$price),
                          ave(dataset$price, FUN = function(x) mean(x, na.rm = TRUE)),
                          dataset$price)




#Handling Categorical data
cols.cat <- c("make","fuel.type","aspiration","num.of.door","body.style","drive.wheels","engine.location",
              "engine.type","num.of.cylinders","fuel.system")
dataset[cols.cat] <- sapply(dataset[cols.cat],as.character)
dataset$make <- as.factor(dataset$make)
dataset$fuel.type <- as.factor(dataset$fuel.type)
dataset$aspiration <- as.factor(dataset$aspiration)
dataset$num.of.door <- as.factor(dataset$num.of.door)
dataset$body.style <- as.factor(dataset$body.style)
dataset$drive.wheels <- as.factor(dataset$drive.wheels)
dataset$engine.location <- as.factor(dataset$engine.location)
dataset$engine.type <- as.factor(dataset$engine.type)
dataset$num.of.cylinders <- as.factor(dataset$num.of.cylinders)
dataset$fuel.system <- as.factor(dataset$fuel.system)
dataset[cols.cat] <- sapply(dataset[cols.cat],as.numeric)

n <- length(dataset$price) # no of observations
#Splitting dataset into train & test
dat <- dataset
train_rows <- sample(1:n, .8*n)
train_set <- dat[train_rows,]
test_set <- dat[-train_rows,]

# Feature Scaling


#SVR execution with linear kernal with initial cost value 5
#MSE, RMSE, R^2 calculation
rmse <- function(test.out,predict.out)
{
  errval=test.out-predict.out
  SSE = sum(errval^2)
  MSE = mean(errval^2)
  RMSE = sqrt(MSE)
  SST = sum((test.out-mean(test.out))^2)
  SSR = SST - SSE
  #SSR = sum((predict.out-mean(test.out))^2)
  print(paste('SSE value is ',SSE))
  print(paste('MSE value is ',MSE))
  print(paste('RMSE value is',RMSE))
  print(paste('SSR value is',SSR))
  print(paste('SST value is',SST))
  print(paste('R-square value is',(1- (SSE/SST))))
}

library(e1071)
svrfit1 = svm(price ~., data=train_set , kernel ="linear", cost =10,type = 'eps-regression',
              scale =TRUE, Cross = 10)
#plot(svmfit , training_set)
#svrfit1$index
#coef(svrfit1)
summary (svrfit1)
#Run on test dataset
ypred=predict (svrfit1 ,test_set)
rmse(test_set$price,ypred)

#SVR execution with less cost value .01
svrfit2 = svm(price ~., data=train_set , kernel ="linear", cost =.01, type = 'eps-regression',
             scale =TRUE, Cross = 10)
#plot(svmfit , training_set)
#svrfit2$index
summary (svrfit2)
#Run on test dataset
ypred=predict (svrfit2 ,test_set)
rmse(test_set$price,ypred)

#Tune parameter with less cost function
set.seed (41)
tune.out=tune(svm ,price~.,data=train_set ,kernel ="linear", type = 'eps-regression',
              ranges =list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100,1000)), Cross = 10)
summary (tune.out)

bestmod =tune.out$best.model
summary (bestmod)

#Run the best model on testing datset
ypred=predict (bestmod ,test_set)
rmse(test_set$price,ypred)

#best model gives highest R-squared 

