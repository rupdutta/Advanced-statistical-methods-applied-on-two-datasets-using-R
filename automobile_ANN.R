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

n <- length(dataset$price)
#Splitting dataset into train & test
dat <- dataset
train_rows <- sample(1:n, .8*n)
train_set <- dat[train_rows,]
test_set <- dat[-train_rows,]

# Feature Scaling

#Load ANN2 library for Neural Network
library(ANN2)

x = data.matrix(train_set[-26])
z = as.vector(train_set$price)
test_set_p = data.matrix(test_set[-26])

################################
## No regularization
################################

NNL0 <- neuralnetwork(x, z, hidden.layers = matrix(c(3,3,3,3),1,4),
                      regression = TRUE, loss.type = "squared", activ.functions = "linear",
                      L1 = 0, L2 = 0, standardize = TRUE, n.epochs = 2000)

predL0 <- predict(NNL0, newdata = test_set_p)



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
  print(paste('R-square value is',(SSR/SST)))
}


rmse(test_set$price,as.numeric(predL0$predictions))

################################
## L1 = 1, Lasso Regression
################################

NNL1 <- neuralnetwork(x, z, hidden.layers = matrix(c(3,3,3,3),1,4),
                    regression = TRUE, loss.type = "squared", activ.functions = "linear",
                    L1 = 1, L2 = 0, standardize = TRUE, n.epochs = 2000)

predL1 <- predict(NNL1, newdata = test_set_p)



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
  print(paste('R-square value is',(SSR/SST)))
}


rmse(test_set$price,as.numeric(predL1$predictions))

#R-squared is 80

################################
## L2 = 1, Ridge Regression
################################

NNL2 <- neuralnetwork(x, z, hidden.layers = matrix(c(3,3,3,3),1,4),
                      regression = TRUE, loss.type = "squared", activ.functions = "linear",
                      L1 = 0, L2 = 1, standardize = TRUE, n.epochs = 2000)

predL2 <- predict(NNL2, newdata = test_set_p)



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
  print(paste('R-square value is',(SSR/SST)))
}


rmse(test_set$price,as.numeric(predL2$predictions))

########################################################
## L1,L2 values between 0 and 1 , Elastic Net Regression
########################################################
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
  print(paste('R-square value is',(SSR/SST)))
}

st <- data.frame()
for (i in 1:10) {
  NNELNET <- neuralnetwork(x, z, hidden.layers = matrix(c(3,3,3,3),1,4),
                           regression = TRUE, loss.type = "squared", activ.functions = "linear",
                           L1 = i/10, L2 = 1-(i/10), standardize = TRUE, n.epochs = 2000)
  predelnet <- predict(NNELNET, newdata = test_set_p)
  print(paste('L1 value is ',i/10))
  print(paste('L2 value is ',1-(i/10)))
  rmse(test_set$price,as.numeric(predelnet$predictions))
  st <- rbind(st, data.frame(L1=i/10,L2=1-(i/10),MSE=mean((test_set$price-as.numeric(predelnet$predictions))^2))              )
}
st
