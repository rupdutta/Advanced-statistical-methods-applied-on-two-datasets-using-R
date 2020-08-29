#Data Preprocessing
library(MASS)
dataset = Boston
n <- length(Boston$crim) # no of observations
#No Missing data
#No Categorical data

#Splitting dataset into train & test
dat <- dataset
train_rows <- sample(1:n, .8*n)
train_set <- dat[train_rows,]
test_set <- dat[-train_rows,]

# # Feature Scaling

#Load ANN2 library for Neural Network
#install.packages('ANN2')
library(ANN2)

x = data.matrix(train_set[-1])
z = as.vector(train_set$crim)
test_set_p = data.matrix(test_set[-1])

#######################################
## No regularization, setting L1,L2 = 0
#######################################

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


rmse(test_set$crim,as.numeric(predL0$predictions))

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


rmse(test_set$crim,as.numeric(predL1$predictions))


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


rmse(test_set$crim,as.numeric(predL2$predictions))

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
  rmse(test_set$crim,as.numeric(predelnet$predictions))
  st <- rbind(st, data.frame(L1=i/10,L2=1-(i/10),MSE=mean((test_set$crim-as.numeric(predelnet$predictions))^2))              )
}
st

