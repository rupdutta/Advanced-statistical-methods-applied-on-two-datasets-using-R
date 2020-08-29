#Function for calculation of model accuracy (MSE, RMSE, R^2)
rmse <- function(test.out,predict.out)
{
  errval=test.out-predict.out
  SSE = sum(errval^2)
  MSE = mean(errval^2)
  RMSE = sqrt(MSE)
  SST = sum((test.out-mean(test.out))^2)
  SSR = SST - SSE
  R_square = (1- (SSE/SST))
  print(paste('SSE value is ',SSE))
  print(paste('MSE value is ',MSE))
  print(paste('RMSE value is',RMSE))
  print(paste('SSR value is',SSR))
  print(paste('SST value is',SST))
  print(paste('R-square value is', R_square))
}


######################
## DATA PREPROCESSING
######################

#Loading Automobile dataset
dataset = read.csv('Automobile_Data_Set.csv')

#Descriptive statistics
summary(dataset)

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

#Splitting the dataset on training and testing data
x = model.matrix(price~.,dataset)[,-1]
y = dataset$price

#############################################
## MULTI-LINEAR REGRESSION WITH LEAST SQUARES
#############################################
multi_regr = lm(price~., data = dataset)
summary(multi_regr)
#write.csv(summary(multi_regr)$coefficients,"coefficients_cars.csv")

#Illustrating each of the 4 plots separately
par(mfrow = c(1,1)) 
plot(multi_regr)

#Multicollinearity 
#Loading Car library that supports computiation of variance inflation factors (VIF)
library(car)
vif(multi_regr)

################################
## REGULARISATION METHODS
################################

#Splitting the dataset on training and testing data
set.seed(2)
train = sample(1:nrow(x), nrow(x)*0.8)
test = (-train)
y.test=y[test]

#Loading the glmnet package that is used to fit ridge/lasso/elastic net models
#install.packages('glmnet')
library(glmnet)

################################
## alpha = 0, Ridge Regression
################################
#Grid of values ranging from  lambda of 10^10 to 0.01
grid = 10^seq(10,-2,length = 100)
ridge.mod = glmnet(x[train,],y[train],alpha = 0, lambda = grid, thresh = 1e-12)

#Experimenting with different values of lambda
lambdaRidge = 30
ridge.pred = predict(ridge.mod, s = lambdaRidge, newx = x[test,])
rmse(y.test,ridge.pred)

#10-folds cross validation to identify the best lambda value 
ridge.cv.out = cv.glmnet(x[train,],y[train],alpha = 0)
plot(ridge.cv.out)

#Identifying the best lambda value for the ridge regression model
bestLambda = ridge.cv.out$lambda.min 
ridge.pred.best = predict(ridge.mod, s =bestLambda, newx=x[test,])
rmse(y.test, ridge.pred.best)

#Multi-linear regression coefficients with L2 regularization
out = glmnet(x,y, alpha = 0, lambda = grid)
ridge.coef = predict(out, type='coefficients', s = bestLambda)[1:26,]


################################
## alpha = 1, Lasso Regression
################################
lasso.mod = glmnet(x[train,],y[train],alpha = 1, lambda = grid)

#Experimenting with different values of lambda
lambdaLasso = 1.46
lasso.pred = predict(lasso.mod, s = lambdaLasso, newx = x[test,])
rmse(y.test,lasso.pred)

#10-folds cross validation to identify the best lambda value 
cv.out.lasso=cv.glmnet(x[train,],y[train],alpha = 1)
plot(cv.out.lasso)

#Identifying the best lambda value for the ridge regression model
bestLambdaLasso = cv.out.lasso$lambda.min

#Testing the model on the test data set with the best lambda parameter
lasso.pred = predict(lasso.mod, s = bestLambdaLasso, newx = x[test,])
rmse(y.test, lasso.pred)

#Multi-linear regression coefficients with L1 regularization
out = glmnet(x,y, alpha = 1, lambda = grid)
lasso.coef= predict(out, type = 'coefficients', s = bestLambdaLasso)[1:26,]



#####################################
## alpha between 0 and 1, Elastic Net
#####################################
#Elastic net model with alpha equal to 0.5
elasticnet = glmnet(x[train,],y[train],alpha = 0.5, lambda = grid)

#Experimenting with different values of lambda
lambdaElasticNet = 0.5
elasticnet.pred = predict(elasticnet, s = lambdaElasticNet, newx = x[test,])
rmse(y.test,elasticnet.pred)

#10-folds cross validation to identify the best lambda value 
elasticnet.cv.out = cv.glmnet(x[train,], y[train], alpha=0.5)
plot(elasticnet.cv.out)

#Identifying the best lambda value for the ridge regression model
bestLambdaElasticnet = elasticnet.cv.out$lambda.min

#Testing the model on the test data set with the best lambda parameter
elasticnet.predicted = predict(elasticnet.cv.out, s=bestLambdaElasticnet, newx=x[test,])
rmse(y.test,elasticnet.predicted)

#Multi-linear regression coefficients with L1 & L2 regularization
out = glmnet(x,y, alpha = 0.5)
predict(out, type='coefficients', s = bestLambdaElasticnet)[1:26,]

#Creating the model using different alphas (0, 0.1, ... , 0.9, 1) to see which one makes a better prediction 
results <- data.frame()
for (i in 1:10) {
  
  #10-folds Cross-validation for identifying the best lambda value
  fit = cv.glmnet(x[train,], y[train], type.measure="mse", alpha=i/10)
  lambda = fit$lambda.min
  pred = predict(fit, s=lambda,  x[test,])
  alpha = i/10
  
  # Model's performance indicators
  errval=y.test-pred
  SSE = sum(errval^2)
  MSE = mean(errval^2)
  RMSE = sqrt(MSE)
  SST = sum((y.test-mean(y.test))^2)
  SSR = SST - SSE
  
  # Storing the results
  results <- rbind(results, data.frame(alpha=alpha,Lambda=lambda, MSE=MSE,Rsquare=SSR/SST))
}
# Printing the table of results for test dataset with different alphas
results
