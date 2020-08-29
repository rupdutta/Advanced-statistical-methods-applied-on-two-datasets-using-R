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

#Loading MASS library that contains Boston dataset
library(MASS)

# Saving the dataset from the library MASS
Boston = Boston

#Descriptive statistics
summary(Boston)

#Correlation between target feature and predictors
cor(Boston, Boston$crim)

#Correlation between dataset features (optional)
#cor(Boston)

#############################################
## MULTI-LINEAR REGRESSION WITH LEAST SQUARES
#############################################
multi_regr = lm(crim~., data = Boston)
summary(multi_regr)

#Illustrate 4 plots simultaneously (optional)
par(mfrow = c(2,2)) 

#Illustrate each of the 4 plots separately
#par(mfrow = c(1,1)) 
plot(multi_regr)

#Outliers 
BostonImproved = Boston
outliers = BostonImproved[c(381,411,419),]


#Remove outliers from the dataset
BostonImproved = BostonImproved[-c(381,411,419),]

#Build a new model excluding outliers
multi_regr_clean = lm(crim~., data = BostonImproved)
summary(multi_regr_clean)

#Illustrate each of the 4 new plots separately
par(mfrow = c(1,1)) 
plot(multi_regr_clean)

# Remove additional outliers from the dataset (hasn't showed further improvement)
# BostonImproved2 =BostonImproved[-c(406,415,428),]
# multi_regr_clean2 = lm(crim~., data = BostonImproved2)
# summary(multi_regr_clean2)

#Multicollinearity 
#Loading Car library that supports computation of variance inflation factors (VIF)
library(car)
#Calculation Variance inflation factors (VIF)
vif(multi_regr_clean)


################################
## REGULARISATION METHODS
################################
#Splitting the dataset on training and testing data
x = model.matrix(crim~.,Boston)[,-1]
y = Boston$crim

set.seed(2)
train = sample(1:nrow(x), nrow(x)*.8)
test = (-train)
#x.test=x[test,]
y.test=y[test]

#Loading the glmnet package that is used to fit ridge/lasso/elastic net models
library(glmnet)

################################
## alpha = 0, Ridge Regression
################################
#Grid of values ranging from  lambda of 10^10 to 0.01
grid = 10^seq(10,-2,length = 100)
ridge.mod = glmnet(x[train,],y[train],alpha = 0, lambda = grid, thresh = 1e-12)

#Experimenting with different values of lambda
lambdaRidge = 4
ridge.pred = predict(ridge.mod, s = lambdaRidge, newx = x[test,])
rmse(y.test,ridge.pred)

#10-folds cross validation to identify the best lambda value 
#set.seed(1)
ridge.cv.out = cv.glmnet(x[train,],y[train],alpha = 0)
plot(ridge.cv.out)

#Identifying the best lambda value for the ridge regression model
bestLambda = ridge.cv.out$lambda.min 

#Testing the model on the test data set with the best lambda parameter
ridge.pred.best = predict(ridge.mod, s =bestLambda, newx=x[test,])
rmse(y.test, ridge.pred.best)

#Multi-linear regression coefficients with L1 regularization
out = glmnet(x,y, alpha = 0)
predict(out, type='coefficients', s = bestLambda)[1:14,]

################################
## alpha = 1, Lasso Regression
################################
lasso.mod = glmnet(x[train,],y[train], alpha = 1, lambda = grid)

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
out = glmnet(x,y, alpha = 1)
predict(out, type = 'coefficients', s = bestLambdaLasso)[1:14,]



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
predict(out, type='coefficients', s = bestLambdaElasticnet)[1:14,]

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

