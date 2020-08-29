#Data Preprocessing
library(MASS)
dataset = Boston
n <- length(Boston$crim) # no of observations
#No Missing data
#No Categorical data
#Convert predictor to binary value
dataset$crim = ifelse(dataset$crim>=.25,1,0)

#Splitting dataset into train & test
dat <- dataset
train_rows <- sample(1:n, .8*n)
train_set <- dat[train_rows,]
test_set <- dat[-train_rows,]


# Feature Scaling

#Load SVM library
#install.packages('sparseSVM') 
library(sparseSVM) 

x = data.matrix(train_set[-1])
z = train_set$crim
test_set_p = data.matrix(test_set[-1])

################################
## alpha = 1, Lasso Regression
################################

fit = cv.sparseSVM(x, z, alpha = 1)
summary(fit)
coef(fit)
pred = predict(fit, test_set_p)

table(Predict=pred,Truth=test_set$crim)
print(paste0("Classification rate for Lasso is ", (sum(pred==test_set$crim)/length(pred))))

#####################################
## alpha between 0 and 1, Elastic Net
#####################################
st <- data.frame()
for (i in 1:10) {
  fit = cv.sparseSVM(x, z, alpha = i/10)
  pred = predict(fit, test_set_p)
  alpha = i/10
  class_rate = sum(pred==test_set$crim)/length(pred)
  st <- rbind(st, data.frame(alpha=i/10,class_rate=class_rate))
}
st

################################
## alpha = 0, Ridge Regression
################################

library(e1071)
svrfit0 = svm(crim ~., data=train_set , kernel ="linear", cost =10,type = 'C-classification',
              scale =FALSE)

ypred=predict (svrfit0 ,test_set)
table(Predict=ypred,Truth=test_set$crim)
print(paste0("Classification rate ", (sum(ypred==test_set$crim)/length(ypred))))

set.seed (41)
tune.out=tune(svm ,as.factor(crim) ~.,data=train_set ,kernel ="linear", type = 'C-classification',
              ranges =list(cost=c(0.001 , 0.01, 0.1,.5, 1,5,10,100,1000) ))
summary (tune.out)

#selecting best model
bestmod =tune.out$best.model
summary (bestmod)

#Run the best model on testing datset
ypred=predict (bestmod ,test_set)

table(Predict=ypred,Truth=test_set$crim)
print(paste0("Classification rate ", (sum(ypred==test_set$crim)/length(ypred))))

#Ridge regression is the winer here 

