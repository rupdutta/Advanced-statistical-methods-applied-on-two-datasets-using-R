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
dataset$price = ifelse(dataset$price>=10595,1,0)
dat <- dataset
train_rows <- sample(1:n, .8*n)
train_set <- dat[train_rows,]
test_set <- dat[-train_rows,]


# Feature Scaling
#train_set[-c(4,14)] = scale(train_set[-c(4,14)])
#test_set[-c(4,14)] = scale(train_set[-c(4,14)])


#Load SVM library
library(sparseSVM) #install.packages('sparseSVM') 

x = data.matrix(train_set[-26])
z = train_set$price
test_set_p = data.matrix(test_set[-26])

################################
## alpha = 1, Lasso Regression
################################

#fit = sparseSVM(x, z)
#pred = predict(fit, test_set_p,lambda = c(0.01), alpha = .5)
#table(Predict=pred,Truth=test_set$medv)

fit = cv.sparseSVM(x, z, alpha = 1)
summary(fit)
coef(fit)
pred = predict(fit, test_set_p)

table(Predict=pred,Truth=test_set$price)
print(paste0("Classification rate for Lasso is ", (sum(pred==test_set$price)/length(pred))))

#####################################
## alpha between 0 and 1, Elastic Net
#####################################
# st <- data.frame()
# for (i in 1:10) {
#   fit = cv.sparseSVM(x, z, alpha = i/10)
#   pred = predict(fit, test_set_p)
#   print(i/10)
#   print(paste0("Classification rate ", (sum(pred==test_set$price)/length(pred))))
# }

st <- data.frame()
for (i in 1:10) {
  fit = cv.sparseSVM(x, z, alpha = i/10)
  pred = predict(fit, test_set_p)
  # print(i/10)
  # print(paste0("Classification rate ", (sum(pred==test_set$price)/length(pred))))
  alpha = i/10
  class_rate = sum(pred==test_set$price)/length(pred)
  st <- rbind(st, data.frame(alpha=i/10,class_rate=class_rate))
}
st
# ealastic net not helping much 

################################
## alpha = 0, Ridge Regression
################################

library(e1071)
svrfit0 = svm(price ~., data=train_set , kernel ="linear", cost =10,type = 'C-classification',
              scale =FALSE)

ypred=predict (svrfit0 ,test_set)
table(Predict=ypred,Truth=test_set$price)
print(paste0("Classification rate ", (sum(ypred==test_set$price)/length(ypred))))

set.seed (41)
tune.out=tune(svm ,as.factor(price) ~.,data=train_set ,kernel ="linear", type = 'C-classification',
              ranges =list(cost=c(0.001 , 0.01, 0.1,.5, 1,5,10,100,1000) ))
summary (tune.out)


bestmod =tune.out$best.model
summary (bestmod)

#Run the best model on testing datset

ypred=predict (bestmod ,test_set)

table(Predict=ypred,Truth=test_set$price)
print(paste0("Classification rate ", (sum(ypred==test_set$price)/length(ypred))))
#cost function changing not influencing the output

