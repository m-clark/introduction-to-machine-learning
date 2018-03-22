
# Preliminaries -----------------------------------------------------------

library(dplyr)


# Concepts ----------------------------------------------------------------


#• Squared error loss ------------------------------------------------------

sqerrloss = function(beta, X, y){
  mu = X %*% beta
  sum((y-mu)^2)
}

# data setup
set.seed(123)                              # for reproducibility
N = 100                                    # sample size
X = cbind(1, X1=rnorm(N), X2=rnorm(N))     # model matrix: intercept, 2 predictors
beta = c(0, -.5, .5)                       # true coef values
y =  rnorm(N, X%*%beta, sd=1)              # target

# results
our_func = optim(par=c(0,0,0),             # starting values
                 fn=sqerrloss,
                 X=X,
                 y=y,
                 method='BFGS')
lm_result = lm(y ~ ., data.frame(X[,-1]))  # check with lm

rbind(optim=c(our_func$par, our_func$value),
      lm=c(coef(lm_result), loss=sum(resid(lm_result)^2)))



#• Regularization ----------------------------------------------------------

# data setup
set.seed(123)
N = 100
X = cbind(1, matrix(rnorm(N*10), ncol=10))
beta = runif(ncol(X))
y =  rnorm(N, X%*%beta, sd=2)

sqerrloss_reg = function(beta, X, y, lambda=.5){
  mu = X%*%beta
  # sum((y-mu)^2) + lambda*sum(abs(beta[-1])) # conceptual
  sum((y-mu)^2) + 2*length(y)*lambda*sum(abs(beta[-1])) # actual for lasso
}

lm_result = lm(y~., data.frame(X[,-1]) )
regularized_result = optim(par=rep(0, ncol(X)),
                           fn=sqerrloss_reg,
                           X=X,
                           y=y,
                           method='BFGS')



# Black Box ---------------------------------------------------------------

# Necessary packages for models
# install.packages('caret')
# install.packages('glmnet')
# install.packages('class')
# install.packages('randomForest')
# install.packages('e1071')


#• Data Setup --------------------------------------------------------------


### Read the data

wine = read.csv('data/wine.csv')  # change to github location


### Implement parallelism

library(doParallel)
cl = makeCluster(7)
registerDoParallel(cl)

### Training-Test Split

library(caret)
set.seed(1234) # so that the indices will be the same when re-run
trainIndices = createDataPartition(wine$good, p=.8, list=F)

wine_train = wine %>% 
  select(-free.sulfur.dioxide, -density, -quality, -color, -white) %>% 
  slice(trainIndices)

wine_test = wine %>% 
  select(-free.sulfur.dioxide, -density, -quality, -color, -white) %>% 
  slice(-trainIndices)


### Predictor plot
wine_trainplot = predict(preProcess(select(wine_train, -good), method='range'),
                         select(wine_train, -good))
wine_trainplot = select(wine_train, -good) %>%
  preProcess(method='range') %>%
  predict(newdata= select(wine_train, -good))
featurePlot(wine_trainplot, wine_train$good, 'box')






#• Models ------------------------------------------------------------------

### For any models, if you hit a snag, you can still load the results of the analysis
# load('data/results_*.RData')



#•• Regularized regression --------------------------------------------------

# Training
set.seed(1234)
cv_opts = trainControl(method='cv', number=10)
regreg_opts = expand.grid(.alpha = seq(.1, 1, length = 5),
                          .lambda = seq(.1, .5, length = 5))
results_regreg <- train(good~.,
                        data=wine_train,
                        method = "glmnet",
                        trControl = cv_opts,
                        preProcess = c("center", "scale"),
                        tuneGrid = regreg_opts)

# Test
preds_regreg = predict(results_regreg, wine_test)
good_observed = wine_test$good
confusionMatrix(preds_regreg, good_observed, positive='Good')



#•• knn regression ----------------------------------------------------------

# Training
set.seed(1234)
knn_opts = data.frame(k=c(seq(3, 11, 2), 25, 51, 101)) # odd to avoid ties
results_knn = train(good~., 
                    data=wine_train, 
                    method='knn',
                    preProcess=c('center', 'scale'), 
                    trControl=cv_opts,
                    tuneGrid = knn_opts)

results_knn

# Test
preds_knn = predict(results_knn, wine_test)
confusionMatrix(preds_knn, good_observed, positive='Good')


#•• Neural Nets -------------------------------------------------------------

# Training
set.seed(1234)
results_nnet = train(good~., 
                     data=wine_train, 
                     method='avNNet',
                     trControl=cv_opts, 
                     preProcess=c('center', 'scale'),
                     tuneLength=5, 
                     trace=F, 
                     maxit=1000)
results_nnet

# Test
preds_nnet = predict(results_nnet, wine_test)
confusionMatrix(preds_nnet, good_observed, positive='Good')


#•• Random Forest -----------------------------------------------------------

# Training
set.seed(1234)
rf_opts = data.frame(mtry=c(2:6))
results_rf = train(good~., 
                   data=wine_train, 
                   method='rf',
                   preProcess=c('center', 'scale'), 
                   trControl=cv_opts, 
                   tuneGrid=rf_opts, 
                   n.tree=1000)
results_rf

# Test
preds_rf = predict(results_rf, wine_test)
confusionMatrix(preds_rf, good_observed, positive='Good')


#•• Support Vector Machines -------------------------------------------------

# Training
set.seed(1234)
results_svm = train(good~., 
                    data=wine_train, 
                    method='svmLinear2',
                    preProcess=c('center', 'scale'), 
                    trControl=cv_opts, 
                    tuneLength=5)
results_svm

# Test
preds_svm = predict(results_svm, wine_test)
confusionMatrix(preds_svm, good_observed, positive='Good')




# Other -------------------------------------------------------------------

stopCluster(cl)  # after you are done with your session



# Wrap-up -----------------------------------------------------------------

install.packages("xgboost")
library(xgboost)
modelLookup("xgbLinear")
modelLookup("xgbTree")

xgb_opts = expand.grid(eta=c(.3,.4),
                       max_depth=c(9, 12),
                       colsample_bytree=c(.6,.8),
                       subsample=c(.5,.75,1),
                       nrounds=1000,
                       min_child_weight=1,
                       gamma=0)

set.seed(1234)
results_xgb = train(good~., 
                    data=wine_train, 
                    method='xgbTree',
                    preProcess=c('center', 'scale'), 
                    trControl=cv_opts, 
                    tuneGrid=xgb_opts)
results_xgb
save(results_xgb, file='data/results_xgb.RData')
preds_gb = predict(results_xgb, wine_test)
confusionMatrix(preds_gb, good_observed, positive='Good')
