
# Concepts ----------------------------------------------------------------

### squared error loss

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



### regularization

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

# read the data
wine = read.csv('data/wine.csv')  # change to github location

library(doParallel)
cl = makeCluster(7)
registerDoParallel(cl)
