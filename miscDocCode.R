
possible data sets: \\
UCI 1994 Census Income, n = ~49k, p=14, y=income>50k \\
Congressional reading level? \\
UCI 2011 Wine Quality, e.g. make binary of 8-10 rating vs. best \\
Communities and Crime 2009 e.g. make binary of crime \\

##################
### Adult Data ###
##################
adult.train_init = read.csv('http://www.nd.edu/~mclark19/learn/data/adult.censusincome/adult.train.csv',  na.strings = "?")
adult.test_init = read.csv('http://www.nd.edu/~mclark19/learn/data/adult.censusincome/adult.test.csv',  na.strings = "?")
adult.train = na.omit(adult.train_init)
adult.test = na.omit(adult.test_init)
levels(adult.train$salary) = c("No","Yes")
levels(adult.test$salary) = c("No","Yes")
adult.train$countryorig = factor(ifelse(adult.train$countryorig=="United-States","US","Other"))
adult.test$countryorig = factor(ifelse(adult.test$countryorig=="United-States","US","Other"))
adult.train$marital = factor(adult.train$marital%in%c("Married-civ-spouse","Married-spouse-absent", "Married-AF-spouse"), labels=c("NotMarried","Married"))
adult.test$marital = factor(adult.test$marital%in%c("Married-civ-spouse","Married-spouse-absent", "Married-AF-spouse"), labels=c("NotMarried","Married"))
levels(adult.train$workclass) = c('Gov',"Gov","Never","Private","Self","Self","Gov","Without"); adult.train = adult.train[adult.train$workclass!="Without",]
levels(adult.test$workclass) = c('Gov',"Gov","Never","Private","Self","Self","Gov","Without"); adult.test = adult.test[adult.test$workclass!="Without",]
adult.train$race = factor(ifelse(adult.train$race=="White","White","Other"))
adult.test$race = factor(ifelse(adult.test$race=="White","White","Other"))

adult.train = droplevels(adult.train[,c(1,2,5,6,9:15)])
adult.test = droplevels(adult.test[,c(1,2,5,6,9:15)])

adult.train[,c(1,3,7:9)] = predict(preProcess(adult.train[,c(1,3,7:9)], 'range'), adult.train[,c(1,3,7:9)])
adult.test[,c(1,3,7:9)] = predict(preProcess(adult.test[,c(1,3,7:9)], 'range'), adult.test[,c(1,3,7:9)])

summary(adult.train)

prop.table(table(adult.train$sal))



cv_opts = trainControl(method="cv", number=10)
testrun_train = train(salary~., data=adult.train, method="svmLinear", trControl=cv_opts, tuneLength=3)
#testrun_test = train(salary~., data=adult.train, method="C5.0")

confusionMatrix(predict(testrun_train, adult.test), adult.test$salary, positive="Yes")
dotPlot(varImp(testrun_train, useModel=F))
filterVarImp(adult.train, adult.train$sal)
pROC:::roc(adult.train$sal, as.numeric(adult.train$marital))$auc  #binary inputs are treated as numeric


##################
### Wine Data ###
##################
red = read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
red$color = "red"
white = read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
white$color = "white"

d = rbind(red, white); d$white = as.numeric(factor(d$color))-1
str(d)
d$good = factor(d$quality>=6, labels=c("No", "Yes"))


summary(d)

write.csv(d, "C:/Users/mclark19/Desktop/miscdata/goodwine.csv", row.names=F)
write.csv(d, "N:/www/learn/data/goodwine.csv", row.names=F)


summary(princomp(wine[,-c(12:14)]))$loadings

wine = read.csv('http://www.nd.edu/~mclark19/learn/data/goodwine.csv')
summary(wine)
white = wine[wine$color=="white",]
red = wine[wine$color=="red",]

summary(lm(white))

library(corrplot)
whitevif = white[,1:11]; whitevif$color = NULL;corrplot(cor(whitevif), "number")
rsq = vector('list', length(colnames(whitevif))); names(rsq) = colnames(whitevif)
for (i in 1:length(colnames(whitevif))){
  varnames = colnames(whitevif)
  y = whitevif[,varnames[i]]
  x = whitevif[,varnames[-i]]
  rsq[[i]] = summary(lm(y~., data=data.frame(x,y)))$r.sq
}
rsq

#remove density, rerun
whitevif = white[,c(1:7,9:11)]; whitevif$color = NULL;corrplot(cor(whitevif), "number")
rsq = vector('list', length(colnames(whitevif))); names(rsq) = colnames(whitevif)
for (i in 1:length(colnames(whitevif))){
  varnames = colnames(whitevif)
  y = whitevif[,varnames[i]]
  x = whitevif[,varnames[-i]]
  rsq[[i]] = summary(lm(y~., data=data.frame(x,y)))$r.sq
}
rsq

#remove total.sulfur, rerun
whitevif = white[,c(1:6,9:13)]; whitevif$color = NULL;corrplot(cor(whitevif), "number")
rsq = vector('list', length(colnames(whitevif))); names(rsq) = colnames(whitevif)
for (i in 1:length(colnames(whitevif))){
  varnames = colnames(whitevif)
  y = whitevif[,varnames[i]]
  x = whitevif[,varnames[-i]]
  rsq[[i]] = summary(lm(y~., data=data.frame(x,y)))$r.sq
}
rsq

groupcomps <- sbf(white[,-c(12:14)], white$good,
                  sizes = c(1:10),
                  sbfControl = sbfControl(functions = rfSBF,
                                         verbose = FALSE, 
                                         method = "cv"))
groupcomps
confusionMatrix(groupcomps)

varsel <- rfe(white[,-c(12:14)], white$good,
              sizes = c(1:10),
              rfeControl = rfeControl(functions = ldaFuncs, method = "cv"))
varsel
plot(varsel, type = c("o", "g"))


confusionMatrix(groupcomps)


library(caret)


set.seed(1234) #so that the indices will be the same when re-run
trainIndices = createDataPartition(white$good, p=.8, list=F)

wine_train = white[trainIndices, -c(8,12:13)]  #remove quality and color variable, as well as density
prep_train = preProcess(wine_train[,-11], method="range") # omit target
wine_train = data.frame(predict(prep_train, wine_train[ ,-11]), good=wine_train[ , 11])

wine_test = white[!1:nrow(white)%in%trainIndices, -c(8,12:13)]
prep_test = preProcess(wine_test[,-11], method="range")
wine_test = data.frame(predict(prep_test, wine_test[,-11]), good=wine_test[ ,11])

nrow(wine_train); nrow(wine_test)

### whole data

trainIndices = createDataPartition(wine$good, p=.8, list=F)
wine_train = wine[trainIndices, -c(6,8,12:14)]  #remove quality and color, as well as density and others
prep_train = preProcess(wine_train[,-10], method="range") # omit target
wine_train = data.frame(predict(prep_train, wine_train[ ,-10]), good=wine_train[ , 10])

wine_test = wine[!1:nrow(wine)%in%trainIndices, -c(6,8,12:14)]
prep_test = preProcess(wine_test[,-10], method="range")
wine_test = data.frame(predict(prep_test, wine_test[,-10]), good=wine_test[ ,10])


findCorrelation(cor(wine_train[,-c(11)]), cutoff=.7) #color is largely captured by acidity, sulfur dioxide

winevif = wine[,c(1:11,14)]; corrplot(cor(winevif), "number")
rsq = vector('list', length(colnames(winevif))); names(rsq) = colnames(winevif)
for (i in 1:length(colnames(winevif))){
  varnames = colnames(winevif)
  y = winevif[,varnames[i]]
  x = winevif[,varnames[-i]]
  rsq[[i]] = summary(lm(y~., data=data.frame(x,y)))$r.sq
}
rsq

library(doSNOW)
registerDoSNOW(makeCluster(3, type = "SOCK"))


set.seed(1234)
cv_opts = trainControl(method="cv", number=10)
knn_opts = data.frame(.k=c(seq(3, 11, 2), 25, 50)) #odd to avoid ties
results_knn = train(good~., data=wine_train, method="knn", trControl=cv_opts, tuneGrid = knn_opts)
results_knn

knn_preds = predict(results_knn, wine_test)
confusionMatrix(knn_preds, wine_test$good, positive="Good")

set.seed(1234)
results_nnet = train(good~., data=wine_train, method="nnet", trControl=cv_opts, tuneLength=3)
results_nnet

nnet_preds = predict(results_nnet, wine_test)
confusionMatrix(nnet_preds, wine_test$good, positive="Good")

set.seed(1234)
test = train(good~., data=wine_train, method="rf", preProcess="range", trControl=cv_opts, tuneLength=5, importance=T) #[,c("alcohol", 'volatile.acidity', 'good')]
test

test_preds = extractPrediction(list(test), testX=wine_test[,-10], testY=wine_test[,10])
confusionMatrix(test_preds[test_preds$dataType=="Test",2], test_preds[test_preds$dataType=="Test",1], positive="Good")


accloss = rep(NA, ncol(wine_train)-1); names(accloss) = colnames(wine_train[,-10])
for (i in 1:(ncol(wine_train)-1)){
  test = train(good~., data=wine_train[,-i], method="knn", trControl=cv_opts, tuneGrid = knn_opts, preProcess="range")
  test_preds = extractPrediction(list(test), testX=wine_test[,-c(i, 10)], testY=wine_test[,10])
  origconf = confusionMatrix(preds_knn[preds_knn$dataType=="Test",2],preds_knn[preds_knn$dataType=="Test",1], positive="Good")
  testconf = confusionMatrix(test_preds[test_preds$dataType=="Test",2], test_preds[test_preds$dataType=="Test",1], positive="Good")
  accloss[i] = origconf$overall[1] - testconf$overall[1]
}

sort(accloss, dec=T)

test = train(good~., data=wine_train[,c("alcohol", 'volatile.acidity', 'good')], method="knn", trControl=cv_opts, tuneGrid = knn_opts, preProcess="range")
test_preds = extractPrediction(list(test), testX=wine_test[c("alcohol", 'volatile.acidity')], testY=wine_test[,10])
testconf = confusionMatrix(test_preds[test_preds$dataType=="Test",2], test_preds[test_preds$dataType=="Test",1], positive="Good")


### tree plot
library(tree)
treemod = tree(good~., data=wine_train, split="deviance")
plot(treemod); text(treemod)


### svm plots
set.seed(123)
x = rnorm(50)
y = rnorm(50)
z = rnorm(50)

#inner product space
x2 = x^2
y2 = y^2
z = sqrt(2)*x*y

lab = factor((x + z) > 0, labels=c("class1","class2"))
lab = factor((x2 + z) > 0, labels=c("class1","class2"))

d = data.frame(x,y,z,lab)
d2 = d[order(x,y),]
d3 = expand.grid(x=x,y=y); d3=d3[order(x,y),]

library(ggplot2)
g = ggplot(aes(x,y), data=d)
g + geom_point(aes(color=lab), size=3, pch=19) + 
  scale_color_manual(values=c("#FF5500", "navy")) +
  theme_minimal() + 
  theme(panel.grid.major=element_line(color=NA),
        panel.grid.minor=element_line(color=NA))

library(lattice)
cloud(y~x+z, data=d, groups=lab, col=c("#FF5500", "navy"), pch=19, screen = list(z = -50, x = -90), 
      pretty=T, par.box = c(col = "gray80"), par.settings = list(axis.line = list(col = "gray50")), frame.plot=F)
cloud(y2~x2+z, groups=lab, col=c("#FF5500", "navy"), pch=19, screen = list(z = 130, x = -80, y=0), 
      pretty=T, par.box = c(col = "gray80"), par.settings = list(axis.line = list(col = "gray50")), frame.plot=F)

newlab = factor(ifelse(x>0 & y>-1, "class1", "class2"))
g = ggplot(aes(x,z), data=d)
g + geom_point(aes(color=lab), size=3, pch=19) + 
  scale_color_manual(values=c("#FF5500", "navy")) +
  ylab("") + xlab("") +
  theme_minimal() + 
  theme(panel.grid.major=element_line(color=NA),
        panel.grid.minor=element_line(color=NA))


### Stacking combine predictions ###
preds_all_names = ls()[grep("preds_", ls())]
preds_all = cbind(sapply(preds_all_names, get))

### Network Example ###
d = read.csv('C:/Users/mclark19/Desktop/miscdata/Senate_Raw.csv')
gdat = d[,c(1,2,6)]
#colnames(gdat) =
library(igraph)
#library(sna)
g <- graph.data.frame(gdat, directed=FALSE) 

adjmat = get.adjacency(g, type="both", attr="Percent_Agreement")
adjmat = as.matrix(adjmat)
diag(adjmat) = 1
g = graph.adjacency(adjmat, mode="undirected", weighted=T)
write.graph(g, "/Figures/senate_graph.graphml", format="graphml")

######################################
### gaussian process bias variance ###
######################################
x = runif(1000)
ytrue = sin(3*pi*x)
basedat = cbind(x,ytrue)[order(x),]

gendatfunc = function(ytrue=sin(3*pi*x), noise=.5, n=1000){
  x = runif(n)
  y = sin(3*pi*x) + rnorm(n, sd=noise)
  d = cbind(x, y)
  d
}
gendat = replicate(100, gendatfunc(n=100))
str(gendat)

library(kernlab)

rbf1 = apply(gendat, 3, function(d) predict(gausspr(y~x, data=data.frame(d), kpar=list(sigma=.5)), newdata = data.frame(x), type='response'))
rbf2 = apply(gendat, 3, function(d) predict(gausspr(y~x, data=data.frame(d)), newdata = data.frame(x), type='response') )


library(scales)
pdf('images/graph64.pdf')
layout(matrix(1:4, ncol=2, byrow=T))
par(mar=c(3,2,3,2) ) # rep(2,4)
###rbf1
plot(basedat, col='white', type='l', lwd=3, ylim=c(-1.5,1.5), ylab="", xlab="", col.axis='gray50', bty="n", tck=-.02)
sapply(sample(1:100, 25), function(ncol) lines(cbind(x,rbf1[,ncol])[order(x),], col=alpha('#FF5500', alpha=.5), new=T))
rbfavg1 = data.frame(x, rowMeans(rbf1))[order(x),]
plot(basedat, col='darkgreen', type='l', lwd=3, ylab="", xlab="", col.axis='gray50', bty="n", tck=-.02)
lines(rbfavg1, col='#FF5500')

###rbf2
plot(basedat, col='white', type='l', lwd=3, ylim=c(-1.5,1.5), ylab="", xlab="", col.axis='gray50', bty="n", tck=-.02)
sapply(sample(1:100, 25), function(ncol) lines(cbind(x,rbf2[,ncol])[order(x),], col=alpha('#FF5500', alpha=.5), new=T))
rbfavg2 = data.frame(x, rowMeans(rbf2))[order(x),]
plot(basedat, col='darkgreen', type='l', lwd=3, ylab="", xlab="", col.axis='gray50', bty="n", tck=-.02)
lines(rbfavg2, col='#FF5500')
dev.off()
layout(1)
detach(package:scales)