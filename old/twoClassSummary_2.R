twoClassSummary_2 <- function (data, lev = NULL, model = NULL) 
{
  require(pROC)
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  rocObject <- try(pROC:::roc(data$obs, data[, lev[1]]), silent = TRUE)
  rocAUC <- if (class(rocObject)[1] == "try-error") 
    NA
  else rocObject$auc
  sens = sensitivity(data[, "pred"], data[, "obs"], lev[1])
  spec = specificity(data[, "pred"], data[, "obs"], lev[2])
  #precision = sum(data[, "pred"]=="Yes" & data[, "obs"] =="Yes") / sum(data[, "pred"]=="Yes")
  #recall = sens
  #f1 = (2*precision*recall) / (precision+recall)
  out <- c(rocAUC, sens, spec, f1, precision, recall)
  names(out) <- c("ROC", "Sens", "Spec")#, "F1", "Prec", "Recall")
  out
}

environment(twoClassSummary_2) <- environment(twoClassSummary)