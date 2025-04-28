# this function evaluates the target fairness metric with a given set of coefficients

evalfairloss <- function(X, y, ftype, beta, beta0, fairness, S, cutoff, version){
  ind <- which(beta != 0)
  n <- nrow(X) 
  
  if (ftype == "LS"){
    # the fairness for linear regression is to be developed
    cat("the fairness for linear regression is to be developed!\n")
    
  } else if (ftype == "Logistic"){
    # the linear predicted value
    l_pred <- as.vector(X[,ind] %*% as.matrix(beta[ind]) + rep(beta0, n))
    # the predicted probability
    prob_pred <- 1 / (1 + exp(-l_pred))
    
    if (version == "discrete"){
      # the predicted binary class with the given cutoff
      y_pred <- ifelse(prob_pred > cutoff, 1, -1)
    } else if (version == "continuous"){
      # the predicted probability
      y_pred <- prob_pred
    }
    fairloss <- fairmetric(S, y, y_pred, fairness, version)
    
  } else{
    cat("function type error!\n")
  }
  return(fairloss)
}