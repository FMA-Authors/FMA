# this function evaluates the prediction loss function value with a given set of coefficients, for linear regression (squared loss) or logistic regression (cross entropy)

evalpredloss <- function(X, y, ftype, beta, beta0, lambda){
  n <- nrow(X) # number of instances
  ind <- which(beta != 0) # the indices of those variables with nonzero coefficients
  if (ftype == "LS"){
    # squared loss + a ridge penalty term
    predloss <- (0.5/n) * sum((rep(beta0, n) + X[,ind] %*% as.matrix(beta[ind]) - y)^2) + 0.5*lambda*sum(beta[ind]^2) 
  } else if (ftype == "Logistic"){
    # cross entropy loss + a ridge penalty term (y is in (-1,1) here to represent the binary response)
    yX <- matrix(rep(y, ncol(X)), nrow = n, ncol = ncol(X), byrow = FALSE) * X # the y*X matrix (check the logistic regression loss function)
    predloss <- (1/n) * sum(log(1 + exp(-y*beta0 - yX[,ind] %*% as.matrix(beta[ind])))) + 0.5*lambda*sum(beta[ind]^2)
  } else{
    cat("function type error!\n")
  }
  return(predloss)
}
