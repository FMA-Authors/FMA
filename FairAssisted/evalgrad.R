# this function evaluates the gradient of the prediction loss with respect to the coefficients w

evalgrad <- function(X, y, ftype, beta, beta0, lambda){
  # make sure that beta0 is in the correct format
  if (is.matrix(beta0)){
    beta0 <- beta0[1,1]
  }
  
  n <- nrow(X) # number of instances
  idx_not0 <- which(beta != 0) # indices of those variables with non-zero coefficients
  
  if (ftype == "LS"){ # lease-squared loss
    if (length(idx_not0) == 0){ # all variables have zero coefficients
      vgrad <- t(X) %*% (beta0 - y) / n + lambda * beta
    } else{ # some variables have non-zero coefficients
      vgrad <- t(X) %*% (X[,idx_not0] %*% as.matrix(beta[idx_not0]) - y + beta0) / n + lambda * beta
    }
  } else if (ftype == "Logistic"){
    if (length(idx_not0) == 0){
      eta <- matrix(beta0, nrow = n, ncol = 1)
    } else{
      eta <- X[,idx_not0] %*% as.matrix(beta[idx_not0]) + beta0
    }
    vgrad <- t(X) %*% (-y / (1 + exp(y * eta))) / n + lambda * beta
  } else{
    cat("function type error!\n")
    vgrad <- NULL
  }
  
  return(vgrad)
}