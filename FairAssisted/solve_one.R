# this function computes the new smallest prediction loss value and the corresponding fairness value by adding a new group of variables in the forward step

solve_one <- function(X, y, ftype, beta, beta0, g, G, lambda, maj, fairness, S, cutoff, version){
  n <- length(y) # number of instances
  
  if (ftype == "LS"){
    nonzero_ind <- which(beta != 0) # indices of those variables having nonzero coefficients in w(k)
    res <- y - rep(beta0, n) - X[,nonzero_ind] %*% beta[nonzero_ind] # keep the previous beta and beta0 unchanged by treating them as the offset term
    idx_g <- which(G == g) # indices of those variables belonging to the group g
    # update the coefficients for those variables belonging to the group g (X^TX+lambdaI)^(-1)X^TY (divide both RSS and penalty by 2, ridge estimator stays the same)
    beta[idx_g] <- solve(t(X[,idx_g]) %*% X[,idx_g] + lambda*diag(length(idx_g))) %*% t(X[,idx_g]) %*% res
    predloss <- evalpredloss(X, y, ftype, beta, beta0, lambda) # the new prediction loss value
    fairloss <- evalfairloss(X, y, ftype, beta, beta0, fairness, S, cutoff, version) # the new fairness loss value
  } else if (ftype == "Logistic"){
    nonzero_ind <- which(beta != 0) # indices of those variables having nonzero coefficients in w(k)
    off_set <- rep(beta0, n) + X[,nonzero_ind] %*% as.matrix(beta[nonzero_ind]) # compute the offset term
    idx_g <- which(G == g) # indices of those variables belonging to the group g
    # update the coefficients for those variables belonging to the group g
    beta[idx_g] <- glmfit_one(X[,idx_g], y, off_set, lambda, maj) # y is from (-1,1), use gradient descent to fit a logistic regression model with l2 penalty term
    predloss <- evalpredloss(X, y, ftype, beta, beta0, lambda) # the new prediction loss value
    fairloss <- evalfairloss(X, y, ftype, beta, beta0, fairness, S, cutoff, version)
  }
  
  return(list(predloss = predloss, fairloss = fairloss))
}