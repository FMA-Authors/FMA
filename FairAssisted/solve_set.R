# this function fits the linear/logistic regression model using a given set of selected groups

solve_set <- function(X, y, ftype, FG, G, lambda, fairness, S, cutoff, version){
  n <- length(y) # number of instances
  indx <- FG_map(FG, G) # variable indices of those variables belonging to the groups selected in FG
  beta <- rep(0, length(G)) # initialize the beta
  
  # linear regression
  if (ftype == "LS"){
    if (length(FG) == 0){ # no group is selected, the model only contains intercept
      beta0 <- mean(y)
      predloss <- evalpredloss(X, y, ftype, beta, beta0, lambda)
      fairloss <- evalfairloss(X, y, ftype, beta, beta0, fairness, S, cutoff, version)
    } else{ # some groups are selected into FG
      ymean <- mean(y)
      Xtemp <- X[,indx] # the predictors used
      Xmean <- colMeans(Xtemp) # the column mean of each predictor
      ytemp <- y - ymean # centralize y
      Xtemp <- Xtemp - matrix(rep(Xmean, n), byrow = TRUE, nrow = n, ncol = length(indx)) # centralize X
      beta[indx] <- solve(t(Xtemp) %*% Xtemp + lambda*diag(length(indx))) %*% t(Xtemp) %*% ytemp # after centralizing X and y, don't need to include the intercept
      beta0 <- ymean - Xmean %*% beta[indx] # compute the intercept
      predloss <- evalpredloss(X, y, ftype, beta, beta0, lambda)
      fairloss <- evalfairloss(X, y, ftype, beta, beta0, fairness, S, cutoff, version)
    }
  } else if (ftype == "Logistic"){ # logistic regression
    if (length(FG) == 0){ # no group is selected, the model only contains intercept
      V2 <- (y+1)/2
      beta0 <- glm(V2 ~ 1, family = "binomial")$coefficients[1] # (y+1)/2 convert y from (-1,1) to (0,1), only the intercept
      predloss <- evalpredloss(X, y, ftype, beta, beta0, lambda)
      fairloss <- evalfairloss(X, y, ftype, beta, beta0, fairness, S, cutoff, version)
    } else{
      temp_fit <- glmfit_set(X[,indx], y, lambda) # fit a penalized logistic regression model using self-defined function
      beta0 <- temp_fit$intercept
      beta[indx] <- temp_fit$beta_FG
      predloss <- evalpredloss(X, y, ftype, beta, beta0, lambda)
      fairloss <- evalfairloss(X, y, ftype, beta, beta0, fairness, S, cutoff, version)
    }
  }
  
  return(list(predloss = predloss, fairloss = fairloss, beta = beta, beta0 = beta0))
}
