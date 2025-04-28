# this function fits a penalized logistic regression model with the l2 penalty term using the gradient descent algorithm

glmfit_set <- function(x, y, lambda){
  n <- length(y) # number of instances
  d <- ncol(x) # number of predictors used
  if (is.vector(x)){
    d <- 1
  }
  
  if (lambda != 0){ # with the ridge penalty term, use our defined function
    maxit <- 1000
    tol <- 1e-10
    res0 <- 0 # initialized intercept
    res <- rep(0, d) # initialized beta
    Xtemp <- cbind(rep(1,n), x) # add a column of 1 to incorporate the intercept
    Hm <- (1/4/n) * (t(Xtemp) %*% Xtemp)
    maj <- eigen(Hm)$values[1] # the largest eigenvalue, determining the step size in the gradient descent
    nyx <- replicate(d+1, -y) * Xtemp # -yixi for each instance i
    yax <- as.vector((res0 + x %*% as.matrix(res)) * y) # (t(xi) %*% res + res0)*yi for each instance
    
    # gradient descent
    for (i in 1:maxit){
      res0_t <- res0 # the intercept/betas for the previous iteration
      res_t <- res
      Gg <- colMeans(nyx / replicate(d+1, (1 + exp(yax)))) # compute the gradient from the cross entropy loss (+lambda*res from the ridge penalty, doesn't penalize intercept)
      res0 <- res0_t - (1/maj)*Gg[1] # update the intercept, doesn't penalize it
      res <- res_t - (1/(maj+lambda)) * (Gg[2:(d+1)] + lambda*res_t) # update the betas, penalize them
      diff <- c(res0 - res0_t, res - res_t)
      if (max(diff^2) < tol){
        break # the change is not significant, we can break
      }
      yax <- as.vector((res0 + x %*% as.matrix(res)) * y)
    }
  } else { # without the ridge penalty term, use the predefined glm function
    x <- as.matrix(x)
    V <- (y+1)/2
    temp_dat <- as.data.frame(cbind(x, V))
    temp_mod <- glm(V ~ ., data = temp_dat, family = binomial())
    coeff <- unname(temp_mod$coefficients)
    res0 <- coeff[1]
    res <- coeff[2:length(coeff)]
  }
  
  return(list(intercept = res0, beta_FG = res))
}