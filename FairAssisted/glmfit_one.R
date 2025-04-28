# this function fits a logistic regression model with linear offset term and the l2 penalty term using the gradient descent algorithm

glmfit_one <- function(x, y, off_set, lambda, maj){
  n <- length(y) # number of instances
  d <- ncol(x) # number of predictors
  if (is.vector(x)){
    d <- 1
  }
  
  if (lambda != 0){ # with the ridge penalty term, using our defined function
    maxit <- 1000
    tol <- 1e-10
    res <- rep(0, d) # initialize the estimated coefficients
    nyx = replicate(d, -y) * x # -yixi for each instance i
    yax = as.vector((off_set + x %*% as.matrix(res)) * y) # (t(xi) %*% res + offset)*yi for each instance
    
    # gradient descent
    for (i in 1:maxit){
      res0 <- res # estimated coefficients from previous step
      Gj <- colMeans(nyx / replicate(d, (1 + exp(yax)))) # compute the gradient from the cross entropy loss (+lambda*res from the ridge penalty)
      res <- res0 - (1/(maj+lambda)) * (Gj + lambda*res0) # the updating rule, here 1/(maj+lambda) is used as the step size
      diff <- res - res0
      if (max(diff^2) < tol){
        break # the change is not significant, we can break
      }
      yax <- as.vector((off_set + x %*% as.matrix(res)) * y)
    }
  } else{ # without the ridge penalty, using the predefined glm function
    x <- as.matrix(x)
    V <- (y+1)/2
    temp_dat <- as.data.frame(cbind(x, V))
    temp_mod <- glm(V ~ . - 1, data = temp_dat, family = binomial(), offset = as.vector(off_set))
    res <- unname(temp_mod$coefficients)
  }
  
  return(res)
}