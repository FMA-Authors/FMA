# this function solves the convex constrained optimization for statistical parity via CVXR

optim_dpar <- function(response, predictors, sensitive, unfairness, unfairness_type, sum_to_one, non_negative, spars, aic_penalty, measure, max.covariance){
  # convert the binary factor (0,1) response into (-1,1)
  response <- ifelse(response == "0", -1, 1)
  n <- length(response)
  n_pos <- sum(response == 1)
  n_neg <- sum(response == -1)
  
  # center the sensitive attribute (z-zbar)
  sensitive <- scale(sensitive, center = TRUE, scale = FALSE)
  
  if (measure == "dpar"){
    # precompute the cross-products ((z-zbar)^T X theta)
    xts <- t(sensitive) %*% predictors
    
    # estimate the regression coefficients subject to a bound on the unfairness
    optimization <- function(my.bound){
      coefs <- CVXR::Variable(rows = ncol(predictors), cols = 1)
      
      if (aic_penalty){
        # convert spars from vector into a matrix
        spars <- matrix(spars, nrow = 1, ncol = length(spars))
        # add the AIC penalty term
        obj <- sum(CVXR::logistic(predictors[response == -1, ] %*% coefs)) + sum(CVXR::logistic(-predictors[response == 1, ] %*% coefs)) + sum(spars %*% coefs)
      } else{
        obj <- sum(CVXR::logistic(predictors[response == -1, ] %*% coefs)) + sum(CVXR::logistic(-predictors[response == 1, ] %*% coefs))
      }
      # define the constraints
      if (sum_to_one == FALSE & non_negative == FALSE){
        constraints <- list(abs(xts %*% coefs) / (n-1) <= my.bound)
      } else if (sum_to_one == TRUE & non_negative == FALSE){
        constraints <- list(abs(xts %*% coefs) / (n-1) <= my.bound, sum(coefs) == 1)
      } else if (sum_to_one == TRUE & non_negative == TRUE){
        constraints <- list(abs(xts %*% coefs) / (n-1) <= my.bound, sum(coefs) == 1, coefs >= 0)
      }
      prob <- CVXR::Problem(CVXR::Minimize(obj), constraints = constraints)
      result <- CVXR::solve(prob, ignore_dcp = TRUE)
    
      if (result$status %in% c("optimal", "optimal_inaccurate")){
        return(result$getValue(coefs))
      } else{
        # try 2: if the default solver fails, try again with a different one
        result <- CVXR::solve(prob, solver = "SCS", ignore_dcp = TRUE)
        if (result$status %in% c("optimal", "optimal_inaccurate")){
          return(result$getValue(coefs))
        } else{
          # try 3: add some slack to the constraint to get a slightly-invalid solution that still looks like a valid one
          if (sum_to_one == FALSE & non_negative == FALSE){
            constraints <- list(abs(xts %*% coefs) / (n-1) <= my.bound*1.01)
          } else if (sum_to_one == TRUE & non_negative == FALSE){
            constraints <- list(abs(xts %*% coefs) / (n-1) <= my.bound*1.01, sum(coefs) == 1)
          } else if (sum_to_one == TRUE & non_negative == TRUE){
            constraints <- list(abs(xts %*% coefs) / (n-1) <= my.bound*1.01, sum(coefs) == 1, coefs >= 0)
          }
          prob <- CVXR::Problem(CVXR::Minimize(obj), constraints = constraints)
          result <- CVXR::solve(prob, ignore_dcp = TRUE)
          if (!(result$status %in% c("optimal", "optimal_inaccurate"))){
            stop("CVXR failed to find a solution (", result$status, ").")
          }
          return(result$getValue(coefs))
        }
      }
    }
    
    # remap the constraint on the covariances to the constraint on the correlations
    constraint_mapping <- function(my.bound){
      coefs <- optimization(my.bound)
      # compute the correlation between each sensitive attribute and the estimated linear predictor
      cc <- abs(cor(sensitive, predictors %*% coefs))
      # return the absolute difference between the maximal correlation and the target unfairness
      return(abs(max(cc) - unfairness))
    }
    
    if (unfairness_type == "correlation"){
      # find the bound on the covariances that satisfies the bound on the correlations (that is, the given unfairness value) using the maximum covariance from
      # the unconstrained model (+ some slack) to limit the search interval
      bound <- optimize(f = constraint_mapping,
                        interval = c(0, max.covariance * 1.1))$minimum
      
    } else if (unfairness_type == "covariance"){
      # unfairness value is already expressed with a bound on the covariances
      bound <- unfairness
    }
    
    # estimate and rename the regression coefficients.
    coefs <- structure(as.vector(optimization(bound)),
                       names = colnames(predictors))
    return(coefs)
  }
}