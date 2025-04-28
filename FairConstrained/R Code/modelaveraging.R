# these functions implement the fairness-constrained model averaging and return the final combined estimator
# each function is for one type of fairness measurement, including statistical parity, equal opportunity, and equal odds

# for statistical parity
ma_dpar <- function(response, predictors, sensitive, unfairness, unfairness_type, sum_to_one, non_negative, spars, aic_penalty){
  # convert the response variable from (-1,1) into a binary (0,1) factor
  response <- factor((response+1)/2, levels = c(0,1))
  
  # if the sensitive attribute is in the vector form, transform it into a dataframe
  if (is.factor(sensitive)){
    sensitive <- structure(data.frame(V = sensitive), names = "S1")
  }
  
  # add the intercept for predictors
  predictors <- design_matrix(predictors, intercept = TRUE)
  # one-hot encoding for the categorical sensitive attribute
  sensitive <- model.matrix(~S1-1, data = sensitive)
  # if add the AIC penalty term, add 0 to spars to incorporate the intercept (the intercept model uses 0 predictor)
  if (aic_penalty){
    spars <- c(0, spars)
  }

  # check whether the unfairness is allowed (if unfairness = 0): if not, special-case:
  # drop all the predictors that are correlated with at least one sensitive attribute
  if (unfairness <= sqrt(.Machine$double.eps)){
    # compute the covariance between each predictor and each sensitive attribute 
    # sum across sensitive attributes for each predictor
    overall.cov <- rowSums(abs(cov(predictors, sensitive)))
    # choose the predictors that are not correlated with any sensitive attribute (the intercept is always allowed)
    allowed.predictors <- predictors[, overall.cov == 0, drop = FALSE]
    # fit a completely fair model and return it.
    completely.fair.model <- glm(response ~ allowed.predictors - 1, family = "binomial")
    # build a vector of coefficients that uses zero coefficients for the predictors we have dropped
    coefs <- structure(rep(0, length(overall.cov)), names = names(overall.cov))
    coefs[colnames(allowed.predictors)] <- coef(completely.fair.model)
    return(build_return_value(completely.fair.model, coefs, response, predictors, sensitive, unfairness, unfairness_type, measure = "dpar"))
  }
  
  # when sum_to_one = FALSE and non_negative = FALSE (only have fairness constraints), fit an unconstrained logistic regression
  # check whether the given unfairness bound is too high (it is not an active constraint, the unconstrained logistic regression satisfies it)
  if ((!sum_to_one) & (!non_negative)){
    unconstrained.model <- glm(response ~ predictors - 1, family = "binomial")
    if (anyNA(coef(unconstrained.model))){
      NAcoefs <- names(which(is.na(coef(unconstrained.model))))
      stop("the coefficient(s) ", q(NAcoefs), " are NA in the unconstrained model.")
    }
    # the model fitted value (linear predictor value)
    unconstrained.fitted <- predictors %*% coef(unconstrained.model)
  
    if (unfairness_type == "correlation"){
      # the correlation between the linear predictor with each of the sensitive attribute
      correlations <- cor(sensitive, unconstrained.fitted)
      # if the constraint w.r.t each sensitive attribute is satisfied, return the unconstrained model
      if (all(abs(correlations) < unfairness)){
        return(build_return_value(model = unconstrained.model, response = response, predictors = predictors, 
                                  sensitive = sensitive, unfairness = unfairness, unfairness_type = unfairness_type, 
                                  measure = "dpar"))
      }
    } else if (unfairness_type == "covariance"){
      # the covariance between the linear predictor with each of the sensitive attribute
      covariances <- cov(sensitive, unconstrained.fitted)
      if (all(abs(covariances) < unfairness)){
        return(build_return_value(model = unconstrained.model, response = response, predictors = predictors, 
                                  sensitive = sensitive, unfairness = unfairness, unfairness_type = unfairness_type, 
                                  measure = "dpar"))
      }
    }
  }
  
  # if use the correlation unfairness measure, get an upper bound for the corresponding covariance measure search interval
  unconstrained.model <- glm(response ~ predictors - 1, family = "binomial")
  unconstrained.fitted <- predictors %*% coef(unconstrained.model)
  max.covariance <- max(abs(cov(sensitive, unconstrained.fitted)))
  
  # perform the constrained optimization.
  coefs <- optim_dpar(response = response, predictors = predictors,
                      sensitive = sensitive, unfairness = unfairness, unfairness_type = unfairness_type,
                      sum_to_one = sum_to_one, non_negative = non_negative, spars, aic_penalty, measure = "dpar",
                      max.covariance = max.covariance)
  
  # fit the logistic regression with the given coefficients (actually not fit model, but compute quantities), computing all the quantities we are going to return
  final.model <- glm(response ~ - 1, offset = predictors %*% coefs, family = "binomial")
  
  return(build_return_value(final.model, coefs, response, predictors, sensitive, unfairness, unfairness_type, measure = "dpar"))
}



# for equalized opportunity
ma_eopp <- function(predictors, response, sensitive, solver_type, EPS, FNR_Bound, FPR_Bound, tau, mu, take_initial_sol, sum_to_one, non_negative, spars, aic_penalty){
  # convert the response variable from (-1,1) into a binary (0,1) factor
  response_factor <- factor((1+response)/2, levels = c(0,1))
  
  # if the sensitive attribute is in the vector form, transform it into a dataframe
  if (is.factor(sensitive)){
    sensitive <- structure(data.frame(V = sensitive), names = "S1")
  }
  
  # add the intercept for predictors
  predictors <- design_matrix(predictors, intercept = TRUE)
  # one-hot encoding for the categorical sensitive attribute
  sensitive <- model.matrix(~S1-1, data = sensitive)
  # if add the AIC penalty term, add 0 to spars to incorporate the intercept (the intercept model uses 0 predictor)
  if (aic_penalty){
    spars <- c(0, spars)
  }
  
  # check whether the unfairness is allowed (if unfairness = 0): if not, special-case:
  # drop all the predictors that are correlated with at least one sensitive attribute
  if (FNR_Bound <= sqrt(.Machine$double.eps)){
    # compute the covariance between each predictor and each sensitive attribute
    # sum across sensitive attributes for each predictor
    overall.cov <- rowSums(abs(cov(predictors, sensitive)))
    # choose the predictors that are not correlated with any sensitive attribute (the intercept is always allowed)
    allowed.predictors <- predictors[, overall.cov == 0, drop = FALSE]
    # fit a completely fair model and return it
    completely.fair.model <- glm(response_factor ~ allowed.predictors - 1, family = "binomial")
    # build a vector of coefficients that uses zero coefficients for the predictors we have dropped
    coefs <- structure(rep(0, length(overall.cov)), names = names(overall.cov))
    coefs[colnames(allowed.predictors)] <- coef(completely.fair.model)
    return(build_return_value(completely.fair.model, coefs, response, predictors, sensitive, FNR_Bound, unfairness_type = "covariance", measure = "eopp"))
  }
  
  # when sum_to_one = FALSE and non_negative = FALSE (only have fairness constraints), fit an unconstrained logistic regression
  # check whether the given FNR_Bound is too high (it is not an active constraint, the unconstrained logistic regression satisfies it)
  if ((!sum_to_one) & (!non_negative)){
    unconstrained.model <- glm(response_factor ~ predictors - 1, family = "binomial")
    if (anyNA(coef(unconstrained.model))){
      NAcoefs <- names(which(is.na(coef(unconstrained.model))))
      stop("the coefficient(s) ", q(NAcoefs), " are NA in the unconstrained model.")
    }
    # the model fitted value (linear predictor value)
    unconstrained.fitted <- predictors %*% coef(unconstrained.model)
  
    # compute the FNR covariance
    covariances <- covariance_fnr(unconstrained.fitted, response, sensitive)
    if (all(covariances < FNR_Bound)){
      return(build_return_value(model = unconstrained.model, response = response, predictors = predictors, 
                                sensitive = sensitive, unfairness = FNR_Bound, unfairness_type = "covariance", 
                                measure = "eopp"))
    }
  }
  
  # convert spars into a matrix
  if (aic_penalty){
    spars <- matrix(spars, nrow = 1, ncol = length(spars))
  }
  
  
  coefs <- optim_eopp(predictors, response, sensitive, solver_type, EPS, "eopp", FNR_Bound, 
                      FPR_Bound, tau, mu, take_initial_sol, sum_to_one, non_negative, spars, aic_penalty)
  # convert the numpy array into a vector in r
  solved_w <- structure(as.vector(coefs), names = colnames(predictors))
  
  # fit the logistic regression with the given coefficients (actually not fit model, but compute quantities), computing all the quantities we are going to return
  final.model <- glm(response_factor ~ - 1, offset = predictors %*% solved_w, family = "binomial")
  
  return(build_return_value(final.model, solved_w, response, predictors, sensitive, FNR_Bound, "covariance", "eopp"))
}



# for equalized odds
ma_eodd <- function(predictors, response, sensitive, solver_type, EPS, FNR_Bound, FPR_Bound, tau, mu, take_initial_sol, sum_to_one, non_negative, spars, aic_penalty){
  # convert the response variable from (-1,1) into a binary (0,1) factor
  response_factor <- factor((1+response)/2, levels = c(0,1))
  
  # if the sensitive attribute is in the vector form, transform it into a dataframe
  if (is.factor(sensitive)){
    sensitive <- structure(data.frame(V = sensitive), names = "S1")
  }
  
  # add the intercept for predictors
  predictors <- design_matrix(predictors, intercept = TRUE)
  # one-hot encoding for the categorical sensitive attribute
  sensitive <- model.matrix(~S1-1, data = sensitive)
  # if add the AIC penalty term, add 0 to spars to incorporate the intercept (the intercept model uses 0 predictor)
  if (aic_penalty){
    spars <- c(0, spars)
  }
  
  # the two upper bounds used for equalized odds
  up_Bound <- c(FNR_Bound, FPR_Bound)
  names(up_Bound) <- c("FNR", "FPR")
  
  # check whether the unfairness is allowed (if unfairness = 0): if not, special-case:
  # drop all the predictors that are correlated with at least one sensitive attribute
  if ((FNR_Bound <= sqrt(.Machine$double.eps)) | (FPR_Bound <= sqrt(.Machine$double.eps))){
    # compute the covariance between each predictor and each sensitive attribute
    # sum across sensitive attributes for each predictor
    overall.cov <- rowSums(abs(cov(predictors, sensitive)))
    # choose the predictors that are not correlated with any sensitive attribute (the intercept is always allowed)
    allowed.predictors <- predictors[, overall.cov == 0, drop = FALSE]
    # fit a completely fair model and return it
    completely.fair.model <- glm(response_factor ~ allowed.predictors - 1, family = "binomial")
    # build a vector of coefficients that uses zero coefficients for the predictors we have dropped
    coefs <- structure(rep(0, length(overall.cov)), names = names(overall.cov))
    coefs[colnames(allowed.predictors)] <- coef(completely.fair.model)
    return(build_return_value(completely.fair.model, coefs, response, predictors, sensitive, up_Bound, unfairness_type = "covariance", measure = "eodd"))
  }
  
  # when sum_to_one = FALSE and non_negative = FALSE (only have fairness constraints), fit an unconstrained logistic regression
  # check whether the given FNR_Bound and FPR_Bound are too high (they are not active constraints, the unconstrained logistic regression satisfies them)
  if ((!sum_to_one) & (!non_negative)){
    unconstrained.model <- glm(response_factor ~ predictors - 1, family = "binomial")
    if (anyNA(coef(unconstrained.model))){
      NAcoefs <- names(which(is.na(coef(unconstrained.model))))
      stop("the coefficient(s) ", q(NAcoefs), " are NA in the unconstrained model.")
    }
    # the model fitted value (linear predictor value)
    unconstrained.fitted <- predictors %*% coef(unconstrained.model)
    
    # compute the FNR covariance and the FPR covariance
    covariances <- covariance_fnr_fpr(unconstrained.fitted, response, sensitive)
    fnr_covariance <- covariances$cov_fnr
    fpr_covariance <- covariances$cov_fpr
    if (all(fnr_covariance < FNR_Bound) & all(fpr_covariance < FPR_Bound)){
      return(build_return_value(model = unconstrained.model, response = response, predictors = predictors,
                                sensitive = sensitive, unfairness = up_Bound, unfairness_type = "covariance",
                                measure = "eodd"))
    }
  }
  
  # convert spars into a matrix
  if (aic_penalty){
    spars <- matrix(spars, nrow = 1, ncol = length(spars))
  }
  

  coefs <- optim_eopp(predictors, response, sensitive, solver_type, EPS, "eodd", FNR_Bound,
                      FPR_Bound, tau, mu, take_initial_sol, sum_to_one, non_negative, spars, aic_penalty)
  # convert the numpy array into a vector in r
  solved_w <- structure(as.vector(coefs), names = colnames(predictors))
  
  # fit the logistic regression with the given coefficients (actually not fit model, but compute quantities), computing all the quantities we are going to return
  final.model <- glm(response_factor ~ - 1, offset = predictors %*% solved_w, family = "binomial")
  
  return(build_return_value(final.model, solved_w, response, predictors, sensitive, up_Bound, unfairness_type = "covariance", measure = "eodd"))
}