# this file provides some preprocessing functions that will be used in the model averaging

# wrap cor() so that it handles zero-variance variables.
safe_cor <- function(x, y){
  suppressWarnings(cor(x, y))
}
# wrap cov() so that it handles zero-variance variables.
safe_cov <- function(x, y){
  suppressWarnings(cov(x, y))
}

# a function to compute the covariance between the sensitive attribute and the decision boundary with respect to FNR
covariance_fnr <- function(fitted, response, sensitive){
  # compute the min(0,y*lp)
  y_sign <- response
  min_val <- pmin(0, y_sign * as.vector(fitted))
  # only focus on the instances with Y = +1
  sub_sens <- sensitive[response == 1, , drop = FALSE]
  sub_min_val <- min_val[response == 1]
  # compute the covariance
  value <- abs(safe_cov(as.vector(sub_min_val), sub_sens))[1, ]
  
  return(value)
}

# a function to compute the covariance between the sensitive attribute and the decision boundary with respect to FPR
covariance_fpr <- function(fitted, response, sensitive){
  # compute the min(0,y*lp)
  y_sign <- response
  min_val <- pmin(0, y_sign * as.vector(fitted))
  # only focus on the instances with Y = -1
  sub_sens <- sensitive[response == -1, , drop = FALSE]
  sub_min_val <- min_val[response == -1]
  # compute the covariance
  value <- abs(safe_cov(as.vector(sub_min_val), sub_sens))[1, ]
  
  return(value)
}

# a function to compute the covariance between the sensitive attribute and the decision boundary with respect to FNR and FPR together (for equalized odds)
covariance_fnr_fpr <- function(fitted, response, sensitive){
  value_fnr <- covariance_fnr(fitted, response, sensitive)
  value_fpr <- covariance_fpr(fitted, response, sensitive)
  
  return(list(cov_fnr = value_fnr, cov_fpr = value_fpr))
}

# construct the return value
build_return_value <- function(model, coefs, response, predictors, sensitive, unfairness, unfairness_type, measure){
  
  # if the coefs is not provided, need to extract it from the model (only for the unconstrained model case)
  if (missing(coefs)){
    coefs <- structure(coef(model), names = colnames(predictors))
  } else{
    coefs <- structure(coefs, names = colnames(predictors))
  }
  
  # compute the linear predictor
  fitted <- predictors %*% coefs
  # compute the unfairness level of the current fitted coefficients
  if (measure == "dpar"){
    if (unfairness_type == "correlation"){
      value <- abs(safe_cor(as.vector(fitted), sensitive))[1, ]
    } else if (unfairness_type == "covariance"){
      value <- abs(safe_cov(as.vector(fitted), sensitive))[1, ]
    }
  }
  
  if (measure == "eopp"){
    if (unfairness_type == "covariance"){
      # compute the FNR covariance
      value <- covariance_fnr(fitted, response, sensitive)
    }
  }
  
  if (measure == "eodd"){
    if (unfairness_type == "covariance"){
      # compute both the FNR covariance and the FPR covariance
      value <- covariance_fnr_fpr(fitted, response, sensitive)
    }
  }
  
  return(structure(list(
    auxiliary = NULL,
    main = list(
      coefficients = coefs,
      residuals = as.vector(residuals(model)),
      fitted.values = as.vector(fitted),
      y = response,
      family = "binomial",
      deviance = model$deviance,
      loglik = logLik(model)
    ),
    fairness = list(
      measure = measure,
      value = value,
      bound = unfairness,
      unfairness_type = unfairness_type
    )
    )))
}

# transform a data set into the corresponding design matrix.
design_matrix <- function(data, intercept = TRUE){
  # gather all the variables (model.matrix() does that internally if we do not).
  if (is.data.frame(data)){
    frame <- model.frame(~ ., data = data)
  } else{
    frame <- model.frame(~ ., data = data.frame(data))
  }
  
  design <- model.matrix(~ ., data = frame)
  
  # ensure column names are syntactically valid.
  colnames(design)[colnames(design) != "(Intercept)"] <-
    make.names(colnames(design)[colnames(design) != "(Intercept)"])
  
  if (!intercept)
    design <- design[, colnames(design) != "(Intercept)", drop = FALSE]
  
  return(design)
}