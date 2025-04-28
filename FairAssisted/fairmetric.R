# this function implements different fairness metrics

fairmetric <- function(S, y_true, y_pred, fairness, version){
  k <- length(unique(S)) # number of sensitive groups
  dat <- as.data.frame(cbind(y_true, y_pred))
  dat$S <- S
  
  if (fairness == "dpar"){
    par_val <- rep(NA, k)
    
    for (i in 1:k){
      subdat <- dat[dat$S == unique(S)[i], ]
      if (version == "discrete"){
        par_val[i] <- sum(subdat$y_pred == 1) / nrow(subdat)
      } else if (version == "continuous") {
        par_val[i] <- mean(subdat$y_pred)
      }
    }
    # compute the maximum pairwise absolute difference 
    metric <- max(abs(outer(par_val, par_val, "-")))
    
  } else if (fairness == "eopp"){
    tpr_val <- rep(NA, k)
    
    for (i in 1:k){
      if (version == "discrete"){
        subdat <- dat[dat$S == unique(S)[i], ]
        tpr_val[i] <- sum((subdat$y_pred == 1) & (subdat$y_true == 1)) / sum(subdat$y_true == 1)
      } else if (version == "continuous"){
        subdat <- dat[((dat$S == unique(S)[i]) & (dat$y_true == 1)), ]
        tpr_val[i] <- mean(subdat$y_pred)
      }
    }
    # compute the maximum pairwise absolute difference
    metric <- max(abs(outer(tpr_val, tpr_val, "-")))
    
  } else if (fairness == "eodd"){
    #p_pos <- sum(y_true == 1) / length(y_true)
    #p_neg <- sum(y_true == -1) / length(y_true)
    p_pos = 1/2
    p_neg = 1/2
    
    tpr_val <- rep(NA, k)
    fpr_val <- rep(NA, k)
    
    for (i in 1:k){
      if (version == "discrete"){
        subdat <- dat[dat$S == unique(S)[i], ]
        tpr_val[i] <- sum((subdat$y_pred == 1) & (subdat$y_true == 1)) / sum(subdat$y_true == 1)
        fpr_val[i] <- sum((subdat$y_pred == 1) & (subdat$y_true == -1)) / sum(subdat$y_true == -1)
      } else if (version == "continuous"){
        subdat1 <- dat[((dat$S == unique(S)[i]) & (dat$y_true == 1)), ]
        subdat2 <- dat[((dat$S == unique(S)[i]) & (dat$y_true == -1)), ]
        tpr_val[i] <- mean(subdat1$y_pred)
        fpr_val[i] <- mean(subdat2$y_pred)
      }
    }
    # compute the maximum pairwise absolute difference
    tpr_diff <- abs(outer(tpr_val, tpr_val, "-")) * p_pos
    fpr_diff <- abs(outer(fpr_val, fpr_val, "-")) * p_neg
    metric <- max(tpr_diff + fpr_diff)
    
  } else{
    cat("the fairness metric is not supported!\n")
  }
  
  return(metric)
}