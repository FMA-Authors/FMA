# this function implements the Fairness-assisted stepwise method to find the variable selection path and the corresponding coefficients

fairassist <- function(X, y, ftype, altype, G, eps_obj, eps_gdt, maxit, flag, nu_forward, nu_backward, lambda, fairness, S, cutoff, version, QUIET){
  # maximal number of groups to be selected (including the intercept)
  maxgp <- max(G) + 1
  
  # check if a valid fairness metric
  if (!(fairness %in% c("dpar", "eopp", "eodd"))){
    stop("This type of fairness metric is not supported!")
  }
  
  if (!QUIET){
    # print some information
    cat(sprintf("%3s\t%3s\t%10s\t%10s\t%10s\t%10s\n", "Iter", "k", "predloss", "fairloss", "forwardNum", "backwardNum"))
  }
  
  # Initialization Part I
  path <- list()
  path$forwardNum <- 0 # forward step
  path$backwardNum <- 0 # backward step
  p <- length(G) # total number of predictors
  n <- length(y) # total number of instances
  beta <- rep(0, p)
  # initialize beta0 (the intercept) for linear regression / logistic regression (at the beginning, no variables in the model yet)
  if (ftype == "LS"){
    beta0 <- mean(y)
  } else if (ftype == "Logistic"){
    V2 <- (y+1)/2
    beta0 <- glm(V2 ~ 1, family = "binomial")$coefficients[1] # (y+1)/2 convert y from (-1,1) to (0,1), only the intercept
  } else {
    stop("function type error!")
  }
  
  FG <- c() # the ids of the selected variable groups, initially it is none
  k <- 0 # current number of groups selected
  predloss <- evalpredloss(X, y, ftype, beta, beta0, lambda) # initial prediction loss function value
  fairloss <- evalfairloss(X, y, ftype, beta, beta0, fairness, S, cutoff, version) # initial fairness loss function value
  top <- c() # the ids of the variable groups that are at our top consideration to add
  
  # Initialization Part II (k ranges from 0 to m)
  path$beta <- vector("list", length = maxgp) # each element stores the fitted beta coefficients at one k
  path$beta0 <- rep(NA, maxgp) # each element stores the fitted intercept at one k
  path$predloss <- rep(NA, maxgp) # each element stores the prediction loss value of the model fitted at one k
  path$fairloss <- rep(NA, maxgp) # each element stores the fairness loss value of the model fitted at one k
  path$FG <- vector("list", length = maxgp) # each element stores the ids of the selected variable groups at one k
  path$top <- vector("list", length = maxgp) # each element stores the ids of the variable groups that are at our top consideration for adding next to achieve the k from k-1
  path$delta <- rep(NA, maxgp) # each element stores the reduction in the prediction loss value by adding a new variable to achieve the k from k-1
  path$solpath <- c() # record the overall solution path
  path$maxs <- 0 # maximum sparsity, maximal number of groups used in the solution path
  path$sparsity <- rep(NA, maxgp) # each element stores the number of groups used (in the model fitting) at one k
  
  # the values at the current k (0 groups is selected)
  path$beta[[k+1]] <- beta
  path$beta0[k+1] <- beta0
  path$predloss[k+1] <- predloss
  path$fairloss[k+1] <- fairloss
  
  path$FG[[k+1]] <- c(NA)
  path$top[[k+1]] <- c(NA)
  path$delta[k+1] <- 0
  path$sparsity[k+1] <- 0
  
  # iteration number
  iter <- 0
  
  # compute the largest eigenvalue of the matrix 1/4/n*t(x[,idx]) %*% x[,idx] for the gth group, where idx indicates the indices of those predictors in the group g
  major = rep(0, max(G))
  if (ftype == "Logistic"){ # for the Logistic case only
    for (g in 1:max(G)){
      idx <- which(G == g)
      Hg <- (1/4/n) * t(X[,idx]) %*% X[,idx]
      major[g] <- eigen(Hg)$values[1]
    }
  } 
  
  # ------------------- forward step ----------------------
  while (iter <= maxit) {
    
    grpval <- rep(-Inf, max(G)) # the change in the prediction loss value by adding a new group of variables in the forward step
    fairval <- rep(-Inf, max(G)) # the change in the fairness loss value by adding a new group of variables in the forward step
    
    # compute the changes by adding a new group of variables
    if (altype == "obj"){ # Plain-vanilla
      for (g in 1:max(G)){
        if (!(g %in% path$FG[[k+1]])){ # the group g of variables has not been selected yet when k groups are being selected
          newvals <- solve_one(X, y, ftype, beta, beta0, g, G, lambda, major[g], fairness, S, cutoff, version)
          # the change in the prediction loss value by adding a group of variables in the forward step
          grpval[g] <- path$predloss[k+1] - newvals$predloss
          # the change in the fairness loss value by adding a group of variables in the forward step
          fairval[g] <- path$fairloss[k+1] - newvals$fairloss
        }
      }
    } else if (altype == "gdt"){ # Gradient-based
      vgrad <- evalgrad(X, y, ftype, beta, beta0, lambda) # the gradient of the prediction loss w.r.t w, the gradient is evaluated at w(k)
      for (g in 1:max(G)){
        if(!(g %in% path$FG[[k+1]])){
          grpval[g] <- norm(vgrad[G == g], "2") # the l2-norm of the gradient w.r.t those coefficients in the group g
          # the fairval has not been computed here, will be computed only for the top groups later to save computational cost
        }
      }
    } else{
      stop("algorithm type error!")
    }
    
    # sort the group by their corresponding grpval values
    sort_result <- sort(grpval, decreasing = TRUE, index.return = TRUE)
    a <- sort_result$x
    I <- sort_result$ix
    
    # the stopping criteria, the smallest reduction in the prediction loss function value acceptable
    # or the smallest norm of the gradient of the prediction loss w.r.t w acceptable
    if (altype == "obj"){
      eps_threshold <- eps_obj
      if (a[1] < eps_obj){
        break
      }
    } else if (altype == "gdt"){
      eps_threshold <- eps_gdt
      if (a[1] < eps_gdt){
        break
      }
    } else{
      stop("algorithm type error!")
    }
    
    # generate the candidate set of groups to add by taking the fairness into consideration
    if (abs(nu_forward-1) < 1e-8){ # if the relaxation parameter is 1, simply select the top one group
      selg <- I[1]
    } else { # if the relaxation parameter less than one, consider the top few groups
      if (a[1] < 0) {
        stop("Forward can lead to increment in prediction loss value!\n")
      }
      top <- which((grpval >= (a[1] * nu_forward))) # no & (grpval >= eps_threshold)
      if (length(top) <= 1) {
        selg <- I[1]
      } else { # take the fairness into consideration, select the group among top groups with the smallest fairness loss
        if (altype == "gdt"){ # to save computational cost, compute coefficients and fairness loss only for top groups 
          for (j in 1:length(top)){
            g <- top[j]
            newvals <- solve_one(X, y, ftype, beta, beta0, g, G, lambda, major[g], fairness, S, cutoff, version)
            fairval[g] <- path$fairloss[k+1] - newvals$fairloss
          }
        }
        selg <- get_one_fair(fairval[top], top)
      }
    }
    
    # add the newly selected group into the model and refit the model
    FG <- sort(c(FG, selg), decreasing = FALSE)
    fit_result <- solve_set(X, y, ftype, FG, G, lambda, fairness, S, cutoff, version) # refit the model with one more selected group
    
    predloss <- fit_result$predloss
    fairloss <- fit_result$fairloss
    beta <- fit_result$beta
    beta0 <- fit_result$beta0
    delta <- path$predloss[k+1] - predloss # the improvement in the prediction loss value
    
    k <- k + 1
    
    # record this forward step
    path$beta[[k+1]] <- beta
    path$beta0[k+1] <- beta0
    path$predloss[k+1] <- predloss
    path$fairloss[k+1] <- fairloss
    
    path$FG[[k+1]] <- FG
    path$delta[k+1] <- delta
    path$top[[k+1]] <- top
    path$solpath <- c(path$solpath, selg) # the solution path of adding in a new group
    path$forwardNum <- path$forwardNum + 1
    path$sparsity[k+1] <- k # the number of groups used (in the model fitting) at k+1 is actually k
    path$maxs <- max(path$maxs, k) # maximum sparsity (number of groups used) so far
    
    iter <- iter + 1 # one more iteration
    
    # print the information
    if (!QUIET){
      cat(sprintf("%3d\t%3d\t%10.4f\t%10.4f\t%10d\t%10d\n", iter, k, path$predloss[k+1], path$fairloss[k+1], path$forwardNum, path$backwardNum))
    }
    
    # ------------------- backward step ----------------------
    while (flag){
      
      deleted_pred <- rep(NA, length(FG)) # record the change in the prediction loss value by eliminating each group
      deleted_fair <- rep(NA, length(FG)) # record the change in the fairness loss value by eliminating each group
      
      for (j in 1:length(FG)){
        temp <- beta # the current fitted coefficients
        temp[G == FG[j]] <- 0 # eliminating the jth group by directly setting the coefficients of those variables in this group to 0
        deleted_pred[j] <- evalpredloss(X, y, ftype, temp, beta0, lambda) - predloss
        deleted_fair[j] <- evalfairloss(X, y, ftype, temp, beta0, fairness, S, cutoff, version) - fairloss
      }
      
      # sort the group by their corresponding deleted_pred values
      sort_result <- sort(deleted_pred, decreasing = FALSE, index.return = TRUE)
      a <- sort_result$x
      I <- sort_result$ix
      
      # if the smallest change is still larger than some tolerance level, we will not eliminate
      if (a[1] >= (path$delta[k+1]/2)){ # the threshold used is the reduction in the prediction loss by adding the last variable
        break
      }
      
      # if the smallest change is lower than the tolerance level, we will eliminate
      # generate the candidate set of groups to remove by taking the fairness into consideration
      if (abs(nu_backward-1) < 1e-8){ # if the relaxation parameter is 1, simply remove the bottom one group
        selg <- FG[I[1]]
      } else { # if the relaxation parameter larger than one, consider the bottom few groups
        if (a[1] < 0) {
          stop("Backward can lead to reduction in prediction loss value!\n")
        }
        bottom <- which((deleted_pred <= (a[1] * nu_backward)) & (deleted_pred < (path$delta[k+1]/2))) # & deleted_pred < (path$delta[k+1]/2)
        if (length(bottom) <= 1) {
          selg <- FG[I[1]]
        } else {
          selg <- remove_one_fair(deleted_fair[bottom], FG[bottom])
        }
      }
      
      path$solpath <- c(path$solpath, -selg)
      FG <- FG[FG != selg]
      
      # remove the newly eliminated group from the model and refit the model
      fit_result <- solve_set(X, y, ftype, FG, G, lambda, fairness, S, cutoff, version)
      
      predloss <- fit_result$predloss
      fairloss <- fit_result$fairloss
      beta <- fit_result$beta
      beta0 <- fit_result$beta0
      
      # update the record for which k-1 groups are used
      k <- k - 1
      
      # record this backward step
      path$beta[[k+1]] <- beta
      path$beta0[k+1] <- beta0
      path$predloss[k+1] <- predloss
      path$fairloss[k+1] <- fairloss
      
      path$FG[[k+1]] <- FG
      path$backwardNum <- path$backwardNum + 1
      
      # don't update the delta, it's only updated in the forward step (used as a threshold for the backward step)
      
      iter <- iter + 1 # one more iteration
      
      # print the information
      if (!QUIET){
        cat(sprintf("%3d\t%3d\t%10.4f\t%10.4f\t%10d\t%10d\n", iter, k, path$predloss[k+1], path$fairloss[k+1], path$forwardNum, path$backwardNum))
      }
      
      if (k == 0){ # no group is left in the model, we should break
        break
      }
    }
  }
  
  # return the fitted result
  return(path)
}















