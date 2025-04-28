# this function finds the variable indices of those variables belonging to the groups in FG

FG_map <- function(FG, G){
  indx <- rep(0, length(G)) # =1 means that a variable's group is in the selected set FG
  
  if (is.list(G)){ # G is a list, each element represents the group index for a variable
    for (g in 1:length(G)){
      indx[g] <- sum(G[[g]] %in% FG) > 0 # variable level selection
    }
  } else{ # G is a p-dimensional vector (what we will use)
    temp_idx <- which(G %in% FG) # their groups are in the selected set
    indx[temp_idx] <- 1
  }
  
  indx_return <- which(indx == 1) # indices of those variables
  return(indx_return)
}