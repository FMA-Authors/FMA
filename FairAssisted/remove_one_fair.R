# this function removes the group among bottom groups with the smallest fairness loss in a backward step

remove_one_fair <- function(chgfair, bottomlist){
  tidx <- which.min(chgfair)
  selg <- bottomlist[tidx]
  
  return(selg)
}