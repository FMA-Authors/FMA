# this function selects the group among top groups with the smallest fairness loss in a forward step

get_one_fair <- function(chgfair, toplist){
  tidx <- which.max(chgfair)
  selg <- toplist[tidx]
  
  return(selg)
}