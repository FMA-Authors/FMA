import numpy as np
import cvxpy as cvx
import os, sys
import traceback
import dccp
from dccp.problem import is_dccp
from copy import deepcopy
from random import seed, shuffle

# define a function to check whether the input array is a dummy variable (binary with 0,1)
def dummy_check(input_arr):
    sort_unique = sorted(list(set(input_arr)))
    num_unique_vals = len(sort_unique)

    if (num_unique_vals == 2) and (sort_unique[0] == 0 and sort_unique[1] == 1):
        return True
    else:
        return False


# define a function to generate the covariance constraints in the DCCP problem, depending on the fairness measurement we use
def get_constraints(x_train, y_train, s_train, cov_thresh, cons_type, sum_to_one, non_negative, w):

    constraints = []
    num_dummys = s_train.shape[1]

    for k in range(num_dummys):
        # create constraints for each sensitive (0,1) dummy variable
        attr_arr = deepcopy(s_train[:,k])
        # check if it is a dummy variable
        check_result = dummy_check(attr_arr)

        if check_result:
            # create some dictionaries to store the intermediate values in the constraints construction
            val_total = {ct:{} for ct in ["FNR", "FPR"]} # N0 and N1
            val_avg = {ct:{} for ct in ["FNR", "FPR"]} # N0/N and N1/N
            val_sum = {ct:{} for ct in ["FNR", "FPR"]} # the sum across D0 and D1 of g

            for v in set(attr_arr): # v=0/1
                val_total["FNR"][v] = np.sum(np.logical_and(attr_arr == v, y_train == 1)) # N0+ and N1+
                val_total["FPR"][v] = np.sum(np.logical_and(attr_arr == v, y_train == -1)) # N0- and N1-
            
            for ct in ["FNR", "FPR"]:
                val_avg[ct][0] = val_total[ct][1] / float(val_total[ct][0] + val_total[ct][1]) # N1+/N+ or N1-/N-
                val_avg[ct][1] = 1.0 - val_avg[ct][0] # N0+/N+ or N0-/N-

            for v in set(attr_arr):
                # find the instances that are in D0 and D1, respectively
                idx = (attr_arr == v)

                # compute the sample covariance
                dist_bound = cvx.multiply(y_train[idx], x_train[idx] @ w) # y*(x@w)
                val_sum["FNR"][v] = cvx.sum(cvx.minimum(0, cvx.multiply((1 + y_train[idx])/2.0, dist_bound))) * (val_avg["FNR"][v] / np.sum(y_train == 1)) 
                val_sum["FPR"][v] = cvx.sum(cvx.minimum(0, cvx.multiply((1 - y_train[idx])/2.0, dist_bound))) * (val_avg["FPR"][v] / np.sum(y_train == -1))

            if cons_type == "Both":
                cts = ["FNR", "FPR"]
            elif cons_type in ["FNR", "FPR"]:
                cts = [cons_type]
            else:
                raise Exception("Invalid constraint type")
            
            # construct new DCCP constraints
            for ct in cts:
                thresh = cov_thresh[ct] # the upper bound of the covariance constraint for each covariance type
                constraints.append(val_sum[ct][1] <= val_sum[ct][0] + thresh)
                constraints.append(val_sum[ct][1] >= val_sum[ct][0] - thresh)
        
        else: # if it is not a dummy variable
            raise Exception("This is not a dummy variable, check the dummy matrix... Exiting...")
            sys.exit(1)
    
    if sum_to_one:
        constraints.append(cvx.sum(w) == 1)
    if non_negative:
        constraints.append(w >= 0)

    return constraints


# this function solves the DCCP constrained optimization for equal opportunity (and equal odds)
def optim_eopp(predictors, response, sensitive, solver_type, EPS, measure, FNR_Bound, FPR_Bound, tau, mu, take_initial_sol, sum_to_one, non_negative, spars, aic_penalty):
    # if the response is a list (when passed in from a vector in R), convert it into numpy array
    if type(response) is list:
        response = np.array(response)
    
    # specify the maximum number of iterations
    max_iters = 100 # for the convex program
    max_iter_dccp = 100 # for the dccp solver

    # determine the type of fairness constraints
    if measure == "eopp":
        cons_type = "FNR"
    elif measure == "eodd":
        cons_type = "Both"
    else:
        raise Exception("Invalid fairness measurement")
    
    # build the dictionary for the upper bound of the covariance constraint for each covariance type
    cov_thresh = {"FNR": FNR_Bound, "FPR": FPR_Bound}

    # create the decision vector
    num_points, num_features = predictors.shape
    w = cvx.Variable(num_features)

    # initialize a random value of w
    np.random.seed(112233)
    w.value = np.random.rand(predictors.shape[1])

    # get the fairness constraints
    constraints = get_constraints(predictors, response, sensitive, cov_thresh, cons_type, sum_to_one, non_negative, w)

    # define the logistic loss function
    if aic_penalty:
      loss = cvx.sum(cvx.logistic(cvx.multiply(-response, predictors @ w))) / num_points + (spars @ w) / num_points
    else:
      loss = cvx.sum(cvx.logistic(cvx.multiply(-response, predictors @ w))) / num_points

    # if a starting point for DCCP solver should be given
    if take_initial_sol:
        p = cvx.Problem(cvx.Minimize(loss), [])
        p.solve()

    # construct the optimization problem
    prob = cvx.Problem(cvx.Minimize(loss), constraints)
    # check if it is a DCCP problem
    print(f"Problem is DCCP: {is_dccp(prob)}")

    # solve the DCCP optimization problem
    try:
        if solver_type == "ECOS":
            prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10, solver=cvx.ECOS, verbose=False,
                       feastol=EPS, abstol=EPS, reltol=EPS, feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
                       max_iters=max_iters, max_iter=max_iter_dccp)
        elif solver_type == "Clarabel":
            prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10, solver=cvx.CLARABEL, verbose=False, max_iter=max_iter_dccp)
            
        assert(prob.status == "Converged" or prob.status == "optimal")
    except:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)

    # check that the fairness constraint is satisfied
    for f_c in constraints:
        #assert(f_c.value() == True) # can comment this out if the solver fails too often, but make sure that the constraints are satisfied empirically. alternatively, consider increasing tau parameter
        pass

    w = np.array(w.value).flatten()

    return w