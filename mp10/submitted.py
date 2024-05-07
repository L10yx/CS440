'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    #raise RuntimeError("You need to write this part!")
    M = model.M
    N = model.N
    P = np.zeros((M, N, 4, M, N))
    for r in range(M):
        for c in range(N):
            
            if model.TS[r,c]:
                P[r,c,:,:,:] = 0
                continue

            intend_p = model.D[r,c,0]
            cc_p = model.D[r,c,1]
            c_p = model.D[r,c,2]
            dest_r = r
            dest_c = c-1
            r_c  = c + 1
            l_c = c - 1
            up_r = r - 1
            down_r = r + 1




            if dest_c >= 0:
                if model.W[r,c - 1]:
                    P[r,c,0,r,c] += intend_p
                else:
                    P[r,c,0, dest_r, dest_c] += intend_p
            else:
                P[r,c,0,r,c] += intend_p
                
            
            if down_r < M:
                if model.W[down_r, c]:
                    P[r,c,0,r,c] += cc_p
                else:
                    P[r,c,0,down_r,c] += cc_p
            else:
                P[r,c,0, r, c] += cc_p
                

            if up_r >= 0 :
                if model.W[up_r, c]:
                    P[r,c,0,r,c] += c_p
                else:
                    P[r,c,0,up_r,c] += c_p
            else:
                P[r,c,0,r,c] += c_p
                
              
            dest_r = r - 1
            dest_c = c

            if dest_r >= 0:
                if model.W[dest_r,dest_c]:
                    P[r,c,1,r,c] += intend_p
                else:
                    P[r,c,1,dest_r, dest_c] += intend_p
            else:
                P[r,c,1,r,c] += intend_p
            if l_c >= 0:
                if model.W[r,l_c]:
                    P[r,c,1,r,c] += cc_p
                else:
                    P[r,c,1,r,l_c] += cc_p
            else:
                P[r,c,1,r,c] += cc_p

            if r_c < N:
                if model.W[r,r_c]:
                    P[r,c,1,r,c] += c_p
                else:
                    P[r,c,1,r,r_c] += c_p
            else:
                P[r,c,1,r,c] += c_p

            dest_r = r
            dest_c = c + 1

            if dest_c < N:
                if model.W[dest_r,dest_c]:
                    P[r,c,2,r,c] += intend_p
                else:
                    P[r,c,2,dest_r,dest_c] += intend_p
            else:
                P[r,c,2,r,c] += intend_p
            
            if up_r >= 0:
                if model.W[up_r,c]:
                    P[r,c,2,r,c] += cc_p
                else:
                    P[r,c,2,up_r,c] += cc_p
            else:
                P[r,c,2,r,c] += cc_p

            if down_r < M:
                if model.W[down_r,c]:
                    P[r,c,2,r, c] += c_p
                else:
                    P[r,c,2,down_r,c] += c_p
            else:
                P[r,c,2,r,c] += c_p

            dest_r = r + 1
            dest_c = c

            if dest_r < M:
                if model.W[dest_r,dest_c]:
                    P[r,c,3,r,c] += intend_p
                else:
                    P[r,c,3,dest_r, dest_c] += intend_p
            else:
                P[r,c,3,r,c] += intend_p

            if r_c < N:
                if model.W[r, r_c]:
                    P[r,c,3,r,c] += cc_p
                else:
                    P[r,c,3,r,r_c] += cc_p
            else:
                P[r,c,3,r,c] += cc_p
            
            if l_c >= 0:
                if model.W[r,l_c]:
                    P[r,c,3,r,c] += c_p
                else:
                    P[r,c,3,r,l_c] += c_p
            else:
                P[r,c,3,r,c] += c_p
                
    return P

def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    
    M = model.M
    N = model.N
    reward = model.R
    U_next = np.zeros((M,N))
    for row in range(M):
        for col in range(N):
            plist = np.array([np.sum(P[row,col,state] * U_current) for state in range(4)])
            U_next[row,col] = reward[row,col] + model.gamma * np.max(plist)
    return U_next

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    
    M = model.M
    N = model.N
    P = compute_transition(model)
    U_cur = np.zeros((M,N))
    for i in range(100):
        U_new = compute_utility(model,U_cur,P)
        if False in (np.abs(U_new - U_cur)< epsilon):
            U_cur = U_new
        else:
            break
    return U_cur

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    #raise RuntimeError("You need to write this part!")
    
    M = model.M
    N = model.N
    U = np.zeros((M,N))
    trans = model.FP
    for iter in range(200):
        U_next = np.zeros((M,N))
        for row in range(M):
            for col in range(N):
                transm = np.sum(trans[row,col] * U)
                U_next[row,col] = model.R[row,col] + model.gamma * transm
        U = U_next
    
    return U