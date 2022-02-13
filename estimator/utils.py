import numpy as np
import scipy.stats as stats

def fix_load_model(model, dataset, size, i):
    
    
    path = "../results/fix_sample_%d/"%size + dataset + '/' + model + '/'
    
    data = np.load(path + "%d.npz"%i)
    
    population = data['origin']
    sample = data['sample']
    
    return sample, population


def get_pr(xu, n):
    
    pr = np.zeros(n)

    for r in xu:
        pr[r] = pr[r] + 1

    pr = pr/pr.sum()
    
    return pr

def A_nN(N, n):
    
    '''
    return the A matrix with shape N x n
    
    A_Rr = P(r|R)
    
    '''
    
    x = np.arange(n)
    x = np.tile(x, (N, 1))# N, n


    R = np.arange(1, N+1)
    R = R.reshape(-1, 1)
    theta = (R-1)/(N-1)
    theta = np.tile(theta, (1, n))# N, n
    
    A = stats.binom.pmf(x, n - 1, theta)

    
    return A.T


def A_Nn(N, n):
    
    '''
    return the A matrix with shape N x n
    
    A_Rr = P(r|R)
    
    '''
    
    x = np.arange(n)
    x = np.tile(x, (N, 1))# N, n


    R = np.arange(1, N+1)
    R = R.reshape(-1, 1)
    theta = (R-1)/(N-1)
    theta = np.tile(theta, (1, n))# N, n
    
    A = stats.binom.pmf(x, n - 1, theta)

    
    return A



def NDCG_k(N, k):
    
    R = np.arange(1, N+1)
    bot = np.log2(R+1)
    mask = (R <= k)*1
    
    MR = mask/bot
    
    return MR
    
def AP_k(N, k):
    
    R = np.arange(1, N+1)
    mask = (R <= k)*1
    MR = mask/R
    
    return MR
    
    
def Recall_k(N, k):
    
    R = np.arange(1, N+1)
    mask = (R <= k)*1
    
    MR = mask
    
    return MR


#-------- for the True R_u input
def NDCG(R, k):
    
    bot = np.log2(R+1)
    mask = (R <= k)*1
    
    MR = mask/bot
    
    return MR.mean()
    
def AP(R, k):
    
    mask = (R <= k)*1
    MR = mask/R
    
    return MR.mean()
    
    
def Recall(R, k):
    
    mask = (R <= k)*1
    
    MR = mask
    
    return MR.mean()