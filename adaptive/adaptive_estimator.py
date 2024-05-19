import numpy as np
import scipy.stats as stats
import torch

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

# precomputing, for repeat use
A100 = A_nN(N, 100)
A200 = A_nN(N, 200)
A400 = A_nN(N, 400)
A800 = A_nN(N, 800)
A1600 = A_nN(N, 1600)
A3200 = A_nN(N, 3200)

A100 = torch.tensor(A100).to(device)
A200 = torch.tensor(A200).to(device)
A400 = torch.tensor(A400).to(device)
A800 = torch.tensor(A800).to(device)
A1600 = torch.tensor(A1600).to(device)
A3200 = torch.tensor(A3200).to(device)

def update_user2(nu, ru, pi):

    #P_rRu = generate_PrR(N, nu, ru)
    P_rRu = eval('A'+str(nu))[ru].clone().detach()
    #P_rRu = torch.tensor(P_rRu).to(device)

    temp = P_rRu * pi
    temp = temp/temp.sum()

    return temp


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

data = np.load("adaptive.npz")
sample_rank = data['rank']
sample_size = data['size']

pi = torch.ones(N)/N
pi = pi.to(device)
epochs = 50

for epoch in (range(epochs)):

    new_pi = torch.zeros_like(pi)

    for i in (range(len(ru))):
        new_pi += update_user2(sample_size, sample_rank, pi)

    pi = new_pi/len(ru)

# pi is the estimated R (which is the rank in global setting)
np.savez('pi.npz', R = pi.cpu().numpy())