def alias_preproc(dist):
    # initialize output
    n = dist.numel()
    U = n * dist
    K = torch.full_like(dist, dtype=torch.long,fill_value=-1) # if val hasn't been initilized yet, mark it by -1
    for i in range(dist.numel()-1): # iterating n-1 times as requested
        if torch.all(torch.logical_or(U == 1.0, K >= 0)): # stopping iterations when reached full distribution (sum up to 1)
            break
        inf_tensor = torch.full_like(U, fill_value=torch.inf)
        i = torch.argmax(U).item()
        j = torch.argmin(torch.where(K < 0, U, inf_tensor)).item() # pick underfull entry j and overfull entry i
        K[j] = i
        U[i] = U[i] - (1-U[j]) # pass probability from U[i] to U[j]
    
    return U, K

def my_sampler(size, dist, requires_grad=False):
    dist = torch.tensor(dist) #instead of list, we're going to use the distribution as a tensor
    assert (len(dist.shape) == 1 and torch.all(dist >= 0) and
            dist.sum() == 1.0), 'the given distribution is not valid.' 
    # checking if until now all probablities are greater or equal to zero, and sum up to 1 as needed
    
    # preprocessing
    U, K = alias_preproc(dist)

    # draw 'x', 'i', and 'y' (in terms of the alias method)
    n = dist.numel()
    x = torch.rand(size)
    i = torch.floor(n*x).long()
    y = n*x - i

    # use the alias tables to draw the actual output
    output = torch.where(y < U[i], i, K[i]).float().clone().detach().requires_grad_(requires_grad)
    
    return output

