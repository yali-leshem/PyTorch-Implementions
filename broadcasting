import torch

def expand_as(A,B):
  A = A.clone()
  i = len(A.shape)-1
  j = len(B.shape)-1
  while i > -1 or j > -1: # Checking until passes B and A dimensions from the right
    if i < 0: # In case needs to add dimensions so their number would be equal
      A.unsqueeze_(0)
      i = i+1
    if (A.shape[i] != 1 and A.shape[i] != B.shape[i]) or j < 0: # When a single dimension wasn't one or two dimensions were unequal
      print("Tensor A cannot be broadcasted to tensor B dimensions. Exit function ")
      return None # The tensors CANNOT be broadcasted as requested - function's finishing by returning None
    if A.shape[i] != B.shape[j] and A.shape[i] == 1:
      A = torch.cat([A] * B.shape[j], dim=i)  # concatenating A by the required dimension as long as dimensions are equal or one
    i = i-1
    j = j-1 # Moving to the next dimension from the right
  return A # Expanded A to B dimensions is returned

def able_broadcast(A,B):
  i = len(A.shape)-1
  j = len(B.shape)-1
  dimensions = []
  while i > -1 or j > -1:
      if i < 0:
        dimensions.insert(0,B.shape[j]) # When there are more dimensions in B than in A, add those dimensions to the broadcasted dimensions tensor
      elif j < 0:
        dimensions.insert(0,A.shape[i]) # When there are more dimensions in A than in B, add those dimensions to the broadcasted dimensions tensor
      elif A.shape[i] == B.shape[j]:
        dimensions.insert(0, B.shape[j]) # When both dimensions are equal, insert tensor B dimension (Could have insert to the list tensor A dimension too)
      elif B.shape[j] == 1:
        dimensions.insert(0, A.shape[i]) # If tensor B dimension is a singleton, then add tensor A dimensions as it couldn't be less
      elif A.shape[i] == 1:
        dimensions.insert(0, B.shape[j]) # If tensor A dimension is a singleton, then add tensor B dimensions as it couldn't be less
      else:
          return False, None # In any other case of the mentioned cases above, the tensors aren't capable of being broadcasted together
      i = i-1
      j = j-1 # Move to the next dimension from the right
  return True, dimensions # If All dimensions of broadcasted tensor were inserted into the list "dimensions", return true and the list representing the dims

def broadcastble(A,B):
  can_broadcast, dims = able_broadcast(A,B) # Checking if broadcasting is avalible and if so, and are the dimensions representing the broadcasted tensor
  if can_broadcast == False: # In case A,B tensors cannot be broadcasted
    return "A and B cannot be broadcasted"
  C = torch.ones(dims) # duplicating the needed broadcasted dims
  D = torch.ones(dims)
  return expand_as(A,C), expand_as(B,D) # expanding A

A = torch.tensor([1])
B = torch.tensor([[1,2,3,4,5,6]])
print(expand_as(A,B))
print(able_broadcast(A,B))
print(broadcastble(A,B))
C = A.expand_as(B)
print(C)
D = torch.broadcast_tensors(A,B)
print(D)

A = torch.tensor([1,2])
B = torch.tensor([[1,2],[3,4]])
print(expand_as(A,B))
print(able_broadcast(A,B))
print(broadcastble(A,B))
C = A.expand_as(B)
print(C)
D = torch.broadcast_tensors(A,B)
print(D)

A = torch.tensor([1,2,3,4,5])
B = torch.tensor([1,2,3,4,5,6])
print(expand_as(A,B))
print(able_broadcast(A,B))
print(broadcastble(A,B))
print("Didn't check with original functions as that'd cause a running time error")
