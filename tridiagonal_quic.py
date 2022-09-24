import time
import jax
import jax.numpy as jnp

'''
def tridiagGeneral(el,d,eu):
  #Create a general tridiagonal matrix out of diagonal d and side diagonal e.
  n = d.shape[0]

  dindices = jnp.stack([jnp.arange(n),jnp.arange(n)],axis=0)
  uIndices = jnp.stack([jnp.arange(n-1),jnp.arange(n-1)],axis=0)
  uIndices[1]+=1 # 0 -> x axis # 1 -> y axis 
  lIndices = jnp.stack([jnp.arange(n-1),jnp.arange(n-1)],axis=0)
  lIndices[0]+=1

  A = jnp.zeros((n,n))
  A = A.at[dindices[0],dindices[1]].set(d)
  A = A.at[uIndices[0],uIndices[1]].set(eu)
  A = A.at[lIndices[0],lIndices[1]].set(el)
  return A

def tridiag(d,e):
  #Create a symm tridiagonal matrix out of diagonal d and side diagonal e.
  n = d.shape[0]

  dindices = jnp.stack([jnp.arange(n),jnp.arange(n)],axis=0)
  uIndices = jnp.stack([jnp.arange(n-1),jnp.arange(n-1)],axis=0)

  uIndices[1]+=1 # 0 -> x axis # 1 -> y axis 
  lIndices = jnp.stack([jnp.arange(n-1),jnp.arange(n-1)],axis=0)

  lIndices[0]+=1

  A = jnp.zeros((n,n))
  A = A.at[dindices[0],dindices[1]].set(d)
  A = A.at[uIndices[0],uIndices[1]].set(e)
  A = A.at[lIndices[0],lIndices[1]].set(e)

  return A
  

def mat2Tridiag(M):
  n = M.shape[0]
  dindices = jnp.stack([jnp.arange(n),jnp.arange(n)],axis=0)
  uIndices = jnp.stack([jnp.arange(n-1),jnp.arange(n-1)],axis=0)

  uIndices[1]+=1 # 0 -> x axis # 1 -> y axis 

  lIndices = jnp.stack([jnp.arange(n-1),jnp.arange(n-1)],axis=0)
  lIndices[0]+=1
  Md = M[dindices[0],dindices[1]]
  Me = M[uIndices[0],uIndices[1]]

  return Md,Me

def invTridiag(b,delt,d):
    
  #W_i,i = d_i+1...d_n*1/delt_i...delt_n  i= 1 to n-1; W_n,n = 1/delt_n
  #W_i,i+1 = b_i+1 * d_i+2 ...d_n/delt_i...delt_n i = 1 to n-2; W_n-1,n = b_n/delt_n-1 delt_n

  n = d.shape[0]

  invd = jnp.concatenate([jnp.cumprod((d[1:]/delt[:-1])[::-1])[::-1],jnp.array([1],dtype=jnp.float32)])/delt[n-1]
  inve = -b*jnp.concatenate([jnp.cumprod((d[2:]/delt[:-2])[::-1])[::-1],jnp.array([1],dtype=jnp.float32)])/(delt[n-2]*delt[n-1])

  return invd,inve


def gradientSW(Sd,Se,Xd,Xe):
  #Evaluate S-W
  #compute inverse of X
  n = Sd.shape[0]
  delt = jnp.empty(shape=(n,),dtype=jnp.float32)
  d = jnp.empty(shape=(n,),dtype=jnp.float32)
  delt,d,_ = luul(n,0,Xd,Xe,delt,d,False)
  Wd,We = invTridiag(Xe,delt,d)
  return Sd - Wd, Se-We


def LogDetDiv(Sd,Se,Xd,Xe):
  #compute -log(det X)-log(det S) + Tr(XS)-d
  n = Xd.shape[0]
  d = jnp.abs(luul(n,1,Xd,Xe,jnp.zeros((1,)).astype(jnp.float32),\
                                       jnp.zeros((n,)).astype(jnp.float32),True)[1])

  logdetX = jnp.sum(jnp.log(d))
  d = jnp.abs(luul(n,1,Sd,Se,jnp.zeros((1,)).astype(jnp.float32),\

                                       jnp.zeros((n,)).astype(jnp.float32),True)[1])

  logdetS = jnp.sum(jnp.log(d))
  trSX = jnp.dot(Sd,Xd)+ 2*jnp.dot(Se,Xe)

  # print('logdetS',logdetS)
  # print('trSX',trSX)
  return -logdetX-logdetS+trSX-n


def LogDetDiff(Sd,Se,X1d,X1e,X2d,X2e):
#     -log(det X2) +log (det X1) +Tr(S(X2-X1))
  n = X1d.shape[0]

  logdetX1 = jnp.sum(jnp.log(jnp.abs(luul(n,1,X1d,X1e,jnp.zeros((1,)).astype(jnp.float32),\
                                       jnp.zeros((n,)).astype(jnp.float32),True)[1])))
  logdetX2 = jnp.sum(jnp.log(jnp.abs(luul(n,1,X2d,X2e,jnp.zeros((1,)).astype(jnp.float32),\
                                          jnp.zeros((n,)).astype(jnp.float32),True)[1])))
  trSX2X1 = Sd@(X2d-X1d)+2*Se@(X2e-X1e)

  return -logdetX2+logdetX1 + trSX2X1


def luul(n, onlyd, a, b, delt, d, ttaken):

  assert a.shape[0]==n and len(a.shape)==1
  assert b.shape[0] == n-1 and len(b.shape)==1
  assert d.shape[0] == n and len(d.shape)==1
  if not onlyd:
    assert delt.shape[0] == n and len(delt.shape)==1

  if onlyd!=1:
    delt = delt.at[0].set(a[0])
    prevDelt = a[0]
    currDelt = prevDelt
    for i in range(1,n):
      bi_1 = b[i-1]
      currDelt = a[i] - (bi_1*bi_1)/prevDelt
      prevDelt = currDelt
      delt = delt.at[i].set(currDelt)
  # print("d rn is:", d)
  # print("a rn is:", a)
  d = d.at[n-1].set(a[n-1])
  prevd = a[n-1]
  currd = prevd
  for i in range(1, n):
    bn_i_1 = b[n-i-1]
    currd = a[n-i-1] - (bn_i_1*bn_i_1)/prevd
    prevd = currd
    d = d.at[n-i-1].set(currd)
    # print("currd:", currd)
    # print("d[n-i-1]:", d[n-i-1])

  return delt,d,None

def isPosDef(Xd,Xe):
  n = Xd.shape[0]
  d = luul(n,1,Xd,Xe,jnp.zeros((1,)).astype(jnp.float32),\
                                           jnp.zeros((n,)).astype(jnp.float32),True)[1]
  return jnp.all(d>0)

def line_search_body_fn(params):
  alpha, Yd, Ye, Xd, Xe, Sd, Se, gd, ge, lineIter, c = params
  alpha*=0.5
  Yd = Xd - alpha*gd
  Ye = Xe - alpha*ge
  lineIter+= 1
  return alpha, Yd, Ye, Xd, Xe, Sd, Se, gd, ge, lineIter, c

def line_search_cond_fn(params):
  #conditions: a) X should be posdef b) f(X-alpha*g)< f(X)-alpha.sigma.||g||^2
  alpha, Yd, Ye, Xd, Xe, Sd, Se, gd, ge, lineIter, c = params
  conda = isPosDef(Yd,Ye)
  condb = (LogDetDiff(Sd,Se,Xd,Xe,Yd,Ye,) < c*alpha)
  return jnp.logical_or(~conda, ~condb) #For now not using lineiter condition

@jax.jit
def GradDescent(Sd,Se,T = 10,sigma = 0.1):
  n = Sd.shape[0]
  Xd0 = 1/(Sd)

  Xe0 = jnp.zeros((n-1,),dtype=jnp.float32)
  print(isPosDef(Xd0,Xe0))

  Xd = Xd0
  Xe = Xe0
  prevalpha =1
  for idx in T:
    gd,ge = gradientSW(Sd,Se,Xd,Xe)
    alpha = prevalpha
    Yd = Xd - alpha*gd
    Ye = Xe - alpha*ge
    c = -sigma*(jnp.linalg.norm(gd)**2+2*jnp.linalg.norm(ge)**2)
    lineIter =0

    alpha, Yd, Ye, Xd, Xe, Sd, Se, gd, ge, lineIter, c = \
    jax.lax.while_loop(line_search_cond_fn, line_search_body_fn, \
                       (alpha, Yd, Ye, Xd, Xe, Sd, Se, gd, ge, lineIter, c))

    prevalpha = alpha
    Xd = Yd
    Xe = Ye
    # print("Xd now is:", Xd)
    # print("Xe now is:", Xe)
    # print("iteration num: ",idx, "logdet divergence",LogDetDiv(Sd,Se,Xd,Xe),"; alpha chosen:", alpha)
    # assert 1==2
    # print("")

  return Xd,Xe
'''

def ldl2tridiag(Lsub,D):
  # n = D.shape[0]
  Xd = jnp.zeros_like(D)
  Xd = Xd.at[1:].set(D[1:]+Lsub*Lsub*D[:-1])
  Xd = Xd.at[0].set(D[0])
  Xe = Lsub*D[:-1]
  return Xd,Xe

# @jax.jit  
def tridiagKFAC(Sd,Se, eps):
  # given diagonal-Sd and subdiagonal-Se
  # find the inverse of pd completion of this tridiagonal matrix
  # interms of Ldiag(D)L^T decomposition
  # outputs Lsub and D, where Lsub-subdiagonal of L
  
  # n = Sd.shape[0]
  Sd = Sd+eps
  psi = Se/Sd[1:]
  condCov = jnp.zeros_like(Sd)
  condCov = condCov.at[:-1].set(Sd[:-1]-Se*(Se/Sd[1:]))
  condCov = condCov.at[-1].set(Sd[-1])
  D = 1/(condCov)
  mask1 = condCov[:-1]<1e-10
  mask2 = condCov < 1e-10
  psi = jnp.where(mask1, 0, psi)
  d = jnp.where(mask2, 1/Sd, D)
  Lsub = -psi
  return ldl2tridiag(Lsub,D)



def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
  ## replace this with https://gist.github.com/num3ric/1357315
  '''Solves a batch of PD matrix linear systems using the preconditioned
  CG algorithm.
  This function solves a batch of matrix linear systems of the form
      A_i X_i = B_i,  i=1,...,K,
  where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
  and X_i is the n x m matrix representing the solution for the ith system.
  Args:
    A_bmm: A callable that performs a batch matrix multiply of A
    and a K x n x m matrix.
    B: A K x n x m matrix representing the right hand sides.
    M_bmm: (optional) A callable that performs a batch matrix multiply
    of the preconditioning matrices M
    and a K x n x m matrix. (default=identity matrix)
    X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
    rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
    atol: (optional) Absolute tolerance for norm of residual. (default=0)
    maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
    verbose: (optional) Whether or not to print status messages. (default=False)
  '''

  K, n, m = B.shape

  if M_bmm is None:
    M_bmm = lambda x: x
  if X0 is None:
    X0 = M_bmm(B)
  if maxiter is None:
    maxiter = 5 * n

  assert B.shape == (K, n, m)
  assert X0.shape == (K, n, m)
  assert rtol > 0 or atol > 0
  assert isinstance(maxiter, int)

  X_k = X0
  R_k = B - A_bmm(X_k)
  Z_k = M_bmm(R_k)

  # P_k = torch.zeros_like(Z_k)
  P_k = jnp.zeros_like(Z_k)

  P_k1 = P_k
  R_k1 = R_k
  R_k2 = R_k
  X_k1 = X0
  Z_k1 = Z_k
  Z_k2 = Z_k
  
  B_norm = jnp.linalg.norm(B, axis=1)
  stopping_matrix = jnp.maximum(rtol*B_norm, atol*jnp.ones_like(B_norm))
  
  for k in range(1, maxiter + 1):
    Z_k = M_bmm(R_k)
    if k == 1:
      P_k = Z_k
      R_k1 = R_k
      X_k1 = X_k
      Z_k1 = Z_k
    else:
      R_k2 = R_k1
      Z_k2 = Z_k1
      P_k1 = P_k
      R_k1 = R_k
      Z_k1 = Z_k
      X_k1 = X_k
      denominator = jnp.sum(R_k2 * Z_k2, axis=1)
      denominator = jnp.where(denominator==0, 1e-8, denominator)
      # denominator = denominator.at[denominator==0].set(1e-8)
      beta = jnp.sum(R_k1 * Z_k1, axis=1) / denominator
      P_k = Z_k1 + jnp.expand_dims(beta, axis=1) * P_k1
    AP_k = A_bmm(P_k)
    denominator = jnp.sum(P_k * AP_k, axis=1)
    denominator = jnp.where(denominator==0, 1e-8, denominator)
    # denominator = denominator.at[denominator==0].set(1e-8)
    alpha = jnp.sum(R_k1 * Z_k1, axis=1) / denominator
    X_k = X_k1 + jnp.expand_dims(alpha, axis=1) * P_k
    R_k = R_k1 - jnp.expand_dims(alpha, axis=1) * AP_k

  if verbose:
      print("%03s | %010s %06s" % ("it", "dist", "it/s"))

  optimal = False
  start = time.perf_counter()
  for k in range(1, maxiter + 1):
    start_iter = time.perf_counter()
    Z_k = M_bmm(R_k)

    if k == 1:
      P_k = Z_k
      R_k1 = R_k
      X_k1 = X_k
      Z_k1 = Z_k
    else:
      R_k2 = R_k1
      Z_k2 = Z_k1
      P_k1 = P_k
      R_k1 = R_k
      Z_k1 = Z_k
      X_k1 = X_k
      denominator = (R_k2 * Z_k2).sum(1)
      denominator = jnp.where(denominator==0, 1e-8, denominator)
      beta = (R_k1 * Z_k1).sum(1) / denominator
      P_k = Z_k1 + jnp.expand_dims(beta, axis=1) * P_k1
    AP_k = A_bmm(P_k)
    denominator = (P_k * AP_k).sum(1)
    denominator = jnp.where(denominator==0, 1e-8, denominator)
    # denominator = denominator.at[denominator==0].set(1e-8)
    alpha = (R_k1 * Z_k1).sum(1) / denominator
    X_k = X_k1 + jnp.expand_dims(alpha, axis=1) * P_k
    R_k = R_k1 - jnp.expand_dims(alpha, axis=1) * AP_k
    end_iter = time.perf_counter()

  info = {
    "niter": k,
    "optimal": optimal
  }
  return X_k, info



def bandedInv(Sd,subDiags,ind,eps,innerIters):
  # given diagonal-Sd and subdiagonals-subDiags
  # find the inverse of pd completion of this banded matrix
  # interms of Ldiag(D)L^T decomposition
  # outputs Lsub and D, where Lsub-subdiagonals of L

  n = Sd.shape[0]
  b = subDiags.shape[1]

  bandvecs = jnp.concatenate((Sd.reshape(-1,1), subDiags), axis=1)
  indX,indY = ind
  epsMat = jnp.zeros((b, b+1))
  epsMat = epsMat.at[:,0].set(eps)
  bandWindows = jnp.concatenate((bandvecs,epsMat), axis=0)
  sig22 = bandWindows[indX[:,1:,1:],indY[:,1:,1:]]
  sig21 = bandWindows[indX[:,1:,0],indY[:,1:,0]]

  def A_bmm(X):
    return jnp.matmul(sig22, X)

  diagSig22 = jnp.diagonal(sig22, axis1=1, axis2=2)

  def M_bmm(X):
    return jnp.broadcast_to(jnp.expand_dims(1/diagSig22, axis=-1), X.shape)*X

  psi,_ = cg_batch(A_bmm, jnp.expand_dims(sig21, axis=-1), M_bmm,
                    maxiter=innerIters if innerIters!=-1 else 5*b, verbose=False)
  psi = psi.squeeze(-1)
  psiSig21 = jnp.matmul(psi.reshape((n,1,b)), sig21.reshape((n,b,1))).squeeze(-1).squeeze(-1)
  condCov = Sd - psiSig21
  faultyIdces = faultyInit[condCov==0.0]
  idcesX = jnp.broadcast_to(jnp.expand_dims(faultyIdces, axis=-1), (faultyIdces.shape[0],b+1))
  idcesY = idcesX-jnp.broadcast_to(jnp.expand_dims(jnp.arange(b+1), axis=0), (idcesX.shape[0],b+1))
  idcesY = jnp.where(idcesY<0.0, 0, idcesY)
  mask = maskInit
  for i in range(b-1,-1,-1):
    if i==(b-1):
      mask = mask.at[idcesY, i].set(True)
    else:
      mask = mask.at[idcesY[:,:-(b-i-1)], i].set(True)
  psi = jnp.where(mask, 0, psi)
  mask = mask.at[:,:].set(False)
  psiSig21 = jnp.matmul(psi.reshape((n,1,b)), sig21.reshape((n,b,1)))
  condCov = Sd - psiSig21.squeeze(-1).squeeze(-1)
  D = 1/(condCov)
  return psi, D


def createInd(n,b):
  b1 = b+1
  offsetX = jnp.broadcast_to(jnp.expand_dims(jnp.arange(b1), axis=-1), (b1,b1))
  print(offsetX)
  print(jnp.transpose(jnp.triu(offsetX,1), (1,0)))
  offsetX = jnp.triu(offsetX)+jnp.transpose(jnp.triu(offsetX,1), (1,0))
  print(offsetX)

  offsetY = jnp.array(scipy.linalg.toeplitz(np.arange(b1)))

  indX = jnp.broadcast_to(jnp.expand_dims(jnp.expand_dims(jnp.arange(n), axis=-1), axis=-1), (n,b1,b1))
  indY = jnp.broadcast_to(jnp.expand_dims(jnp.expand_dims(jnp.zeros(n, dtype=jnp.int32), axis=-1), axis=-1), (n,b1,b1))

  indX = indX+jnp.expand_dims(offsetX, 0)
  indY = indY+jnp.expand_dims(offsetY, 0)

  return jnp.array([indX, indY])
