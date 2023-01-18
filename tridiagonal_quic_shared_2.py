import time
import jax
import jax.numpy as jnp
import scipy
import numpy as np
# MACHINE_EPS = 0.0078

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

def getl1norm_tds(sd, se):
  """Get l1norm of tridiagonal matrix given diagonal and off-diag statistics."""
  temp = sd
  temp = temp.at[:-1].set(sd[:-1] + jnp.abs(se))
  temp = temp.at[1:].set(sd[1:] + jnp.abs(se))
  return jnp.max(temp)


def ldl2tridiag(lsub, d):
  """Use L and D to compute tridiag=LDL^T.

  Args:
   lsub: L is bidiagonal, where diag(L)=1. So providing subdiagonal of L.
   d: inverse conditional covariance. Used as "D" in LDL^T.

  Returns:
   xd, xe -- diagonal and subdiagonal preconditioner.
  """
  xd = jnp.zeros_like(d)
  xd = xd.at[1:].set(d[1:] + lsub * lsub * d[:-1])
  xd = xd.at[0].set(d[0])
  xe = lsub * d[:-1]
  return xd, xe



def tridiag_kfac(sd, se, eps, min_eps=1e-16):
  """Tridiagonal approximation.

  Given diagonal (sd) and subdiagonal(se), find the inverse
  of PD completion of this tridiag matrix using LDL^T
  Ref: https://arxiv.org/pdf/1503.05671.pdf, Sec 4.3.

  Args:
   sd: Diagonal statistics
   se: Subdiagonal statistcs
   eps: eps added to the diagonal statistic
   min_eps: minimum eps to be added to the diag statistics

  Returns:
   xd, xe -- diagonal and subdiagonal preconditioner
  """
  l1norm = getl1norm_tds(sd, se)
  sd = sd + jnp.maximum(eps * l1norm, min_eps)
  psi = se / sd[1:]
  cond_cov = jnp.zeros_like(sd)
  cond_cov = cond_cov.at[:-1].set(sd[:-1] - se*(se/sd[1:]))
  cond_cov = cond_cov.at[-1].set(sd[-1])
  d = 1 / cond_cov
  mask1 = cond_cov[:-1] <= 0.0
  mask2 = cond_cov <= 0.0
  psi = jnp.where(mask1, 0, psi)
  d = jnp.where(mask2, 1 / sd, d)
  lsub = -psi
  return ldl2tridiag(lsub, d)



def GENP_jax(a, b):
  """Gaussian elimination with no pivoting.

  % input: a is an n x n nonsingular matrix
  %        b is an n x 1 vector
  % return: x is the solution of Ax=b.
  % post-condition: A and b have been modified.
  """
  n = a.shape[1]

  orig_a = a
  mach_eps = jnp.finfo(orig_a.dtype).eps
  if b.shape[1] != n:
    raise ValueError("Invalid argument: " +
                     "incompatible sizes between A & b.", b.shape[-1], n)
  for pivot_row in range(n-1):
    for row in range(pivot_row+1, n):
      den = a[:, pivot_row, pivot_row]
      den = jnp.where(den == 0, mach_eps*orig_a[:, pivot_row, pivot_row], den)
      # den = jnp.where(den == 0, 0.0078*orig_a[:, pivot_row, pivot_row], den)
      a = a.at[:, pivot_row, pivot_row].set(den)
      multiplier = a[:, row, pivot_row]/den
      multiplier = multiplier.reshape(-1, 1)
      a = a.at[:, row, pivot_row:].set(a[:, row, pivot_row:]-
                                       multiplier*a[:, pivot_row, pivot_row:])
      b = b.at[:, row].set(b[:, row] - multiplier*b[:, pivot_row])
  batches = a.shape[0]
  x = jnp.zeros((batches, n), dtype=a.dtype)
  k = n-1
  
  # a = a.at[:, k, k].set(jnp.where(a[:, k, k] == 0, 0.0078*orig_a[:, k, k],
  #                                 a[:, k, k]))
  a = a.at[:, k, k].set(jnp.where(a[:, k, k] == 0, mach_eps*orig_a[:, k, k],
                                  a[:, k, k]))
  orig_a = a
  den = a[:, k, k].reshape(-1, 1)
  temp = b[:, k] / den
  temp = temp.reshape(-1)
  x = x.at[:, k].set(temp)
  k = k-1
  while k >= 0:
    first = a[:, k, k+1:].reshape((batches, 1, -1))
    second = x[:, k+1:].reshape((batches, -1, 1))
    second_term = jnp.matmul(first, second)
    temp = second_term.reshape(-1)
    den = a[:, k, k].reshape(-1)
    x = x.at[:, k].set((b[:, k].reshape(-1) - temp.reshape(-1))/den)
    k = k-1
  return x.reshape((batches, -1, 1))


def get_band_pencil(M,n,b):

  M_new = jnp.zeros((n,b+1,b+1))
#     print(M_new.shape)
  n = M_new.shape[0]
  
  for diag_num in range(b+1):
    diag = M[:,diag_num]
    for offset in range(b+1-diag_num):
      diag_sliced = diag[offset:offset+n]
      # print(diag_sliced.shape)
      # print(M_new[offset,offset+diag_num].shape)

      M_new = M_new.at[:,offset,offset+diag_num].set(diag_sliced)
      M_new = M_new.at[:,offset+diag_num,offset].set(diag_sliced)
  return M_new

def bandedInv(Sd,subDiags,eps,innerIters):
  # given diagonal-Sd and subdiagonals-subDiags
  # find the inverse of pd completion of this banded matrix
  # interms of Ldiag(D)L^T decomposition
  # outputs Lsub and D, where Lsub-subdiagonals of L

  n = Sd.shape[0]
  b = subDiags.shape[1]

  bandvecs = jnp.concatenate((Sd.reshape(-1,1), subDiags), axis=1)
  # indX,indY = ind
  epsMat = jnp.zeros((b, b+1), dtype=Sd.dtype)
  epsMat = epsMat.at[:,0].set(eps)
  bandWindows = jnp.concatenate((bandvecs,epsMat), axis=0)
  sig_full = get_band_pencil(bandWindows,n,b)
  # sig22 = bandWindows[indX[:,1:,1:],indY[:,1:,1:]]
  sig22 = sig_full[:,1:,1:]
  # sig21 = bandWindows[indX[:,1:,0],indY[:,1:,0]]
  sig21 = sig_full[:,1:,0]

  def A_bmm(X):
    return jnp.matmul(sig22, X)

  diagSig22 = jnp.diagonal(sig22, axis1=1, axis2=2)

  def M_bmm(X):
    return jnp.broadcast_to(jnp.expand_dims(1/diagSig22, axis=-1), X.shape)*X

  psi = GENP_jax(M_bmm(sig22), M_bmm(jnp.expand_dims(sig21, axis=-1)))
  # psi = GENP_jax(sig22, jnp.expand_dims(sig21, axis=-1))
  psi = psi.squeeze(-1)
  psiSig21 = jnp.matmul(psi.reshape((n,1,b)), sig21.reshape((n,b,1))).squeeze(-1).squeeze(-1)
  condCov = Sd - psiSig21
  mach_eps = jnp.finfo(Sd.dtype).eps
  condCov = jnp.where(condCov<2*mach_eps*Sd, 2*mach_eps*Sd,condCov)
  # D = jnp.where(condCov<2*mach_eps*Sd, jnp.zeros_like(condCov),1/condCov)

  ################# Edge Removal Start ################
  '''
  condCovFail = (condCov<=0.0078*Sd).reshape((-1,1))
  # condCovFail = (condCov<=1e-7*Sd).reshape((-1,1))
  condCovFail = jnp.broadcast_to(condCovFail, (condCovFail.shape[0], b))
  condCovFail = condCovFail.at[:-1,0].set(jnp.logical_or(condCovFail[:-1,0],
                                                          condCovFail[1:,0]))
  for i in range(1, b):
    condCovFail = condCovFail.at[:-(i+1),i].set(
        jnp.logical_or(condCovFail[:-(i+1), i], condCovFail[1:-(i), i-1]))

  psi = jnp.where(condCovFail, 0.0, psi)
  psiSig21 = jnp.matmul(psi.reshape((n, 1, b)), sig21.reshape((n, b, 1)))
  psiSig21 = psiSig21.squeeze(-1).squeeze(-1)
  condCov = Sd - psiSig21
  '''
  # def cond(arguments):
  #   condCov, psi, psiSig21 = arguments
  #   # return jnp.any((condCov <= 0.0078*Sd))
  #   return jnp.any((condCov <= 1e-6*Sd))

  # def body(arguments):
  #   condCov, psi, psiSig21 = arguments
  #   # condCovFail = (condCov<=0.0078*Sd).reshape((-1,1))
  #   condCovFail = (condCov<=1e-7*Sd).reshape((-1,1))
  #   condCovFail = jnp.broadcast_to(condCovFail, (condCovFail.shape[0], b))
  #   condCovFail = condCovFail.at[:-1,0].set(jnp.logical_or(condCovFail[:-1,0],
  #                                                          condCovFail[1:,0]))
  #   for i in range(1,b):
  #     condCovFail = condCovFail.at[:-(i+1),i].set(
  #         jnp.logical_or(condCovFail[:-(i+1), i], condCovFail[1:-(i), i-1]))

  #   psi = jnp.where(condCovFail, 0.0, psi)
  #   psiSig21 = jnp.matmul(psi.reshape((n, 1, b)), sig21.reshape((n, b, 1)))
  #   psiSig21 = psiSig21.squeeze(-1).squeeze(-1)
  #   condCov = Sd - psiSig21
  #   return (condCov, psi, psiSig21)
  # ret = (condCov, psi, psiSig21)
  # ret = jax.lax.while_loop(cond, body, ret)
  # condCov, psi, psiSig21 = ret
  # ################## Edge Removal End ###############
  
  D = jnp.where(condCov==0,jnp.zeros_like(condCov),1/(condCov))
  return psi.astype(Sd.dtype), D.astype(Sd.dtype)



def bandedInv2(Sd,subDiags,eps):
  # given diagonal-Sd and subdiagonals-subDiags
  # find the inverse of pd completion of this banded matrix
  # interms of Ldiag(D)L^T decomposition
  # outputs Lsub and D, where Lsub-subdiagonals of L

  n = Sd.shape[0]
  b = subDiags.shape[1]

  bandvecs = jnp.concatenate((Sd.reshape(-1,1), subDiags), axis=1)
  # indX,indY = ind
  epsMat = jnp.zeros((b, b+1), dtype=Sd.dtype)
  epsMat = epsMat.at[:,0].set(eps)
  bandWindows = jnp.concatenate((bandvecs,epsMat), axis=0)
  def linsys_2x2(d1,d2,d3,b3,b4):
    mach_eps = jnp.finfo(d1.dtype).eps
    denomhat = (d3-d2*(d2/d1))
    denomhat = jnp.where(denomhat<mach_eps*d3,mach_eps*d3,denomhat)
    y2 = (b4- b3*(d2/d1))/denomhat
    
    y1 = (b3-d2*y2)/d1
    return jnp.concatenate([y1,y2],axis=1)
  d1,d2,d3 = bandWindows[1:n+1,0:1],bandWindows[1:n+1,1:2],bandWindows[2:n+2,0:1]
  # b3,b4 = bandWindows[:n,1:2],bandWindows[:n,2:]
  sig21 = bandWindows[:n,1:]
  b3,b4 = sig21[:,0:1],sig21[:,1:]
  psi = linsys_2x2(d1,d2,d3,b3,b4)
  # psi = GENP_jax(M_bmm(sig22), M_bmm(jnp.expand_dims(sig21, axis=-1)))
  # psi = GENP_jax(sig22, jnp.expand_dims(sig21, axis=-1))
  # psi = psi.squeeze(-1)
  psiSig21 = jnp.matmul(psi.reshape((n,1,b)), sig21.reshape((n,b,1))).squeeze(-1).squeeze(-1)
  condCov = Sd - psiSig21
  mach_eps = jnp.finfo(Sd.dtype).eps
  condCov = jnp.where(condCov<2*mach_eps*Sd, 2*mach_eps*Sd,condCov)
  # D = jnp.where(condCov<2*mach_eps*Sd, jnp.zeros_like(condCov),1/condCov)
  D = jnp.where(condCov==0,jnp.zeros_like(condCov),1/(condCov))
  return psi.astype(Sd.dtype), D.astype(Sd.dtype)


def bandedInv3(Sd,subDiags,eps):
  # given diagonal-Sd and subdiagonals-subDiags
  # find the inverse of pd completion of this banded matrix
  # interms of Ldiag(D)L^T decomposition
  # outputs Lsub and D, where Lsub-subdiagonals of L

  n = Sd.shape[0]
  b = subDiags.shape[1]

  bandvecs = jnp.concatenate((Sd.reshape(-1,1), subDiags), axis=1)
  # indX,indY = ind
  epsMat = jnp.zeros((b, b+1), dtype=Sd.dtype)
  epsMat = epsMat.at[:,0].set(eps)
  bandWindows = jnp.concatenate((bandvecs,epsMat), axis=0)
  def linsys_3x3(bands):
    mach_eps = jnp.finfo(bands.dtype).eps
    
    a = bands[1:n+1,0:1]
    c = bands[1:n+1,1:3]
    b1,b2 = bands[:n,1:2],bands[:n,2:]
    b2hat = b2-b1*c/a
    b2hat = jnp.where(b2hat< mach_eps*b2,mach_eps*b2,b2hat)
    
    b2 = b2hat
    
    d1,d2,d3 = bands[2:n+2,0:1],bands[2:n+2,1:2],bands[3:n+3,0:1]
    
    
    c1,c2 = c[:,0:1],c[:,1:]
    d1hat,d2,d3 =  d1 - c1*c1/a, d2-c1*c2/a, d3- c2*c2/a
    d1hat = jnp.where(d1hat<mach_eps*d1,mach_eps*d1,d1hat)
    d1 = d1hat
    ## solving 2x2 linear system here. with matrix [d1, d2,d2,d3] [x1,x2] = [b3,b4]
    b3,b4 = b2[:,0:1],b2[:,1:]

    def linsys_2x2(d1,d2,d3,b3,b4):
        
      denomhat = (d3-d2*(d2/d1))
      denomhat = jnp.where(denomhat<mach_eps*d3,mach_eps*d3,denomhat)
      y2 = (b4- b3*(d2/d1))/denomhat

      y1 = (b3-d2*y2)/d1
      return y1,y2
    
    y1,y2 = linsys_2x2(d1,d2,d3,b3,b4)
    
    y0 = (b1 - c1*y1-c2*y2)/a
    y = jnp.concatenate([y0,y1,y2],axis=1)
    return y
  sig21 = bandWindows[:n,1:]
  psi = linsys_3x3(bandWindows)
  ## verification script
  # def residual(A,x,b):
  #   return jnp.linalg.norm(A@x-b)
  # bandPencil = get_band_pencil(bandWindows,n,b)
  # jax.debug.print("residual: {x}",x=(jax.vmap(residual)(bandPencil[:,1:,1:],psi,bandPencil[:,1,1:])).mean())
  
  psiSig21 = jnp.matmul(psi.reshape((n,1,b)), sig21.reshape((n,b,1))).squeeze(-1).squeeze(-1)
  condCov = Sd - psiSig21
  mach_eps = jnp.finfo(Sd.dtype).eps
  condCov = jnp.where(condCov<2*mach_eps*Sd, 2*mach_eps*Sd,condCov)
  # D = jnp.where(condCov<2*mach_eps*Sd, jnp.zeros_like(condCov),1/condCov)
  D = jnp.where(condCov==0,jnp.zeros_like(condCov),1/(condCov))
  return psi.astype(Sd.dtype), D.astype(Sd.dtype)


def bandedMult(psi, D, vecv):
  normalize = jnp.ones_like(D)
  b = psi.shape[1]
  normalize = jnp.sqrt(normalize)
  update = vecv*normalize
  for i in range(b):
    update = update.at[:-i-1].set(update[:-i-1] - vecv[i+1:]*psi[:-i-1, i])
  update = update*D
  vecv2 = update
  for i in range(b):
    update = update.at[i+1:].set(update[i+1:] - vecv2[:-i-1]*psi[:-i-1, i])
  update = update*normalize
  return update

def getl1norm(Sd, Se):
  bandSize = Se.shape[1]
  n = Sd.shape[0]
  temp = Sd
  for b in range(bandSize):
    temp = temp.at[:-(b+1)].set(Sd[:-(b+1)] + jnp.abs(Se[:, b][:-(b+1)]))
    temp = temp.at[(b+1):].set(Sd[(b+1):] + jnp.abs(Se[:, b][:-(b+1)]))
  return jnp.max(temp)

def bandedUpdates(Sd, subDiags, eps, innerIters, mu):
  l1norm = getl1norm(Sd, subDiags)
  # psi, D = bandedInv(Sd+(eps*l1norm), subDiags, eps, innerIters)
  # print(subDiags.shape[1])
  if subDiags.shape[1]>3:
    psi, D = bandedInv(Sd+(eps*l1norm), subDiags, eps, innerIters)
  elif subDiags.shape[1]==2:
    psi, D = bandedInv2(Sd+(eps*l1norm), subDiags, eps)
  # psi, D = bandedInv(Sd, subDiags, eps, innerIters)
  elif subDiags.shape[1]==3:
    psi, D = bandedInv3(Sd+(eps*l1norm), subDiags, eps)
  return bandedMult(psi, D, mu)

def createInd(n,b):
  b1 = b+1
  offsetX = jnp.broadcast_to(jnp.expand_dims(jnp.arange(b1), axis=-1), (b1,b1))
  offsetX = jnp.triu(offsetX)+jnp.transpose(jnp.triu(offsetX,1), (1,0))

  offsetY = jnp.array(scipy.linalg.toeplitz(np.arange(b1)))

  indX = jnp.broadcast_to(jnp.expand_dims(jnp.expand_dims(jnp.arange(n), axis=-1), axis=-1), (n,b1,b1))
  indY = jnp.broadcast_to(jnp.expand_dims(jnp.expand_dims(jnp.zeros(n, dtype=jnp.int32), axis=-1), axis=-1), (n,b1,b1))

  indX = indX+jnp.expand_dims(offsetX, 0)
  indY = indY+jnp.expand_dims(offsetY, 0)

  return jnp.array([indX, indY])
