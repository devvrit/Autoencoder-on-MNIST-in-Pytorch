import time
import jax
import jax.numpy as jnp


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
    inve = -b*jnp.concatenate(\
            [jnp.cumprod((d[2:]/delt[:-2])[::-1])[::-1],jnp.array([1],dtype=jnp.float32)])/(delt[n-2]*delt[n-1])

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
