# SDE utilities and guiding

import jax
import jax.numpy as jnp

# for 2d + 3d cases with factorizable matrices
# multiply on the factorized matrix, e.g. covariance matrix
dot = lambda A,v: jnp.einsum('ij,jd->id',A,v.reshape((A.shape[0],-1))).flatten()
# multiple on inverse factorized matrix, e.g. inverse covariance matrix
solve = lambda A,v: jnp.linalg.solve(A,v.reshape((A.shape[0],-1))).flatten()

# time increments
def dts(T=1.,n_steps=100):
    return jnp.array([T/n_steps]*n_steps)

# Euler-Maruyama SDE integration
def forward(x,dts,dWs,b,sigma,params):
    def SDE(carry, val):
        t,X = carry
        dt,dW = val
        
        # SDE
        Xtp1 = X + b(t,X,params)*dt + dot(sigma(x,params),dW)
        tp1 = t + dt
        
        return((tp1,Xtp1),(t,X))    

    # sample
    (T,X), (ts,Xs) = jax.lax.scan(SDE,(0.,x),(dts,dWs))
    Xs = jnp.vstack((Xs,X))
    return Xs

# forward guided sampling, assumes already backward filtered (H,F parameters)
def forward_guided(x,H_T,F_T,tildea,dts,dWs,b,sigma,params):
    tildebeta = lambda t,params: 0.
    tildeb = lambda t,x,params: tildebeta(t,params) #+jnp.dot(tildeB,x) #tildeB is zero for now

    T = jnp.sum(dts)
    Phi_inv = lambda t: jnp.eye(H_T.shape[0])+H_T@tildea*(T-t)
    #Phi = lambda t: jnp.linalg.inv(Phi_inv(t))
    #Phi_0 = Phi(0.); Phi_T = Phi(T)#jnp.eye(H[1].shape[0])
    #Phi = lambda t: (T-t)/t*Phi_0+t/T*Phi_T
    Ht = lambda t: solve(Phi_inv(t),H_T).reshape(H_T.shape) 
    Ft = lambda t: solve(Phi_inv(t),F_T).reshape(F_T.shape) 
    #Ht = lambda t: dot(Phi(t),H[1]).reshape(H[1].shape) 
    #Ft = lambda t: dot(Phi(t),F[1]).reshape(F[1].shape) 
    #Ht = lambda t: #(t*Phi_T+(T-t)*Phi_0)@H[1]
    #Ft = lambda t: #dot(t*Phi_T+(T-t)*Phi_0,F[1])
    #Ht = lambda t: (T-t)/T*H[0]-t/T*H[1]
    #Ft = lambda t: (T-t)/T*F[0]-t/T*F[1]

    def bridge_SFvdM(carry, val):
        t, X, logpsi = carry
        #dt, dW, H, F = val
        dt, dW = val
        H = Ht(t); F = Ft(t)
        tilderx =  F-dot(H,X)
        _sigma = sigma(x,params)
        _a = jnp.einsum('ij,kj->ik',_sigma,_sigma)
        n = _a.shape[0]
        
        # SDE
        Xtp1 = X + b(t,X, params)*dt + dot(_a,tilderx)*dt + dot(_sigma,dW)
        tp1 = t + dt
        
        # logpsi
        amtildea = _a-tildea
        logpsicur = logpsi+(
                jnp.dot(b(t,X,params)-tildeb(t,X,params),tilderx)
                -.5*jnp.einsum('ij,ji->',amtildea,H)
                +.5*jnp.einsum('ij,jd,id->',
                           amtildea,tilderx.reshape((n,-1)),tilderx.reshape((n,-1)))
                    )*dt
        return((tp1,Xtp1,logpsicur),(t,X,logpsi))    

    # sample
    (T,X,logpsi), (ts,Xs,logpsis) = jax.lax.scan(bridge_SFvdM,(0.,x,0.),(dts,dWs))#,H,F))
    Xscirc = jnp.vstack((Xs, X))
    return Xscirc,logpsi