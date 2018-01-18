"""
Created on Mon Jan 18 2018 

by Brochini L. 


Implementation of procedure presented in "SYSTEMS OF INTERACTING RANDOM MARKOV CHAINS:
CONCENTRATION INEQUALITIES AND LASSO"
First working version on 18 jan 2018

________________________________________________________


1o passo- Sorteio uniformemente x 
Se x cai em lambda(vazio,0) decido se i dispara 
com probabilidade pi(vazio,zero)(1)

Lembrando que lambda_i(vazio,0)=(1-sum_j sum_s lambda_i(j,s)) e que p_i(vazio,0)=delta_i/(1-lambda_i(vazio,0)(1))

2o passo- Caso x caia em setor qualquer lambda_i(j,s), entao escolho se i 
dispara com probabilidade p_i(j,s)(1|x).
Lembrando que  p_i(j,s)(1|x)=(x_s(j) se j E Vi+) || (1-x_s(j) se j E Vi-)

Assumindo decaimento geometrico de taxa mu

"""

import numpy as np
import pickle
from matplotlib import pyplot as plt
import pylab

N=5
nsteps=5
mu=0.5
factnu=20

#%%
#Wmatrix=np.ones([N,N]) ## starting from all to all grid

def Build_nu(N,fact):
    nu=np.random.rand(N)/fact
    return nu
    
def BuildWmat(N,Nerase):
    Wmatrix=np.random.rand(N,N)
    Wmatrix=Wmatrix*2-1
    l1=np.random.randint(0,N,Nerase)
    l2=np.random.randint(0,N,Nerase)
    for n in range(N):#erase diagonal
        Wmatrix[n,n]=0
    for x,y in zip(l1,l2):
        Wmatrix[x,y]=0
    return Wmatrix    

def gfunc(s):
    return mu**s
    

#%%
def CheckprobAndReturnWG_i(Wmat,nu,i):
    
    #Check if values are compatible to a probability measure:
    myinf=nu[i]
    mysup=nu[i]
    #print("nu[i]=",nu[i])
    
    wgmat=np.zeros([N,nsteps-1])#porque comeca de 1
    for j in range(N): # including i itself because wi->i is zero by construction
        for s in range(nsteps-1):
            wg=Wmat[i,j]*gfunc(s+1)  #porque range comeca de 0      
            wgmat[j,s]=wg
            if wg>0:
                mysup=mysup+wg
            elif wg<0:
                myinf=myinf+wg
            
    #print(mysup,myinf)  # Tem que estar entre 1 e zero senao nao da uma medida de probabilidade              
    
    return wgmat,mysup,myinf


#%%

def FindValidWmatNu():    
    while True: 

        Wmat=BuildWmat(N,15)
        nu=Build_nu(N,factnu)
        wgmat=[]
        OK,wgmat=CheckprobAndReturnWG_all(Wmat,nu)
        if not OK:
            print("Doesnt lead to a prob measure. Keep trying.")
        else:
            print("Worked. Stick to this one.\n\n")
            break
    
    return wgmat,Wmat,nu
    
#%%

def CheckprobAndReturnWG_all(Wmat,nu):
    wgmat=[]
    OK=True
    for i in range(N):        
        wgaux,sup,inf=CheckprobAndReturnWG_i(Wmat,nu,i)
        wgmat.append(wgaux)
        if sup>1 or inf<=0:
            print("NOT PROB MEASURE. REDO.")
            OK=False
    
    return OK, wgmat           

#pickle.dump({"Wmat":Wmat,"nu":nu},open("W_nu_valid.pkl","wb"))

D=pickle.load(open("W_nu_valid.pkl","rb"))
Wmat=D["Wmat"]
nu=D["nu"]

OK,wgmat=CheckprobAndReturnWG_all(Wmat,nu)


#%%
delta=[]
lambda_grid=[]
p_Empty0_1=[]
js_inds=[]

for i in range(N):    
#for i in [0]:
    wgmat_neg_sum=[it for it in Wmat]    
    
    delta.append(nu[i]+wgmat[i].sum())
        
    lambda_i_js=[]
    js_inds_i=[]
    jlist=[j for j in range(N) if j is not i]
    for j in jlist: 
        wabs=np.abs(Wmat[i,j]);
        if wabs!=0: 
            for s in range(1,nsteps+1):
                js_inds_i.append((j,s))
                lambda_i_js.append(wabs*gfunc(s)) #CUIDADO PARA NAO COLOCAR VALORES NULOS NESSA LISTA
                
                
    lambda_i_Empty0=1-np.sum(lambda_i_js)    
    lambda_i_grid=[0,lambda_i_Empty0]
    for it in lambda_i_js:
        aux=lambda_i_grid[-1]
        lambda_i_grid.append(aux+it)
    
    p_i_Empty0_1=delta[i]/(lambda_i_Empty0)
    lambda_grid.append(lambda_i_grid)
    p_Empty0_1.append(p_i_Empty0_1)    
    js_inds.append(js_inds_i)

#%%

def RecursionUntilEmptyZero(i,s):
    r=np.random.rand()
    partition=[ind-1 for ind in range(len(lambda_grid[i])) if r<lambda_grid[i][ind] and r>=lambda_grid[i][ind-1]][0]
    if partition == 0:
        r_spk=np.random.rand()
        if r_spk<=p_Empty0_1[i]:
            x[i,-1-s]=1 # last element, more recent in time (time zero reference)
            return Path_j_s
        else:
            x[i,-1-s]=0
            return Path_j_s
    else:
        (j,snext)=js_inds[i][partition-1]
        s=s+snext        
        print("\n\nenter recursion\n r, j, s, partition,Wmat=",j,s)
        Path_j_s.append((j,s))
        RecursionUntilEmptyZero(j,s) 

#%%
def AssignValsToPath(Path_j_s,ref_neuron):
    Path_j_s.reverse()
    Path_j_s.append((ref_neuron,0))
    for (i_prev,s_prev),(i_now,s_now) in zip(Path_j_s[:-1],Path_j_s[1:]):
        #print("\nanterior=",(i_prev,s_prev),'\t atual',(i_now,s_now),'\tw[i2,i1]',Wmat[i_now,i_prev])
        if Wmat[i_now,i_prev]>0:
            x[i_now,-s_now-1]=x[i_prev,-s_prev-1]
        elif Wmat[i_now,i_prev]<0:
            x[i_now,-s_now-1]=1-x[i_prev,-s_prev-1]
        else:
            print("BUG in neuron",ref_neuron)
#%%
def PlotRaster(X,figname="Sample",myaspect=5):

    print("\nPlase check a Raster plot representation in ",figname,".eps")
    plt.matshow(X,cmap='Greys',  interpolation='nearest',aspect=myaspect)
    #plt.xlim(100,min(1000,X.shape[1]))
    pylab.savefig(figname+'.eps', format='eps', dpi=1000)

#%%
initsize=1000
fillvalue=0.5

x=np.zeros([N,initsize])
x.fill(fillvalue)  
maxsteps=100

for step in range(maxsteps):
    for ref_neuron in range(N):
        Path_j_s=[]
        RecursionUntilEmptyZero(ref_neuron,0)
        AssignValsToPath(Path_j_s,ref_neuron)
        
    newrow=[[0] for it in range(N)]
    x=np.append(x,np.array([[0], [0], [0], [0], [0]]),axis=1)

#%%
    

sumaux=[x[:,i].sum() for i in range(initsize)]
initraster=max(next((i for i, x in enumerate(sumaux) if x!=fillvalue*N), None)-10,0)
        
PlotRaster(x[:,initraster:])    


    
