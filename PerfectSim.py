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
import sys




#%%

def Build_nu(N,fact): 
    ''' builds array of spontaneous firing rates '''
    nu=np.random.rand(N)/fact
    return nu
    
def BuildWmat(N,Nerase):
    ''' builds matrix of random connections in the form W[postsynaptic,presynaptic] 
        of weigth between -1 and 1
        of NxN where Nerase connections are forced to zero    
    '''
    Wmatrix=np.random.rand(N,N)
    Wmatrix=Wmatrix*2-1
    l1=np.random.randint(0,N,Nerase)
    l2=np.random.randint(0,N,Nerase)
    for n in range(N):#erase diagonal
        Wmatrix[n,n]=0
    for x,y in zip(l1,l2):#erase other randomly selected elements
        Wmatrix[x,y]=0
    return Wmatrix    

def gfunc(s,mu):
    return mu**s
    
#%%

    
#%%
def CheckprobAndReturnWG_i(Wmat,nu,i,nsteps,mu):
    
    '''Checks if values are compatible to a probability measure 
        and returns the wgmat object for each neuron '''
    myinf=nu[i] #at the end myinf has to be greater than zero and mysup smaller than 1
    mysup=nu[i]
    
    N=len(Wmat)
    wgmat=np.zeros([N,nsteps])
    for j in range(N): # including i itself because wi->i is zero by construction
        for s in range(nsteps):
            wg=Wmat[i,j]*gfunc(s+1,mu)  #because range starts at 0     
            wgmat[j,s]=wg
            if wg>0:
                mysup=mysup+wg
            elif wg<0:
                myinf=myinf+wg
                
    return wgmat,mysup,myinf


    
#%%

def CheckprobAndReturnWG_all(Wmat,nu,nsteps,mu):
    '''Checks if values are compatible to a probability measure 
        and returns the wgmat for all neurons '''

    wgmat=[]
    N=len(Wmat)
    OK=True
    for i in range(N):        
        wgaux,sup,inf=CheckprobAndReturnWG_i(Wmat,nu,i,nsteps,mu)
        wgmat.append(wgaux)
        if sup>1 or inf<=0:
            print("NOT PROB MEASURE. REDO.")
            OK=False
    
    return OK, wgmat    
       

#%%

def FindValidWmatNu(N,factnu,nsteps,mu):    
    '''Searches for a configuration of Wmat, nu,nsteps and mu that yields a valid probability measure'''
    Nerase=15
    while True: 
        Wmat=BuildWmat(N,Nerase)
        nu=Build_nu(N,factnu)
        wgmat=[]
        OK,wgmat=CheckprobAndReturnWG_all(Wmat,nu,nsteps,mu)
        if not OK:
            print("Doesnt lead to a prob measure. Keep trying.")
        else:
            print("Worked. Stick to this one.\n\n")
            break
    
    return wgmat,Wmat,nu
    
#%%
def MakeAndSaveValidWmat(N=5,nsteps=4,mu=0.5,factnu=2,filename="W_nu_valid.pkl"):
    '''Searches for a configuration of Wmat, nu,nsteps and mu that yields a valid probability measure
    and saves as dictionary in a pickle file'''
    
    wgmat,Wmat,nu=FindValidWmatNu(N,factnu,nsteps,mu)
    pickle.dump({"Wmat":Wmat,"nu":nu,'nsteps':nsteps,'mu':mu},open(filename,"wb"))


#%%

def RecursionUntilEmptyZero(x,i,s,lambda_grid,p_Empty0_1,js_inds,Path_j_s,fillvalue):
    r=np.random.rand()
    partition_pos=[ind-1 for ind in range(len(lambda_grid[i])) if r<lambda_grid[i][ind] and r>=lambda_grid[i][ind-1]][0]
    if partition_pos == 0:
        r_spk=np.random.rand()
        if r_spk<=p_Empty0_1[i]:
            x[i,-1-s]=1 # last element, more recent in time (time zero reference)
        else:
            x[i,-1-s]=0
    else:
        (j,snext)=js_inds[i][partition_pos-1]
        s=s+snext                
        Path_j_s.append((j,s))
        pastpos=-1-s
        if abs(pastpos)>len(x[0]):
            print("Path to past too long for existent array")
            sys.exit()
        else:
            if x[j,pastpos]==fillvalue: 
#                RecursionUntilEmptyZero(j,s) 
                RecursionUntilEmptyZero(x,j,s,lambda_grid,p_Empty0_1,js_inds,Path_j_s,fillvalue)
  
  
#%%
            
def AssignValsToPath(x,Path_j_s,ref_neuron,Wmat):
    Path_j_s.reverse()
    Path_j_s.append((ref_neuron,0))
    for (i_prev,s_prev),(i_now,s_now) in zip(Path_j_s[:-1],Path_j_s[1:]):
        #print("\nanterior=",(i_prev,s_prev),'\t atual',(i_now,s_now),'\tw[i2,i1]',Wmat[i_now,i_prev])
        #try:        
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
    plt.ylabel("particle state")
    plt.xlabel("time")
    pylab.savefig(figname+'.eps', format='eps', dpi=100)





#%%

def BuildLambdaGrid(wgmat,nsteps,N,nu):
    delta=[]
    lambda_grid=[]
    p_Empty0_1=[]
    js_inds=[]
     
    for i in range(N):    
           
        wgmatneg=[it for l in wgmat[i] for it in l if it<0]
        
        delta.append(nu[i]+np.sum(wgmatneg))
           
        lambda_i_js=[]
        js_inds_i=[]
        jlist=[j for j in range(N) if j is not i]

        for j in jlist: 
            if np.abs(wgmat[i][j,0])!=0: 
                for s in range(1,nsteps+1): # not interested in s=0 because that is lambda0 that is not determined yet
                    js_inds_i.append((j,s))
                    lambda_i_js.append(abs(wgmat[i][j,s-1])) #CUIDADO PARA NAO COLOCAR VALORES NULOS NESSA LISTA
                    
                    
        lambda_i_Empty0=1-np.sum(lambda_i_js)    
        lambda_i_grid=[0,lambda_i_Empty0]
        for it in lambda_i_js:
            aux=lambda_i_grid[-1]
            lambda_i_grid.append(aux+it)
        
        p_i_Empty0_1=delta[i]/(lambda_i_Empty0)
        lambda_grid.append(lambda_i_grid)
        p_Empty0_1.append(p_i_Empty0_1)    
        js_inds.append(js_inds_i)
    
    return js_inds,lambda_grid,p_Empty0_1


if __name__ == "__main__":    

    D=pickle.load(open("W_nu_valid.pkl","rb"))
    Wmat=D["Wmat"]
    nu=D["nu"]
    N=len(Wmat)
    nsteps=D["nsteps"]
    mu=D["mu"]
    OK,wgmat=CheckprobAndReturnWG_all(Wmat,nu,nsteps,mu)
#    js_inds,lambda_grid,p_Empty0_1=BuildLambdaGrid(wgmat,nsteps,Wmat,mu)
    js_inds,lambda_grid,p_Empty0_1=BuildLambdaGrid(wgmat,nsteps,N,nu)

    initsize=100
    fillvalue=0.5 # DO NOT USE NAN because of if comparisons. Also do not use 0 or one for obvious reasons.
    maxsteps=100
    
    x=np.zeros([N,initsize])
    x.fill(fillvalue)      
    
    for step in range(maxsteps):        
        newrow=np.array([[fillvalue] for it in range(N)])
        x=np.append(x,newrow,axis=1)        
        for ref_neuron in range(N):
            Path_j_s=[]
            RecursionUntilEmptyZero(x,ref_neuron,0,lambda_grid,p_Empty0_1,js_inds,Path_j_s,fillvalue)
            AssignValsToPath(x,Path_j_s,ref_neuron,Wmat)

    PlotRaster(x[:,:]) 
    
    
    