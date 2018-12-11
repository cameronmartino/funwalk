from __future__ import division
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from matplotlib.colors import rgb2hex

def MonodExt(k1,k2,kt,l,Stmp,ks=200):
    # source: Lejeune et al 1995, Morphology of Trichoderma reesei QM 9414 in Submerged Cultures
    return (k1+k2*(l/(l+kt)))*(Stmp/(Stmp+ks))

class hyphal_walk(object):
    
    def __init__(self,minTheta=10,maxTheta=100,avgRate=28,H=1440,N=1200,M=1e10,tstep=.0005,
                 q=0.004,S0=5e5,k1=50,maxktip=None,k2=None,kt=5,init_n=20,width=100,
                 set_start_center=True,use_monod=True,normal_theta=True):

        """

        minTheta = 10           #*pi/180 - minimum angle a branch can occur
        maxTheta = 100          #*pi/180 - maximum angle a branch can occur
        H = 1440                # number of hours
        N = 1200               # max simulation rounds
        M = 1e10               # max hyphae (carrying capacity)
        tstep = .0005       # time step (hours/step)
        q = 0.004            # branching frequency (maybe need to be scaled of what the time step is)
        S0 = 5e5                # intital conc. of substrate mg in whole grid (evenly dist.)
        k1 = 50                 # (µm/h) initial tip extension rate, value estimated from Spohr et al 1998 figure 5
        maxktip = 2*k1           # (µm/h) maximum tip extension rate, value estimated from Spohr et al 1998 figure 5
        k2 = maxktip - k1    # (µm/h) difference between k1 and maxktip
        kt = 5                  # saturation constant
        init_n = 20      # starting spores 
        width = 100  # view window (um) (this is just 1 cm)
        set_start_center = True # if you want the model to start all spores at (0,0)

        """
        
        if maxktip is None:
            maxktip = k1
        if k2 is None:
            k2 = maxktip - k1

        self.minTheta = minTheta
        self.maxTheta = maxTheta
        #self.avgRate = avgRate
        self.H = H
        self.N = N
        self.M = M
        self.tstep = tstep
        self.q = q
        self.S0 = S0
        self.k1 = k1
        self.maxktip = maxktip
        self.k2 = k2
        self.kt = kt
        self.init_n = init_n
        self.width = width
        self.set_start_center = set_start_center
        self.use_monod = use_monod
        self.normal_theta = normal_theta
        self.hyphae = self.intialize_hyphae()
        self.Sgrid = self.intialize_subtrate()
    
    def intialize_hyphae(self):
        hyphae = {}
        centers = np.array([0,0]).reshape(1, 2)
        # for each spore make and intital random walk direction (no movement yet)
        if self.normal_theta==True:
            theta_init = {i:angle_ for i,angle_ in enumerate(np.linspace(0,360,self.init_n))}
        for spore_i in range(0,self.init_n):
            if self.set_start_center==False:
                rxy = np.random.uniform(0,round(self.width),2) + centers
            else:
                rxy = centers
            if self.normal_theta==True:
                iTheta = theta_init[spore_i]
            else:
                iTheta = np.around(np.random.uniform(0,360),1)
            hyphae[spore_i] = {'x0':rxy[:,0], 'y0':rxy[:,1], 'x':rxy[:,0], 
                         'y':rxy[:,1], 'angle':iTheta, 'biomass':0, 't':0, 'l':0}
        return hyphae

    def intialize_subtrate(self,block_div=2):
        # make a substrate grid 
        Sgrid = []
        # make a substrate grid 
        size_of_block = round(self.width/block_div)  
        grid_min = list(np.linspace(-self.width,self.width,
                                    size_of_block)[:-1])
        grid_max = list(np.linspace(-self.width,self.width,
                                    size_of_block)[1:])
        for i in range(len(grid_max)):
            Sgrid.append(pd.DataFrame([[self.S0/len(grid_max)]*len(grid_max),grid_min,grid_max,
                                       [grid_min[i]]*len(grid_max),[grid_max[i]]*len(grid_max)],
                                      index=['S','X_Gmin','X_Gmax','Y_Gmin','Y_Gmax']).T)
        Sgrid = pd.concat(Sgrid,axis=0).reset_index()
        return Sgrid
    
    def run_simulation(self):
        
        # run until i exceeds limits 
        time_snapshot_hy = {}
        time_snapshot_sub = {}
        for i in tqdm(range(0,self.N)):
            bio_mass = 0 
            if len(self.hyphae)>=self.M:
                # hit carrying cpacity of the system
                print('broke capacity first')
                break
            # otherwise continue to model
            for j in range(0,len(self.hyphae)):
                # find tip in substrate grid 
                grid_index = self.Sgrid[((self.Sgrid['Y_Gmin']<=self.hyphae[j]['y'][0])&\
                                    (self.Sgrid['X_Gmin']<=self.hyphae[j]['x'][0]))==True].index.max()
                if np.isnan(grid_index):
                    # left view space
                    continue
                if round(self.Sgrid.loc[grid_index,'S'])!=0:
                    # get current extention
                    if self.use_monod==True:
                        ext = MonodExt(self.k1,self.k2,self.kt,
                                       self.hyphae[j]['l'],
                                       self.Sgrid.loc[grid_index,'S'])
                    else:
                        ext = self.maxktip
                    # extend in x and y
                    dx = ext * self.tstep * np.cos(self.hyphae[j]['angle']*np.pi/180) # new coordinate in x-axis
                    dy = ext * self.tstep * np.sin(self.hyphae[j]['angle']*np.pi/180) # new coordinate in y-axis
                    # biomass created for hyphae j
                    dl_c = np.sqrt(dx**2 + dy**2)
                    # (constant to scale biomass density)
                    dl_c *= 1
                    bio_mass += dl_c
                    # subtract used substrate
                    if self.use_monod==True:
                        self.Sgrid.loc[grid_index,'S'] = self.Sgrid.loc[grid_index,'S'] - dl_c
                    # update location
                    self.hyphae[j]['x'] = self.hyphae[j]['x']+dx
                    self.hyphae[j]['y'] = self.hyphae[j]['y']+dy
                    self.hyphae[j]['l'] = np.sqrt((self.hyphae[j]['x'][0]-self.hyphae[j]['x0'][0])**2 \
                                            +(self.hyphae[j]['y'][0]-self.hyphae[j]['y0'][0])**2 )
                    self.hyphae[j]['biomass'] = self.hyphae[j]['biomass'] + dl_c
                    # randomly split
                    if np.random.uniform(0,1) < self.q:
                        direction = [-1,1][round(np.random.uniform(0,1))]
                        newangle = direction*round(np.random.uniform(self.minTheta,self.maxTheta))
                        newangle += self.hyphae[j]['angle'] 
                        self.hyphae[len(self.hyphae)] = {'x0':self.hyphae[j]['x'], 'y0':self.hyphae[j]['y'], 
                                                 'x':self.hyphae[j]['x'], 'y':self.hyphae[j]['y'], 
                                                 'angle':newangle, 'biomass':0, 't':i, 'l':0}
            time_snapshot_hy[i] = pd.DataFrame(self.hyphae.copy()).copy()
            time_snapshot_sub[i] = pd.DataFrame(self.Sgrid.copy()).copy() 
        return time_snapshot_hy,time_snapshot_sub