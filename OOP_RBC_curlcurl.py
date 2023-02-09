import numpy as np
import dedalus.public as d3
import logging

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

import os
logger = logging.getLogger(__name__)


###globals

dealias = 3/2
dtype = np.float64


class Laplacian_Problem:
    def __init__(self,Nx,Nz,alpha):
        self.Nx = Nx
        self.Nz = Nz
        self.alpha = alpha
    def initialize(self):
        self.coords = d3.CartesianCoordinates('x', 'z')
        self.dist = d3.Distributor(self.coords, dtype=dtype)
        self.xbasis = d3.RealFourier(self.coords['x'], size=self.Nx, bounds=(-1*np.pi/self.alpha, np.pi/self.alpha))
        self.zbasis = d3.Chebyshev(self.coords['z'], size=self.Nz, bounds=(-1, 1))
        
        u = self.dist.Field(name='u', bases=(self.xbasis, self.zbasis))
        v = self.dist.Field(name='v', bases=(self.xbasis, self.zbasis))
        phi = self.dist.Field(name='phi', bases=(self.xbasis, self.zbasis))
        tau_1 = self.dist.Field(name='tau_1', bases=self.xbasis)
        tau_2 = self.dist.Field(name='tau_2', bases=self.xbasis)
        
        self.u = u
        self.v = v
        self.phi = phi
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        
        dz = lambda A: d3.Differentiate(A, self.coords['z'])
        dx = lambda A: d3.Differentiate(A, self.coords['x'])
        lift_basis = self.zbasis.clone_with(a=3/2, b=3/2) # Natural output basis
        lift = lambda A, n: d3.Lift(A, lift_basis, n)

        
        self.problem = d3.LBVP([self.u,self.v,self.tau_1,self.tau_2],namespace=locals())
        self.problem.add_equation("lap(v) + lift(tau_1,-1) + lift(tau_2,-2) = phi")
        self.problem.add_equation("dx(u) + dz(v) + lift(tau_1,-1) = 0",condition='nx!=0')
        self.problem.add_equation("u=0",condition='nx==0')
        self.problem.add_equation("v(z=-1) = 0")
        self.problem.add_equation("v(z=1) = 0")
        
        self.solver = self.problem.build_solver()
    
    def getVel(self,phiArr):
        self.phi.load_from_global_grid_data(phiArr)
        self.solver.solve()
        uArr = self.u.allgather_data('g')
        vArr = self.v.allgather_data('g')
        return uArr, vArr

        
        

class RBC_Problem:
    def __init__(self,Ra,Pr,alpha,Nx,Nz,bcs,time_step=None,initial_u=None,initial_v=None,initial_phi=None,initial_b=None):
        self.Ra = Ra
        self.Pr = Pr
        self.alpha = alpha
        self.Nx = Nx
        self.Nz = Nz
        self.bcs = bcs
        self.time_step = time_step
        self.time = 0
        
        self.tVals = []
        self.NuVals = []
        
        self.init_u = initial_u
        self.init_v = initial_v
        self.init_phi = initial_phi
        self.init_b = initial_b
        self.init_dt = time_step
        
        self.volume = ((2*np.pi)/alpha)*2    
        
    def initialize(self):
        self.time = 0
        self.tVals = []
        self.NuVals = []
        
        ###making laplacian problem that will find u,v from phi for this case
        self.phi_lap = Laplacian_Problem(self.Nx, self.Nz, self.alpha)
        self.phi_lap.initialize()
        
        ##making Bases
        self.coords = d3.CartesianCoordinates('x','z')
        self.dist = d3.Distributor(self.coords, dtype=dtype)
        self.xbasis = d3.RealFourier(self.coords['x'], size=self.Nx, bounds=(-1*np.pi/self.alpha, np.pi/self.alpha), dealias=dealias)
        self.zbasis = d3.ChebyshevT(self.coords['z'], size=self.Nz, bounds=(-1, 1), dealias=dealias)
        
        # Fields
        phi = self.dist.Field(name='phi', bases=(self.xbasis,self.zbasis))
        u = self.dist.Field(name='u', bases=(self.xbasis,self.zbasis))
        v = self.dist.Field(name='v', bases=(self.xbasis,self.zbasis))
        b = self.dist.Field(name='b', bases=(self.xbasis,self.zbasis))


        tau_v1 = self.dist.Field(name='tau_v1', bases=self.xbasis)
        tau_v2 = self.dist.Field(name='tau_v2', bases=self.xbasis)

        tau_phi1 = self.dist.Field(name='tau_phi1', bases=self.xbasis)
        tau_phi2 = self.dist.Field(name='tau_phi2', bases=self.xbasis)

        tau_b1 = self.dist.Field(name='tau_b1', bases=self.xbasis)
        tau_b2 = self.dist.Field(name='tau_b2', bases=self.xbasis)

        tau_u1 = self.dist.Field(name='tau_u1', bases=self.xbasis)
        tau_u2 = self.dist.Field(name='tau_u2', bases=self.xbasis)
            
        
        self.phi = phi
        self.u = u
        self.v = v
        self.b = b
        
        self.tau_v1 = tau_v1
        self.tau_v2 = tau_v2
        self.tau_phi1 = tau_phi1
        self.tau_phi2 = tau_phi2
        self.tau_b1 = tau_b1
        self.tau_b2 = tau_b2
        
        ## Substitutions
        Ra = self.Ra
        Pr = self.Pr
        kappa = 4*(self.Ra * self.Pr)**(-1/2)
        nu = 4*(self.Ra / self.Pr)**(-1/2)
        self.x, self.z = self.dist.local_grids(self.xbasis, self.zbasis)
        self.ex, self.ez = self.coords.unit_vector_fields(self.dist)
        lift_basis = self.zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        lift = lambda A, n: d3.Lift(A, lift_basis, n)

        grad_v = d3.grad(self.v) + self.ez*lift(self.tau_v1,-1)
        grad_phi = d3.grad(self.phi) + self.ez*lift(self.tau_phi1,-1)
        grad_b = d3.grad(self.b) + self.ez*lift(self.tau_b1,-1) # First-order reduction
        dz = lambda A: d3.Differentiate(A, self.coords['z'])
        dx = lambda A: d3.Differentiate(A, self.coords['x'])
        
        ##defining problem
        self.problem = d3.IVP([self.phi, self.u, self.v, self.b, self.tau_v1, self.tau_v2, self.tau_phi1, self.tau_phi2, self.tau_b1, self.tau_b2], namespace=locals())
        
        if self.bcs == 'RB1':
            self.problem.add_equation("div(grad_v) + lift(tau_v2,-1) - phi= 0")
            self.problem.add_equation("dt(phi) - nu*div(grad_phi)-  dx(dx(b)) + lift(tau_phi2,-1) = -dx(u*phi - v*lap(u))  ")
            self.problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2,-1) = -u*dx(b)-v*dz(b)")
            self.problem.add_equation("dx(u) + dz(v)+ lift(tau_v1,-1) = 0", condition='nx!=0')
            self.problem.add_equation("u = 0", condition='nx==0')
            self.problem.add_equation("b(z=1) = -1")
            self.problem.add_equation("v(z=1) = 0")
            self.problem.add_equation("b(z=-1) = 1")
            self.problem.add_equation("v(z=-1) = 0")
            self.problem.add_equation("dz(v)(z=1) = 0")
            self.problem.add_equation("dz(v)(z=-1) = 0")

        if self.bcs == 'IH1':
            Ra = self.Ra
            Pr = self.Pr
            self.problem.add_equation("div(grad_v) + lift(tau_v2,-1) - phi= 0")
            self.problem.add_equation("dt(phi) - Pr*div(grad_phi)-  Pr*Ra*dx(dx(b)) + lift(tau_phi2,-1) = -dx(u*phi - v*lap(u))  ")
            self.problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2,-1) = -u*dx(b)-v*dz(b)+1")
            self.problem.add_equation("dx(u) + dz(v)+ lift(tau_v1,-1) = 0", condition='nx!=0')
            self.problem.add_equation("u = 0", condition='nx==0')
            self.problem.add_equation("b(z=1) = 0")
            self.problem.add_equation("v(z=1) = 0")
            self.problem.add_equation("b(z=-1) = 0")
            self.problem.add_equation("v(z=-1) = 0")
            self.problem.add_equation("dz(v)(z=1) = 0")
            self.problem.add_equation("dz(v)(z=-1) = 0")
        
        ##put in initial conditions if given
        if self.init_u is not None:
            #self.u['g'] = self.init_u
            #self.u.data = self.init_u
            self.u.load_from_global_grid_data(self.init_u)
            
        if self.init_v is not None:
            #self.u['g'] = self.init_u
            #self.u.data = self.init_u
            self.v.load_from_global_grid_data(self.init_v)
        
        if self.init_b is not None:
            #self.b['g'] = self.init_b
            #self.b.set_global_data(self.init_b)
            self.b.load_from_global_grid_data(self.init_b)
            
        if self.init_phi is not None:
            #self.p['g'] = self.init_p
            #self.p.set_global_data(self.init_p)
            self.phi.load_from_global_grid_data(self.init_phi)
        
        #if no initial conditions given, use a perturbed conduction state
        if self.init_u is None and self.init_v is None and self.init_b is None and self.bcs=='RB1':
            #self.b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
            #self.b['g'] *= (1+self.z) * (1 - self.z) # Damp noise at walls
            # self.b['g'] += 0.01*np.cos((1/2)*np.pi*self.x)*np.sin(np.pi*self.z*self.alpha)
            self.b['g'] += self.conduction_state() + np.cos(self.alpha*self.x)*np.cos(np.pi*self.z/2) # Add appropriate conduction state
            self.init_u = np.copy(self.u['g'])
            self.init_v = np.copy(self.v['g'])
            self.init_b = np.copy(self.b['g'])
            self.init_phi = np.copy(self.phi['g'])
        if self.init_u is None and self.init_v is None and self.init_b is None and self.bcs=='IH1':
            self.b.fill_random('g', seed=42, distribution='normal', scale=1e-5) # Random noise
            self.b['g'] *= (1+self.z) * (1 - self.z) # Damp noise at walls
            # self.b['g'] += 0.01*np.cos((1/2)*np.pi*self.x)*np.sin(np.pi*self.z*self.alpha)
            self.b['g'] += self.conduction_state() # Add appropriate conduction state
            self.init_u = np.copy(self.u['g'])
            self.init_v = np.copy(self.v['g'])
            self.init_b = np.copy(self.b['g'])
            self.init_phi = np.copy(self.phi['g'])
        
        
        ##defining timestepper
        timestepper = d3.RK222
        self.solver = self.problem.build_solver(timestepper)
        if self.bcs == 'IH1':
            max_timestep = 0.05
        else:
            max_timestep = 0.125
        
        # CFL
        self.CFL = d3.CFL(self.solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
                     max_change=1.5, min_change=0.5, max_dt=max_timestep)
        self.CFL.add_velocity(self.ex*self.u + self.ez*self.v)
        
        self.flow = d3.GlobalFlowProperty(self.solver, cadence=10)
        self.flow.add_property(1 + (self.b*self.v)/kappa,name='Nu')
        
        
    def conduction_state(self):
        if self.bcs == 'RB1':
            return -1*self.z
        elif self.bcs == 'IH1':
            return (1+self.z)*(self.z-1)
        
    def reset(self):
        ### NEED TO CHANGE
        self.u.load_from_global_grid_data(self.init_u)
        self.v.load_from_global_grid_data(self.init_v)
        self.phi.load_from_global_grid_data(self.init_phi)
        self.b.load_from_global_grid_data(self.b)
        self.time_step = self.init_dt
        self.time = 0
        self.tVals = []
        self.NuVals = []
        
    
    def solve_system(self,end_time,trackNu=None,anim_frames=False,write=False):
        assert(end_time > self.time)
        self.solver.sim_time = self.time
        self.solver.stop_sim_time = end_time
        
        if anim_frames:
            plt.figure()
            framecount = 0
            X,Z = np.meshgrid(self.x.ravel(),self.z.ravel())
        
        first_step = True
        if not write:
            for system in ['solvers']:
                logging.getLogger(system).setLevel(logging.WARNING)
        try:
            if write:
                logger.info('Starting main loop')
            while self.solver.proceed:
                if self.time_step != None and first_step:
                    timestep = self.time_step
                else:
                    timestep = self.CFL.compute_timestep()
                self.solver.step(timestep)
                if (self.solver.iteration-1) % 10 == 0:
                    if anim_frames:
                        self.b.change_scales(1)
                        plt.pcolormesh(X.T,Z.T,self.b['g'],cmap='seismic')
                        plt.colorbar()
                        plt.xlabel('x')
                        plt.ylabel('z')
                        plt.title('t='+str(self.solver.sim_time))
                        image_name = 'animations/frame' + str(framecount).zfill(10) + '.jpg'
                        plt.savefig(image_name)
                        plt.clf()
                        framecount += 1
                    flow_Nu = self.flow.volume_integral('Nu')/self.volume
                    if trackNu:
                         self.tVals.append(self.solver.sim_time)
                         self.NuVals.append(flow_Nu)
                    if write:
                        logger.info('Iteration=%i, Time=%e, dt=%e, Nu=%f' %(self.solver.iteration, self.solver.sim_time, timestep,flow_Nu))
        except:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.solver.log_stats()
        self.time_step = timestep
        self.time = end_time
    
    def calc_Nu(self):
        Nu = self.flow.volume_integral('Nu')/self.volume
        return Nu
    
    def plot(self):
        X,Z = np.meshgrid(self.x.ravel(),self.z.ravel())
        
        
        self.u.change_scales(1)
        self.v.change_scales(1)
        self.b.change_scales(1)
        
        uArr = self.u['g']
        vArr = self.v['g']
        bArr = self.b['g']
        
        ##just for coolness, cmap='coolwarm' also looks alright
        
        title = 'Ra: ' + str(self.Ra) + '  Pr: ' + str(self.Pr) + "  alpha: " + str(self.alpha)
        
        fig, axs = plt.subplots(2, 2)
        p1 = axs[0,0].pcolormesh(X.T,Z.T,bArr,cmap='seismic')
        axs[0,0].quiver(X.T,Z.T,uArr,vArr)
        fig.colorbar(p1,ax=axs[0,0])
        
        p2 = axs[0,1].pcolormesh(X.T,Z.T,bArr,cmap='seismic')
        fig.colorbar(p2,ax=axs[0,1])
        
        p3 = axs[1,0].contourf(X.T,Z.T,bArr,cmap='seismic')
        fig.colorbar(p3,ax=axs[1,0])
        
        p4 = axs[1,1].plot(self.tVals,self.NuVals)
        axs[1,1].set_title('Nusselt vs. Time')
        # p4 = axs[1,1].contourf(X.T,Z.T,np.linalg.norm(uArr,axis=0),cmap='BrBG')
        # fig.colorbar(p4,ax=axs[1,1])
        
        plt.suptitle(title)
    
    def loadFromFile(self,time,path=None):
        if path is None:
            path = 'Outputs/' + self.bcs + '/Ra_' + str(self.Ra) + '/Pr_' + str(self.Pr) +   '/' + 'alpha_' + str(self.alpha) + '/' + 'Nx_' + str(self.Nx) + '/' + 'Nz_' + str(self.Nz) + '/T_' + str(time) + '/'
            path = path + 'data.npy'
        
        loadFile = path
        
        with open(loadFile,'rb') as l_File:
            uFromFile = np.load(l_File)
            vFromFile = np.load(l_File)
            bFromFile = np.load(l_File)
            phiFromFile = np.load(l_File)
            dt = np.load(l_File)
        
        # self.u['g'] = uFromFile
        # self.v['g'] = vFromFile
        # self.b['g'] = bFromFile
        # self.phi['g'] = phiFromFile
        self.u.load_from_global_grid_data(uFromFile)
        self.v.load_from_global_grid_data(vFromFile)
        self.b.load_from_global_grid_data(bFromFile)
        self.phi.load_from_global_grid_data(phiFromFile)
        self.time_step = dt
        
    
    def saveToFile(self,path=None):
        self.u.change_scales(1)
        self.v.change_scales(1)
        self.b.change_scales(1)
        self.phi.change_scales(1)
        
        
        
        if path == None:
            path = 'Outputs/' + self.bcs + '/Ra_' + str(self.Ra) + '/Pr_' + str(self.Pr)  +  '/' + 'alpha_' + str(self.alpha) + '/' + 'Nx_' + str(self.Nx) + '/' + 'Nz_' + str(self.Nz) + '/T_' + str(self.time) + '/'
            try:
                os.makedirs(path)
            except FileExistsError:
                ans = input('File already exists, you could possibly overrwrite! Do you want to proceed? (y/n)')
                if ans == 'y':
                    pass
                else:
                    raise FileExistsError
            path = path + 'data.npy'
            
        outputFile = path
        with open(outputFile,'wb') as outFile:
            np.save(outFile,self.u.allgather_data('g'))
            np.save(outFile,self.v.allgather_data('g'))
            np.save(outFile,self.b.allgather_data('g'))
            np.save(outFile,self.phi.allgather_data('g'))
            np.save(outFile,self.time_step)
            
        
    def __repr__(self):
        repr_string = "RBC Problem in configuration " + self.bcs + " with following params: \n" + "Ra= " + str(self.Ra) + "\n" + "Pr= " + str(self.Pr) + "\n" + "alpha= " + str(self.alpha) + "\n"
        repr_string = repr_string + "\n" + "Simulation params: \n" + "Nx= " + str(self.Nx) + "\n" + "Nz= " + str(self.Nz)
        return repr_string
    



######################################
### to load arrays from .npy files ###
######################################

def open_fields(file_name):
    with open(file_name,'rb') as l_File:
        uFromFile = np.load(l_File)
        vFromFile = np.load(l_File)
        bFromFile = np.load(l_File)
        phiFromFile = np.load(l_File)
        dt = np.load(l_File)
    return uFromFile, vFromFile,bFromFile, phiFromFile, dt

#########################
### finding steady state ###
#########################

def probToStateVec(RBCProb):
    Nx = RBCProb.Nx
    Nz = RBCProb.Nz
    
    RBCProb.phi.change_scales(1)
    RBCProb.b.change_scales(1)
    bArr = RBCProb.b.allgather_data('g')
    phiArr = RBCProb.phi.allgather_data('g')

    X = np.zeros(2*Nx*Nz)
    bArr = bArr.flatten()
    phiArr = phiArr.flatten()
    X[0:Nx*Nz] = phiArr
    X[Nx*Nz:] = bArr
    return X

def arrsToStateVec(phiArr,bArr):
    Nx,Nz = bArr.shape
    X = np.zeros(2*Nx*Nz)
    bArr = bArr.flatten()
    phiArr = phiArr.flatten()
    X[0:Nx*Nz] = phiArr
    X[Nx*Nz:] = bArr
    return X

def stateToArrs(X,Nx,Nz):
    phiArr = X[0:Nx*Nz]
    bArr = X[Nx*Nz:]
    
    #get them into a format we can put into the flow map function
    phiArr = np.reshape(phiArr,(-1,Nz))
    bArr = np.reshape(bArr,(-1,Nz))
    return phiArr, bArr

def Gt(X,T,problem):
    phiArr, bArr = stateToArrs(X,problem.Nx,problem.Nz)
    uArr, vArr = problem.phi_lap.getVel(phiArr)
    problem.time = 0
    problem.phi.load_from_global_grid_data(phiArr)
    problem.u.load_from_global_grid_data(uArr)
    problem.v.load_from_global_grid_data(vArr)
    problem.b.load_from_global_grid_data(bArr)
    problem.solve_system(T)
    Gt_Vec = probToStateVec(problem)
    Gt_Vec = (Gt_Vec - X)/T
    return Gt_Vec

def jac_approx(X,dX,F,T,problem):
    print(dX)
    mach_eps = np.finfo(float).eps
    normX = np.linalg.norm(X)
    normdX = np.linalg.norm(dX)
    dotprod = np.dot(X,dX)
    eps = (np.sqrt(mach_eps)/(normdX**2))*max(dotprod,normdX)
    #eps = 1e-3
    GtArr = Gt(X+eps*dX,T,problem).T
    return (GtArr + F.T)/eps


def steady_state_finder(problem,guess,T,tol,max_iters,write):
    #problem is an RBC_problem
    #guess is a guess for the state vec
    #T is time we are integrating out to
    #tol is tolerance for Newton method 
    #max_iters is max Newton iterations that will be done
    err = 1e10
    iters = 0

    Nx = problem.Nx
    Nz = problem.Nz

    X = guess
    while err > tol and iters < max_iters:
        if write == 'y':
            print("iter: ",iters)
            print(X)
            print("-------------")

        F = -1*Gt(X,T,problem)
        A = lambda dX : jac_approx(X,dX,F,T,problem)
        A_mat = LinearOperator((2*Nx*Nz,2*Nx*Nz),A)
        delta_X,code =gmres(A_mat,F,tol=1e-3)
        print(delta_X)
        if code != 0:
            raise("gmres did not converge")
        X= X+delta_X
        iters += 1
        err = np.linalg.norm(Gt(X,T,problem))
    phiStead, bStead = stateToArrs(X,Nx,Nz)
    problem.phi.load_from_global_grid_data(phiStead)
    problem.b.load_from_global_grid_data(bStead)
    return iters

###############################################################################

                                        #########
                                        #TESTING#
                                        #########


########################
### testing restarts ###
########################

def test_restart():
    Nx = 128
    Nz = 64
    test_start = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test_start.initialize()
    test_start.solve_system(100)
    test_start.saveToFile('test_save')
    
    test1 = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test1.initialize()
    test1.loadFromFile(1,'test_save')
    test1.solve_system(100)
    
    test_full = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test_full.initialize()
    test_full.solve_system(200)
    
    test1.u.change_scales(1)
    test1.v.change_scales(1)
    test1.b.change_scales(1)
    test1.phi.change_scales(1)
    
    test_full.u.change_scales(1)
    test_full.v.change_scales(1)
    test_full.b.change_scales(1)
    test_full.phi.change_scales(1)
    
    
    uErr = np.max(abs(test1.u['g']-test_full.u['g']))
    vErr = np.max(abs(test1.v['g']-test_full.v['g']))
    bErr = np.max(abs(test1.b['g']-test_full.b['g']))
    phiErr = np.max(abs(test1.phi['g']-test_full.phi['g']))
    print('u error: ', uErr)
    print('v error: ', vErr)
    print('b error: ', bErr)
    print('phi error: ', phiErr)
    
def test_restart_phi():
    Nx = 128
    Nz = 64
    test_start = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test_start.initialize()
    test_start.solve_system(100)
    test_start.saveToFile('test_save')
    
    uArr, vArr, bArr, phiArr, dt = open_fields('test_savedata.npy')
    
    test1 = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test1.initialize()
    test1.phi.load_from_global_grid_data(phiArr)
    test1.solve_system(100)
    
    test_full = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test_full.initialize()
    test_full.solve_system(200)
    
    test1.u.change_scales(1)
    test1.v.change_scales(1)
    test1.b.change_scales(1)
    test1.phi.change_scales(1)
    
    test_full.u.change_scales(1)
    test_full.v.change_scales(1)
    test_full.b.change_scales(1)
    test_full.phi.change_scales(1)
    
    
    uErr = np.max(abs(test1.u['g']-test_full.u['g']))
    vErr = np.max(abs(test1.v['g']-test_full.v['g']))
    bErr = np.max(abs(test1.b['g']-test_full.b['g']))
    phiErr = np.max(abs(test1.phi['g']-test_full.phi['g']))
    print('u error: ', uErr)
    print('v error: ', vErr)
    print('b error: ', bErr)
    print('phi error: ', phiErr)
    


#######################################
### testing the array manipulations ###
#######################################


def test_steady_finder():
    uArr1,vArr1,bArr1, phiArr1,dt1 = open_fields("test_files_curlcurl/steady_state_tests/Ra5000Pr100alpha1.5585Nx128Nz64T1000.npy")
    Nx = 128
    Nz = 64
    
    test1 = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test1.initialize()
    guess1 = arrsToStateVec(phiArr1, bArr1)
    iters1 = steady_state_finder(test1,guess1,2,1e-3,2,True)
    
    test1.b.change_scales(1)
    test1.phi.change_scales(1)
    
    steady1_b = test1.b.allgather_data('g')
    steady1_phi = test1.phi.allgather_data('g')
    
    
    error1_b = np.max(abs(steady1_b-bArr1))
    error1_phi = np.max(abs(steady1_phi-phiArr1))
    
    logging.info("number of iters: %i", iters1)
    logging.info("Error in b: %f",error1_b)
    logging.info("Error in phi: %f",error1_phi)
    logging.info("test 4 over \n ----------------------")
    
    
    uArr2,vArr2,bArr2, phiArr2,dt2 = open_fields("test_files_curlcurl/steady_state_tests/Ra5000Pr100alpha1.5585Nx128Nz64T250.npy")
    
    test2 = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    test2.initialize()
    guess2 = arrsToStateVec(phiArr2, bArr2)
    iters2 = steady_state_finder(test2,guess2,2,1e-3,2,True)
    
    test2.b.change_scales(1)
    test2.phi.change_scales(1)
    
    steady2_b = test1.b.allgather_data('g')
    steady2_phi = test1.phi.allgather_data('g')
    
    error2_b = np.max(abs(steady2_b-bArr1))
    error2_phi = np.max(abs(steady2_phi-phiArr1))
    
    logging.info("number of iters: %i", iters2)
    logging.info("Error in b: %f",error2_b)
    logging.info("Error in phi: %f",error2_phi)
    logging.info("test 4 over \n ----------------------")
    
    return test2, bArr1,phiArr1

def follow_branch():
    #RaVals = [5000,7000,10000,13000,17000,20000]
    #RaVals = [6000,7000,8000,8500,9000,9500,10000]
    #RaVals = [10000,15000,20000,25000,30000,35000,40000]
    #RaVals = [40000,45000,50000,]
    #RaVals1 = np.arange(2e3,5e3,1e3)
    #RaVals2 = np.arange(1e4,10e4,5e3)
    #RaVals = np.block( [RaVals1 , RaVals2])
    #RaVals = [100000,105000,110000]
    #RaVals = [2000,3000]
    RaVals = [3000,4000]
    steady_states = []
    Nu_Vals = []
    Nx = 128
    Nz = 64
    uArr,vArr,bArr, phiArr,dt = open_fields('RB1_steady_states/Pr100/primary_box/Ra3000.0Pr100alpha1.5585Nx128Nz64data.npy')
    guess = arrsToStateVec(phiArr, bArr)
    #guess = np.zeros(2*Nx*Nz)
    #dt = 0.125
    filename = 'branch_tester/Pr100'
    for Ra in RaVals:
        steady = RBC_Problem(Ra,100,1.5585,Nx,Nz,'RB1',time_step=dt)
        steady.initialize()
        iters = steady_state_finder(steady, guess, 2, 1e-7, 50, False)
        #print('Ra= ',Ra)
        #print('steady state found . Iters = ', iters)
        steady_states.append(steady)
        Nu = steady.calc_Nu()
        Nu_Vals.append(Nu)
        logging.info("Ra= %i", Ra)
        logging.info('Steady State Found. Iters = %i', iters)
        logging.info("Nu= %f", Nu)
        saveFile = filename +'Ra'+str(Ra)+'.npy'
        steady.saveToFile(saveFile)
        
        
        steady.phi.change_scales(1)
        steady.b.change_scales(1)
        steady_b = steady.b.allgather_data('g')
        steady_phi = steady.phi.allgather_data('g')
        guess = arrsToStateVec(steady_phi, steady_b)
        dt = steady.time_step
        
    return RaVals, Nu_Vals, steady_states
        
    
def optimize_alpha():
    Ra = 80000
    Nx = 128
    Nz = 64
    uArr,vArr,bArr, phiArr,dt = open_fields('RB1_steady_states/Pr100/primary_box/Ra80000.0Pr100alpha1.5585Nx128Nz64data.npy')
    guess = arrsToStateVec(phiArr, bArr)
    alpha_vals = np.linspace(2,10,17)
    steady_states = []
    Nu_Vals = []
    for alpha in alpha_vals:
        steady = RBC_Problem(Ra,100,alpha,Nx,Nz,'RB1',time_step=dt)
        steady.initialize()
        iters = steady_state_finder(steady, guess, 2, 1e-7, 50, False)
        print('alpha= ', alpha)
        print("steady state found. Iters= ", iters)
        steady_states.append(steady)
        Nu = steady.calc_Nu()
        Nu_Vals.append(Nu)
        
        steady.phi.change_scales(1)
        steady.b.change_scales(1)
        steady_b = steady.b.allgather_data('g')
        steady_phi = steady.phi.allgather_data('g')
        guess = arrsToStateVec(steady_phi, steady_b)
        dt = steady.time_step
    
    return alpha_vals, Nu_Vals, steady_states
        
        
        
        
        
    
    
    