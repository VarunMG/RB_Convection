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


class RBC_Problem:
    def __init__(self,Ra,Pr,alpha,Nx,Nz,bcs,time_step=None,initial_u=None,initial_b=None,initial_p=None):
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
        self.init_b = initial_b
        self.init_p = initial_p
        self.init_dt = time_step
        
        self.volume = ((2*np.pi)/alpha)*2
        
        # ##making Bases
        # self.coords = d3.CartesianCoordinates('x','z')
        # self.dist = d3.Distributor(self.coords, dtype=dtype)
        # self.xbasis = d3.RealFourier(self.coords['x'], size=Nx, bounds=(-1*np.pi/alpha, np.pi/alpha), dealias=dealias)
        # self.zbasis = d3.ChebyshevT(self.coords['z'], size=Nz, bounds=(-1, 1), dealias=dealias)
        
        # ##making fields
        # p = self.dist.Field(name='p', bases=(self.xbasis,self.zbasis))
        # b = self.dist.Field(name='b', bases=(self.xbasis,self.zbasis))
        # u = self.dist.VectorField(self.coords, name='u', bases=(self.xbasis,self.zbasis))
        # tau_p = self.dist.Field(name='tau_p')
        # tau_b1 = self.dist.Field(name='tau_b1', bases=self.xbasis)
        # tau_b2 = self.dist.Field(name='tau_b2', bases=self.xbasis)
        # tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.xbasis)
        # tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.xbasis)
        
        # self.p = p
        # self.b = b
        # self.u = u
        # self.tau_p = tau_p
        # self.tau_b1 = tau_b1
        # self.tau_b2 = tau_b2
        # self.tau_u1 = tau_u1
        # self.tau_u2 = tau_u2
            
        
        # # Substitutions
        # kappa = 4*(Ra * Pr)**(-1/2)
        # nu = 4*(Ra / Pr)**(-1/2)
        # self.x, self.z = self.dist.local_grids(self.xbasis, self.zbasis)
        # ex, ez = self.coords.unit_vector_fields(self.dist)
        # lift_basis = self.zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        # lift = lambda A: d3.Lift(A, lift_basis, -1)
        # grad_u = d3.grad(self.u) + ez*lift(self.tau_u1) # First-order reduction
        # grad_b = d3.grad(self.b) + ez*lift(self.tau_b1) # First-order reduction
        
        # ##defining problem
        # self.problem = d3.IVP([self.p, self.b, self.u, self.tau_p, self.tau_b1, self.tau_b2, self.tau_u1, self.tau_u2], namespace=locals())
        
        # if self.bcs == 'RB1':
        #     self.problem.add_equation("trace(grad_u) + tau_p = 0")
        #     self.problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - dot(u,grad(b))")
        #     self.problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - dot(u,grad(u))")
        #     self.problem.add_equation("b(z=-1) = 1")
        #     self.problem.add_equation("u(z=-1) = 0")
        #     self.problem.add_equation("b(z=1) = -1")
        #     self.problem.add_equation("u(z=1) = 0")
        # if self.bcs == 'IH1':
        #     self.problem.add_equation("trace(grad_u) + tau_p = 0")
        #     self.problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2) = - dot(u,grad(b))+1")
        #     self.problem.add_equation("dt(u) - Pr*div(grad_u) + grad(p) - Pr*Ra*b*ez + lift(tau_u2) = - dot(u,grad(u))")
        #     self.problem.add_equation("b(z=-1) = 0")
        #     self.problem.add_equation("b(z=1) = 0")
        #     self.problem.add_equation("u(z=1) = 0")
        #     self.problem.add_equation("u(z=-1) = 0")
        # self.problem.add_equation("integ(p) = 0") # Pressure gauge
        
        # ##put in initial conditions if given
        # if initial_u != None:
        #     self.u['g'] = initial_u
        
        # if initial_b != None:
        #     self.b['g'] = initial_b
        
        # #if no initial conditions given, use a perturbed conduction state
        # if initial_u == None and initial_b == None:
        #     self.b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
        #     self.b['g'] *= (1+self.z) * (1 - self.z) # Damp noise at walls
        #     self.b['g'] += self.conduction_state() # Add appropriate conduction state
        #     self.init_u = np.copy(u['g'])
        #     self.init_b = np.copy(b['g'])
        
        
        # ##defining timestepper
        # timestepper = d3.RK222
        # self.solver = self.problem.build_solver(timestepper)
        # if self.bcs == 'IH1':
        #     max_timestep = 0.05
        # else:
        #     max_timestep = 0.125
        
        # # CFL
        # self.CFL = d3.CFL(self.solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
        #              max_change=1.5, min_change=0.5, max_dt=max_timestep)
        # self.CFL.add_velocity(self.u)
        
        # self.flow = d3.GlobalFlowProperty(self.solver, cadence=10)
        # self.flow.add_property(1 + d3.dot(self.b*self.u,ez)/kappa,name='Nu')
        
    def initialize(self):
        self.time = 0
        self.tVals = []
        self.NuVals = []
        
        ##making Bases
        self.coords = d3.CartesianCoordinates('x','z')
        self.dist = d3.Distributor(self.coords, dtype=dtype)
        self.xbasis = d3.RealFourier(self.coords['x'], size=self.Nx, bounds=(-1*np.pi/self.alpha, np.pi/self.alpha), dealias=dealias)
        self.zbasis = d3.ChebyshevT(self.coords['z'], size=self.Nz, bounds=(-1, 1), dealias=dealias)
        
        ##making fields
        p = self.dist.Field(name='p', bases=(self.xbasis,self.zbasis))
        b = self.dist.Field(name='b', bases=(self.xbasis,self.zbasis))
        u = self.dist.VectorField(self.coords, name='u', bases=(self.xbasis,self.zbasis))
        tau_p = self.dist.Field(name='tau_p')
        tau_b1 = self.dist.Field(name='tau_b1', bases=self.xbasis)
        tau_b2 = self.dist.Field(name='tau_b2', bases=self.xbasis)
        tau_u1 = self.dist.VectorField(self.coords, name='tau_u1', bases=self.xbasis)
        tau_u2 = self.dist.VectorField(self.coords, name='tau_u2', bases=self.xbasis)
        
        self.p = p
        self.b = b
        self.u = u
        self.tau_p = tau_p
        self.tau_b1 = tau_b1
        self.tau_b2 = tau_b2
        self.tau_u1 = tau_u1
        self.tau_u2 = tau_u2
    
            
        
        # Substitutions
        Ra = self.Ra
        Pr = self.Pr
        kappa = 4*(self.Ra * self.Pr)**(-1/2)
        nu = 4*(self.Ra / self.Pr)**(-1/2)
        self.x, self.z = self.dist.local_grids(self.xbasis, self.zbasis)
        ex, ez = self.coords.unit_vector_fields(self.dist)
        lift_basis = self.zbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        grad_u = d3.grad(self.u) + ez*lift(self.tau_u1) # First-order reduction
        grad_b = d3.grad(self.b) + ez*lift(self.tau_b1) # First-order reduction
        
        ##defining problem
        self.problem = d3.IVP([self.p, self.b, self.u, self.tau_p, self.tau_b1, self.tau_b2, self.tau_u1, self.tau_u2], namespace=locals())
        
        if self.bcs == 'RB1':
            self.problem.add_equation("trace(grad_u) + tau_p = 0")
            self.problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - dot(u,grad(b))")
            self.problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - dot(u,grad(u))")
            self.problem.add_equation("b(z=-1) = 1")
            self.problem.add_equation("u(z=-1) = 0")
            self.problem.add_equation("b(z=1) = -1")
            self.problem.add_equation("u(z=1) = 0")
        if self.bcs == 'IH1':
            Ra = self.Ra
            Pr = self.Pr
            self.problem.add_equation("trace(grad_u) + tau_p = 0")
            self.problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2) = - dot(u,grad(b))+1")
            self.problem.add_equation("dt(u) - Pr*div(grad_u) + grad(p) - Pr*Ra*b*ez + lift(tau_u2) = - dot(u,grad(u))")
            self.problem.add_equation("b(z=-1) = 0")
            self.problem.add_equation("b(z=1) = 0")
            self.problem.add_equation("u(z=1) = 0")
            self.problem.add_equation("u(z=-1) = 0")
        self.problem.add_equation("integ(p) = 0") # Pressure gauge
        
        ##put in initial conditions if given
        if self.init_u is not None:
            #self.u['g'] = self.init_u
            #self.u.data = self.init_u
            self.u.load_from_global_grid_data(self.init_u)
        
        if self.init_b is not None:
            #self.b['g'] = self.init_b
            #self.b.set_global_data(self.init_b)
            self.b.load_from_global_grid_data(self.init_b)
            
        if self.init_p is not None:
            #self.p['g'] = self.init_p
            #self.p.set_global_data(self.init_p)
            self.p.load_from_global_grid_data(self.init_p)
        
        #if no initial conditions given, use a perturbed conduction state
        if self.init_u is None and self.init_b is None:
            self.b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
            self.b['g'] *= (1+self.z) * (1 - self.z) # Damp noise at walls
            self.b['g'] += self.conduction_state() # Add appropriate conduction state
            self.init_u = np.copy(u['g'])
            self.init_b = np.copy(b['g'])
            self.init_p = np.copy(p['g'])
        
        
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
        self.CFL.add_velocity(self.u)
        
        self.flow = d3.GlobalFlowProperty(self.solver, cadence=10)
        self.flow.add_property(1 + d3.dot(self.b*self.u,ez)/kappa,name='Nu')
        
        
    def conduction_state(self):
        if self.bcs == 'RB1':
            return -1*self.z
        elif self.bcs == 'IH1':
            return (1+self.z)*(self.z-1)
        
    def reset(self):
        #self.u.change_scales(1)
        #self.b.change_scales(1)
        self.u.load_from_global_grid_data(self.init_u)
        self.b.load_from_global_grid_data(self.init_b)
        self.p.load_from_global_grid_data(self.init_p)
        self.time_step = self.init_dt
        self.time = 0
        self.tVals = []
        self.NuVals = []
        
    
    def solve_system(self,end_time,trackNu=None):
        assert(end_time > self.time)
        self.solver.sim_time = self.time
        self.solver.stop_sim_time = end_time
        
        first_step = True
        try:
            logger.info('Starting main loop')
            while self.solver.proceed:
                if self.time_step != None and first_step:
                    timestep = self.time_step
                else:
                    timestep = self.CFL.compute_timestep()
                self.solver.step(timestep)
                if (self.solver.iteration-1) % 10 == 0:
                    flow_Nu = self.flow.volume_integral('Nu')/self.volume
                    if trackNu:
                         self.tVals.append(self.solver.sim_time)
                         self.NuVals.append(flow_Nu)
                    logger.info('Iteration=%i, Time=%e, dt=%e, Nu=%f' %(self.solver.iteration, self.solver.sim_time, timestep,flow_Nu))
        except:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.solver.log_stats()
        self.time_step = timestep
        self.time = end_time
        # if save:
        #     path = 'Outputs_restart/Ra_' + str(Ra) + '/Pr_' + str(Pr)  +  '/' + 'alpha_' + str(alpha) + '/' + 'Nx_' + str(Nx) + '/' + 'Nz_' + str(Nz) + '/T_' + str(T) + '/'
        #     try:
        #         os.makedirs(path)
        #     except FileExistsError:
        #         pass
        #     file_name = path + 'Nu_data.npy'
        #     with open(file_name,'wb') as nu_file:
        #         np.save(nu_file,np.array(tVals)) 
        #         np.save(nu_file,np.array(Nu_vals))
        
    
    def plot(self):
        X,Z = np.meshgrid(self.x.ravel(),self.z.ravel())
        
        self.u.change_scales(1)
        self.b.change_scales(1)
        
        uArr = self.u['g']
        bArr = self.b['g']
        
        ##just for coolness, cmap='coolwarm' also looks alright
        
        title = 'Ra: ' + str(self.Ra) + '  Pr: ' + str(self.Pr) + "  alpha: " + str(self.alpha)
        
        fig, axs = plt.subplots(2, 2)
        p1 = axs[0,0].pcolormesh(X.T,Z.T,bArr,cmap='seismic')
        axs[0,0].quiver(X.T,Z.T,uArr[0,:,:],uArr[1,:,:])
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
        if path == None:
            path = 'Outputs/Ra_' + str(self.Ra) + '/Pr_' + str(self.Pr) +   '/' + 'alpha_' + str(self.alpha) + '/' + 'Nx_' + str(self.Nx) + '/' + 'Nz_' + str(self.Nz) + '/T_' + str(time) + '/'
        
        # uFile = path + 'u.npy'
        # bFile = path + 'b.npy'
        # pFile = path + 'p.npy' 
        
        # uFromFile = np.load(uFile)
        # bFromFile = np.load(bFile)
        # pFromFile = np.load(pFile)
        
        loadFile = path + '.npy'
        
        with open(loadFile,'rb') as l_File:
            uFromFile = np.load(l_File)
            bFromFile = np.load(l_File)
            pFromFile = np.load(l_File)
            dt = np.load(l_File)
        
        self.u['g'] = uFromFile
        self.b['g'] = bFromFile
        self.p['g'] = pFromFile
        self.time_step = dt
    
    def saveToFile(self,path=None):
        # self.u.change_scales(1)
        # self.b.change_scales(1)
        # self.p.change_scales(1)
        
        # if path == None:
        #     path = 'Outputs/Ra_' + str(self.Ra) + '/Pr_' + str(self.Pr)  +  '/' + 'alpha_' + str(self.alpha) + '/' + 'Nx_' + str(self.Nx) + '/' + 'Nz_' + str(self.Nz) + '/T_' + str(self.time) + '/'
        # try:
        #     os.makedirs(path)
        # except FileExistsError:
        #     pass
        # uFile = path + 'u.npy'
        # bFile = path + 'b.npy'
        # pFile = path + 'p.npy' 
        
        # with open(uFile,'wb') as u_file:
        #     np.save(u_file, self.u['g'])
            
        # with open(bFile,'wb') as b_file:
        #     np.save(b_file, self.b['g'])
        
        # with open(pFile,'wb') as p_file:
        #     np.save(p_file, self.p['g'])
        
        self.u.change_scales(1)
        self.b.change_scales(1)
        self.p.change_scales(1)
        
        if path == None:
            path = 'Outputs/Ra_' + str(self.Ra) + '/Pr_' + str(self.Pr)  +  '/' + 'alpha_' + str(self.alpha) + '/' + 'Nx_' + str(self.Nx) + '/' + 'Nz_' + str(self.Nz) + '/T_' + str(self.time) + '/'
            try:
                os.makedirs(path)
            except FileExistsError:
                ans = input('File already exists, you could possibly overrwrite! Do you want to proceed? (y/n)')
                if ans == 'y':
                    pass
                else:
                    raise FileExistsError
            
        outputFile = path + '.npy'
        with open(outputFile,'wb') as outFile:
            np.save(outFile,self.u['g'])
            np.save(outFile,self.b['g'])
            np.save(outFile,self.p['g'])
            np.save(outFile,self.time_step)
            
        
    def __repr__(self):
        repr_string = "RBC Problem in configuration " + self.bcs + " with following params: \n" + "Ra= " + str(self.Ra) + "\n" + "Pr= " + str(self.Pr) + "\n" + "alpha= " + str(self.alpha) + "\n"
        repr_string = repr_string + "\n" + "Simulation params: \n" + "Nx= " + str(self.Nx) + "\n" + "Nz= " + str(self.Nz)
        return repr_string

######################################
### to load arrays from .npy files ###
######################################

def open_fields(file_name):
    with open(file_name,'rb') as testFile1:
        uTest1 = np.load(testFile1)
        bTest1 = np.load(testFile1)
        pTest1 = np.load(testFile1)
        dt1 = np.load(testFile1)
    return uTest1,bTest1,pTest1,dt1


#################################
### testing the time marching ###
#################################

def problems_close(rbc1,rbc2,tol):
    assert(rbc1.u['g'].shape == rbc2.u['g'].shape)
    assert(rbc1.b['g'].shape == rbc2.b['g'].shape)
    assert(rbc1.p['g'].shape == rbc2.p['g'].shape)
    
    if np.allclose(rbc1.u['g'].all(),rbc2.u['g'].all(),atol=tol) and np.isclose(rbc1.b['g'].all(),rbc2.b['g'].all(),atol=tol) and np.isclose(rbc1.p['g'].all(),rbc2.p['g'].all(),atol=tol):
        return True
    return False

def rbc_test():
    ##############
    ### TEST 1 ###
    ##############
    
    print("Return to Conduction state test. Ra = 1000, Pr = 1")
    test_problem1 = RBC_Problem(1000,1,1.5585,256,128,'RB1')
    test_problem1.initialize()
    test_problem1.solve_system(30,True)
    test_problem1.plot()
    assert(np.allclose(test_problem1.b['g'].all(),test_problem1.conduction_state().all(),atol=1e-12))
    print("test 1 passed")
    print("-------------")
    
    ##############
    ### TEST 2 ###
    ##############
    
    print("Return to Conduction state test. Ra = 1600, Pr = 1")
    test_problem2 = RBC_Problem(1600,1,1.5585,256,128,'RB1')
    test_problem2.initialize()
    test_problem2.solve_system(30,True)
    #test_problem2.plot()
    assert(np.allclose(test_problem2.b['g'].all(),test_problem2.conduction_state().all(),atol=1e-12))
    print("test 2 passed")
    print("-------------")
    
    ##############
    ### TEST 3 ###
    ##############
    
    print("Return to Conduction state test. Ra = 1700, Pr = 100")
    test_problem3 = RBC_Problem(1700,100,1.5585,256,128,'RB1')
    test_problem3.initialize()
    test_problem3.solve_system(30,True)
    #test_problem3.plot()
    assert(np.allclose(test_problem3.b['g'].all(),test_problem3.conduction_state().all(),atol=1e-12))
    print("test 3 passed")
    print("-------------")
    
    ##############
    ### TEST 4 ###
    ##############
    
    print("Reference Output 1 Test. Ra = 80000, Pr = 1")
    test_problem4 = RBC_Problem(80000,1,1.5585,256,128,'RB1')
    test_problem4.initialize()
    test_problem4.solve_system(30,True)
    
    with open('test_files/Ra80000Pr1alpha1.5585Nx256Nz128T30.npy','rb') as testFile1:
        uTest1 = np.load(testFile1)
        bTest1 = np.load(testFile1)
        pTest1 = np.load(testFile1)
        dt1 = np.load(testFile1)
    
    assert(np.allclose(test_problem4.u['g'].all(),uTest1.all(),atol=1e-12))
    assert(np.allclose(test_problem4.b['g'].all(),bTest1.all(),atol=1e-12))
    assert(np.allclose(test_problem4.p['g'].all(),pTest1.all(),atol=1e-12))
    
    print("test 4 passed")
    print("-------------")
    
    ##############
    ### TEST 5 ###
    ##############
    
    print("Reference Output 2 Test. Ra = 80000, Pr = 100")
    test_problem5 = RBC_Problem(80000,100,2,128,64,'RB1')
    test_problem5.initialize()
    test_problem5.solve_system(100,True)
    
    with open('test_files/Ra80000Pr100alpha2Nx128Nz64T100.npy','rb') as testFile2:
        uTest2 = np.load(testFile2)
        bTest2 = np.load(testFile2)
        pTest2 = np.load(testFile2)
        dt2 = np.load(testFile2)
    
    assert(np.allclose(test_problem5.u['g'].all(),uTest2.all(),atol=1e-12))
    assert(np.allclose(test_problem5.b['g'].all(),bTest2.all(),atol=1e-12))
    assert(np.allclose(test_problem5.p['g'].all(),pTest2.all(),atol=1e-12))
    
    print("test 5 passed")
    print("-------------")

    ##############
    ### TEST 6 ###
    ##############

    print("Loading From File Test. Ra=80000, Pr = 100")
    
    test_problem6 = RBC_Problem(80000,100,2,128,64,'RB1',dt1,uTest1,bTest1,pTest1)
    test_problem6.initialize()
    test_problem6.solve_system(20,True)

    assert(np.allclose(test_problem6.u['g'].all(),uTest2.all(),atol=1e-12))
    assert(np.allclose(test_problem6.b['g'].all(),bTest2.all(),atol=1e-12))
    assert(np.allclose(test_problem6.p['g'].all(),pTest2.all(),atol=1e-12))

    print("test 6 passed")
    print("-------------") 

#########################
### optimization part ###
#########################

def probToStateVec(RBCProb):
    Nx = RBCProb.Nx
    Nz = RBCProb.Nz
    
    RBCProb.u.change_scales(1)
    RBCProb.b.change_scales(1)
    RBCProb.p.change_scales(1)
    uArr = RBCProb.u.allgather_data('g')
    bArr = RBCProb.b.allgather_data('g')
    pArr = RBCProb.p.allgather_data('g')

    X = np.zeros(4*Nx*Nz)
    uArr1 = uArr[0,:,:].flatten()
    uArr2 = uArr[1,:,:].flatten()
    bArr = bArr.flatten()
    pArr = pArr.flatten()
    X[0:Nx*Nz] = uArr1
    X[Nx*Nz:2*Nx*Nz] = uArr2
    X[2*Nx*Nz:3*Nx*Nz] = bArr
    X[3*Nx*Nz:] = pArr
    return X

def arrsToStateVec(uArr,bArr,pArr):
    Nx,Nz = bArr.shape
    X = np.zeros(4*Nx*Nz)
    uArr1 = uArr[0,:,:].flatten()
    uArr2 = uArr[1,:,:].flatten()
    bArr = bArr.flatten()
    pArr = pArr.flatten()
    X[0:Nx*Nz] = uArr1
    X[Nx*Nz:2*Nx*Nz] = uArr2
    X[2*Nx*Nz:3*Nx*Nz] = bArr
    X[3*Nx*Nz:] = pArr
    return X

def stateToArrs(X,Nx,Nz):
    uArr1 = X[0:Nx*Nz]
    uArr2 = X[Nx*Nz:2*Nx*Nz]
    bArr = X[2*Nx*Nz:3*Nx*Nz]
    pArr = X[3*Nx*Nz:]
    
    #get them into a format we can put into the flow map function
    uArr1 = np.reshape(uArr1,(-1,Nz))
    uArr2 = np.reshape(uArr2,(-1,Nz))
    uArr = np.zeros((2,Nx,Nz))
    
    uArr[0,:,:] = uArr1
    uArr[1,:,:] = uArr2
    bArr = np.reshape(bArr,(-1,Nz))
    pArr = np.reshape(pArr,(-1,Nz))
    
    return uArr,bArr, pArr

def Gt(X,T,problem):
    uArr,bArr,pArr = stateToArrs(X,problem.Nx,problem.Nz)
    problem.time = 0
    problem.u.load_from_global_grid_data(uArr)
    problem.b.load_from_global_grid_data(bArr)
    problem.p.load_from_global_grid_data(pArr)
    problem.solve_system(T)
    Gt_Vec = probToStateVec(problem)
    Gt_Vec = (Gt_Vec - X)/T
    return Gt_Vec
    
def jac_approx(X,dX,F,T,problem):
    #mach_eps = np.finfo(float).eps
    #normX = np.linalg.norm(X)
    #normdX = np.linalg.norm(dX)
    #dotprod = np.dot(X,dX)
    #eps = (np.sqrt(mach_eps)/(normdX**2))*max(dotprod,normdX)
    eps = 1e-3
    return (Gt(X+eps*dX,T,problem).T + F.T)/eps


def optimization(problem,guess,T,tol,max_iters,write):
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
            #print("iter: ",iters)
            #print(X)
            #print("-------------")
            pass

        F = -1*Gt(X,T,problem)
        A = lambda dX : jac_approx(X,dX,F,T,problem)
        A_mat = LinearOperator((4*Nx*Nz,4*Nx*Nz),A)
        delta_X,code =gmres(A_mat,F)
        if code != 0:
            raise("gmres did not converge")
        X= X+delta_X
        iters += 1
        err = np.linalg.norm(Gt(X,T,problem))
    uStead,bStead,pStead = stateToArrs(X,Nx,Nz)
    problem.u.load_from_global_grid_data(uStead)
    problem.b.load_from_global_grid_data(bStead)
    problem.p.load_from_global_grid_data(pStead)
    return iters

    
    

#######################################
### testing the array manipulations ###
#######################################

def test_Gt():
    with open('test_files/Ra80000Pr100alpha2Nx128Nz64T100.npy','rb') as testInit:
        uTest1 = np.load(testInit)
        bTest1 = np.load(testInit)
        pTest1 = np.load(testInit)
        dt1 = np.load(testInit)
    Nx = 128
    Nz = 64
    test_problem1 = RBC_Problem(80000,100,2,Nx,Nz,'RB1',dt1,uTest1,bTest1,pTest1)
    test_problem1.initialize()
    X = arrsToStateVec(uTest1, bTest1, pTest1)
    T = 20
    Gt_Vec = Gt(X,T,test_problem1)
    print(Gt_Vec.shape)
    state_T = Gt_Vec*T + X
    u_T,b_T,p_T = stateToArrs(state_T, Nx, Nz)
    
    with open('test_files/Ra80000Pr100alpha2Nx128Nz64T120.npy','rb') as testFinal:
        uFinal1 = np.load(testFinal)
        bFinal1 = np.load(testFinal)
        pFinal1 = np.load(testFinal)
        dtFinal1 = np.load(testFinal)
    
    assert(np.allclose(u_T,uFinal1,atol=1e-12))
    assert(np.allclose(b_T,bFinal1,atol=1e-12))
    assert(np.allclose(p_T,pFinal1,atol=1e-12))
    
    print('test 1 passed')
    print("-------------")


def testing_allgather_data():
    uTest1,bTest1,pTest1,dt1 = open_fields('test_files/Ra80000Pr100alpha2Nx128Nz64T100.npy')
    testProb = RBC_Problem(80000,100,2,128,64,'RB1',dt1,uTest1,bTest1,pTest1)
    testProb.initialize()
    testProb.u.change_scales(1)
    uStuff = testProb.u.allgather_data('g')
    assert(np.allclose(uStuff,uTest1,atol=1e-12))
    print("test passed!")


def test_array_manipulations():
    with open('test_files/Ra80000Pr100alpha2Nx128Nz64T100.npy','rb') as testFile1:
        uTest1 = np.load(testFile1)
        bTest1 = np.load(testFile1)
        pTest1 = np.load(testFile1)
        dt1 = np.load(testFile1)
    Nx = 128
    Nz = 64
    test_problem1 = RBC_Problem(80000,100,2,Nx,Nz,'RB1',dt1,uTest1,bTest1,pTest1)
    test_problem1.initialize()
    X1 = probToStateVec(test_problem1)
    uArr1,bArr1,pArr1 = stateToArrs(X1,Nx,Nz)
    assert(np.allclose(bTest1-bArr1,np.zeros((Nx,Nz)),atol=1e-12))
    assert(np.allclose(pTest1-pArr1,np.zeros((Nx,Nz)),atol=1e-12))
    assert(np.allclose(uTest1 - uArr1, np.zeros((2,Nx,Nz)),atol=1e-12))
    print("manipulation test 1 passed")
    print("-------------")
    
    test_problem1.solve_system(20)
    test_problem1.u.change_scales(1)
    test_problem1.b.change_scales(1)
    test_problem1.p.change_scales(1)
    
    uTest2 = test_problem1.u.allgather_data('g')
    bTest2 = test_problem1.b.allgather_data('g')
    pTest2 = test_problem1.p.allgather_data('g')
    
    X2 = probToStateVec(test_problem1)
    uArr2,bArr2,pArr2 = stateToArrs(X2,Nx,Nz)
    assert(np.allclose(bTest2-bArr2,np.zeros((Nx,Nz)),atol=1e-12))
    assert(np.allclose(pTest2-pArr2,np.zeros((Nx,Nz)),atol=1e-12))
    assert(np.allclose(uTest2 - uArr2, np.zeros((2,Nx,Nz)),atol=1e-12))
    print("manipulation test 2 passed")
    print("-------------")

############################
### Optimization Testing ###
############################

def test_optimization():
    Nx = 128
    Nz = 64
    print("STARTING TEST 1")
    testProb = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    testProb.initialize()
    uArr,bArr,pArr,dt = open_fields("test_files/optim_test/Ra5000Pr100alpha1.5585Nx128Nz64T1000.npy")
    guess = arrsToStateVec(uArr, bArr, pArr)
    iters = optimization(testProb,guess,2,1e-3,10,True)
    
    print("Number of iterations:")
    print(iters)
    #testProb.plot()
    #testProb.saveToFile('opti_output_remote/opti_test1')
    print("Error:")
    err1 = np.max(abs(testProb.u.allgather_data()-uArr))
    print(err1)
    assert(err1 < 1e-12)
    print("test 1 over")
    
    print("STARTING TEST 2")
    testProb2 = RBC_Problem(5000,100,1.5585,Nx,Nz,'RB1')
    testProb2.initialize()
    uArr,bArr,pArr,dt = open_fields("test_files/optim_test/Ra5000Pr100alpha1.5585Nx128Nz64T500.npy")
    guess = arrsToStateVec(uArr, bArr, pArr)
    iters = optimization(testProb,guess,2,1e-3,10,True)
    
    print("Number of iterations:")
    print(iters)
    testProb2.plot()
    #testProb.saveToFile('opti_output_remote/opti_test1')
    print("Error:")
    err2 = np.max(abs(testProb2.u.allgather_data()-uArr))
    print(err2)
    return testProb2
    
    

###################################
### run every test successively ###
###################################

def test_all():
    rbc_test()
    print("passed rbc_test")
    print("-------------")
    test_array_manipulations()
    print("passed array manipulation test")
    print("-------------")
    test_Gt()
    print("passed Gt Test")
    print("-------------")

    print("wow congrats you actually passed everything for once buddy")

#test_all()
#test_rbc()
#test_Gt()
#test_array_manipulations()      
#testing_functions()        
#test_optimization()

#IH_test = RBC_Problem(5000,100,1.5585,1024,512,'IH1')
#IH_test.initialize()
#IH_test.solve_system(0.01)
#IH_test.saveToFile('IH_output/Ra5000Pr100alpha1.5585Nx1024Nz512')
