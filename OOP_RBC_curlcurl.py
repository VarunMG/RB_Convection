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
            self.b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
            self.b['g'] *= (1+self.z) * (1 - self.z) # Damp noise at walls
            # self.b['g'] += 0.01*np.cos((1/2)*np.pi*self.x)*np.sin(np.pi*self.z*self.alpha)
            self.b['g'] += self.conduction_state() # Add appropriate conduction state
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
        self.u.load_from_global_grid_data(self.init_u)
        self.b.load_from_global_grid_data(self.init_b)
        self.p.load_from_global_grid_data(self.init_p)
        self.time_step = self.init_dt
        self.time = 0
        self.tVals = []
        self.NuVals = []
        
    
    def solve_system(self,end_time,trackNu=None,anim_frames=False):
        assert(end_time > self.time)
        self.solver.sim_time = self.time
        self.solver.stop_sim_time = end_time
        
        if anim_frames:
            plt.figure()
            framecount = 0
            X,Z = np.meshgrid(self.x.ravel(),self.z.ravel())
        
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
                    logger.info('Iteration=%i, Time=%e, dt=%e, Nu=%f' %(self.solver.iteration, self.solver.sim_time, timestep,flow_Nu))
        except:
            logger.error('Exception raised, triggering end of main loop.')
            raise
        finally:
            self.solver.log_stats()
        self.time_step = timestep
        self.time = end_time
        
    
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