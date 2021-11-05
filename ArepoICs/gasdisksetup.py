import numpy as np
import h5py
import random
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.integrate import simps,quad
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from astropy.constants import G as Gconst
from astropy.constants import M_sun,pc,kpc

from ..utils.hcpmesh import background_mesh,active_mesh
from ..utils.gasproperties import Temperature_to_IsoCs

# Used for generating gas disk in vertical equalibrium
# reference: https://bitbucket.org/tepper/gas_disc_ics/src 
 

class ArepoGasDisk(object):
  
  def __init__(self,SurfDenfunc,R0,Sigma0,GasTemp,Potfunc,Unitbase):

    # Input: R0          should be in [kpc], 
    #        Sigma0      should be in [Msun/pc^2]
    #        GasTemp     shoule be in [K]

    # define units (should be in cgs units)   
 
    self.unitL = Unitbase['unitL']
    self.unitM = Unitbase['unitM']
    self.unitV = np.sqrt(Gconst.cgs.value*self.unitM/(self.unitL))
    self.unitE = self.unitV**2
    self.unitG = 1.0 

    # define paramters
    
    self.Sigma0 = Sigma0*(M_sun.cgs.value/pc.cgs.value**2)/(self.unitM/self.unitL**2)
    self.R0     = R0*kpc.cgs.value/self.unitL
    self.cs     = Temperature_to_IsoCs(GasTemp)/self.unitV
    self.z0     = self.cs**2/(2*np.pi*self.unitG*self.Sigma0) 

    # define function

    self.SurfDenfunc = SurfDenfunc
    self.Potfunc     = Potfunc

    # print basic information
    print("------------------------")
    print("------------------------")


  def PotDiffOverCs(self,Rp,zp):

      Rad = Rp*self.R0
      zad = zp*self.z0

      return -(self.Potfunc(Rad,0.0,zad)-self.Potfunc(Rad,0.0,0.))/self.cs**2

  def poisson_derivs(self,zp,W,Rp):

      dW = np.zeros_like(W)

      dW[0] = W[1]
      dW[1] = -1.*W[2]*(self.SurfDenfunc(Rp)/self.sfDensNorm[1])*np.exp(self.PotDiffOverCs(Rp,zp))
      dW[2] = W[1]*W[2]

      return dW

  def density_derivs(self,zp,Den,Rp):

      W1 = self.W1zp(zp)
      W3 = np.exp(W1)

      dDen = W3*np.exp(self.PotDiffOverCs(Rp,zp)) if (W3 >=0.0) else 0.0

      return [dDen]  


  def GasDensityfunc(self):

    Rp_array = np.linspace(0., 20.*self.R0,1001)
    zp_array = np.linspace(0.,100.*self.z0,1001)
    Denp     = np.zeros([1001,1001])

    for i in range(len(Rp_array)):

      Rp = Rp_array[i]
      self.sfDensNorm = np.array([0.0,1.0])

      count = 0

      while (np.abs(self.sfDensNorm[1]-self.sfDensNorm[0])/self.sfDensNorm[1] > 1.0e-3):

        if (count <= 20):

           self.sfDensNorm[0] = self.sfDensNorm[1]

           Winitial = np.array([0.0,0.0,1.0])
           sol1 = solve_ivp(lambda zp, W : self.poisson_derivs(zp,W,Rp), [0,100.*self.z0], Winitial, \
                            method='Radau', dense_output=True)

           Wsol = sol1.sol(zp_array)

           self.W1zp = InterpolatedUnivariateSpline(zp_array, Wsol[0,:])
           self.W2zp = InterpolatedUnivariateSpline(zp_array, Wsol[1,:])
           self.W3zp = InterpolatedUnivariateSpline(zp_array, Wsol[2,:])

           sol2 = solve_ivp(lambda zp, Den: self.density_derivs(zp,Den,Rp), [0,100.*self.z0], [0.0], \
                            method='Radau', dense_output=True)
           self.sfDensNorm[1] = sol2.y[0][-1]

           count = count + 1

        else:

           print('Warning: Maximum number of iterations reached (without convergence)!\n')
           print('Rp =',Rp,'Difference = ',np.abs(sfDensNorm[1]-sfDensNorm[0])/sfDensNorm[1],' \n')
           break

        Denp[i,:] = self.SurfDenfunc(Rp)/self.sfDensNorm[1]\
                *np.exp(self.W1zp(zp_array))*np.exp(self.PotDiffOverCs(Rp,zp_array))

    Den_grid = Denp*self.Sigma0/(2.*self.z0)

    Rad_array = Rp_array*self.R0
    zad_array = zp_array*self.z0

    self.totalM = 2.*simps(np.array([simps(den,zad_array) for den in Den_grid])*2*np.pi*Rad_array,Rad_array)

    return Densityfunc = rgi((Rad_array,zad_array),Den_grid)

  # -------------------------
  # velocity setup
  # -------------------------

  def innerIntRp(self,a):
      innerInt = lambda w,a: self.SurfDenfunc(np.sqrt(w**2+a**2))
      return quad(innerInt,0,np.inf,args=(a,),epsabs = 1e-4, limit=100)[0]

  def outerIntA(self,Rp):
      outerInt = lambda theta: -4.*self.unitG*np.pi/2.*self.a_func(Rp*np.sin(0.5*np.pi*theta))
      return quad(outerInt,0.0,1.0,epsabs = 1e-4, limit=100)[0]


  def GasVelocityfunc(self):

    Rbin = 1001
    Radarrayedge = np.linspace(0.,20.*self.R0,Rbin+1)
    Radarray     = [(Radarrayedge[i]+Radarrayedge[i+1])/2. for i in range(len(Radarrayedge)-1)]
    deltaR       = Radarray[1]-Radarray[0]

    # external potential
    # azmuthally averaged:

    self.Potfunc(Rad,0.0,zad)

    Potext_array = np.zeros_like(Radarray)


    # potential due to gas disk
    # (Eq.39 of Cuddeford 1993):

    a_array = np.linspace(0.,100.*self.R0,1001)
    innerInta_array = np.array([innerIntRp(a0) for a0 in a_array])
    self.a_func = InterpolatedUnivariateSpline(a_array,innerInta_array)
    gaspot_tmparray = np.array([outerIntA(Rp) for Rp in Rp_array])
    gaspot_func = InterpolatedUnivariateSpline(Rp_array*R0,gaspot_tmparray*Sigma0*R0)

    Potgas_array = gaspot_func(Radarray)
    Potall_array = Potext_array+Potgas_array

    # calculate rotation curve 

    Vc_smooth = np.sqrt(np.gradient(Potall_array)/(deltaR)*Radarray)
    Vc_smooth_filter = savgol_filter(Vc_smooth, 15, 1)

    Radarray = np.insert(Radarray,0,1.0e-6)
    Vc_smooth_filter = np.insert(Vc_smooth_filter,0,0.0)

    return rotcurve_func = interp1d(Radarray,Vc_smooth_filter,kind='linear',bounds_error=False,fill_value=0.0)





