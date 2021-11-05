import numpy as np

# generate hexagon close-packed (hcp) particle distributions
# used for creating Arepo initial condition 


def hcp_mesh(n,boxsize):

#  input: number of particles and the boxsize (dimensionless)
#         n is the number of paticles along x-axis 
#         boxsize centered at (0,0,0)
# output: particle positions 

    n = int(n)
    boxsize = float(boxsize)

    dim = 3

    xcenter = 0.0
    ycenter = 0.0
    zcenter = 0.0

    nx = n
    delta = float(boxsize/nx)
    deltax = delta
    deltay = np.sqrt(3.)/2.*delta
    deltaz = np.sqrt(6.)/3.*delta

    ny = int(boxsize/deltay)+1
    nz = int(boxsize/deltaz)+1

    ny = 2*int(ny/2)
    nz = 3*int(nz/3)

    deltay = boxsize/ny
    deltaz = boxsize/nz

    nmax = np.max([nx,ny,nz])

    k, j, i = [v.flatten()
              for v in np.meshgrid(*([range(nmax)] * dim), indexing='ij')]

    x = (2. * i + (j + k) % 2)*delta/2 + 0.5*deltax
    y = (2  * deltay * (j + 1/3 * (k % 3)))/2 + 0.5*deltay
    z = (2. * deltaz * k)/2 + 0.5*deltaz

    idx = np.where((x<boxsize) & (y<boxsize) & (z<boxsize))

    x = x[idx]
    y = y[idx]
    z = z[idx]

    return np.c_[x-boxsize/2,y-boxsize/2,z-boxsize/2]


def background_mesh(gridsize,boxsize):

    return hcp_mesh(gridsize,boxsize)


def active_mesh(geometry,boxsize,nx,bound):

    if (geometry == 'cyl'):

      pos = hcp_mesh(nx,boxsize)
      Rad = np.sqrt(pos[:,0]**2+pos[:,1]**2)
      zad = pos[:,2]

      idx = np.where((Rad<bound[0]) & (np.abs(zad)<bound[1]))

      x = pos[idx[0],0]
      y = pos[idx[0],1]
      z = pos[idx[0],2]

      return np.c_[x,y,z]

