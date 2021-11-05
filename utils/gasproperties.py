import numpy as np
from astropy.constants import k_B,m_p

# functions to calculate gas properties such as 
# equation of state, sound speed, entropy, etc. 



# ---------
# constants
# ---------

# mass fraction of hydrogen
# this is the default value in arepo

HYDROGEN_MASSFRAC = 0.76 



# ---------
# functions
# ---------

def mean_molecular_weight(Temperature):

# calculate mean moleculate weight of gas based on temperature
#  input: gas temperature
# output: mean molecular gas

    if (Temperature > 1.0e4):

       return 4. / (8. - 5. * (1. - HYDROGEN_MASSFRAC))

    else:

       return 4. / (1. + 3. * HYDROGEN_MASSFRAC)


def Temperature_to_IsoCs(Temperature):

# calculate gas sound speed based on temperature
# works for isothermal EoS
# input : gas temperature [K]
# output: sound speed [cm/s]
    
    gmw = mean_molecular_weight(Temperature)

    cs = np.sqrt((k_B.cgs.value*Temperature/(gmw*m_p.cgs.value))) 

    return cs 



