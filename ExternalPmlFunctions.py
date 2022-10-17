from ExternalFunctions import Mesh1D, Lobatto, LobattoDerivative
import numpy as np
def FT1(xi, rpml, k):
	a = 1/(1j*k*rpml)
	return np.log( (1-xi)*(1+a) / (1+a*(1-xi)) )

def Mesh1D_PML(DuctLength, NrOfElem, R0, Xin, NrOfElemPML, Lpml):
    NrOfNodes, Coord = Mesh1D(DuctLength, NrOfElem, R0)[0:2]
    NrOfNodesPML = NrOfElemPML+1
    CoordPML = np.linspace(Xin, Xin+Lpml, NrOfNodesPML)
    NrOfNodesT = NrOfNodes + NrOfNodesPML - 1
    CoordT = np.concatenate((Coord, CoordPML))
    CoordT = np.delete(CoordT, NrOfNodes)

    Element = np.zeros((2, NrOfNodesT-1))
    Element[0, :] = np.arange(0, NrOfNodesT-1)
    Element[1, :] = np.arange(1, NrOfNodesT)
    return NrOfNodesT, CoordT, Element

def PML(Coord, Xin, Lpml, k):
	import numpy as np
	xi = Lpml*Coord + Xin
	F = np.log(1-xi)
	rpml = Coord + F/(1j*k)
	F_prime = 1/(1-xi)*(1/Lpml)
	return rpml, F, F_prime

def MassAndStiffness_spherical_1D_PML(iElem, Order, Coord, Element, F, Fprime, delta, Rin, k):
  import numpy as np

  Ke = np.zeros((Order+1, Order+1), dtype=np.complex128)
  Ce = np.zeros((Order+1, Order+1), dtype=np.complex128)
  Me = np.zeros((Order+1, Order+1), dtype=np.complex128)
  Me_r2 = np.zeros((Order+1, Order+1), dtype=np.complex128)
  
  x1 = Coord[Element[0, iElem-1].astype(int)]
  x2 = Coord[Element[1, iElem-1].astype(int)]
  Le = (x2-x1)
  # quadrature rule (integration order 2*Order
  GaussPoints, GaussWeights = np.polynomial.legendre.leggauss(int(2*Order))
  NrOfGaussPoints = GaussWeights.size
  # Loop over the Gauss points
  for n in np.arange(0, NrOfGaussPoints):
      xi = GaussPoints[n]
      # High-order Lobatto shape functions
      L = Lobatto(xi, Order)
      dLdxi = LobattoDerivative(xi, Order)
      dLdx = 2/Le*dLdxi
      r = x1 + (Le/2)*(1+xi)
      if (x1 >= Rin):
        xi_pml = (r - Rin)/(delta)
        r_tilde = r + F(xi_pml)/(1j*k)
        gamma_r = 1. + Fprime(xi_pml)/(1j*k*delta)
      else: 
        r_tilde = r
        gamma_r = 1.
      # Elementary matrices
      Ke += GaussWeights[n]*np.outer(dLdx, dLdx)*Le/2*(1./gamma_r)
      Ce += GaussWeights[n]*(1/r_tilde)*np.outer(L, dLdx)*Le/2
      Me += GaussWeights[n]*np.outer(L, L)*Le/2*gamma_r
      Me_r2 += GaussWeights[n]*np.outer(L, L)*Le/2*( 1/(r_tilde**2) )*gamma_r

  return Ke, Ce, Me, Me_r2