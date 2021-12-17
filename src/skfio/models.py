'''
 skfio
 Convert Slater-Koster files, DFTB+ <=> analytical representation
 
 Copyright (C) 2021 Ahmad W. Huran

 ahmad.houran@gmail.com


 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.'''

from math import factorial
from scipy.special import binom
import numpy as np
from .parsing import atomizeModel


def polynomial(xx, *cc, x0=0.0):
    """
    Evaluate polynomial of degree n or its derivatives.
    The polynomial is represented in the power basis:

         n
        ===
        \                   m
        /    cc[m] (xx - x0)
        ===
       m = 0
    
    Parameters:
    -----------
        xx : scalar or array-like of floats. 
             The points at which the function
             is evaluated.

        cc : scalar or array-like of shape(n+1) of floats.
             The coefficients vector defining the polynomial.
    
        x0: scalar float.
            The origin of the local power basis.
    
        order: scalar int.
               order of the derivative.


    Returns:
    -------
        array of shape (xx.size,)

    No type checks happen here.
    You are on your own.

    WARNING:
    Not intended for large degrees
    and differentiation orders.

    """
    cc = np.array(cc)
    size = cc.size


    iis = np.arange(size)[None,:]
    result = (xx - x0)[:,None]**iis
    result *= cc[None,:]
    result = result.sum(axis=1)

    return result 

def jacPolynomial(xx, *cc, x0=0.0):
    """
    Compute the Jacobian matrix of skfio.models.polynomial
    with respect to *cc.

    Returns :
    ---------
        array of shape (xx.size, coeff.size). J[ii, jj] is the 
        jacobian at xx[ii] wrt cc[jj].
    """
    cc = np.array(cc)
    size = cc.size
    iis = np.arange(1, size)[None,:]
    result = np.ones((xx.size, size), dtype=float)
    result[:,1:] = (xx - x0)[:,None]**iis

    return result

def rational(xx, *cc, pp=1):
    """
    Evaluate the rational function: 

                pp - 1
                 ___
                 ╲               ii
                 ╱    cc[ii]   xx
                 ‾‾‾
               ii = 0
    ────────────────────────────────────────
          nn - 1
           ___
           ╲           2     (jj - pp + 1)
    1  +   ╱     cc[jj]    xx
           ‾‾‾
         jj = pp


    
    Parameters:
    -----------
        xx : scalar or array-like of floats. 
             The points at which the function
             is evaluated.

        cc : scalar or array-like of shape(nn) of floats.
             The coefficients vector defining the rational function.
    
        pp: int.
            Number of coefficients defining the numerator.
            0 < pp < len(cc)
    

    Returns:
    -------
        array of shape (xx.size,)

    No type checks happen here.
    You are on your own.

    WARNING:
    Not intended for large degrees.
    """
    cc = np.array(cc)
    nn = cc.size
    assert pp > 0 and pp < nn

    ###
    #overriding pp by spliting the coefficients in half:
    pp = int(np.ceil(nn / 2))
    ###

    num_cc = cc[:pp]
    denom_cc = np.ones(nn - pp + 1)
    denom_cc[1:] = cc[pp:]

    return polynomial(xx, *num_cc) / polynomial(xx, *denom_cc) 
    
def jacRational(xx, *cc, pp=1):
    """Similar to jacPolynomial"""
    cc = np.array(cc)
    nn = cc.size
    assert pp > 0 and pp < nn

    ###
    #overriding pp by spliting the coefficients in half:
    pp = int(np.ceil(nn / 2))
    ###

    num_cc = cc[:pp]
    denom_cc = np.ones(nn - pp + 1)
    denom_cc[1:] = cc[pp:]

    num = polynomial(xx, *num_cc) 
    denom = polynomial(xx, *denom_cc)

    jac_num = jacPolynomial(xx, *num_cc)
    jac_denom = jacPolynomial(xx, *denom_cc)[:,1:]
    jac = np.zeros((xx.size, nn))
    jac[:,:pp] = jac_num/denom[:,None]
    jac[:,pp:] = -jac_denom*num[:,None]/denom[:,None]**2

    return jac


def rational2(xx, *cc, pp=1):
    cc = np.array(cc)
    nn = cc.size
    assert pp > 0 and pp < nn

    ###
    #overriding pp by spliting the coefficients in half:
    pp = int(np.ceil(nn / 2))
    ###

    iis = np.arange(pp)[None,:]
    result = xx[:,None]**iis
    result *= cc[None,:pp]
    result = result.sum(axis=1)

    jjs = np.arange(pp,nn)[None,:]
    result2 = xx[:,None]**(jjs - pp + 1)
    result2 *= cc[None,pp:]
    result2 = result2.sum(axis=1)
    return result / (1 + result2)

def exponential(xx, *param):
    """
    Evaluate linear combination of epxonential
    functions:

    n = len(param)//3
    cc = param[:n]
    aa = param[n:2n]
    x0 = param[2n:]

    n - 1
     ===
     \            -aa[m] |(xx - x0[m])|
     /     cc[m] e
     ===
    m = 0                         
    
    xx: scalar or array-like of floats.
        The points at which the function
        is evaluated.

    param: array-like of floats whose 
           length is a multiple of 3.
           The first third of the entries
           is the coefficients vector cc, 
           the second third is the exponents 
           vector aa, and the last third 
           is the vector x0 specifying the 
           respective centers of the 
           exponential functions.

    No type or size checks happen here.
    You are on your own.

    Return: array of shape (xx.size,)
    """
    param = np.array(param)
    nn = param.size
    assert nn > 0, "len(param) = 0"
    assert nn % 3 == 0, "len(param) % 3 != 0."
    nn = nn // 3
    cc = param[None,:nn]
    aa = param[None,nn:2*nn]
    x0 = param[None,2*nn:]
    res = (cc * np.exp(-aa * np.abs(xx[:,None]-x0))).sum(axis=1)
    return res

def gaussian(xx, *param):
    """
    Evaluate linear combination of gaussians.

    n = len(param)//3
    cc = param[:n]
    aa = param[n:2n]
    x0 = param[2n:]

    n - 1
     ===                2              2
     \            -aa[m]   (xx - x0[m])
     /     cc[m] e
     ===
    m = 0                         
    
    xx: scalar or array-like of floats.
        The points at which the function
        is evaluated.

    param: array-like of floats whose 
           length is a multiple of 3.
           The first third of the entries
           is the coefficients vector cc, 
           the second third is the exponents 
           vector aa, and the last third 
           is the vector x0 specifying the 
           respective centers of the 
           gaussians.

    No type or size checks happen here.
    You are on your own.

    Return: array of shape (xx.size,)
    """
    param = np.array(param)
    nn = param.size
    assert nn > 0, "len(param) = 0"
    assert nn % 3 == 0, "len(param) % 3 != 0."
    nn = nn // 3
    cc = param[None,:nn]
    aa = param[None,nn:2*nn]
    x0 = param[None,2*nn:]
    xx_x0 = xx[:,None] - x0
    xx_x02 = xx_x0 * xx_x0
    aa2 = aa * aa
    res = (cc * np.exp(-aa2 * xx_x02)).sum(axis=1)
    return res

def jacGaussian(xx, *param):
    """
    Compute the Jacobian matrix of skfio.models.gaussian
    with respect to x0 and *cc.

    Return: array of shape (xx.size, param.size).
            J[ii, jj] is the jacobian at xx[ii] wrt param[jj]
    """
    param = np.array(param)
    nn = param.size
    assert nn > 0, "len(param) = 0"
    assert nn % 3 == 0, "len(param) % 3 != 0."
    nn = nn // 3
    cc = param[None,:nn]
    aa = param[None,nn:2*nn]
    x0 = param[None,2*nn:]
    xx_x0 = xx[:,None] - x0
    xx_x02 = xx_x0 * xx_x0
    aa2 = aa * aa
    res = np.zeros((xx.size, param.size))

    res[:,:nn] = np.exp(-aa2 * xx_x02 )

    res[:,nn:2*nn] = (-2 * aa * cc 
                      * res[:,:nn]
                      * xx_x02 
                      )

    res[:,2*nn:] = (2 * aa2 * cc 
                    * res[:,:nn]
                    * xx_x0 
                    )

    return res


allModels = {
    "p": (polynomial, jacPolynomial),
    "px": (polynomial, "3-point"),
    "g": (gaussian, jacGaussian),
    "gx": (gaussian, "3-point"),
    "ex": (exponential, "3-point"),
    "rx": (rational, "3-point"),
    "r": (rational, jacRational),
        }

def modelFuncSum(functions, paramLengths):
    """
    Return model function from summing 
    functions; and the total number of
    parameters defining the new model. 

    functions: a list of callables with signature 
               (xx, *param). The model functions 
               to be summed.

    paramLengths: a list of int. The number of
             parameters defining the models to 
             be summed.

    """

    assert len(functions) >= 1
    assert len(functions) == len(paramLengths)

    if len(functions) == 1:
        return functions[0], paramLengths[0]

    nPtot = sum(paramLengths)
    def __sum(xx, *param):
        msg = "len(param) != sum(paramLengths), "\
              f"{len(param)} != {nPtot}"
        assert len(param) == nPtot, msg
        nP = 0
        res = np.zeros_like(xx)
        for nP_i, func_i in zip(paramLengths, functions):
            param_i = param[nP:nP+nP_i]
            if len(param_i) == 0:
                param_i = param[-nP_i:]
            nP += nP_i
            res += func_i(xx, *param_i)
        return res
    return __sum, nPtot

def modelJacSum(jacobians, paramLengths):
    """
    Return model Jacobian from summing 
    jacobians; and the total number of
    parameters defining the new Jacobian. 

    jacobians: a list of callables with signature 
               (xx, *param). The model Jacobians
               to be summed.

    paramLengths: a list of int. The number of
             parameters defining the models to 
             be summed.
             ***NOTE: if any of the entries in 
                      jacobians is not callable,
                      then the retained Jacobian
                      is a None object.
    """

    assert len(jacobians) >= 1
    assert len(jacobians) == len(paramLengths)

    if len(jacobians) == 1:
        return jacobians[0], paramLengths[0]

    nPtot = sum(paramLengths)
    if any([not callable(jac) for jac in jacobians]):
        return None, nPtot

    def __sum(xx, *param):
        msg = "len(param) != sum(paramLengths)"
        assert len(param) == nPtot, msg
        assert len(xx.shape) == 1
        nP = 0
        res = np.zeros((xx.size, nPtot))
        for nP_i, jac_i in zip(paramLengths, jacobians):
            param_i = param[nP:nP+nP_i]
            if len(param_i) == 0:
                param_i = param[-nP_i:]
                res[:,-nP_i:] = jac_i(xx, *param_i)
            else:
                res[:,nP: nP+nP_i] = jac_i(xx, *param_i)
            nP += nP_i
        return res
    return __sum, nPtot

def modelFuncProd(functions, paramLengths):
    """
    Return model function from the product
    of functions; and the total number of
    parameters defining the new model. 

    functions: a list of callables with signature 
               (xx, *param). The model functions 
               forming the product model function.

    paramLengths: a list of int. The number of
             parameters defining the models to 
             in the product.

    """

    assert len(functions) >= 1
    assert len(functions) == len(paramLengths)

    if len(functions) == 1:
        return functions[0], paramLengths[0]

    nPtot = sum(paramLengths)
    def __prod(xx, *param):
        msg = "len(param) != sum(paramLengths)"
        assert len(param) == nPtot, msg
        nP = 0
        res = np.ones_like(xx)
        for nP_i, func_i in zip(paramLengths, functions):
            param_i = param[nP:nP+nP_i]
            if len(param_i) == 0:
                param_i = param[-nP_i:]
            res *= func_i(xx, *param_i)
            nP += nP_i
        return res
    return __prod, nPtot

def modelJacProd(jacobians, functions, paramLengths):
    """
    Return model Jacobian from the product 
    jacobians; and the total number of
    parameters defining the new Jacobian. 

    jacobians: a list of callables with signature 
               (xx, *param). The Jacobians
               of the functions in the product.

    functions: a list of callables with signature 
               (xx, *param). The model functions
               in the product.

    paramLengths: a list of int. The number of
             parameters defining the models.

             ***NOTE: if any of the entries in 
                      jacobians is not callable,
                      then the retained Jacobian
                      is a None object.
    """

    assert len(jacobians) >= 1
    assert len(jacobians) == len(paramLengths)

    if len(jacobians) == 1:
        return jacobians[0], paramLengths[0]
    assert len(jacobians) == len(functions)

    nPtot = sum(paramLengths)
    if any([not callable(jac) for jac in jacobians]):
        return None, nPtot

    def __prod(xx, *param):
        msg = "len(param) != sum(paramLengths)"
        assert len(param) == nPtot, msg
        assert len(xx.shape) == 1
        nP = 0
        res = np.zeros((xx.size, nPtot))
        res_i = np.zeros((xx.size, len(functions)))
        zipped = zip(paramLengths, functions)
        for nn, (nP_i, func_i) in enumerate(zipped):
            param_i = param[nP:nP+nP_i]
            if len(param_i) == 0:
                param_i = param[-nP_i:]
            res_i[:, nn] = func_i(xx, *param_i)
            nP += nP_i
        nP = 0
        zipped = zip(paramLengths, jacobians)
        for nn, (nP_i, jac_i) in enumerate(zipped):
            param_i = param[nP:nP+nP_i]
            if len(param_i) == 0:
                param_i = param[-nP_i:]
                res[:,-nP_i:] = jac_i(xx, *param_i)
                res[:,-nP_i:] *= np.prod(
                                    res_i[:,:nn],
                                    axis=1,
                                    keepdims=True
                                    )
            elif nn == 0:
                res[:,nP: nP+nP_i] = jac_i(xx, *param_i)
                res[:,nP: nP+nP_i] *= np.prod(
                                    res_i[:,nn+1:],
                                    axis=1,
                                    keepdims=True
                                    )
            else:
                res[:,nP: nP+nP_i] = jac_i(xx, *param_i)
                res[:,nP: nP+nP_i] *= np.prod(
                                    res_i[:,:nn],
                                    axis=1,
                                    keepdims=True
                                    )
                res[:,nP: nP+nP_i] *= np.prod(
                                    res_i[:,nn+1:],
                                    axis=1,
                                    keepdims=True
                                    )
            nP += nP_i
        return res
    return __prod, nPtot

def makeModel(modelStr):
    """
    Return the model function the corresponding 
    Jacobian from the str modelStr.
    See skfio.parsing.atomizeModel for syntax.

    Since the supported operators in modelStr
    are * and + only, we can split at + after
    removing the white space, then construct 
    all the products, and finally get the sums.
    """

    #syntax check:
    _ = atomizeModel(modelStr)

    #now for real
    _str = "".join(modelStr.split())
    modelTerms = _str.split("+")

    #products
    funcFin = []
    jacFin = []
    paramLenFin = []
    for term in modelTerms:
        funcProd = []
        jacProd = []
        paramLen = []
        factors = term.split("*")
        for factor in factors:
            model_i, nParam_i = factor.split("_")
            func, jac = allModels[model_i.lower()]
            funcProd.append(func)
            jacProd.append(jac)
            paramLen.append(int(nParam_i))
        func, nP = modelFuncProd(funcProd, paramLen)
        jac, _ = modelJacProd(jacProd, funcProd, paramLen)
        funcFin.append(func)
        jacFin.append(jac)
        paramLenFin.append(nP)
    # now sum
    func, nPtot = modelFuncSum(funcFin, paramLenFin)
    jac, _ = modelJacSum(jacFin, paramLenFin)
    return func, jac, nPtot

