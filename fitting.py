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


import numpy as np
from scipy.optimize import curve_fit
from .models import allModels

class SK_Function:
    """
    Represent a Slater-Koster integral
    (one column in the *.skf) by some analytical
    representation.
    Just a wrapper around fitting and calling for
    bookkeeping purposes.
    """

    def __init__(self, param, cutoff, modelFunc, modelJac, err=None):
        """
        param: array-like, float.
               parameters defining the analytical
               fit model.

        cutoff: scalar, float.
                integral cutoff distance.

        modelFunc: callable signature (rr, *param)
                   Evaluate the model function at rr.

        modelJac: callable signature (rr, *param)
                  Evaluate the Jacobian of model 
                  function at rr.

        err : do not set explicitly, there's no point.
              max absoulte error in the fit.

           """
        self.param = param
        self.cutoff = cutoff
        self.func = modelFunc
        self.Jacobian = modelJacobian
        self.err = err

    @classmethod
    def fromData(cls, rr, ff, model, ir0=0, kwargs={}):
        """
        Carry out the fitting and store relevant data.
 
        rr: one-dimensional array-like, with elements 
            interprtable as floats.
            Inter-nuclear distance grid.
 
        ff: one-dimensional array-like, with elements 
            interprtable as floats.
            Integral values at rr.
 
        model: str, see examples in the models module.
               definition of the fit model.
               
        ir0: scalar, interpretable as int. 
             Determines the first point in 
             rr that enters the fit, 
             avoiding problems with 
             integrals padded with zeros at 
             inter-nuclear distances."""
        modelFunc, modelJac, nP = makeModel(model)

        pOpt, pCov, err = fit(rr, 
                              ff, 
                              nP, 
                              modelFunc, 
                              modelJac, 
                              ir0,
                              kwargs) 

        return cls(pOpt, rr[-1], modelFunc, modelJac, err)


    def __call__(self, rGrid, order=0, ir0=0, cutoff=None):
        """
        Evaluate the fitted model function and/or 
        its derivatives at the points rGrid.

        rGrid: scalar or array-like interpretable as floats.
               the point(s) at which to evaluate the model.
        order: scalar or array-like interpretabel as ints.
               orders of the required dervatives.
        ir0: scalar, interpretable as int. 
             Determines the first point in 
             rr that enters the fit, 
             avoiding problems with 
             integrals padded with zeros at 
             inter-nuclear distances.
        cutoff: scalar, float.
                integral cutoff distance.
        """
        try:
            rGrid = np.array(rGrid, dtype=float)
        except:
            msg = "rGrid must be a scalar or array-like, " \
                  "with elements interprtable as floats."
            raise TypeError(msg)
        try:
            order = np.array(order, dtype=int)
            order = order.flatten()
        except:
            msg = "order must be a scalar or array-like, " \
                  "with elements interprtable as ints."
            raise TypeError(msg)
        msg = "rGrid must be one-dimensional."
        assert len(rGrid.shape) == 1, msg
        msg = "order must be one-dimensional."
        assert len(order.shape) == 1, msg
        result = np.zeros((order.size, rGrid.size))
        if cutoff == None: cutoff = self.cutoff
        icutoff = np.where(rGrid >= cutoff)[0]
        for nn, ii in enumerate(order):
            result[nn,ir0:icutoff] = self.func(rGrid[ir0:icutoff], 
                                                *self.param, order=ii)

        return result




def fit(rr, ff, nP, modelFunc, modelJac, ir0=0, kwargs={}):
    """
    Return the parameters defining the fit according to model.

    Parameters:
    -----------
    rr : one-dimensional array-like, with elements interprtable as 
        floats. Inter-nuclear distance grid.

    ff : one-dimensional array-like, with elements interprtable as 
        floats. Integral values at rr.

    nP : scalar, interpretable as int. Number of parameters in the 
        fit model.

    modelFunc : callable signature (rr, *param) Evaluate the model 
        function at rr.

    modelJac : callable signature (rr, *param) Evaluate the Jacobian 
        of model function at rr.
           
    ir0 : scalar, interpretable as int. Determines the first point in
        rr that enters the fit, avoiding problems with integrals padded
        with zeros at inter-nuclear distances.
    
    Returns:
    --------
    """

    try:
        rr = np.array(rr, dtype=float)
    except:
        msg = "rr must be array-like, " \
              "with elements interprtable as floats."
        raise TypeError(msg)
    try:
        ff = np.array(ff, dtype=float)
    except:
        msg = "ff must be array-like, " \
              "with elements interprtable as floats."
        raise TypeError(msg)
    try:
        nP = int(nP)
    except:
        msg = "nP must be a scalar interprtable as int." 
        raise TypeError(msg)
    try:
        ir0 = int(ir0)
    except:
        msg = "ir0 must be a scalar interprtable as int." 
        raise TypeError(msg)

    msg = "rr must be one-dimensional."
    assert len(rr.shape) == 1, msg
    msg = "ff must be one-dimensional."
    assert len(ff.shape) == 1, msg
    msg = "rr.size={} != ff.size={}."
    msg = msg.format(rr.size, ff.size)
    assert rr.size == ff.size, msg
    msg = "nP <= 0, that's just mean."
    assert nP > 0, msg
    p0 = np.ones(nP)
    pOpt, pCov = curve_fit(modelFunc, 
                           rr, 
                           ff,
                           jac=modelJac,
                           p0=p0,
                           **kwargs)
    err = np.max(np.abs(ff - modelFunc(rr, *pOpt)))
    return pOpt, pCov, err
