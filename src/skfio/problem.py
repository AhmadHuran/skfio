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


from .parsing import readConfig
from .models import makeModel
from .skf import readSKfile, skfOrder
from .fitting import fit
import numpy as np
from itertools import product


class Problem:
    """
    Main entry point to skfio's basic functionality.

    Instance variables:
    -------------------
      
      species : list of strings denoting the symbols of chemical 
          elements present in species.

      eigenValues : array-like of shape (len(species), 3). Contains 
          atomic-like eigenalues of the s, p and d angular momentum
          shells in a.u.
      
      model : str, denoting the model used to represent the 
          Slater-Koster tables.

      pairs : list of strings, used for *.skf files lookup.
          "-" delimited pairs of the symbols in species.

      data : dictionary with keys denoting the different tables in 
          slater-Koster files-set.
          > key formatting:
              - Vl1l2m, Sl1l2m: denot Hamiltonian and overlap tables
                  respectively.

              - l1, l2: angular momentum shells, with l1<=l2. However, 
                  using s, p, and d instead of 0,1, and 2 respectively.

              - m: |m1|=|m2| with mi the projection of li.
                  m=s for sigma with m1 = m2 = 0
                  m=p for pi with    m1 = m2 = +/- 1
                  m=d for delta with m1 = m2 = +/- 2

              e.g. Vsss, Spdp 

          > value formatting:
              - values are arrays of floats with shape (nSpecies, 
                  nSpecies, nPoint) with nPoints the number of 
                  inter-nuclear distances in the tables. 

              E. g. data["Vspp"][ii, jj, :] is the table corresponding
              to the s-p hamiltonian integral of type pi of 
              "spieces[ii]-species[jj].skf"

          > an additional key "grid" points to the 
              inter-nuclear distance grid associated 
              with the tables.

          ***NOTE: all tables are assumed to have the same cutoff 
              distance and nPoints.
                     
      modelData : same as data, but with model parameters instead of 
          integral tables.

    """
    def __init__(self, **kwargs):

        species = kwargs["species"]
        msg = "Expected a list."
        assert isinstance(species, list), msg
        msg = "Entries of species must be of type str."
        assert all([isinstance(ii, str) 
                    for ii in species]
                    ), msg
        msg = "Duplicate species found."
        assert len(species) == len(set(species)), msg

        nSpecies = len(species)
        self.species = species[:]

        eigenValues = kwargs["eigenValues"]
        msg = "eigenValues must be array-like of floats."
        try:
            eigenValues = np.array(eigenValues, dtype=float)
        except:
            raise TypeError(msg)
        msg = "eigenValues must be of shape (nSpecies, 3)."
        assert eigenValues.shape == (nSpecies, 3), msg
        self.eigenValues = eigenValues.copy()
        
        model = kwargs["model"]
        msg = "Expected a str."
        assert isinstance(model, str), msg
        self.model = model
        self.pairs = product(species, repeat=2)
        self.pairs = ["-".join(pair) for pair in self.pairs]
        self.data = None
        self.modelData = None
        ff, jj, nP = makeModel(model)
        self.modelFunc = ff
        self.modelJac = jj
        self.nParam = nP

    @classmethod
    def fromDictionary(cls, **kwargs):
        """
        Initialize from a simple format dictionary kawrgs. A valid 
        example:

            >>> kawargs = {
            >>> 
            >>>  "speciesWhatever": {
            >>>  
            >>>      "symbol": "Aa",
            >>>      "Es": -0.2,
            >>>      "Ep": -0.1,
            >>>      "Ed": 0.0,
            >>>       },
            >>> 
            >>> "models":{
            >>> 
            >>>     "global": True,
            >>>     "model": "P_4 + G_2",
            >>>     }
            >>> }
        """

        keys = kwargs.keys()
        msg = "Expecting at least 2 keys"
        assert len(keys) >= 2, msg
        msg = "At least one species must be specified."
        assert any(["species" in ii.lower()  
                    for ii in keys]
                   ), msg

        species = []
        eigVal = []
        model = None
        for key in keys:
            if "species" in key.lower():
                spec = kwargs[key]
                species.append(spec["symbol"])
                vals = []
                ls = ["Es", "Ep", "Ed"]
                for lstr in ls:
                    vals.append(spec[lstr])
                eigVal.append(vals[:])
            elif "models" == key.lower():
                model = kwargs[key]["model"]
        msg = "section 'models' was not found"
        assert model != None, msg

        inpDict = {
                   "model": model,
                   "species": species,
                   "eigenValues": eigVal[:],
                   }
        return cls(**inpDict)

    @classmethod
    def fromFile(cls, fileName):
        """
        Initialize from a configuration file. See standard library 
        configparser for general description. 
        
        Basically the same as cls.fromDictionary .

        fileName : str, specifying the configuration file to be read.
            E.g.

                cat <<EOF > fileName
    
                    [species1]
                       symbol = Aa
                       maxL = 2
                       Es = 0.0
                       Ep = 0.0
                       Ed = 0.0
                    
                    [models]
                       global = yes
                       model = g_1 * p_9 + p_5 
                    
                EOF
        """
        config = readConfig(fileName)
        sections = config.items()
        msg = "Expecting at least 2 sections"
        assert len(sections) >= 2, msg
        msg = "At least one species must be specified."
        assert any(["species" in ii[0].lower()  
                    for ii in sections]
                   ), msg

        species = []
        eigVal = []
        model = None
        for name, sec in sections:
            if "species" in name.lower():
                species.append(sec["symbol"])
                vals = []
                ls = ["Es", "Ep", "Ed"]
                for lstr in ls:
                    vals.append(sec.getfloat(lstr))
                eigVal.append(vals[:])
            elif "models" == name.lower():
                model = sec["model"]
        msg = "section 'models' was not found"
        assert model != None, msg

        inpDict = {
                   "model": model,
                   "species": species,
                   "eigenValues": eigVal[:],
                   }
        return cls(**inpDict)

    def fetchData(self, ir0=0):
        """
        Read all relevant *.skf and extract the electronic part. 
        """
        sk = {}
        for pp in self.pairs:
            sk[pp]= readSKfile("{}.skf".format(pp))
        ang = "spd"
        data = {}
        grid = sk[self.pairs[0]]["grid"][ir0:]
        data["grid"] = grid
        nPoints = grid.size
        nSpec = len(self.species)
        for nn, (l1, l2, mm)  in enumerate(skfOrder):
            matrix = np.zeros((nSpec, nSpec, nPoints))
            for pp in self.pairs:
                spec1, spec2 = pp.split("-")
                isp1 = self.species.index(spec1)
                isp2 = self.species.index(spec2)
                matrix[isp1, isp2, :] = sk[pp]["data"][ir0:,nn]
            suffix = ang[l2].join([ang[l1], ang[mm]])
            if nn < 10: key = "V{}" 
            else: key = "S{}"
            key = key.format(suffix)
            data[key] = matrix.copy()
        return data  

    def makeModelData(self, model=None, ir0=0):
        """
        Fit the model to the Slater-Koster tables.
        """
        if self.data == None:
            self.data = self.fetchData(ir0)
        modelData = {}
        grid = self.data["grid"]
        nPoints = grid.size
        nSpec = len(self.species)
        if model == None:
            modelFunc = self.modelFunc
            modelJac = self.modelJac
            nP = self.nParam
        else:
            modelFunc, modelJac, nP = makeModel(model)
        for key, val in self.data.items():
            if key == "grid": continue
            matrix = np.zeros((nSpec, nSpec, nP))
            for ii in range(nSpec):
                for jj in range(nSpec):
                    pOpt, pCov, err = fit(grid,
                                          val[ii,jj,:],
                                          nP,
                                          modelFunc,
                                          modelJac,
                                         kwargs={
                                          "maxfev":1500,
                                           "verbose": 0,
                                           "tr_solver": "exact",
                                           "loss": "linear",
                                           "method": "trf",
                                           "ftol": 1.e-3,
                                           "x_scale":"jac",
                                          
                                          }
                             )
                    matrix[ii, jj, :] = pOpt.copy()
                    text = f"max abs error in {key}[{ii},{jj}]: {err}"
                    print(text)
            modelData[key] = matrix.copy()
        return modelData


