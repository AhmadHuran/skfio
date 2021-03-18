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



"""
This module deals with reading *.skf according 
to v1.0 Slater-Koster files  in its "simple format"
form, i.e. can not handle f orbitals.

the format specifications can be found at
<https://dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf>

skfOrder stores (l1, l2, mm) in the order of columns of the
Slater-Koster tables found in the *.skf.
"""

import numpy as np

skfOrder = [(2,2,0),
            (2,2,1),
            (2,2,2),
            (1,2,0),
            (1,2,1),
            (1,1,0),
            (1,1,1),
            (0,2,0),
            (0,1,0),
            (0,0,0), 
            (2,2,0),
            (2,2,1),
            (2,2,2),
            (1,2,0),
            (1,2,1),
            (1,1,0),
            (1,1,1),
            (0,2,0),
            (0,1,0),
            (0,0,0),]

def readSKfile(fileName):
    """
    Read Slater-Koster file fileName.
    The file name is assumed to be of the
    form x-y.skf.
    Only the elctronic part is considered here.

    Return {
      "data": array of shape (nPoints, 20),

      "grid": array of shape (nPoints),
      
      "homo": boolean, True if the pair x-y
              is homo-nuclear.,

      "header": list of the appropriate header
                lines.
    }
    """

    name = fileName.split("/")[-1]
    name = name.split(".")[0]
    aa, bb = name.split("-")
    homo = (aa == bb)
    with open(fileName,"r") as ff:
        lines = ff.readlines()

    r0 = float(lines[0].split()[0])
    nPoints = int(lines[0].split()[1]) - 1
    r2 = r0 * (nPoints + 1)
    grid = np.arange(r0, r2, r0)
    if homo: ind = 3
    else: ind = 2
    header = lines[:ind]
    HS=[[float(j) for j in i.split()] for i in lines[ind:ind+nPoints]]
    HS=np.stack(HS)
    return {"data": HS.copy(), 
            "grid": grid.copy(),
            "homo": homo,
            "header": header[:]
              }

