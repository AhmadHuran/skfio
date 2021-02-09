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

from tokenize import generate_tokens
from token import tok_name, NEWLINE, NAME, OP, ENDMARKER
from io import StringIO
from configparser import ConfigParser

def readConfig(fileName):
    """
    Read the configuration file <fileName>
    and return the corresponding 
    configparser.ConfigParser()
    """
    config = ConfigParser()
    inputs = config.read(fileName)
    msg = "Panicking. Give one input file only."
    assert len(inputs) == 1, msg
    config["input_file"] = {1: inputs[0]}
    return config


def atomizeModel(string):
    """
    Tokenize the model name <string>.
    
    The input is case insensitive and 
    white space is ignored, i.e.

        X Y => xy

    Allowed tokens are model names
    (valid python variable name), and 
    two binary operators, namely, + 
    and * . 

    Allowed models can be checked in 
    skfio.models.allModels .

    A number determining the number of  
    parameters for each model is appended 
    to the model type with "_" as a 
    delimiter. E.g.

        'P_5 * G_1 + G_3'

         P_n: an skfio.models.polynomial 
               model function with n parameters
               in total, i.e. x0 and (n - 1) 
               coefficients.
         
         G_n: an skfio.models.gaussian
               model function with n gaussians,
               i.e. 3n parameters in total.

    """
    msg = """
          The used value {} of type {}
          is not allowed. Here is a legal
          example:
          
          'P_5 * G_1 + G_3'
          """

    allowed = ["NAME", "PLUS", "STAR"]

    string = "".join(string.split())
    tokens = generate_tokens(StringIO(string.lower()).readline)
    tokens = list(tokens)

    msg2 = "An empty string was given as a model type."
    assert len(tokens) > 2, msg2

    types = [ii.type for ii in tokens]
    msg2 = "A multi-line string was given as a model type."
    assert types.count(NEWLINE) == 1, msg2

    accepted = []
    for token in tokens:
        typ, val, _, _, _ = token 
        if typ == NEWLINE: break
        name = tok_name[token.exact_type]
        assert name in allowed, msg.format(val, name)
        if typ == NAME: checkModelNameSyntax(val)
        accepted.append((name, val))
    return accepted

def checkModelNameSyntax(name):
    """
    Check if model name conforms 
    with the expected formt.

    Expected format:
    NAME_NUMBER

    NAME can not have "_" in it.

    NUMBER is a non-zero, base 10,integer 
    literal, it determines the number of 
    parameters for each model. It is appended 
    to the model type with "_" as a delimiter.

    """
    splits = name.split("_")
    msg = "Illegal model specification format.\n"\
          "Expected format: NAME_NUMBER. "\
          "Received '{}'.".format(name)
    assert len(splits) == 2, msg
    msg = "Illegal model specification format.\n"\
          "Expected format: NAME_NUMBER.\n"\
          "NUMBER must be a non-zero, base 10, "\
          "integer literal. Received '{}'.".format(splits[1])
    try:
        num = int(splits[1], base=10)
        assert num > 0, msg
    except ValueError:
        raise ValueError(msg)
