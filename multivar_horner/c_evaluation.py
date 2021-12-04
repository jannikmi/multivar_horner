import ctypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from multivar_horner.classes.factorisation import BasePolynomialNode
from multivar_horner.classes.helpers import FactorContainer

if os.name == "nt":
    COMPILED_C_ENDING = ".dll"
else:
    COMPILED_C_ENDING = ".so"

DOUBLE = "double"
# array for evaluation results of both scalar and monomial factors
FACTORS = "f"
COEFFS = "c"
EVAL_FCT = "eval"
C_TYPE_DOUBLE = ctypes.c_double


def write_c_file(
    nr_coeffs: int,
    nr_dims: int,
    path_out: Path,
    factorisation: Tuple[BasePolynomialNode, FactorContainer],
    verbose: bool = False,
) -> int:
    tree, factor_container = factorisation

    # count and return the required amount of operations
    num_ops = 0

    instr = "#include <math.h>\n"
    # NOTE: the coefficient array will be used to store intermediary results
    # -> copy to use independent instance of array (NO pointer to external array!)
    func_def = f"{DOUBLE} {EVAL_FCT}({DOUBLE} x[{nr_dims}], {DOUBLE} {COEFFS}[{nr_coeffs}])"
    # declare function ("header")
    instr += f"{func_def};\n"
    # function definition
    instr += f"{func_def}"
    instr += "{\n"
    # NOTE: the order of computing the values important: scalar factors, monomial factors, monomials,...
    scalars = factor_container.scalar_factors
    monomials = factor_container.monomial_factors
    nr_factors = len(factor_container)
    # initialise arrays to capture the intermediary results of factor evaluation to 0
    if nr_factors > 0:
        instr += f"{DOUBLE} {FACTORS}[{nr_factors}]= {{0.0}};\n"
    idx: int = 0
    # set the index of each factor in order to remember it for later reference
    # ATTENTION: different from the ones used in the recipes!
    for idx, scalar in enumerate(scalars):
        scalar.value_idx = idx
        instr += scalar.get_instructions(FACTORS)
        num_ops += scalar.num_ops
    for monomial in monomials:
        idx += 1
        monomial.value_idx = idx
        monomial.factorisation_idxs = [f.value_idx for f in monomial.scalar_factors]
        instr += monomial.get_instructions(FACTORS)
        num_ops += monomial.num_ops
    # compile the instructions for evaluating the Horner factorisation tree
    instr += tree.get_instructions(COEFFS, FACTORS)
    num_ops += tree.num_ops
    # return the entry of the coefficient array corresponding to the root node of the factorisation tree
    root_idx = tree.value_idx
    instr += f"return {COEFFS}[{root_idx}];\n"
    instr += "}\n"
    if verbose:
        print(f"writing in file {path_out}")
    with open(path_out, "w") as c_file:
        c_file.write(instr)
    return num_ops


def get_compiler() -> str:
    compiler = "gcc"
    if shutil.which(compiler) is None:
        compiler = "cc"
        if shutil.which(compiler) is None:
            raise ValueError("compiler (`gcc` or `cc`) not present. install one first.")
    return compiler


def compile_c_file(compiler: str, path_in: Path, path_out: Path):
    cmd = [compiler, "-shared", "-o", str(path_out), "-fPIC", str(path_in)]
    subprocess.call(cmd)
    if not path_out.exists():
        raise ValueError(f"expected compiled file missing: {path_out}")
