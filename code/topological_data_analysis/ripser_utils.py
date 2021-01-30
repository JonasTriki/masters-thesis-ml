import re

import numpy as np

test_output = """
------------RIPSER++ WITH PYTHON BINDINGS CALLED------------ <_io.TextIOWrapper name='<stderr>' mode='w' encoding='UTF-8'>
loaded python dense user matrix
value range: [17,175]
distance matrix with 10 points
GPU memory before full rips memory calculation, total mem: 11554717696 bytes, free mem: 11385896960 bytes
gpu memory needed for full rips by calculation in bytes for dim: 1: 3677 bytes
recalculated dim_max based on GPU free DRAM capacity: 1
max possible num simplices over all dim<=dim_max (without clearing) for memory allocation: 45
GPU memory after full rips memory calculation and allocation, total mem: 11554717696 bytes, free mem: 11383799808 bytes
CUDA PREPROCESSING TIME (e.g. memory allocation time): 0.001031s
num edges filtered by diameter: 34
persistence intervals in dim 0:
 [0,17)
 [0,26)
 [0,33)
 [0,44)
 [0,44)
 [0,45)
 [0,50)
 [0,60)
 [0,63)
 [0, )
num cols to reduce: dim 1, 25
0-dimensional persistence total computation time with GPU: 0.028513s
max possible num simplices: 45
gpu scan kernel time for dim: 1: 1.8e-05s
num apparent for dim: 1 is: 21
INSERTION POSTPROCESSING FOR GPU IN DIM 1: 0.000123s
-SUM OF GPU MATRIX SCAN and post processing time for dim 1: 0.000321s
persistence intervals in dim 1:
 [84,88)
 [78,86)
 [68,107)
 [61,73)
SUBMATRIX REDUCTION TIME for dim 1: 2.7e-05s

GPU ACCELERATED COMPUTATION from dim 0 to dim 1: 0.030024s
total time: 0.030087s
total GPU memory used: 0.00209715GB
"""


def parse_ripser_plus_plus_output(rpp_output: str, dims: int) -> list:
    """
    Parses Vietoris-Rips diagrams from Ripser++ output into Numpy matrices.

    Parameters
    ----------
    rpp_output : str
        Output from Ripser++
    dims : int
        Homology dimensionality

    Returns
    -------
    diagrams : list
        List of Vietoris-Rips diagrams (similar to Ripsers "dgms" output)
    """
    diagrams = []
    for dim in range(dims + 1):
        dim_header = re.findall(fr"persistence intervals in dim {dim}:\n", rpp_output)
        if len(dim_header) == 1:
            output_dim_data = rpp_output.split(dim_header[0])[1]
            output_dim_data_lines = output_dim_data.split("\n")
            line_i = 0
            dim_diagram = []
            while output_dim_data_lines[line_i].startswith(" ["):
                current_line = output_dim_data_lines[line_i]
                birth_str, death_str = re.split(r" \[|,|\)|]", current_line)[1:3]
                birth = float(birth_str)
                if death_str == " ":
                    death = np.inf
                else:
                    death = float(death_str)
                dim_diagram.append([birth, death])
                line_i += 1
            dim_diagram = np.array(dim_diagram)
            diagrams.append(dim_diagram)
        else:
            diagrams.append([])
    return diagrams


if __name__ == "__main__":

    # Testing
    parse_ripser_plus_plus_output(rpp_output=test_output, dims=2)
