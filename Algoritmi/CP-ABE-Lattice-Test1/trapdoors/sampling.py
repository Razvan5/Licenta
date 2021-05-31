import numpy


def randomVectorFromMatrix(arr: numpy.array, size: int = 1) -> numpy.array:
    return arr[numpy.random.choice(len(arr), size=size, replace=False)][0]


def __genericAlgSampleD(_b1_lattice_base, _rounding_parameter, _sigma_matrix, c_vector):
    """

    :param _b1_lattice_base: The base of the lattice Λ
    :param _rounding_parameter: used for the final rounding
    :param _sigma_matrix: a positive definite covariance matrix Σ > Σ1 = r^2*B1B1t
    :param c_vector:
    :return:
    """
    pass


def genericAlgoSampleD(_b1):
    _r = numpy.sqrt(len(_b1))*len(_b1)
    _sigma_matrix = numpy.sqrt(len(_b1))*len(_b1)*2 * numpy.array(_b1)*numpy.array(_b1).transpose()
    __genericAlgSampleD(_b1, _r, _sigma_matrix, )
