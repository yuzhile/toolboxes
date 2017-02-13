import numpy as np
def diff_values(value1,value2):
    '''
    compute the differences between value1 and value2 by norming the difference of value1 and value2
    inputs
    - value1: np.ndarray
    - value2: np.ndarray with the same dim as value1
    returns:
    - diff_norm
    '''
    return np.linalg.norm(value1.flatten()-value2.flatten())

