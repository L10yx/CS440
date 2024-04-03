import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    # raise RuntimeError("You need to write this")
    cX = len(PX)
    EX = np.sum(np.arange(1, cX + 1) * PX)
    p = 1 / EX
    PY = [(1-p)**y * p for y in range(cX)]
    return p, PY
