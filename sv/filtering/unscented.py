import warnings

import numpy as np
import scipy.linalg

from filtering.nearpd import nearpd

def sigmaPoints(x, P, alef=3.0):
    n = np.shape(x)[0]
    lmbda = alef - n
    # lambda is a keyword in Python, hence "lmbda".
    nPlusLambda = n + lmbda

    # NB! Depending on convention, by a matrix square root pf P we may mean
    # either (i) A such that A^T * A = P or (ii) A such that A * A^T = P.
    # Unfortunately, [Wan-2000]_ does not explain why it is important to use the
    # right convention. As pointed our in [Julier-2004]_, page 406, footnote 5,
    # if the matrix square root follows convention (i), then the sigma points
    # are formed from the ROWS of A. However, if the matrix square root follows
    # convention (ii), then the sigma points are formed from the COLUMNS of A.
    # Python's scipy.linalgo.cholesky follows convention (i):
    #
    # >>> P = np.array([[1.,2.],[2.,5.]])
    # >>> A = scipy.linalg.cholesky(P)
    # >>> np.dot(A.T, A)    # matches P; convention (i)
    # >>> np.dot(A, A.T)    # does not match P; convention (ii)
    #
    # However, we would like to use convention (ii), in which the sigma points
    # are formed from the columns of A. Therefore we transpose the result of
    # scipy.linalgo.cholesky.

    sqrtP = scipy.linalg.cholesky(nPlusLambda * P).T

    # The above is a Cholesky square root. Could also use a symmetric square
    # root:
    #
    #     sqrtP = scipy.linalg.matfuncs.toreal(
    #         scipy.linalg.matfuncs.sqrtm(nPlusLambda * P))
    #
    # This method relies on the work by Higham [Higham-1986]_, [Higham-1984]_.

    sigmaPointCount = 2*n+1
    X = np.zeros((n, sigmaPointCount))
    Wm = np.zeros(sigmaPointCount)
    Wc = np.zeros(sigmaPointCount)
    X[:,0] = x.reshape(n)
    Wm[0] = lmbda / nPlusLambda
    Wc[0] = 1.0/(2.0*nPlusLambda)


    weight = 1.0 / (2.0 * nPlusLambda)
    for i in range(0, n):
        index0 = i+1
        index1 = i+1+n
        X[:,index0] = x.reshape(n) + sqrtP[:,i].reshape(n)
        X[:,index1] = x.reshape(n) - sqrtP[:,i].reshape(n)
        Wm[index0]  = weight
        Wm[index1]  = weight
        Wc[index0]  = weight
        Wc[index1]  = weight

    return X, Wm, Wc

def unscentedTransform(X, Wm, Wc, f):
    Y = None
    Ymean = None
    fdim = None
    N = np.shape(X)[1]
    for j in range(0,N):
        fImage = f(X[:,j])
        if Y is None:
            fdim = np.size(fImage)
            Y = np.zeros((fdim, np.shape(X)[1]))
            Ymean = np.zeros(fdim)
        Y[:,j] = fImage
        Ymean += Wm[j] * Y[:,j]
    Ycov = np.zeros((fdim, fdim))
    for j in range(0, N):
        meanAdjustedYj = Y[:,j] - Ymean
        Ycov += np.outer(Wc[j] * meanAdjustedYj, meanAdjustedYj)
    return Y, Ymean, Ycov

class UnscentedKalmanFilter(object):
    MINUS_HALF_LN_2PI = -.5 * np.log(2. * np.pi)

    def __init__(self, x0, P0, Q, R, cor, f, h):
        self.Q = Q
        self.R = R
        self.cor = cor
        self.fa = lambda col: f(col[0], col[2])
        self.ha = lambda col: h(col[0], col[1])
        
        Pxx = P0
        Pxv = 0.
        self.xa = np.array( ((x0,), (0.,), (0.,), (0.,)) )
        self.Pa = np.array( ((Pxx, Pxv   , 0.      , 0.      ),
                             (Pxv, self.R, 0.      , 0.      ),
                             (0. , 0.    , self.Q  , self.cor),
                             (0. , 0.    , self.cor, self.R  )) )
        
        self.lastobservation = np.NAN
        self.predictedobservation = np.NAN
        self.innovation = np.NAN
        self.innovationvar = np.NAN
        self.gain = np.NAN
        
        self.loglikelihood = 0.0
        
    def predict(self):
        try:
            X, Wm, Wc = sigmaPoints(self.xa, self.Pa)
        except:
            warnings.warn('Encountered a matrix that is not positive definite in the sigma points calculation at the predict step')
            self.Pa = nearpd(self.Pa)
            X, Wm, Wc = sigmaPoints(self.xa, self.Pa)
        fX, x, Pxx = unscentedTransform(X, Wm, Wc, self.fa)
        x = np.asscalar(x)
        Pxx = np.asscalar(Pxx)

        Pxv = 0.
        N = np.shape(X)[1]
        for j in range(0, N):
            Pxv += Wc[j] * fX[0,j] * X[3,j]
        
        self.xa = np.array( ((x,), (0.,), (0.,), (0.,)) )
        self.Pa = np.array( ((Pxx, Pxv   , 0.      , 0.      ),
                             (Pxv, self.R, 0.      , 0.      ),
                             (0. , 0.    , self.Q  , self.cor),
                             (0. , 0.    , self.cor, self.R  )) )

    def observe(self, y):
        self.lastobservation = y
        
        xa = self.xa[0:2:1,0:1:1]
        Pa = self.Pa[0:2:1,0:2:1]
        try:
            X, Wm, Wc = sigmaPoints(xa, Pa)
        except:
            warnings.warn('Encountered a matrix that is not positive definite in the sigma points calculation at the observe step')
            Pa = nearpd(Pa)
            X, Wm, Wc = sigmaPoints(xa, Pa)
        hX, self.predictedobservation, Pyy = \
                unscentedTransform(X, Wm, Wc, self.ha)
        self.predictedobservation = np.asscalar(self.predictedobservation)
        Pyy = np.asscalar(Pyy)
        self.innovationvar = Pyy

        x = self.xa[0,0]
        Pxy = 0.
        Pvy = 0.
        M = np.shape(X)[1]
        for j in range(0, M):
            haImage = self.ha(X[:,j])
            Pxy += Wc[j] * (X[0,j] - x) * (haImage - self.predictedobservation)
            Pvy += Wc[j] * X[1,j] * haImage

        Pa = np.array( ((Pxy,), (Pvy,), (0.,), (0.,)) )
        K = Pa * (1./Pyy)
        self.gain = K[0,0]
        
        self.innovation = y - self.predictedobservation        

        self.xa += K * self.innovation
        self.Pa -= np.dot(K, Pa.T)
        
        self.loglikelihood += UnscentedKalmanFilter.MINUS_HALF_LN_2PI - .5 * (np.log(self.innovationvar) + self.innovation * self.innovation / self.innovationvar)
        
    @property
    def mean(self): return self.xa[0,0]
    
    @property
    def var(self): return self.Pa[0,0]
