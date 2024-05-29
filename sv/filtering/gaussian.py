import numpy as np

class GaussianFilter(object):
    MINUS_HALF_LN_2PI = -.5 * np.log(2. * np.pi)
    
    def __init__(self, x0, P0, params):
        self._params = params
        self.x = x0
        self.P = P0
        self._constterm = self._params.meanlogvar * (1. - self._params.persistence)
        self._cv = self._params.cor * self._params.voloflogvar
        self._cv2 = self._cv * self._cv
        self._p2 = self._params.persistence * self._params.persistence
        self._v2 = self._params.voloflogvar * self._params.voloflogvar
        self.predictedobservation = np.NAN
        self.lastobservation = None
        self.innovation = np.NAN
        self.innovationvar = np.NAN
        self.gain = np.NAN
        self.loglikelihood = 0.0

    def predict(self):
        self.x = self._constterm + self._params.persistence * self.x
        self.P = self._p2 * self.P + self._v2
                
    def _observeimpl(self):
        raise NotImplementedError('Pure virtual method')
    
    def observe(self, observation):
        self.innovation, self.innovationvar, crosscov = self._observeimpl(observation)
        self.gain = crosscov / self.innovationvar
        self.x += self.gain * self.innovation
        self.P -= self.gain * crosscov
        self.loglikelihood += GaussianFilter.MINUS_HALF_LN_2PI - .5 * (np.log(self.innovationvar) + self.innovation * self.innovation / self.innovationvar)
        self.lastobservation = observation
                
    @property
    def mean(self): return self.x
    
    @property
    def var(self): return self.P

class SVLGaussianFilter(GaussianFilter):
    def _observeimpl(self, observation):
        efactor = np.exp(.5 * self.x + .125 * self.P)
        self.predictedobservation = .5 * efactor * self._cv
        innovation = observation - self.predictedobservation
        innovationvar = np.exp(self.x + .5 * self.P) * \
                (1. + self._cv2 * (1. - .25 * np.exp(-.25 * self.P)))
        crosscov = efactor * self._cv * (.5 * self.x + .25 * self.P + 1.) - \
                self.x * self.predictedobservation
        return innovation, innovationvar, crosscov

class SVL2GaussianFilter(GaussianFilter):
    def _observeimpl(self, observation):
        self.predictedobservation = 0.
        innovation = observation - self.predictedobservation
        innovationvar = np.exp(self.x + .5 * self.P) * (1. + .25 * self._cv2)
        crosscov = np.exp(.5 * self.x + .125 * self.P) * self._cv - self.x * self.predictedobservation        
        # !!! crosscov = np.exp(.5 * self.x + .125 * self.P) * (.25 + self._params.cor) * self._params.voloflogvar - self.x * self.predictedobservation
        # crosscov = np.exp(.5 * self.x + .125 * self.P) * (1. + .25 * self._cv2) - self.x * self.predictedobservation
        return innovation, innovationvar, crosscov
