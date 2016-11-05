import numpy as np
import scipy as sc
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


class Vasicek(object):
    """ VASICEK model
        mean reverting normal distributed process
    """
    def __init__(self, init, target, speed, vol):
        self.init   = init
        self.target = target
        self.speed  = speed
        self.vol    = vol

    def mean(self, maturity, init=None):
        if not(init):
            init=self.init
        return( init*np.exp(-self.speed*maturity) + self.target*(1-np.exp(-self.speed*maturity)) )

    def variance(self, maturity, init=None):
        if not(init):
            init=self.init
        return( self.vol**2/(2*self.speed)*(1-np.exp(-2*self.speed*maturity)) )

    def generate_brownian(self, maturity, nb_simulations, nb_steps):
        """ Generates normalized brownian increments
        """
        self.dB = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))

    def simuate(self, maturity, nb_simulations, nb_steps):
        """ Simulation via explicit solution
        """
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt        = 1.0/nb_steps
        for j in range(res.shape[1]-1):
            res[:,j+1]=res[:,j]+self.speed*(self.target-res[:,j])*self.dB[:,j]
        self.sim = res


class CIR(object):
    """ COX INGERSOL ROSS model
        mean reverting positive process
    """
    def __init__(self, init, target, speed, vol_of_vol):
        self.init       = init
        self.target     = target
        self.speed      = speed
        self.vol_of_vol = vol_of_vol

    def mean(self, maturity, init=None):
        """ mean of the process at maturity when initial point is init
        """
        if not(init):
            init=self.init
        return(self.target+(init-self.target)*np.exp(-self.speed*maturity))

    def variance(self, maturity, init=None):
        """ variance of the process at maturity when initial point is init
        """
        if not(init):
            init=self.init
        res = init*self.vol_of_vol*self.vol_of_vol*np.exp(-self.speed*maturity)*(1-np.exp(-self.speed*maturity))/self.speed
        res = res + 0.5*self.target*self.vol_of_vol*self.vol_of_vol*(1-np.exp(-self.speed*maturity))**2/self.speed
        return(res)

    def generate_brownian(self, maturity, nb_simulations, nb_steps):
        """ Generates normalized brownian increments
        """
        self.dB = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))

    def simulate(self, maturity, nb_simulations, nb_steps):
        """ Simulation with QE discretisation scheme
        """
        #non optimal QE algo
        #check if works
        #norm_rand = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt        = 1.0/nb_steps
        for j in range(res.shape[1]-1):
            for i in range(res.shape[0]):
                m  = self.mean(dt, res[i,j])
                s2 = self.variance(dt, res[i,j])
                psi = s2/(m*m)
                if (psi<1.5):
                    b2  = 2/psi-1+np.sqrt(2/psi*(1/psi-1))
                    a   = m/(1+b2)
                    b   = np.sqrt(b2)
                    res[i,j+1] = a*(b+self.dB[i,j])**2
                else:
                    beta = 2/(m*(psi+1))
                    p    = (psi-1)/(psi+1)
                    u    = norm.cdf(self.dB[i,j])
                    res[i,j+1]  = self.__inverse_psi__(u,p, beta)
        self.sim = res

    def __simulate_euler__(self, maturity, nb_simulations, nb_steps):
        """ Simulaton with Euler Scheme
        """
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt        = 1.0/nb_steps
        for j in range(res.shape[1]-1):
            res[:,j+1]=self.speed*(self.target-res[:,j])*dt+self.vol_of_vol*np.sqrt(np.maximum(res[:,j],0)*dt)*self.dB[:,j]
        self.sim=res

    #help tools for CIR class
    def __inverse_psi__(self, u, p, beta):
        if 0<=u<=p:
            return(0)
        else:
            return(1/beta*np.log((1-p)/(1-u)))

class HestonModel(object):
    """ Heston Model
    """

    def __init__(self, init, rate, dividend, vol_init, target, speed, vol_of_vol, correlation):
        self.init        = init
        self.rate        = rate
        self.dividend    = dividend
        self.volatility  = CIR(vol_init, target, speed, vol_of_vol)
        self.correlation = correlation

    def generate_brownian(self, maturity, nb_simulations, nb_steps):
        """ Generates normalized brownian increments
        """
        self.dB = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))

    def simulate(self, maturity, nb_simulations, nb_steps):
        """ Simulation using Andersen Scheme
            source: Efficient Simulation of the Heston Stochastic Volatility Model
                    Leif Andersen
                    p19
        """
        self.volatility.simulate(maturity, nb_simulations, nb_steps)
        self.generate_brownian(maturity, nb_simulations, nb_steps)
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt        = 1.0/nb_steps
        rootdt    = np.sqrt(dt)
        gamma1    = 0.5
        gamma2    = 0.5
        k0        = -self.correlation*self.volatility.speed*self.volatility.target*dt/self.volatility.vol_of_vol
        k1        = gamma1*dt*(self.volatility.speed*self.correlation/self.volatility.vol_of_vol-0.5)-self.correlation/self.volatility.vol_of_vol
        k2        = gamma2*dt*(self.volatility.speed*self.correlation/self.volatility.vol_of_vol-0.5)+self.correlation/self.volatility.vol_of_vol
        k3        = gamma1*dt*(1-self.correlation**2)
        k4        = gamma2*dt*(1-self.correlation**2)
        for j in range(res.shape[1]-1):
            res[:,j+1]=res[:,j]*np.exp(
                                        (self.rate-self.dividend)*dt+
                                        k0+k1*self.volatility.sim[:,j]+k2*self.volatility.sim[:,j+1]
                                        +np.sqrt(k3*self.volatility.sim[:,j]+k4*self.volatility.sim[:,j+1])*dB[:,j]
                                       )
        self.sim = res

    def call_price(self, maturity, strike):
        """ Price of the Call Contract
        """
        def integrand(w):
            return(np.exp(-1j*w*np.log(strike*np.exp(-self.rate*maturity)/self.init))*(self.characteristic_function(w-1j, maturity)-1)/(1j*w*(1+1j*w)))
        #integral=sc.integrate.quad(integrand, -1000, 1000)[0]
        integral  = sc.integrate.trapz(integrand(np.arange(-100,100,0.1)),dx=0.1)
        res       = 0.5*self.init/np.pi*integral+np.max(0, self.init-strike*np.exp(-self.rate*maturity))
        return(float(res))

    def put_price(self, maturity, strike):
        """ Price of Put contract
            uses the call/put parity formula
        """
        return(self.call_price(maturity, strike)-self.init+strike*np.exp(-self.rate*maturity))

    def characteristic_function(self, w, maturity):
        """ characteristic function
            w: point of evaluation
            maturity: maturity
        """
        gamma = np.sqrt(self.volatility.vol_of_vol**2*(w*w+1j*w)+(self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w)**2)
        d     = -(w*w+1j*w)/(gamma*1/np.tanh(0.5*gamma*maturity)+self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w)
        c     = self.volatility.speed*maturity*(self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w-gamma)/(self.volatility.vol_of_vol**2)
        c     = c-2*self.volatility.speed/(self.volatility.vol_of_vol**2)*np.log(0.5*(1+np.exp(-gamma*maturity))+0.5*(self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w)*(1-np.exp(-gamma*maturity))/gamma)
        return(np.exp(self.volatility.target*c+self.volatility.init*d))

class BlackScholes(object):
    """ Black Scholes model
    """

    def __init__(self, init=100.0, mu=0.05, sigma=0.20):
        """ instantiation of BlackScholes Object
        """
        self.mu    = mu
        self.sigma = sigma
        self.init  = init

    def simulate(self, maturity, nb_simulations, nb_steps):
        """ simulates the model accross time and generates scenarios
            inputs
            ------
            initial_value:  initial starting point of the simulations
            maturity: horizon of the simulation
            nb_simulations: number of simulations
            nb_steps: number of steps per year
            results
            -------
            numpy array with simulations whose dimension is (nb_simulations, nb_steps*maturity+1)
        """
        #brownian increments
        dB        = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt        = 1.0/nb_steps
        root_dt   = np.sqrt(dt)
        #loop over columns
        for j in range(res.shape[1]-1):
            res[:,j+1] = res[:,j]*np.exp(self.mu*dt+self.sigma*root_dt*dB[:,j]-0.5*self.sigma*self.sigma*dt)
        return(res)

    def get_next(self, underlying_t, brownian_dt, dt):
        """
        """
        root_dt         = np.sqrt(dt)
        next_underlying = underlying_t*np.exp(self.mu*dt+self.sigma*root_dt*brownian_dt-0.5*self.sigma*self.sigma*dt)
        return(next_underlying)

    def fit_history(self, historical_values, time_interval=1.0):
        """ Calibrates model parameters from historical values
            inputs
            -------
            historical_values: historical observations of the variable
            time_interval: time betwen two observations (by default :1 year)
        """
        hv         = pd.Series(historical_values)
        hv         = np.log(hv).diff()
        self.mu    = hv.mean()/time_interval
        self.sigma = hv.std()/time_interval

    def call_price(self, spot, strike, maturity, dividend, rate, volatility):
        forward = spot*np.exp(rate*maturity)
        d1      = (np.log(forward/strike)+0.5*volatility*volatility*maturity)/(volatility*np.sqrt(maturity))
        d2      = d1-volatility*np.sqrt(maturity)
        res     = np.exp(-rate*maturity)*(forward*norm.cdf(d1)-norm.cdf(d2)*strike)
        return(res)
