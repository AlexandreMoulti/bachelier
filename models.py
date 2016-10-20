import numpy as np
import scipy as sc
from scipy.stats import norm


class CIR(object):
    """ CIR mean reverting process
    """
    def __init__(self, init, target, speed, vol_of_vol):
        self.init       = init
        self.target     = target
        self.speed      = speed
        self.vol_of_vol = vol_of_vol

    def mean(self, init, dt):
        """ mean of the process at time t
        """
        return(self.target+(init-self.taget)*np.exp(-self.speed*dt))

    def variance(self, init, dt):
        """ variance of the process at time t
        """
        res = init*self.vol_of_vol*self.vol_of_vol*np.exp(-self.speed*dt)*(1-np.exp(-self.speed*dt))/self.speed
        res = res + 0.5*self.target*self.vol_of_vol*self.vol_of_vol*(1-np.exp(-self.speed*dt))**2/self.speed
        return(res)

    def simulate(self, maturity, nb_simulations, nb_steps):
        #non optimal QE algo
        #check if works
        urand     = np.random.uniform(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        norm_rand = np.random.normal(loc=0, scale=1, size=(nb_simulations, nb_steps*maturity))
        res       = np.zeros((nb_simulations, nb_steps*maturity+1))
        res[:, 0] = self.init
        dt        = 1.0/nb_steps
        root_dt   = np.sqrt(dt)
        for j in range(res.shape[1]-1):
            for i in range(res.shape[0]):
                m  = self.mean(res[i,j], dt)
                s2 = self.variance(res[i,j], dt)
                psi = s2/(m*m)
                if (psi<1.5):
                    b2  = 2/psi-1+np.sqrt(2/psi*(1/psi-1))
                    a   = m/(1+b2)
                    b   = np.sqrt(b)
                    res[i,j+1] = a*(b+norm_rand[i,j])**2
                else:
                    beta = 2/(m*(psi+1))
                    p    = (psi-1)/(psi+1)
                    u    = urand[i,j]
                    res[i,j+1]  = self.__inverse_psi__(u,p, beta)


    def simulate_integral(self, maturity, nb_simulations, nb_steps):
        #simulates the integrated variance process as described in Andersen paper
        #useful for heston
        return(0)

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

    def simulate(self, maturity, nb_simulations, nb_steps):
        #to do
        return(0)

    def __call_integrand__(self, w, maturity, strike):
        """ Integrand for call price, Attari 2004 formula
        """
        lK=np.log(strike*np.exp(-self.rate*maturity)/self.init)
        return(
            (
              (np.real(self.characteristic_function(w,maturity))+np.imag(self.characteristic_function(w, maturity))/w)*np.cos(w*lK)
              +(np.imag(self.characteristic_function(w,maturity))-np.real(self.characteristic_function(w,maturity))/w)*np.sin(w*lK)
            )/(1+w*w)
        )

    def call_price(self, maturity, strike):
        """ Price of the Call Contract
        """
        def integrand(w):
            return(np.exp(-1j*w*np.log(strike*np.exp(-self.rate*maturity)/self.init))*(self.characteristic_function(w-1j, maturity)-1)/(1j*w*(1+1j*w)))
        integral=sc.integrate.quad(integrand, -100, 100)[0]
        res=0.5*self.init/np.pi*integral+np.max(0, self.init-strike*np.exp(-self.rate*maturity))
        return(res)


    def put_price(self, maturity, strike):
        """ Price of Put contract
            uses the call/put parity formula
        """
        return(self.call_price(maturity, strike)-self.init+strike*np.exp(-r*maturity))

    def characteristic_function(self, w, maturity):
        """ characteristic function
            w: point of evaluation
            maturity: maturity
        """
        gamma = np.sqrt(self.volatility.vol_of_vol**2*(w*w+1j*w)+(self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w)**2)
        d=-(w*w+1j*w)/(gamma*1/np.tanh(0.5*gamma*maturity)+self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w)
        c=self.volatility.speed*maturity*(self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w-gamma)/(self.volatility.vol_of_vol**2)
        c=c-2*self.volatility.speed/(self.volatility.vol_of_vol**2)*np.log(0.5*(1+np.exp(-gamma*maturity))+0.5*(self.volatility.speed-1j*self.correlation*self.volatility.vol_of_vol*w)*(1-np.exp(-gamma*maturity))/gamma)
        return(np.exp(self.volatility.target*c+self.volatility.init*d))


class BlackScholes(object):
    """ Black Scholes model
    """

    def __init__(self, init=100.0, mu=0.05, sigma=0.20):
        """ instantiation of BlackScholes Object
        """
        self.mu = mu
        self.sigma = sigma
        self.init = init

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
        root_dt=np.sqrt(dt)
        next_underlying = underlying_t*np.exp(self.mu*dt+self.sigma*root_dt*brownian_dt-0.5*self.sigma*self.sigma*dt)
        return(next_underlying)

    def fit_from_history(self, historical_values, time_interval=1.0):
        """ Calibrates model parameters from from historical values

            INPUTS
            ------
            historical_values: historical observations of the variable
            time_interval: time betwen two observations (by default :1 year)

            RESULT
            -------
            nothing
        """
        hv  = pd.Series(historical_values)
        hv  = np.log(hv).diff()
        self.mu = hv.mean()/time_interval
        self.sigma = hv.std()/time_interval
        return(1)

    def call_price(self, spot, strike, maturity, dividend, rate, volatility):
        forward = spot*np.exp(rate*maturity)
        d0      = (np.log(forward/strike)+0.5*volatility*volatility*maturity)/(volatility*np.sqrt(maturity))
        d1      = d0-volatility*np.sqrt(maturity)
        res     = np.exp(-rate*maturity)*(forward*norm.cdf(d1)-norm.cdf(d2)*strike)
        return(res)
