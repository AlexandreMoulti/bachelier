# -*- coding: utf-8 -*-

import numpy as np
from math import *
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sc

class DiscretisationGrid(object):
    """ Discretisation to be used for the Monte-Carlo simulations
        Time grid, discretisation of the time 
             horizon: Horizon of the simulation
        nb_simulations represents the number of Monte-Carlo paths to simulate
    """
    
    def __init__(self, nb_simulations, horizon, nb_steps):
        """ instantiation of the object
        """
        self.nb_steps       = nb_steps #per year
        self.dt             = 1.0/self.nb_steps
        self.horizon        = horizon
        self.nb_simulations = nb_simulations
        self.time_grid      = np.linspace(0,self.horizon, nb_steps*self.horizon+1, endpoint=True)
        self.shape          = (self.nb_simulations, self.time_grid.shape[0])

class Vasicek(object):
    """ Vasicek Model
        Equation of the Vasicek process: dr=speed*(target-r)dt+sigma*dB
        initial : initial point of the process
        target  : target to which the process converges in average
        speed: speed of mean reversion
        sigma : volatility of the mean reversion
    """
    def __init__(self, initial, target, speed, sigma):
        """ Instantation of the object
            initial: initial point
        """
        self.initial         = initial
        self.sigma           = sigma
        self.target          = target
        self.speed           = speed
        self.dB              = None

    def generate_brownian(self, sim_grid):
        """ Generates normalized brownian increments
        """
        dt              = sim_grid.dt
        if self.dB is None:
            self.dB = np.random.normal(loc=0, scale=1, size=(sim_grid.nb_simulations, sim_grid.nb_steps*sim_grid.horizon))*sqrt(dt)
            
    def simulate(self, sim_grid):
        dt              = sim_grid.dt
        self.sim_grid   = sim_grid
        self.generate_brownian(sim_grid)
        self.paths      = np.ndarray(sim_grid.shape)
        self.paths[:,0] = self.initial
        for z in np.arange(1,self.paths.shape[1]):
            self.paths[:,z]     = self.paths[:,z-1]*np.exp(-dt*self.speed) + self.target*(1-np.exp(-self.speed*dt)) + self.sigma*np.sqrt((1-np.exp(-2*self.speed*dt))/(2*self.speed*dt))*self.dB[:,z-1]
        self.simulations = pd.DataFrame(self.paths, columns=self.sim_grid.time_grid)
    
    def get_simulations(self):
        return(self.simulations)

    def __theoretical_mean__(self, target_time):
        res = np.exp(-self.speed*(target_time-self.duree_palier))*self.initial + self.target*(1-np.exp(-self.speed*(target_time-self.duree_palier)))
        return(res)

    def __theoretical_variance__(self, target_time):
        return(self.sigma**2/(2*self.speed)*(1-np.exp(-2*self.speed*target_time)))

class HW1F(Vasicek):
    """ Hull White 1F
        short rate is r_t = x_t + \alpha_t
        x_t : Vasicek process 
        \alpha_t : correction term whose puropose is to fit the model to the spot zero coupon curve
    """
    
    def __init__(self, speed, sigma, zc_spot_prices):
        """ speed: mean reversion of the vasicek process
            sigma : volatility of the vasicek process
            zc_spot_prices: pd.series with columns tenor and prices
        """
        Vasicek.__init__(self, 0.0, 0.0, speed, sigma)
        self.zc_spot_prices = zc_spot_prices
        #self.compute_instantaneous_forward_rates = self.compute_instantaneous_forward_rates(zc_spot_prices)
    

    def get_short_rates_simulations(self):          
        phi = self.compute_forward_rates(self.sim_grid.time_grid)
        res = pd.DataFrame(np.zeros(self.simulations.shape), columns=self.sim_grid.time_grid)
        for j in res.columns:
            tt = j
            res.loc[:,j] = self.simulations.loc[:,j]+phi[j]+self.sigma**2*(1-np.exp(-self.speed*tt))/(2*self.speed**2)
        return(res)
    
    
    def get_zero_coupon_prices_simulations(self, tenor):
        """ Zero coupon prices with respect to the simulations previously performed
        """
        dt                 = self.sim_grid.dt
        time_grid          = np.linspace(0,self.sim_grid.horizon+tenor, self.sim_grid.nb_steps*(self.sim_grid.horizon+tenor)+1, endpoint=True)
        zero_coupon_prices = self.interpolate_spot_zc_prices(time_grid)
        b                  = (1-np.exp(-self.speed*tenor))/self.speed
        res                = self.simulations.copy()
        index_time=0
        for j in self.simulations.columns:
            tt             = j
            a              = np.exp(-0.5*b*self.sigma**2*(1-np.exp(-self.speed*tt))**2/(self.speed**2)-0.25*(1-np.exp(-2*self.speed*tt))*self.sigma**2*b**2/self.speed)
            res.loc[:,j]   = np.exp(-b*self.simulations.loc[:,j])*a*zero_coupon_prices.iloc[index_time+tenor*self.sim_grid.nb_steps]/zero_coupon_prices.iloc[index_time]
            index_time     = index_time+1
        return(res)
    
    def get_actuarial_rates_simulations(self, tenor):
        """ returns: simulations of actuarial rates
            by definition Price(t, t+\tau)= exp(-ActuarialRate(t, \tau) * \tau)
        """
        zc_simulations= self.get_zero_coupon_prices_simulations(tenor)
        return(-np.log(zc_simulations)/tenor)
        
    def compute_forward_rates(self, time_grid):
        """ computes instantaneous forward rates
            time_grid: the time grid on which forward rates must be computed
            retruns: Pandas.Series with instantaneous forward rates
        """
        actuarial_rates_dot_mat  = -np.log(self.zc_spot_prices)
        actuarial_rates_dot_mat  = actuarial_rates_dot_mat.reindex(time_grid)
        actuarial_rates_dot_mat  = actuarial_rates_dot_mat.interpolate(method='slinear')
        #actuarial_rates_dot_mat = actuarial_rates_dot_mat*actuarial_rates_dot_mat.index.values
        res                      = actuarial_rates_dot_mat.copy()
        res[time_grid[:-1]]     = actuarial_rates_dot_mat.diff()[time_grid[1:]]/(time_grid[1:]-time_grid[:-1])
        res                      = res.fillna(method='ffill')
        return(res)
    
    def interpolate_spot_zc_prices(self, timegrid):
        """ returns: interpolation of zero coupon prices with respect to the timegrid
            interpolation is performed via the computation of forward rates
        """
        forward_rates          = self.compute_forward_rates(timegrid)
        res                    = forward_rates.copy()
        tenors                 = forward_rates.index
        res[0]                 = 1.0
        integral_forward_rates = forward_rates[tenors[:-1]]*(tenors[1:]-tenors[:-1])
        res[tenors[1:]]        = np.exp(-integral_forward_rates.cumsum())
        return(res)


class CIR(object):
    """ COX INGERSOL ROSS model
        mean reverting positive process
    """
    def __init__(self, initial, target, speed, sigma):
        self.initial    = initial
        self.target     = target
        self.speed      = speed
        self.sigma      = sigma
    
    @classmethod
    def instantiate_via_target_std(cls, initial, target, speed, std):
        """ Alternative instantiation via target standard deviation of the process
        """
        sigma = std*sqrt(2*speed/target)
        return(cls(initial, target, speed, sigma))
        

    def mean(self, maturity, init=None):
        """ mean of the process at maturity when initial point is init
        """
        if not(init):
            init=self.initial
        return(self.target+(init-self.target)*np.exp(-self.speed*maturity))

    def variance(self, maturity, init=None):
        """ variance of the process at maturity when initial point is init
        """
        if not(init):
            init=self.initial
        res = init*self.sigma*self.sigma*np.exp(-self.speed*maturity)*(1-np.exp(-self.speed*maturity))/self.speed
        res = res + 0.5*self.target*self.sigma*self.sigma*(1-np.exp(-self.speed*maturity))**2/self.speed
        return(res)
        
    def generate_brownian(self, sim_grid):
        """ Generates normalized brownian increments
        """
        dt              = sim_grid.dt
        if self.dB is None:
            self.dB = np.random.normal(loc=0, scale=1, size=(sim_grid.nb_simulations, sim_grid.nb_steps*sim_grid.horizon))*sqrt(dt)

    def simulate(self, sim_grid):
        """ Simulation with Alfonsi discretisation scheme
        """
        dt              = sim_grid.dt
        self.sim_grid   = sim_grid
        self.generate_brownian(sim_grid)
        self.paths      = np.ndarray(sim_grid.shape)
        self.paths[:,0] = self.initial
        for j in range(1,self.paths.shape[1]):
            self.paths[:,j] = np.power(
                                  (self.sigma*self.dB[:,j-1]+np.sqrt(
                                                                  np.power(self.sigma*self.dB[:,j-1],2)+
                                                                  4*(self.paths[:,j-1]+dt*(self.speed*self.target-0.5*self.sigma**2))*(1+self.speed*dt)
                                                                 )
                                  )
                                  /(2+2*self.speed*dt) 
                                 ,2)
        self.simulations = pd.DataFrame(self.paths, columns=self.sim_grid.time_grid)


class BlackScholes(object):
    """ Black Scholes Model
    """
    def __init__(self, initial, rate, sigma, dividend=0.0):
        self.initial    = initial
        self.rate       = rate
        self.dividend   = dividend
        self.sigma      = sigma
        
    def generate_brownian(self, sim_grid):
        """ Generates normalized brownian increments
        """
        dt              = sim_grid.dt
        if self.dB is None:
            self.dB = np.random.normal(loc=0, scale=1, size=sim_grid.shape)*sqrt(dt)
             
    def simulate(self, sim_grid):
        dt              = sim_grid.dt
        self.sim_grid   = sim_grid
        self.generate_brownian(sim_grid)
        self.paths      = np.ndarray(sim_grid.shape)
        self.paths[:,0] = self.initial
        for j in np.arange(1,self.paths.shape[1]):
            self.paths[:,j] = self.paths[:,j-1]*np.exp((self.rate-self.dividend)*dt+self.sigma*self.dB[:,j-1]-0.5*self.sigma*self.sigma*dt)
        self.simulations = pd.DataFrame(self.paths, columns=self.sim_grid.time_grid)
        
    
    