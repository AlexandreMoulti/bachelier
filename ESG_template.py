# -*- coding: utf-8 -*-
import xlwings as xw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sc
from bachelier import *

def launch_simulations():
    wb = xw.Book.caller()
    horizon = wb.sheets['Main'].range('B3').value
    nb_steps = wb.sheets['Main'].range('B4').value
    nb_sim = wb.sheets['Main'].range('B5').value
    
    out_repo = wb.sheets['Main'].range('B6').value
    
    grid = DiscretisationGrid(nb_sim,horizon,nb_steps)
    zc_prices = wb.sheets['market_data'].range('F2:G33').options(pd.Series, index_col=0, numbers= np.float64).value
    
    speed = wb.sheets['model_parameters'].range('B2').value
    vol = wb.sheets['model_parameters'].range('B3').value
    
    hw = HW1F(speed, vol, zc_prices)
    hw.simulate(grid)
    short_rates = hw.get_short_rates_simulations()
    short_rates.to_csv(out_repo+'simulations_short_rates.csv')
    r1 = hw.get_actuarial_rates_simulations(1)
    r1.to_csv(out_repo+'simulation_actuarial_rates_1Y.csv')
    r10 = hw.get_actuarial_rates_simulations(10)
    r10.to_csv(out_repo+'simulation_actuarial_rates_10Y.csv')   
    

if __name__ == '__main__':
    # Expects the Excel file next to this source file, adjust accordingly.
    xw.Book('ESG_template.xlsm').set_mock_caller()
    launch_simulations()
    
    
    