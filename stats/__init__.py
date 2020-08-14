from config import *

import numpy as np
import datetime

from stats import utils

"""
    Return sales quantiy, if start time + horizon is in the future return the actual past sales with the future forecasts
    Parameters:
    unit: w for weekly data, 2w for half monthly data, m for monthly data
    category: the category of the products you want to fetch
    Horizon: horizon of the prediction in the future
    Start: beginning date of the data, default values is the current date (or the hypothetical current date) minus 49 which is 7 weeks
            that mean, the default behaviour of the function to fetch the first 7 weeks' data and to forecast the 8 following weeks

"""
def GetSalesQuantity(unit="w", category="GZ1900211", Horizon=15, 
                                                            Start=(datetime.datetime.strptime(hypoth_current_date ,"%Y-%m-%d") - datetime.timedelta(days=49)).strftime("%Y-%m-%d")):


    #Get the unit in both the numerical form and the character form
    Unit = {"w": 7, "m":30}

    try:
        num_unit = Unit[unit]
    except:  
        print("Unit doesn't exist")  


    #Make sure that the inserted date is not in the future
    #TODO: Make an error page for this    
    assert(datetime.datetime.strptime(Start ,"%Y-%m-%d") < datetime.datetime.strptime(hypoth_current_date ,"%Y-%m-%d"))


    #Clip the value of the horizons so it doesn't exceed the maximum
    horizon = 15 
    if num_unit == 30:
        horizon = np.clip(Horizon, 3, MONTHLY_LIMITS)
    elif num_unit == 15:
        horizon = np.clip(Horizon, 3, WEEKLY2_LIMITS)
    else:
        horizon = np.clip(Horizon, 3, WEEKLY_LIMITS)

    
    #Parse the date and figure out the exact beginning date and end date
    #Beginning date must be the beginning of a the week or beginning of the month (6th day)
    start, end, forecast, forecastStarts, futureSteps, monthlyForecastStart = utils.getFirstDay(num_unit, Start, horizon)


    forecasted = utils.getForecast(futureSteps, forecast ,forecastStarts, start, end, category, unit, monthlyForecastStart, horizon)
    
    assert(forecasted.index.shape[0] == horizon)

    return (forecasted.T, category, unit, end, start, horizon, "w", futureSteps, forecastStarts)