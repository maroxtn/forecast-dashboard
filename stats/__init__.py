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



def indexVals():

    cnx = connectDatabase()



    #Getting the monthly revenues of this year and last year
    currdate = datetime.datetime.strptime(hypoth_current_date ,"%Y-%m-%d")

    currdate = datetime.date(currdate.year - 1, 1, 1)

    query = "select YEAR(posting_date) , MONTH(posting_date) ,sum(sales_amount_expected + sales_amount_actual) from transactions WHERE posting_date BETWEEN '{0}' and '{1}' AND entry_type LIKE '%sale%' GROUP BY MONTH(posting_date), YEAR(posting_date);"
    query = query.format(currdate, hypoth_current_date)

    dataCursor = cnx.cursor()
    dataCursor.execute(query)
    dataResults = dataCursor.fetchall()

    results = pd.DataFrame(dataResults, columns = ['year', 'month', 'qty'])

    year1 = np.round((results["qty"].values[0:12].astype("float")/1000), decimals=3) #Convert unity to dinar since it is in millime
    year2 = np.round((results["qty"].values[12:-1].astype("float")/1000), decimals=3)


    year_1 = ' '.join(str(e) for e in year1)   #Stringify the array
    year_2 = ' '.join(str(e) for e in year2) 



    #Getting all columns of this month
    currdate = datetime.datetime.strptime(hypoth_current_date ,"%Y-%m-%d")
    currdate = datetime.date(currdate.year, currdate.month, 1)

    query = "select item_no, posting_date, source_no, document_no, (sales_amount_expected + sales_amount_actual) as revenue from transactions WHERE posting_date BETWEEN '{0}' and '{1}' AND entry_type LIKE '%sale%' AND document_type LIKE '%Sales Shipment%';"
    query = query.format(currdate, hypoth_current_date)
    
    dataCursor = cnx.cursor()
    dataCursor.execute(query)
    dataResults = dataCursor.fetchall()   

    results = pd.DataFrame(dataResults, columns = ['productName', 'date', 'client', 'facture', 'revenue'])
    results['date'] = pd.to_datetime(results['date'])  

    #Get revenue per product and select top 10
    top = results[['productName', 'revenue']]
    top = top.groupby("productName").sum().sort_values(by=['revenue'], ascending=False)[:10]
    tops = np.round(top.values.astype('float') / 1000, decimals=3)

    tops_values = ' '.join(str(e) for e in np.squeeze(tops))
    tops_columns = ' '.join(str(e) for e in np.squeeze(top.index.values))


    #Get top 10 clients
    clientsTop = results[['client', 'revenue']]
    clientsTop = clientsTop.groupby('client').sum().sort_values(by=['revenue'], ascending=False)[:7]

    clientsTop_values = np.squeeze(np.round(clientsTop.values.astype('float') / 1000, decimals=3))
    clientsTop_column = np.squeeze(clientsTop.index.values)


    #day values
    ordersPerDay = results.groupby("date")["facture"].nunique()
    revenuePerDay = results[['date', 'revenue']].groupby('date').sum()

    revenuePerDay["per"] = revenuePerDay.pct_change()["revenue"]*100
    ordersPerDay = pd.DataFrame({'values': ordersPerDay.values, 'per':ordersPerDay.pct_change()*100}, index=revenuePerDay.index)


    #Month single values
    monthlyOrders = results[['facture', 'revenue']].groupby("facture").sum().index.shape[0]
    monthlyRevenue = results['revenue'].values.sum()

    return (year_1, year_2, tops_values, tops_columns, clientsTop_values, clientsTop_column, revenuePerDay, ordersPerDay, monthlyOrders, monthlyRevenue)