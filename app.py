from flask import Flask, render_template
from flask import Flask, url_for

import mysql.connector
import pandas as pd
import os

import pickle
import time
import datetime
from flask import request
from dateutil.relativedelta import relativedelta

import config #Import the configuration file, same directory
import stats

app = Flask(__name__)


hypoth_current_date = datetime.datetime.strptime(config.hypoth_current_date,"%Y-%m-%d")



@app.route("/", methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
def index():


    #Check if request method is post or get, if GET use the default values and apss it to the GetSalesQuantity
    #if POST get the values the user specified and pass them to the function 
    if(request.method == 'POST'):

        unit = request.form['unit'] #Possible Values w, 2w, m

        #An idea: arrange items not depending on their category but on their item_cateogry, then color the items in the dashboard with the same category
        category = request.form['category'] #Categories of the products
            
        horizon = int(request.form['horizon']) #How many steps of the unit to predict in the future

        start = request.form["start"] #Starting date of these predictions

        results, category, unit, end, start, horizon, Unit, futureSteps, forecastStarts = stats.GetSalesQuantity(unit, category, horizon, start)


    elif (request.method == 'GET'):

        results, category, unit, end, start, horizon, Unit, futureSteps, forecastStarts = stats.GetSalesQuantity()


    #request.method goes as a parameter to GetSalesQuantity
    #Create a list of the variables needed to rende the template, and think about a way to represent it in one big dictionary
    #Read more about the templating system of flask

    return render_template ('index.html', title="Main", selectedCat=category, unit=unit, groups= config.categories_options,
                           endDate = end,  startDate=start, products=results, horizon=horizon,
                            Unit=Unit, predictHorizon=futureSteps, forecast_start = forecastStarts, sum=results.sum(axis=0))



@app.route("/forecasts", methods=['GET', 'POST'])
def forecasts():


    #Check if request method is post or get, if GET use the default values and apss it to the GetSalesQuantity
    #if POST get the values the user specified and pass them to the function 
    if(request.method == 'POST'):

        unit = request.form['unit'] #Possible Values w, 2w, m

        #An idea: arrange items not depending on their category but on their item_cateogry, then color the items in the dashboard with the same category
        category = request.form['category'] #Categories of the products
            
        horizon = int(request.form['horizon']) #How many steps of the unit to predict in the future

        start = request.form["start"] #Starting date of these predictions

        results, category, unit, end, start, horizon, Unit, futureSteps, forecastStarts = stats.GetSalesQuantity(unit, category, horizon, start)


    elif (request.method == 'GET'):

        results, category, unit, end, start, horizon, Unit, futureSteps, forecastStarts = stats.GetSalesQuantity()


    #request.method goes as a parameter to GetSalesQuantity
    #Create a list of the variables needed to rende the template, and think about a way to represent it in one big dictionary
    #Read more about the templating system of flask

    return render_template ('forecasts.html', title="Main", selectedCat=category, unit=unit, groups= config.categories_options,
                           endDate = end,  startDate=start, products=results, horizon=horizon,
                            Unit=Unit, predictHorizon=futureSteps, forecast_start = forecastStarts, sum=results.sum(axis=0))



if __name__ == '__main__':
    app.run(debug=True)