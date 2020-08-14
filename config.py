"""
    All configuration variable goes here, imported from app.py
"""
from flask import Flask, render_template
from flask import Flask, url_for


import mysql.connector
import pandas as pd
import os

app = Flask(__name__)


#Database Config
def connectDatabase():
    return  mysql.connector.connect(
                host="127.0.0.1",
                port=3306,
                user="root",
                password="root",
                database="pfe")


#Limits forecasting horizons
MONTHLY_LIMITS = 24 
WEEKLY_LIMITS = 54 
WEEKLY2_LIMITS = 27


#Model sequence length for both input and output
input_sequence_length = 15
output_sequence_length = 15


#This is the date, where hypothetically the prediction is gonna start in
hypoth_current_date = "2015-12-20"   


#Max number of data points that was used to filter products' quantity for forecasting them
#This means that only products with more than 20 data points in the last two years were selected for forecasting 
MAX_DATA_POINTS = 20



#Categories that will show up as an option in the main page
categories_options = ["O1010","O1050","M1030","SD390080","SD310120","O1020","M1010","M1020","L1030","","O1040","L1090","J1030","O1070","A1020","GZ0200121","GZ0600333","GZ0501703","GZ0800312","GZ0200112","GZ1100113","GZ0200111","GZ0800311","GZ1900211","GZ1000414","GZ1200413","GZ0600323","GZ1801262","GZ0301362","GZ2600443","GZ1801261","GZ1000411","GZ0200122","GZ0800313","GZ0801162","GZ1400611","GZ0600330","GZ0801161","GZ0401801","GZ1300511","GZ0401701","GZ0401401","GZ1300521","GZ0800314","GZ1801241","GZ0401411","GZ0401702","GZ0800315","GZ1000412","GZ0401416","GZ0401412","GZ0801421","GZ0301351","GZ2201662","GZ2600444","GZ1100123","GZ0301363","GZ0401413","GZ2201661","GZ0301191","GZ1500613","GZ0301981","GZ1100133","GZ0501803","GZ0402270","GZ1300522","DIVERS","5267"]