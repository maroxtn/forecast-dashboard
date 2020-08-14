from dateutil.relativedelta import relativedelta
import datetime

import pandas as pd
import re

import numpy as np

from config import *


# Let's define a small function that predicts based on the trained encoder and decoder models
def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict, num_y_signals):
    """Predict time series with encoder-decoder.
    
    Uses the encoder and decoder models previously trained to predict the next
    num_steps_to_predict values of the time series.
    
    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder model.
    decoder_predict_model: The Keras decoder model.
    num_steps_to_predict: The number of steps in the future to predict
    
    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)

    # The states must be a list
    if not isinstance(states, list):
        states = [states]

        
    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0], 1, num_y_signals))


    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
        [decoder_input] + states, batch_size=1)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)
    

"""Function to parse the start date, and return the Start date, 
End date, forecasting end date, and if there is any forecast to be done"""
def getFirstDay(unit, Start, horizon):

    start = datetime.datetime.strptime(Start ,"%Y-%m-%d")
    end = datetime.datetime.strptime(Start ,"%Y-%m-%d") #Default value its going to change
    forecast = False
    monthlyForecastStart = datetime.datetime.strptime(hypoth_current_date ,"%Y-%m-%d")
    forecastStart = datetime.datetime.strptime(hypoth_current_date ,"%Y-%m-%d")

    if unit == 7 or unit == 14:
        
        while start.weekday() != 6:  #Substract a day till you reach day 6 which is sunday (start of the week)
            start = start - datetime.timedelta(days=1)

        end = start + datetime.timedelta(days=int(unit*horizon))

    elif unit == 30:
        while start.day != 1:
            start = start - datetime.timedelta(days=1)

        end = start + relativedelta(months=+horizon)


    #Keep substracting till you get to the first day of the month or week
    #After this we have a concrete end date of the forecast, a concrete ending date of the data query
    #and a concrete starting date of the query
  
    while forecastStart.weekday() != 6: 
        forecastStart = forecastStart - datetime.timedelta(days=1)

    while monthlyForecastStart.day != 1:
        monthlyForecastStart = monthlyForecastStart - datetime.timedelta(days=1)
    
    futureSteps = 0
    #Check doing forecast is required or not
    if(end > forecastStart):
        forecast = True
        futureSteps = int((end - forecastStart).days/7) #Steps to forecast into the future

    return (start, end, forecast, forecastStart, futureSteps, monthlyForecastStart)
    

#If forecast get the products that were forecasted
#Select the last input_seq_len weeks before the current time to use it for the prediction
#Put them in the right format and do the prediction
#Parse the prediction, assign for each prediction the product name in a dictionary
#Callbacks an array with a list of functions, every function will be passed with the results as parameter
def getForecast(futureSteps, forecast, forecastStarts, start, end, category, unit, monthlyForecastStart, horizon, callbacks = []):


    if forecast:

        import pickle

        from sklearn.preprocessing import MinMaxScaler
        import tensorflow.keras as keras


    cnx = connectDatabase()

    #The start and end date of the sequence that is going to be extracted to feed the model to do the prediction
    seqStart = (forecastStarts - datetime.timedelta(days=7*input_sequence_length))
    seqEnd = forecastStarts

    queryStart = None
    queryEnd = None

    #If the start of the query is more recent than the needed sequence to feed the model, then let the queryStart be in  
    #input_sequence_length in past, and then take a slice of the needed time period

    #if fetch start is going to be before length of the necessary sequence AND there is forecasting
    if start > seqStart and forecast:  
        queryStart = seqStart.strftime("%Y-%m-%d")
    else:
        queryStart = start.strftime("%Y-%m-%d")

    #If there is going to be forecasting, let queryEnd be in the present date, otherwise let the query end in the user specified date
    if forecast:
        queryEnd = seqEnd.strftime("%Y-%m-%d")
    else:
        queryEnd = end.strftime("%Y-%m-%d")




    #Only products to predict for now are "G" nature product because only these product have enough data points (To Change)
    #This Query contains products with item_category_code that are in the G nature
    query = """SELECT item_no, item_category_code, product_group_code ,posting_date, quantity*-1  FROM transactions WHERE entry_type LIKE '%sale%' AND (posting_date BETWEEN '{0}' AND '{1}')"""
    
    query = query.format(queryStart, queryEnd)

    if not forecast: #If there is not forecast, no need to fetch all the products, just fetch the products with the chosen category
        query = query + " AND product_group_code LIKE '%"+ category +"%';"


    dataCursor = cnx.cursor()
    dataCursor.execute(query)
    dataResults = dataCursor.fetchall()

    #Parse the results into a panda DataFrame
    results = pd.DataFrame(dataResults, columns = ['product', 'group', 'group2', 'date', 'quantity'])
    results["date"] = pd.to_datetime(results['date'])
    results["group"] = results.group.apply(lambda x: re.sub(r'\W+', '', x))


    for callback in callbacks: #Callback functions to handle more statistical data, so we don't waste time retrieving data again
        callback(results)


    #In the case of no forecast
    if not forecast:

        index = pd.date_range(start=queryStart, end=queryEnd, freq=unit) 

        products = results.groupby("product")
        total = pd.DataFrame(index=index)

        for i, g in products:
            
            product = products.get_group(i)
            productName = product["product"].array[0]

            #Remove them we no longer need them
            del product['group']
            del product["product"]
            del product['group2']

            product = product.groupby("date").sum()
            product = product.astype("float64")

            
            product = product.resample(unit).sum().reindex(index).fillna(0)
            product[productName] = product["quantity"].apply(lambda x: 0.0 if x<0 else x) #Make all negative values zeros

            del product["quantity"]

            total = pd.concat([total, product], axis=1, sort=False)


        if unit == "w": #Because if the weekly data there is one unneed extra step at the start
            total = total[1:]
        
        return total


    index = pd.date_range(start=seqStart, end=seqEnd, freq='w')[1:]

    if unit == "w":
        index0 = pd.date_range(start=start, end=forecastStarts, freq='w')[1:]
    else:
        index0 = pd.date_range(start=start, end=monthlyForecastStart, freq='m')


    colsToBePredicted = []

    #Index must be 15 steps in the past weekly
    #Index must be n steps in the past monthly

    #Data fram for holding the residual data
    residual = pd.DataFrame({"residual" : np.zeros(len(index))}, index=index)

    total = pd.DataFrame(index=index)
    total0 = pd.DataFrame(index=index0)

    total = total.assign(Yearday=total.index.dayofyear)  #Re-create the features, might be changed
    total = total.assign(Yearweek=total.index.weekofyear)
    total = total.assign(Monthday=total.index.day)

    qtys = ["residual"]

    products = results.groupby("product")

    for i, g in products:
        
        product = products.get_group(i)
        
        productName = product["product"].array[0]
        productGroup = product["group"].array[0]
        productGroupCode = product["group2"].array[0]
    
        del product["group"]
        del product["product"]
        del product["group2"]

        product = product.groupby("date").sum()
        product = product.astype("float64")

        product0 = product.resample(unit).sum().reindex(index0).fillna(0)  #Product Dataframe with the past data
        product0["quantity"] = product0["quantity"].apply(lambda x: 0.0 if x<0 else x)

        product = product.resample('w').sum().reindex(index).fillna(0)
        product["quantity"] = product["quantity"].apply(lambda x: 0.0 if x<0 else x) #Make all negative values zeros

        admissibleCols = ['G00005', 'G00006', 'G00008', 'G00009', 'G00011', 'G00012', 'G00016', 'G00019', 'G00033', 'G00034', 'G00041', 'G00045', 'G00058', 'G00064', 'G00096', 'G00121', 'G00124', 'G00133', 'G00145', 'G00159', 'G00169', 'G00174', 'G00176', 'G00181', 'G00184', 'G00185', 'G00187', 'G00188', 'G00197', 'G00207', 'G00210', 'G00255', 'G00257', 'G00267', 'G00279', 'G00282', 'G00324', 'G00328', 'G00331', 'G00339', 'G00340', 'G00346', 'G00347', 'G00348', 'G00349', 'G00352', 'G00397', 'G00493', 'G00507', 'G00568', 'G00672', 'G00676', 'G00679', 'G00686', 'G00687', 'G00692', 'G00697', 'G00708', 'G00717', 'G00718', 'G00725', 'G00733', 'G00735', 'G00739', 'G00745', 'G00746', 'G00747', 'G00760', 'G00770', 'G00771', 'G00954', 'G00969']



        #Create a seperate dataframe that fetch the values of the quantities of the specified group from the bigger dataframe
        if productGroupCode == category:
            
            product0[productName] = product0["quantity"]
            del product0["quantity"]

            total0 = pd.concat([total0, product0], axis=1, sort=False)

            if productName in admissibleCols:
                colsToBePredicted.append(productName)


        #If the number of the weeks bigger than the input_sequence_length
        if start < seqStart:
            #Then trim the product DataFrame to only the timesteps needed to feed to the model (input_sequence_length steps)
            product = product[:input_sequence_length]
            
        if productName in admissibleCols:

            qtColName = productName
            ctColName = productName+ "_group"

            groupList = ['70', 'GZ0200011', 'GZ0300020', 'GZ0400018', 'GZ0500019', 'GZ0600033',  #List of the groups of the product
                            'GZ0800031', 'GZ1000041', 'GZ1100013', 'GZ1200043', 'GZ1300051',  #We need that to recreate the features
                            'GZ1400061', 'GZ1500063', 'GZ1800012', 'GZ1900021', 'GZ2200023',
                            'GZ2600044']


            k=0 #The index of the group in the array
            for i in range(len(groupList)):
                if groupList[i] == productGroup:
                    k = i
            
            tempDf = pd.DataFrame(
                {qtColName: product["quantity"]
                ,ctColName: k
                })      

            total = pd.concat([total, tempDf], axis=1, sort=False)
            qtys.append(qtColName)  

        else:

            residual = residual.add(product.values)
    
    
    total = pd.concat([total, residual], axis=1, sort=False)



    #Now we reproduced the Dataframe like it was done during the train
    #Next step to shape it in the right format to feed it to the model
    #Shape is (1, input_seq_len, number of cols of the dataframe)

    input_seq = np.expand_dims(total.values, axis=0)
    
    #import minmax scalers
    path = '../static/models/{0}'
    basedir = os.path.abspath(os.path.dirname(__file__))

    x_scaler = pickle.load( open( os.path.join(basedir, path.format("x_scaler.pkl")) , "rb" ) )
    y_scaler = pickle.load( open( os.path.join(basedir, path.format("y_scaler.pkl")) , "rb" ) )

    input_seq[0] = x_scaler.transform(input_seq[0])

    #Rebuild the model as in the training  
    layers = [35,35]
    num_x_signals = input_seq.shape[2]
    num_y_signals = 73

    encoder_inputs = keras.layers.Input(shape=(None, num_x_signals))
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons))
    encoder = keras.layers.RNN(encoder_cells, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]
    decoder_inputs = keras.layers.Input(shape=(None, num_y_signals))

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons))
        
    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_outputs_and_states[0]
    decoder_dense = keras.layers.Dense(num_y_signals, activation='linear')
    decoder_outputs = decoder_dense(decoder_outputs)

    learning_rate = 1.0e-2
    optimiser = keras.optimizers.RMSprop(lr=learning_rate)

    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss="mse")

    #import the model now
    try:
        model.load_weights(  os.path.join(basedir, path.format("model.keras"))  )
    except Exception as error:
        print("Error trying to load weights.")
        print(error)
    
    encoder_predict_model = keras.models.Model(encoder_inputs,
                                            encoder_states)

    decoder_states_inputs = []

    for hidden_neurons in layers[::-1]:
        # One state for GRU
        decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

    decoder_outputs_and_states = decoder(
        decoder_inputs, initial_state=decoder_states_inputs)

    decoder_outputs = decoder_outputs_and_states[0]
    decoder_states = decoder_outputs_and_states[1:]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_predict_model = keras.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)


    #Now that we got everything settled, lets predict the values
    predictions = predict(input_seq, encoder_predict_model, decoder_predict_model, futureSteps, num_y_signals)
    predictions_unscaled = y_scaler.inverse_transform(predictions[0])

    predictions_unscaled[predictions_unscaled < 0] = 0  #Negative value become zero


    index = pd.date_range(start=queryEnd, end=(forecastStarts + datetime.timedelta(days=int(7*futureSteps))).strftime("%Y-%m-%d"), freq='w')[1:]
    predictions_df = pd.DataFrame(predictions_unscaled, index=index, columns=qtys)

    print(predictions_df)
    #If the forecast is monthly, and the forecast start date doesn't match the monthly forecast start date, then merge the first weeks 
    # of the month with forecast
    if unit == "m" and monthlyForecastStart != forecastStarts:
        predictions_df = pd.concat([total[colsToBePredicted][monthlyForecastStart:], predictions_df], axis=0, sort=False)


    #If the data is monthly, then resample the prediction from being weekly to month 
    if unit == "m":
        predictions_df = predictions_df[colsToBePredicted].resample("m").sum()


    #Concatenate the past values dataframe and the forecasted dataframe together
    final_df = pd.concat([total0, predictions_df[colsToBePredicted]], axis=0, sort=False)   

    if unit == "m":
        if final_df.index.shape[0] != horizon:  #Sometimes the model predicts one extra step (this is because of futureSteps calculation process),
            final_df = final_df[:-1]            # so when it happens we trim it


    return final_df