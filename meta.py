import numpy as np
import pandas as pd
import re
import datetime
import mysql.connector

import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error


from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from tensorflow.keras import backend as K

from sklearn.preprocessing import MinMaxScaler, RobustScaler, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import tensorflow
import pickle


print("Libraries Loaded")


dim_learning_rate = Real(low=1e-3, high=1e-1, prior="log-uniform", name="learning_rate")
dim_num_dense_layers = Integer(low=1, high=4, name='num_layers')
dim_num_dense_nodes = Integer(low=5, high=100, name='num_nodes')
optimizer_dim = Categorical(categories=[0, 1], name='optimiser')
month = Categorical(categories=[False, True], name='month')
quarter = Categorical(categories=[False, True], name='quarter')
year = Categorical(categories=[False, True], name='year')
dayyear = Categorical(categories=[False, True], name='dayyear')
weekyear = Categorical(categories=[False, True], name='weekyear')
day = Categorical(categories=[False, True], name='day')
residual = Categorical(categories=[False, True], name='residualV')
dim_num_batch_size = Integer(low=20, high=150, name='batch_size')
dim_num_steps_epoch = Integer(low=20, high=100, name='steps_epoch')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              optimizer_dim,
              dim_num_batch_size,   
              dim_num_steps_epoch,
              month,
              quarter,
              year, dayyear, weekyear, day, residual]

default_parameters = [1e-2, 2, 35, 0, 128, 50, True, True, True, True, True, True, True]

def log_dir_name(learning_rate, num_layers,
                 num_nodes, opt, batch_size, steps_epoch, stri):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}_batch{4}_epoch{5}{6}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_layers,
                       num_nodes,opt, batch_size, steps_epoch, stri)

    return log_dir

cnx = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="root",
    database="pfe")

nature = "G"
nQuery = "select item_category_code from nature_map WHERE nature = '"+ nature +"';"

natureQuery = cnx.cursor()
natureQuery.execute(nQuery)
itemCategoryCode = natureQuery.fetchall()
extension = ""

for code in itemCategoryCode:
    c = code[0]
    extension = extension + " OR item_category_code LIKE '%"+ c +"%'"


extension = extension[3:]
query = "SELECT item_no, item_category_code ,posting_date, quantity*-1  FROM transactions WHERE entry_type LIKE '%sale%' AND (" + extension + ");"

dataCursor = cnx.cursor()
dataCursor.execute(query)
dataResults = dataCursor.fetchall()

results = pd.DataFrame(dataResults, columns = ['product', 'group', 'date', 'quantity'])
results["date"] = pd.to_datetime(results['date'])
results["group"] = results.group.apply(lambda x: re.sub(r'\W+', '', x))

print("SQL queried")

def mapGroups(group, groupList, inverse = False):
        
        if inverse:
            groupList[group]
        else:
            for i in range(len(groupList)):
                if groupList[i] == group:
                    return i

products = results.groupby("product")



print("Data processed")

print("DL libraries are loaded")


highest_per = 1
path_best_model = 'abdessalem.h5'


@use_named_args(dimensions=dimensions)

def fitnenss(learning_rate, num_layers,
            num_nodes, optimiser, batch_size, steps_epoch, month, quarter, year, dayyear, weekyear, day, residualV):   



    def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict):

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
    

    groupList = results[["group"]].groupby("group").sum().index #Number of groups

    index = pd.date_range(start="2012-03-04", end="2016-12-04", freq='w')
    residual = pd.DataFrame({"residual" : np.zeros(len(index))}, index=index)

    qtys = []
    if residualV:
        qtys = ["residual"]

    total = pd.DataFrame(index=index)

    if dayyear:
        total = total.assign(Yearday=total.index.dayofyear)  #Time related features
    if weekyear:
        total = total.assign(Yearweek=total.index.weekofyear)
    if day:
        total = total.assign(Monthday=total.index.day)
    if year:
        total = total.assign(Year=total.index.year)
    if month:
        total = total.assign(month=total.index.month)
    if quarter:
        total = total.assign(quarter=total.index.quarter)

    k=0
    groupsList = []

    for i, g in products:
        
        product = products.get_group(i)
        
        productName = product["product"].array[0]
        productGroup = product["group"].array[0]
        
        del product["group"]
        del product["product"]
        
        product = product.groupby("date").sum()
        product = product.astype("float64")

        #roduct.resample("w").sum()["2014-12-04":"2016-12-04"].astype(bool).sum(axis=0).values[0]
        dataPoints = product.resample("w").sum()["2014-12-04":"2016-12-04"].astype(bool).sum(axis=0).values[0] #Data Points in the last two years
                                                            #If more than 10 we'll consider into our prediction
                                                            #If not I'm gonna put it in the residual column

            
        #Fill empty spots and make them same size
        product = product.resample('w').sum().reindex(index).fillna(0)    
        product["quantity"] = product["quantity"].apply(lambda x: 0.0 if x<0 else x) 
        
        
        if(dataPoints < 20):
            residual = residual.add(product.values)
        else:
            qtColName = productName + "_qty"
            ctColName = productName+ "_group"
            groupsList.append(mapGroups(productGroup, groupList))
            
            tempDf = pd.DataFrame(
                {qtColName: product["quantity"]
                ,ctColName: mapGroups(productGroup, groupList)
                })
                
            
            total = pd.concat([total, tempDf], axis=1, sort=False)
            
            qtys.append(qtColName)

    if residualV:
        total = pd.concat([total, residual], axis=1, sort=False)

    products_2 = results.groupby("product")
    from sklearn.metrics import mean_squared_error

    index = pd.date_range(start="2012-03-04", end="2016-12-04", freq='w')

    prods = []
    for i in range(1,len(qtys)):
        prods.append(qtys[i][:-4])
        
    t = 0
    k=0

    naive = {}

    for i, g in products_2:
        
        product = products_2.get_group(i)
        prod= product["product"].array[0]
        product = product.groupby("date").sum()
        
        
        
        del product["group"]
        del product["product"]
        
        product = product.resample('w').sum().reindex(index).fillna(0)    
        product["quantity"] = product["quantity"].apply(lambda x: 0.0 if x<0 else x) 
        
        coef = 0.8
        delimiter = int((1-coef) *index.shape[0])
        
        validate = product.tail(delimiter)
        
        prediction = product[index.shape[0] - delimiter*2:index.shape[0] - delimiter]

        if(prod in prods):
            rmse = np.sqrt(mean_squared_error(prediction.values.astype("float"), validate.values.astype("float")))
            naive[prod] = rmse
            
            t = t + rmse
            k = k + 1
            


    product = total


    target = product[qtys]

    x_data = product.values #trimming the end of the serie because of the nan values resulted from the shift
    y_data = target.values



    data_count = len(x_data)
    train_split = 0.8

    num_train = int(train_split * data_count)
    num_test = data_count - num_train

    #Creating the test and the train data
    x_train = x_data[:num_train]
    x_test = x_data[num_train:]

    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]



    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]



    x_scaler = MaxAbsScaler() #Normalize the data
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    y_scaler = MaxAbsScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)



    def batch_generator(batch_size, input_seq_len, target_seq_len):
        """
        Generator function for creating random batches of training-data.
        """

        while True:

            x_shape = (batch_size, input_seq_len, num_x_signals)
            y_shape = (batch_size, target_seq_len, num_y_signals)
            
            encoder_input = np.zeros(shape=x_shape, dtype=np.float16)
            decoder_output = np.zeros(shape=y_shape, dtype=np.float16)
            decoder_input = np.zeros(shape=y_shape, dtype=np.float16)
            
            total_length = input_seq_len + target_seq_len

            for i in range(batch_size):

                idx = np.random.randint(num_train - total_length)

                encoder_input[i] = x_train_scaled[idx:idx+input_seq_len]
                decoder_output[i] = y_train_scaled[idx+input_seq_len:idx+total_length]
            
            yield ([encoder_input, decoder_input], decoder_output)

    
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('Number of layers:', num_layers)
    print('Number of nodes:', num_nodes)

    stri = ""
    if(month):
        stri = stri + "Month "
    if(quarter):
        stri = stri + "Quarter "
    if(year):
        stri = stri + "Year "
    if(dayyear):
        stri = stri + "Dayyear "
    if(weekyear):
        stri = stri + "Weekyear "
    if(day):
        stri = stri + "Day "
    if(residualV):
        stri = stri + "Residual "
    
    print("State " + stri)

    if optimiser == 0:
        print("Optimiser: RMSProp")
    else:
        print("Optimiser: Adam")

    print('Batch size:', batch_size)
    print('Steps epoch', steps_epoch)

    generator = batch_generator(batch_size=batch_size,
                                input_seq_len=15, target_seq_len=15)


    input_seq_len = 15
    target_seq_len = 15
    validation_data = ([np.expand_dims(x_test_scaled[:input_seq_len], axis=0), np.zeros(shape=(1, target_seq_len, num_y_signals), dtype=np.float16)],
                    np.expand_dims(y_test_scaled[input_seq_len:input_seq_len+target_seq_len], axis=0))





    layers = []
    for i in range(num_layers):
        layers.append(num_nodes)

    #Encoder
    encoder_inputs = keras.layers.Input(shape=(None, num_x_signals))

    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons))

    encoder = keras.layers.RNN(encoder_cells, return_state=True)

    encoder_outputs_and_states = encoder(encoder_inputs)

    encoder_states = encoder_outputs_and_states[1:]



    #Decoder
    decoder_inputs = keras.layers.Input(shape=(None, num_y_signals))

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons))
        
    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    decoder_outputs = decoder_outputs_and_states[0]

    decoder_dense = keras.layers.Dense(num_y_signals, activation='linear')

    decoder_outputs = decoder_dense(decoder_outputs)



    if(optimiser == 0):
        optimiser = keras.optimizers.RMSprop(lr=learning_rate)
    else:
        optimiser = keras.optimizers.Adam(lr=learning_rate)
    
    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss="mse")





    
    log_dir = log_dir_name(learning_rate, num_layers,
                           num_nodes, optimiser, batch_size, steps_epoch, stri)

    callback_early_stopping = EarlyStopping (monitor='val_loss',
                                        patience=10, verbose=1)
    
    checkpoint_name = "checks/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}_batch{4}_epoch{5}{6}.keras".format(learning_rate,
                       num_layers,
                       num_nodes,optimiser, batch_size, steps_epoch, stri)

    path_checkpoint = checkpoint_name
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
                                      
    callback_log = [callback_early_stopping , callback_checkpoint,TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)]
   
    # Use Keras to train the model.
    model.fit_generator(generator=generator,
                    epochs=200,
                    steps_per_epoch=steps_epoch,
                    validation_data=validation_data,
                    callbacks=callback_log)





    
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)  


    loss = model.evaluate(validation_data[0], validation_data[1])

    print()
    print("Loss: " + str(loss))
    print()

    encoder_predict_model = keras.models.Model(encoder_inputs,
                                            encoder_states)

    decoder_states_inputs = []

    for hidden_neurons in layers[::-1]:
        # One state for GRU
        decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

    decoder_outputs_and_states = decoder(
        decoder_inputs, initial_state=decoder_states_inputs, training=True)

    decoder_outputs = decoder_outputs_and_states[0]
    decoder_states = decoder_outputs_and_states[1:]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_predict_model = keras.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)





    horizon = 50

    X_for_pred = np.expand_dims(x_train_scaled[x_train_scaled.shape[0]-input_seq_len:], axis=0)
    y_test_predicted = predict(X_for_pred, encoder_predict_model, decoder_predict_model, horizon)

    pred= y_scaler.inverse_transform(y_test_predicted[0]).T
    true = y_test[:horizon].T

    t_rmse = 0

    rnn = {}

    for i in range(len(pred)):

        pred[i][pred[i] < 0] = 0

        rmse = np.sqrt(mean_squared_error(true[i], pred[i]))
        t_rmse = t_rmse+rmse
        
        rnn[qtys[i][:-4]] = rmse


    valsArr_old = [66.6851666549545, 67.88958611597897, 69.21681535327671, 66.48135020266528, 64.27709785778202, 60.72749709928205, 63.701465954928295, 65.17733810711289, 64.78680046203984, 54.03158148686864, 50.996898683969604, 54.064319848800224, 42.69338976116232, 7.7145067699141645, 58.73075744706745, 42.21435723311817, 65.8428871293241, 55.54466048398387, 37.459384554198074, 65.91935078588978, 59.44024944642978, 66.75112087030772, 55.109361656982436, 48.873453068349015, 46.018164004825145, 63.55124830274102, 58.51154392727636, 65.20080941236168, 20.582912304727387, 65.47745592815562, 29.05502729752538, 31.88538428636566, 66.19052965893539, 25.209622432309466, 65.30813282106377, 48.47010482324254, 55.35896540587586, 67.8970786052471, 63.95296694703776, 65.58669446483145, 69.07507501099678, 53.27732807353019, 59.13327495000382, 49.961020439339, 36.43109621690061, 58.25244506735996, 73.47924335249874, 48.24591638102002, 15.306627521117743, 54.619525753773665, 33.460658318138734, 63.38118223469059, 21.4684065315376, 51.57643764331531, 38.35348241078786, 57.49193432508641, 52.378319918141834, 3.2970115809005662, 45.070092238160676, 50.820191022437356, 20.009408502370174, 66.76133851423023, 15.381852603930446, 55.844775832748425, 63.3844295901193, 62.06249668996134, 18.796244866954524, 61.151276543758314, 69.76259052461357, 32.22915185003426, 52.164895526117284, 55.51180174042507]
    rnn["total"] = t_rmse/len(pred)
    
    valsArr = []
    for f in naive:
        
        val = (1-(rnn[f]/naive[f]))*100
        #print(f + ": " + str(np.round(val,decimals=2)) + "%")
        valsArr.append(val)

    print("Current : " + str(np.array(valsArr[:]).mean()))
    print("Old: " +  str(np.array(valsArr_old[:]).mean()))

    percentage = np.array(valsArr).mean()
        

    
    global highest_per
    if percentage > highest_per:
        model.save(path_best_model)
        highest_per = percentage

    print("Highest Percentage: " + str(highest_per))
    print()

    del model
    K.clear_session()
    return -percentage
    

search_result = gp_minimize(func=fitnenss,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=100,
                            x0=default_parameters)

pickle.dump(search_result, open("FINAL2.p", "wb"))

plot_convergence(search_result)