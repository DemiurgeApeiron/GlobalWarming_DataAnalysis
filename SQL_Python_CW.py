import mysql.connector
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.style as style
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def makeConnection():
    try:
        cnx = mysql.connector.connect(user="root", password="42admin", host="127.0.0.1", database="GlobalWarming")
        return cnx

    except mysql.connector.Error as err:

        if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)


def SQL_Querry(cursor):
    queryGetName = f"SELECT dt, LandAverageTemperature, LandAverageTemperatureUncertainty FROM GlobalWarming.global_temperatures WHERE dt <= 2020"
    cursor.execute(queryGetName)
    data = cursor.fetchall()
    return data


def data_processing(list):
    df = pd.DataFrame(list, columns=["Date", "AverageTemperature", "LandAverageTemperatureUncertainty"])
    df["Date"] = pd.to_datetime(df["Date"])
    print(df.info())
    print(df.describe())
    print(df.head())
    return df


def dataDisplay(DataFrameProcessed):
    DataFrameProcessed["AverageTemperature + Uncertainty"] = DataFrameProcessed["AverageTemperature"] + DataFrameProcessed["LandAverageTemperatureUncertainty"]
    DataFrameProcessed["AverageTemperature - Uncertainty"] = DataFrameProcessed["AverageTemperature"] - DataFrameProcessed["LandAverageTemperatureUncertainty"]
    DataFrameProcessed = DataFrameProcessed[["Date", "AverageTemperature", "AverageTemperature + Uncertainty", "AverageTemperature - Uncertainty"]]
    DataFrameProcessed = DataFrameProcessed.groupby(pd.Grouper(key="Date", freq="Y")).mean()
    print(DataFrameProcessed.head())
    sns.lineplot(data=DataFrameProcessed)
    plt.show()


def DataModel(DataFrameProcessed):
    DataFrameProcessed["Year"] = DataFrameProcessed["Date"].dt.year
    DataFrameProcessed["Month"] = DataFrameProcessed["Date"].dt.month
    DataFrameProcessed["Day"] = DataFrameProcessed["Date"].dt.day
    x = DataFrameProcessed[["Year", "Month", "Day"]].to_numpy()
    y = DataFrameProcessed["AverageTemperature"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    print(X_train)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train.reshape(-1, 1))
    # X_test = scaler.transform(X_test.reshape(-1, 1))
    N, D = X_train.shape

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(D,)))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    opt = tf.keras.optimizers.Adam(0.01)
    model.compile(optimizer=opt, loss="mse")
    # learning rate schededuler (variable lr)
    def schedule(epoch, lr):
        if epoch >= 50:
            print(lr)
            return lr - lr / 50
        return 0.01

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70, callbacks=[scheduler])

    print(f"Train score: {model.evaluate(X_train, y_train)}")
    print(f"Test score: {model.evaluate(X_test, y_test)}")

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title("loss")
    axs[0, 0].plot(r.history["loss"], label="loss")

    yhat = model.predict(X_test).flatten()
    rdmc = np.sqrt(metrics.mean_squared_error(y_test, yhat))
    print(f"Raíz de la desviación media al cuadrado: {rdmc}")
    plt.show()

    X_year = 2020
    futureYears = []
    for i in range(50):
        X_year += 1
        futureYears.append([X_year, 1, 1])

    futureYears = np.array(futureYears)
    # futureYears = scaler.fit_transform(futureYears.reshape(-1, 1))
    yhatFutureYears = model.predict(futureYears).flatten()
    print(futureYears[0], yhatFutureYears)
    print(len(futureYears[0]), len(yhatFutureYears))

    sns.lineplot(x=np.array([2020 + x for x in range(50)]), y=yhatFutureYears)
    plt.show()


def main():
    cnx = makeConnection()
    cursor = cnx.cursor()
    list = SQL_Querry(cursor)
    DataFrameProcessed = data_processing(list)
    dataDisplay(DataFrameProcessed)
    DataModel(DataFrameProcessed)
    # cnx.commit()
    cnx.close()


main()
