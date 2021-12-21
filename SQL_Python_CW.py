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
    queryGetGlobalTemperatures = f"SELECT dt, LandAverageTemperature, LandAverageTemperatureUncertainty FROM GlobalWarming.global_temperatures"
    cursor.execute(queryGetGlobalTemperatures)
    dataGlobalTemperatures = cursor.fetchall()
    queryGetGlobalIndicators = f"SELECT Year, Value, CountryName, IndicatorName FROM GlobalWarming.world_indicators WHERE IndicatorCode LIKE '%CO2E.KT%' ORDER BY Year;"
    cursor.execute(queryGetGlobalIndicators)
    dataGlobalIndicators = cursor.fetchall()

    return dataGlobalTemperatures, dataGlobalIndicators


def data_processing(dataGlobalTemperatures, dataGlobalIndicators):
    dfGlobalTemperatures = pd.DataFrame(dataGlobalTemperatures, columns=["Date", "AverageTemperature", "LandAverageTemperatureUncertainty"])
    dfGlobalIndicators = pd.DataFrame(dataGlobalIndicators, columns=["Date", "Value", "CountryName", "IndicatorName"])

    dfGlobalTemperatures["Date"] = pd.to_datetime(dfGlobalTemperatures["Date"])
    dfGlobalIndicators["Date"] = pd.to_datetime(dfGlobalIndicators["Date"])

    dfGlobalTemperatures = dfGlobalTemperatures.dropna()

    print(dfGlobalTemperatures.info())
    print(dfGlobalTemperatures.describe())
    print(dfGlobalTemperatures.head())

    print(dfGlobalIndicators.info())
    print(dfGlobalIndicators.describe())
    print(dfGlobalIndicators.head())
    return dfGlobalTemperatures, dfGlobalIndicators


def dataDisplay(DataFrameProcessed):
    DataFrameProcessed["AverageTemperature + Uncertainty"] = DataFrameProcessed["AverageTemperature"] + DataFrameProcessed["LandAverageTemperatureUncertainty"]
    DataFrameProcessed["AverageTemperature - Uncertainty"] = DataFrameProcessed["AverageTemperature"] - DataFrameProcessed["LandAverageTemperatureUncertainty"]
    DataFrameProcessed = DataFrameProcessed[["Date", "AverageTemperature", "AverageTemperature + Uncertainty", "AverageTemperature - Uncertainty"]]
    DataFrameProcessed = DataFrameProcessed.groupby(pd.Grouper(key="Date", freq="Y")).mean()
    print(DataFrameProcessed.head())
    sns.lineplot(data=DataFrameProcessed)
    plt.show()


def DataModel(dfGlobalTemperatures, dfGlobalIndicators):
    dfGlobalTemperatures["Year"] = dfGlobalTemperatures["Date"].dt.year
    dfGlobalTemperatures["Month"] = dfGlobalTemperatures["Date"].dt.month
    dfGlobalTemperatures["Day"] = dfGlobalTemperatures["Date"].dt.day
    x = dfGlobalTemperatures[["Year", "Month", "Day"]].to_numpy()
    y = dfGlobalTemperatures["AverageTemperature"].to_numpy()
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
        if epoch >= 10:
            print(lr)
            return lr - lr / 10
        return 0.01

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
    r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, callbacks=[scheduler])

    print(f"Train score: {model.evaluate(X_train, y_train)}")
    print(f"Test score: {model.evaluate(X_test, y_test)}")

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title("loss")
    axs[0, 0].plot(r.history["loss"], label="loss")

    yhat = model.predict(X_test).flatten()
    rdmc = np.sqrt(metrics.mean_squared_error(y_test, yhat))
    print(f"Raíz de la desviación media al cuadrado: {rdmc}")
    plt.show()

    X_year = 2017
    X_Year_Month_Day = []
    futureYears = []
    for i in range(50):
        X_year += 1
        X_Month = 1
        # for j in range(11):
        # X_Month += 1
        X_Day = 1
        # for k in range(27):
        # X_Day += 1
        futureYears.append([X_year, X_Month, X_Day])
        X_Year_Month_Day.append(f"{X_year}-{X_Month}-{X_Day}")

    futureYears = np.array(futureYears)
    df = pd.DataFrame(X_Year_Month_Day, columns=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    # futureYears = scaler.fit_transform(futureYears.reshape(-1, 1))
    yhatFutureYears = model.predict(futureYears).flatten()
    # x=np.array([2020 + x for x in range(50)])
    df["AverageTemperature"] = pd.DataFrame(yhatFutureYears, columns=["AverageTemperature"])
    df = df.set_index("Date")
    ndf = dfGlobalTemperatures[["Date", "AverageTemperature"]]
    ndf = ndf.groupby(pd.Grouper(key="Date", freq="Y")).mean()
    df = pd.concat([df, ndf])
    print(df.head())
    sns.lineplot(data=df)
    plt.show()


def main():
    cnx = makeConnection()
    cursor = cnx.cursor()
    dataGlobalTemperatures, dataGlobalIndicators = SQL_Querry(cursor)
    dfGlobalTemperatures, dfGlobalIndicators = data_processing(dataGlobalTemperatures, dataGlobalIndicators)
    dataDisplay(dfGlobalTemperatures)
    DataModel(dfGlobalTemperatures, dfGlobalIndicators)
    # cnx.commit()
    cnx.close()


main()
