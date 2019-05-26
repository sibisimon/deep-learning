"""
Author: Sibi Simon
Date: 25-11-2018
"""

# Importing modules

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import plot_model

from bokeh.plotting import output_notebook, show, figure, gmap
from bokeh.models import ColumnDataSource, GMapOptions
import holoviews as hv
hv.extension('bokeh')

PATH = "keras_kc_house_data.csv"


def main_process():

    # Reading dataset
    price_data_set = pd.read_csv(PATH)

    # Data Analysis using summary, histogram & scatter plots
    print('----------------------------------- DATA ANALYSIS -----------------------------------------')
    analysis = DataAnalysis(df=price_data_set)
    analysis.summary()
    analysis.housing_location_by_pos()
    analysis.histogram()

    # Data pre-processing (wrangling)
    print('----------------------------------- DATA WRANGLING -----------------------------------------')
    pre_process = PreProcessing(df=price_data_set)
    pre_process.checking_null_values()
    pre_process.remove_unwanted_columns()
    pre_process.remove_outliers()
    pre_process.scaling()
    x_train, x_test, y_train, y_test = pre_process.test_train_split()

    # Modelling
    print('----------------------------------- DATA MODELLING -----------------------------------------')
    sq_model = SequentialModel(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    sq_model.create()
    sq_model.compile()
    history = sq_model.fit()
    sq_model.evaluate()
    y_pred = sq_model.predict()

    # Plotting model running history
    print('----------------------------------- MODEL DIAGNOSIS -----------------------------------------')
    analysis.plot_history(history)
    analysis.predict_plot(y_pred, y_test)


class PreProcessing(object):
    """
    Class for performing data pre-processing operations such as data cleansing, wrangling etc.
    """

    def __init__(self, *args, **kwargs):
        self.df = kwargs['df']

    def checking_null_values(self):
        if not self.df.isnull().any().any():
            print("CHECKING NULL VALUES: No null vlaues in dataset")

    def remove_unwanted_columns(self):
        self.df = pd.DataFrame(self.df.drop(["date", "id", "zipcode"], axis=1))

    def remove_outliers(self):
        # Avoiding outlier check for these columns because they are not continuous or normally distributed
        total_data_points = len(self.df)
        df_temp = self.df.drop(["yr_renovated", "sqft_basement", "waterfront", "view"], axis=1)
        self.df = pd.DataFrame(self.df[(np.abs(stats.zscore(df_temp)) <= 4).all(axis=1)])
        print("Number of Outliers:", total_data_points - len(self.df))
        print("Number of rows without outliers:", len(self.df))

    def scaling(self):
        col_train = list(self.df.columns)
        mat_train = np.matrix(self.df)
        mat_new = np.matrix(self.df.drop('price', axis=1))
        mat_y = np.array(self.df.price).reshape((len(self.df), 1))

        pre_processing_y = MinMaxScaler()
        pre_processing_y.fit(mat_y)

        pre_processing_df = MinMaxScaler()
        pre_processing_df.fit(mat_train)

        pre_processing_test = MinMaxScaler()
        pre_processing_test.fit(mat_new)

        self.df = pd.DataFrame(pre_processing_df.transform(mat_train), columns=col_train)
        print(self.df.head(5))

    def test_train_split(self):
        columns = list(self.df.columns)
        features = list(self.df.columns)
        features.remove('price')
        training_set = self.df[columns]
        prediction_set = self.df.price
        x_train, x_test, y_train, y_test = train_test_split(training_set[features], prediction_set, test_size=0.33,
                                                            random_state=42)
        return x_train, x_test, y_train, y_test


class DataAnalysis(object):
    """
    Class for performing various data analysis and interpretations on data using bokeh plotting
    """

    def __init__(self, *args, **kwargs):
        self.df = kwargs['df']
        self.key = "AIzaSyAM1OHVm6Yr_i54Kt01mylfxyNxQdxmxHQ"

    def summary(self):
        print(self.df.describe())

    def histogram(self):
        """
        Plotting histogram of the response variable
        """
        hist, edges = np.histogram(self.df['price'].tolist(), density=False, bins=100)
        p = figure(title="Response variable Histogram", tools='', background_fill_color="#fafafa")
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white")
        p.xaxis.axis_label = 'Price'
        p.yaxis.axis_label = 'Frequency'
        show(p)

    def housing_location_by_pos(self):
        # Top 25 house with highest price
        high_price_pos = self.df.sort_values(by='price', ascending=0).head(25)[['lat', 'long']]
        self._geo_plot(high_price_pos['lat'].tolist(), high_price_pos['long'].tolist(), 'blue', 47.65, -122.25,
                       "King County, Washington - Highest Price")

        # Top 25 with lowest price
        lowest_price_pos = self.df.sort_values(by='price', ascending=1).head(25)[['lat', 'long']]
        self._geo_plot(lowest_price_pos['lat'].tolist(), lowest_price_pos['long'].tolist(), 'red', 47.42, -122.25,
                       "King County, Washington - Lowest Price")

    def plot_history(self, history):
        p = figure(plot_width=400, plot_height=400, title="Epoch vs Loss (Test and Train)")
        p.line(history.epoch, history.history['loss'], line_color="blue", line_width=2, legend='Train')
        p.line(history.epoch, history.history['val_loss'], line_width=2, line_color='orange', legend='Test')
        p.xaxis.axis_label = 'Loss'
        p.yaxis.axis_label = 'Epoch'
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        show(p)

    def predict_plot(self, y_pred, y_test):
        abs_error = [abs(n) for n in list(y_pred.reshape(len(y_pred))) - y_test]
        p = figure(plot_width=500, plot_height=500, title="Absolute error for test data")
        p.line(range(0, len(y_pred)), abs_error, line_color="blue", line_width=2)
        p.xaxis.axis_label = 'Data points'
        p.yaxis.axis_label = 'Error'
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        show(p)

    def _geo_plot(self, lat, long, col, ini_lat, ini_long, title):
        map_options = GMapOptions(lat=ini_lat, lng=ini_long, map_type="hybrid", zoom=11)
        p = gmap(self.key, map_options, title=title)
        source = ColumnDataSource(
            data=dict(lat=lat,
                      lon=long)
        )
        p.circle(x="lon", y="lat", size=15, fill_color=col, fill_alpha=0.8, source=source)
        show(p)


class SequentialModel(object):
    """
    class for making sequential deep learning model using keras
    """
    def __init__(self, *args, **kwargs):
        self.model = Sequential()
        self.x_train = kwargs['x_train']
        self.x_test = kwargs['x_test']
        self.y_train = kwargs['y_train']
        self.y_test = kwargs['y_test']

    def create(self):
        self.model.add(Dense(32, activation='relu', input_shape=(17,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(32, activation='relu', input_shape=(50,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(32, activation='relu', input_shape=(50,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(1))

    def summary(self):
        print(self.model.summary())
        plot_model(self.model, to_file='model.png')

    def compile(self):
        self.model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mse'])

    def fit(self):
        history = self.model.fit(self.x_train, self.y_train, batch_size=100, epochs=150, verbose=1,
                                 validation_data=(self.x_test, self.y_test))
        return history

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("\n")
        print("Test loss:", score[0])

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        return y_pred


if __name__ == "__main__":
    main_process()
