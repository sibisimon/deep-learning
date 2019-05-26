"""
Author: Sibi Simon
Date: 25-11-2018
"""

# Importing modules

import numpy as np
import pandas as pd
from sympy import *
from scipy import stats

# Importing modules for pytorch
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init

# Importing modules for keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Visualization import
from plotly.offline import plot
import plotly.figure_factory as ff
import plotly.graph_objs as go

PATH = "pytorch_breast_cancer.csv"


def main_process():
    # Reading dataset
    data_set = pd.read_csv(PATH)

    # Data Analysis using summary, bar & confusion
    print('------------------------------------------ VISUALIZATION --------------------------------------------')
    plts = Visualize(df=data_set)
    plts.summary()
    plts.bar()

    # Data pre-processing (wrangling)
    print('----------------------------------------- PRE-PROCESSING --------------------------------------------')
    pre_process = PreProcessing(df=data_set)
    pre_process.remove_outliers()
    pre_process.modify_col()
    pre_process.test_train_split()
    data_loader, test_data, test_validation, y_test_pyt = pre_process.scaling()

    # Modelling using pytorch
    print('--------------------------------------- PYTORCH MODELLING ------------------------------------------')
    pyt_sq_model = PyTorchModel(data_loader=data_loader, test_data=test_data, test_validation=test_validation)
    pyt_sq_model.create()
    pyt_sq_model.run()
    predictd_pyt = pyt_sq_model.predict()
    history_pyt = pyt_sq_model.get_history

    # Modelling using keras
    print('--------------------------------------- KERAS MODELLING --------------------------------------------')
    x_train, x_test, y_train, y_test = pre_process.krs_test_train_split()
    krs_sq_model = KerasModel(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    krs_sq_model.create()
    krs_sq_model.compile()
    history_krs=krs_sq_model.fit()
    krs_sq_model.evaluate()
    predictd_krs = krs_sq_model.predict()

    # Diagnosing Model
    print('-------------------------------- DIAGNOSING AND COMPARING MODELS --------------------------------')

    # Plots for pytorch
    plts.history_pytorch(history_pyt, type_plt='loss', title="Model Loss plot(Pytorch) in Percentage")
    plts.history_pytorch(history_pyt, type_plt='accuracy', title="Pytorch Model Accuracy plot")
    plts.confusion_matrix(predictd_pyt, y_test_pyt, title="Confusion matrix(Pytorch)")

    # Plots for keras
    plts.history_keras(history_krs, type_plt='loss', title="Keras Model Loss plot")
    plts.history_keras(history_krs, type_plt='acc', title="Keras Model Loss plot")
    plts.confusion_matrix(predictd_krs, y_test, title="Confusion matrix(Keras)")


class Visualize(object):
    """
    class for visualizingfor
    """
    def __init__(self, *args, **kwargs):
        self.df = kwargs['df']

    def summary(self):
        print(self.df.describe())

    def bar(self):
        output_col = self.df['diagnosis'].tolist()
        data = [go.Bar(
            x=['Malignant', 'Benign'],
            y=[output_col.count('M'), output_col.count('B')],
            text=[output_col.count('M'), output_col.count('B')],
            textposition='auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.8

        )]
        layout = go.Layout(
            title='Frequency of Malignant and Benign',
            width=600,
            height=600
        )

        fig = go.Figure(data=data, layout=layout)

        plot(fig, filename='cancer_type.html')

    def confusion_matrix(self, y_pred, y_actual, title=''):

        labels = ['Malignant', 'Benign']
        c_matrix = confusion_matrix(y_actual, y_pred)
        fig = ff.create_annotated_heatmap(c_matrix, x=labels, y=labels)
        fig.layout.title = title
        fig.layout.width = 400
        fig.layout.height = 400
        plot(fig, filename='confusion_matrix.html')


    def history_pytorch(self, history, type_plt='loss', title=''):

        # Accuracy plot and Loss plot
        trace0 = go.Scatter(
            x=list(range(0, history['epochs'])),
            y=history[type_plt],
            mode='lines',
            name='Train'
        )
        trace1 = go.Scatter(
            x=list(range(0, history['epochs'])),
            y=history[type_plt + '_val'],
            mode='lines+markers',
            name='Test'
        )
        data = [trace0, trace1]
        layout = go.Layout(
            title=title,
            width=600,
            height=600,
            xaxis=dict(
                title='Epochs',
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            ),
            yaxis=dict(
                title=type_plt,
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            )
        )
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename='history_py.html')


    def history_keras(self, history, type_plt='loss', title=''):

        # Accuracy plot and Loss plot
        trace0 = go.Scatter(
            x=history.epoch,
            y=history.history[type_plt],
            mode='lines',
            name='Train'
        )
        trace1 = go.Scatter(
            x=history.epoch,
            y=history.history['val_' + type_plt],
            mode='lines+markers',
            name='Test'
        )
        data = [trace0, trace1]
        layout = go.Layout(
            title=title,
            width=600,
            height=600,
            xaxis=dict(
                title='Epochs',
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            ),
            yaxis=dict(
                title=type_plt,
                titlefont=dict(
                    size=18,
                    color='#7f7f7f'
                )
            )
        )
        fig = go.Figure(data=data, layout=layout)
        plot(fig, filename='history_kr.html')


class PreProcessing(object):
    """
    Class for performing data pre-processing operations such as data cleansing, wrangling etc.
    """

    def __init__(self, *args, **kwargs):
        self.df = kwargs['df']
        self.y_out = []
        self.x_in = []
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

    def modify_col(self):
        bin_output = {"M": 1, "B": 0}  # modifying output data to binary(0, 1)
        self.y_out = self.df["diagnosis"].replace(bin_output)
        self.x_in = self.df.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)

    def remove_outliers(self):
        # Avoiding outlier check for these columns because they are not continuous or normally distributed
        total_data_points = len(self.df)
        df_temp = self.df.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
        self.df = pd.DataFrame(self.df[(np.abs(stats.zscore(df_temp)) <= 4).all(axis=1)])
        print("Number of Outliers:", total_data_points - len(self.df))
        print("Number of rows without outliers:", len(self.df))

    def test_train_split(self):
       self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_in, self.y_out, test_size=0.2,
                                                                               random_state=85)

    def scaling(self):

        # Scaling train data
        scaling_train = StandardScaler()
        transformed = scaling_train.fit_transform(self.x_train)
        train_data = data_utils.TensorDataset(torch.from_numpy(transformed).float(), torch.from_numpy(
            self.y_train.as_matrix()).float())
        data_loader = data_utils.DataLoader(train_data, batch_size=128, shuffle=False)

        # Scaling test data
        scaling_test = StandardScaler()
        transformed = scaling_test.fit_transform(self.x_test)
        test_data = torch.from_numpy(transformed).float()
        test_validation = torch.from_numpy(self.y_test.as_matrix()).float()

        return data_loader, test_data, test_validation, self.y_test

    def test_train_split(self):
        scaling_test = StandardScaler()
        transformed = scaling_test.fit_transform(self.x_in)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(transformed, self.y_out, test_size=0.3,
                                                                                random_state=85)

    def krs_test_train_split(self):
        scaling_train = StandardScaler()
        x_in = scaling_train.fit_transform(self.x_in)
        return train_test_split(x_in, self.y_out, test_size=0.3, random_state=85)


class PyTorchModel(object):
    """
    Pytorch model building, diagnosing
    """

    def __init__(self, *args, **kwargs):
        self.train_data_loader = kwargs['data_loader']
        self.test_data = kwargs['test_data']
        self.test_validation = kwargs['test_validation']
        self.model = ''
        self.learning_rate = 0.0006
        self.epochs = 350
        self.history = {"epochs": self.epochs, "loss": [], "accuracy": [], "loss_val": [], "accuracy_val": []}

    def create(self):
        self.model = torch.nn.Sequential()

        module = torch.nn.Linear(30, 20)
        init.xavier_normal(module.weight)
        self.model.add_module("linear - 1", module)

        module = torch.nn.Linear(20, 10)
        init.xavier_normal(module.weight)
        self.model.add_module("linear - 2", module)

        module = torch.nn.Linear(10, 1)
        init.xavier_normal(module.weight)
        self.model.add_module("linear - 3", module)

        self.model.add_module("rel", torch.nn.ReLU())
        self.model.add_module("sig", torch.nn.Sigmoid())

    def run(self):
        loss_func = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            loss = None

            for idx, (batch, target) in enumerate(self.train_data_loader):
                y_pred = self.model(Variable(batch))
                loss = loss_func(y_pred, Variable(target.float()).reshape(len(Variable(target.float())), 1))
                prediction = self._n_correct(y_pred)
                corrected_n = (prediction == target.numpy()).sum()

                y_val_pred = self.model(Variable(self.test_data))
                loss_val = loss_func(y_val_pred, Variable(self.test_validation.float()).reshape(len(Variable(
                    self.test_validation.float())), 1))
                prediction_val = self._n_correct(y_val_pred)
                corrected_val_n = (prediction_val == self.test_validation.numpy()).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.history["loss"].append(loss.data[0])
            self.history["accuracy"].append(100 * corrected_n / len(prediction))
            self.history["loss_val"].append(loss_val.data[0])
            self.history["accuracy_val"].append(100 * corrected_val_n / len(prediction_val))

            print("Loss, accuracy, val loss, val acc at epoch", epoch + 1, self.history["loss"][-1],
                  self.history["accuracy"][-1], self.history["loss_val"], self.history["accuracy_val"][-1])

    def _n_correct(self, out):
        return [1 if x > 0.5 else 0 for x in out.data.numpy()]

    def predict(self):
        y_pred = self.model(Variable(self.test_data))
        prediction_val = self._n_correct(y_pred)

        return prediction_val

    @property
    def get_history(self):
        return self.history


class KerasModel(object):
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
        self.model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=30))
        self.model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
        self.model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    def summary(self):
        print(self.model.summary())
        plot_model(self.model, to_file='model.png')

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self):
        history = self.model.fit(self.x_train, self.y_train, batch_size=50, epochs=400, verbose=1,
                                 validation_data=(self.x_test, self.y_test))
        return history

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print("\n")
        print("Test loss:", score[0])

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        prediction_val_krs = self._n_correct(y_pred)

        return prediction_val_krs

    def _n_correct(self, out):
        return [1 if x > 0.5 else 0 for x in out]


if __name__ == "__main__":
    main_process()
