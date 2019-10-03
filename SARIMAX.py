import copy
import numpy as np
import os
import pandas as pd
import random
import statsmodels.api as sm
import sys
from bayes_opt import BayesianOptimization
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.externals import joblib
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm
from kpi_forecast.processors.data_processor import DataProcessor
from kpi_forecast.utils.file_organizer import check_create_directory

class Sarimax(object):
    """
    SARIMAX time series model

    Attributes
    ----------
    __model : sarimax.MLEResults
        SARIMAX fitted model instance
    __models : list of sarimax.MLEResults
        List of SARIMAX fitted model instances
    __lower_quantile_model : sarimax.MLEResults
        SARIMAX lower quantile regression model instance
    __lower_quantile_models : list of sarimax.MLEResults
        List of SARIMAX lower quantile regression model instances
    __upper_quantile_model : sarimax.MLEResults
        SARIMAX upper quantile regression model instance
    __upper_quantile_models : list of sarimax.MLEResults
        List of SARIMAX upper quantile regression model instances
    __best_score : float
        Metric score of the best trained model
    __best_models : list of sarimax.MLEResults
        Best trained models
    __aic_score : float
        AIC score of the current trained model
    __tuner : kpi_forecast.utils.hyperparameter_tuner.HyperparameterTuner
        Hyperparameter tuner
    __data_processor : kpi_forecast.processors.data_processor.DataProcessor
        Data processor
    __is_incremental : bool
        True if the model is incremental else False
    __is_trained : bool
        True if the regression model has been trained else False if otherwise
    __exog_data :
        Current loaded exogeneous dataframe
    __parameters : dict
        Dictionary of parameters
    __defaults : dict
        Dictionary of default parameters
    """
    __model = None
    __models = []
    __lower_quantile_model = None
    __lower_quantile_models = []
    __upper_quantile_model = None
    __upper_quantile_models = []
    __best_score = None    
    __best_models = None
    __aic_score = None
    __data_processor = None
    __is_trained = False
    __exog_data = None
    __parameters = None
    __tuner_parameters = None
    __defaults = {
        'model': {
            'endog': None,                      # The observed time-series process y
            'exog': None,                       # Array of exogenous regressors, shaped nobs x k
            'order_p': 1,                       # Order of the model for the number of AR parameters
            'order_d': 0,                       # Order of the model for the number of differences
            'order_q': 0,                       # Order of the model for the number of MA
            'seasonal_order_p': 0,              # Order of the seasonal component of the model for the AR parameters
            'seasonal_order_d': 0,              # Order of the seasonal component of the model for differences
            'seasonal_order_q': 0,              # Order of the seasonal component of the model for the MA parameters
            'seasonal_order_s': 0,              # Order of the seasonal component of the model for the periodicity ( 4 for quarterly data, 12 for monthly data)
            'trend': None,                      # Parameter controlling the deterministic trend polynomial A(t)
            'measurement_error': False,         # Whether or not to assume the endogenous observations endog were measured with error (True, False)
            'time_varying_regression': False,   # Whether or not coefficients on the exogenous regressors are allowed to vary over time. (True, False)
            'mle_regression': True,             # Whether or not to use estimate the regression coefficients for the exogenous variables as part of maximum likelihood estimation or through the Kalman filter (True, False)
            'simple_differencing': False,       # Whether or not to use partially conditional maximum likelihood estimation (True, False)
            'enforce_stationarity': True,       # Whether or not to transform the AR parameters to enforce stationarity in the autoregressive component of the model (True, False)
            'enforce_invertibility': True,      # Whether or not to transform the MA parameters to enforce invertibility in the moving average component of the model (True, False)
            'hamilton_representation': False,   # Whether or not to use the Hamilton representation of an ARMA process (if True) or the Harvey representation (True, False)
            'concentrate_scale': False,         # Whether or not to concentrate the scale (variance of the error term) out of the likelihood (True, False)
            'trend_offset': 1,                  # The offset at which to start time trend values. Default is 1
            'metric': 'rmse'                    # Metric used for comparison of models with different parameters
        },
        'extras': {
            'include_exog_data': False,         # Whether or not to include exogenous data during model training and prediction
            'total_top_aic_scores': 10,         # Total top AIC scores to be considered during hyperparameter tuning process
            'overfit_rmse_threshold': 0.01,     # RMSE threshold value that is used to decide if overfitting has occurred during training
            'random_seed': 0                    # Random seed
        },
        'data_imputation': {
            'max_iteration': 100                # Max iterations to perform data imputation
        },
        'file': {
            'path': 'models',                   # Path to load and save the model
            'name': 'sarimax',                  # Filename of the model to be saved
            'format': 'pkl'                     # File format of the model file
        }
    }

    def __init__(self, model_parameters=None, tuner_parameters=None):
        """
        Constructor that copies the model and hyperparameter tuner parameters

        Parameters
        ----------
        model_parameters : dict
            Dictionary of model parameters
        tuner_parameters : dict
            Dictionary of hyperparameter tuner parameters

        Returns
        -------
        None
        """
        self.__parameters = copy.deepcopy(self.__defaults)
        self.__tuner_parameters = copy.deepcopy(tuner_parameters)
        # Set defined parameter values if present
        if model_parameters is not None:
            self.__parameters['model'].update(model_parameters['model'])
        # Fix the random seed to achieve consistent prediction results
        random.seed(self.__parameters['extras']['random_seed'])
        self.__data_processor = DataProcessor()

    def create_model(self, endog_data, exog_data):
        """
        Create a new SARIMAX model

        Parameters
        ----------
        endog_data : pandas.DataFrame
            Parsed data labels
        exog_data : pandas.DataFrame
            Parsed exogeneous data

        Returns
        -------
        model : sarimax.MLEResults
            Fitted SARIMAX model
        """
        # Clear exogenous data
        self.__exog_data = pd.DataFrame()
        return sm.tsa.statespace.SARIMAX(endog_data, exog_data,
            order=(
                self.__parameters['model']['order_p'],
                self.__parameters['model']['order_d'],
                self.__parameters['model']['order_q']
            ),
            seasonal_order=(
                self.__parameters['model']['seasonal_order_p'],
                self.__parameters['model']['seasonal_order_d'],
                self.__parameters['model']['seasonal_order_q'],
                self.__parameters['model']['seasonal_order_s']
            ),
            measurement_error=self.__parameters['model']['measurement_error'],
            time_varying_regression=self.__parameters['model']['time_varying_regression'],
            mle_regression=self.__parameters['model']['mle_regression'],
            simple_differencing=self.__parameters['model']['simple_differencing'],
            enforce_stationarity=self.__parameters['model']['enforce_stationarity'],
            enforce_invertibility=self.__parameters['model']['enforce_invertibility'],
            hamilton_representation=self.__parameters['model']['hamilton_representation'],
            concentrate_scale=self.__parameters['model']['concentrate_scale']
        )

    def load_models(self, identifier=None):
        """
        Load the trained SARIMAX models from files

        Parameters
        ----------
        identifier : str
            Identifier name

        Returns
        -------
        is_trained : bool
            True if the trained models are loaded successfully else False if otherwise
        """
        print('Loading the model from a file')
        # Clear exogenous data
        self.__exog_data = pd.DataFrame()
        self.__is_trained = True
        identifier = ('_' + identifier) if identifier is not None else ''
        name = self.__parameters['file']['path'] + '/' + \
               self.__parameters['file']['name'] + identifier + '.' + self.__parameters['file']['format']
        if os.path.isfile(name):
            self.__model = joblib.load(name)
        else:
            self.__is_trained = False
            return self.__is_trained
        name = self.__parameters['file']['path'] + '/' + \
                self.__parameters['file']['name'] + '_incremental' + identifier + '.' + self.__parameters['file'][
                    'format']
        if os.path.isfile(name):
            self.__models = joblib.load(name)
        else:
            self.__is_trained = False
        return self.__is_trained

    def save_models(self, identifier=None):
        """
        Save the trained regression models to files

        Parameters
        ----------
        identifier : str
            Identifier name

        Returns
        -------
        None
        """
        print('Saving the model to a file')
        identifier = ('_' + identifier) if identifier is not None else ''
        if self.__is_trained == True:
            name = self.__parameters['file']['path'] + '/' + \
                   self.__parameters['file']['name'] + identifier + '.' + self.__parameters['file']['format']
            check_create_directory(name)
            joblib.dump(self.__model, name)
            name = self.__parameters['file']['path'] + '/' + \
                    self.__parameters['file']['name'] + '_incremental' + identifier + '.' + \
                    self.__parameters['file']['format']
            check_create_directory(name)
            joblib.dump(self.__models, name)

    def is_trained(self):
        """
        Check if the models are trained

        Parameters
        ----------
        None

        Returns
        -------
        is_trained : bool
            True if the models have been trained else False if otherwise
        """
        return self.__is_trained

    def get_parameters(self):
        """
        Get the model parameters

        Parameters
        ----------
        None

        Returns
        -------
        parameters : dict
            Dictionary of parameters
        """
        return self.__parameters

    def parse_train_data(self, data, labels):
        """
        Parse the training data to the format that is suppoerted by the SARIMAX model

        Parameters
        ----------
        data : pandas.DataFrame
            Data
        labels : pandas.DataFrame
            Labels

        Returns
        -------
        endog_data : pandas.DataFrame
            Parsed training label data
        exog_data : pandas.DataFrame
            Parsed training exogeneous data
        """
        exog_data = data.copy()
        # Perform data imputation to ensure that the exogenous data does not contain NaN values
        exog_data = self.__data_processor.impute_data_columns(exog_data, method='zero')
        imputer = IterativeImputer(max_iter=self.__parameters['data_imputation']['max_iteration'])
        exog_data = pd.DataFrame(
            imputer.fit_transform(exog_data), columns=exog_data.columns.values
        )
        endog_data = labels.copy().reset_index(drop=True)
        dates = dict(year=data['Year'], month=data['Month'])
        # For daily data
        if 'DayOfMonth' in data.columns:
            dates['day'] = data['DayOfMonth']
        else:
            dates['day'] = 1
        endog_data['Date'] = list(pd.to_datetime(dates))
        if 'DayOfMonth' in data.columns:
            endog_data["Date"] = endog_data["Date"].dt.to_period("D")
        else:
            endog_data["Date"] = endog_data["Date"].dt.to_period("M")
        exog_data['Date'] = endog_data['Date']
        endog_data = endog_data.set_index(["Date"])
        exog_data = exog_data.set_index(["Date"])
        for column in ['Year', 'Month', 'DayOfMonth', 'Date']:
            if column in endog_data.columns:
                del endog_data[column]
            if column in exog_data.columns:
                del exog_data[column]
        return endog_data, exog_data

    def parse_test_data(self, data, start_date):
        """
        Parse the testing data to the format that is supported by the SARIMAX model

        Parameters
        ----------
        data : pandas.DataFrame
            Data
        start_date : datetime.datetime
            One unit date after end of training dats date (Start date of out of sample date)

        Returns
        -------
        exog_data : pandas.DataFrame
            Parsed testing exogeneous data
        start_date : datetime.datetime
            Start date of testing data
        end_date : datetime.datetime
            End date of testing data
        """
        exog_data = data.copy()
        # Perform data imputation to ensure that the exogenous data does not contain NaN values
        exog_data = self.__data_processor.impute_data_columns(exog_data, method='zero')
        imputer = IterativeImputer(max_iter=self.__parameters['data_imputation']['max_iteration'])
        exog_data = pd.DataFrame(
            imputer.fit_transform(exog_data), columns=exog_data.columns.values
        )
        dates = dict(year=data['Year'], month=data['Month'])
        if 'DayOfMonth' in data.columns:
            dates['day'] = data['DayOfMonth']
        else:
            dates['day'] = 1
        exog_data['Date'] = list(pd.to_datetime(dates))
        if exog_data[exog_data['Date'] >= start_date].shape[0] > 0:
            exog_data = exog_data[exog_data['Date'] >= start_date]
        if 'DayOfMonth' in data.columns:
            exog_data["Date"] = exog_data["Date"].dt.to_period("D")
        else:
            exog_data["Date"] = exog_data["Date"].dt.to_period("M")
        start_date = exog_data['Date'].min()
        end_date = exog_data['Date'].max()
        return exog_data, start_date, end_date

    def check_parameters_validity(self, parameters):
        """
        Check the validity of the model parameters

        Parameters
        ----------
        parameters : dict
            Dictionary of model parameters

        Returns
        -------
        boolean : bool
            True if the parameters are valid else False if otherwise
        """
        if (parameters['order_p'] > 0 or parameters['order_q'] > 0 or parameters['order_d'] > 0 
            or parameters['seasonal_order_p'] > 0 or parameters['seasonal_order_q'] > 0 or parameters['seasonal_order_d'] > 0):
            return True
        else:
            return False

    def tune_hyperparameters(self, parameters_list, train_data, train_labels, validate_data, validate_labels, methods=["random"]):
        """
        Perform hyperparameter tuning

        Parameters
        ----------
        parameters_list : dict
            Dictionary of hyperparameter values
        train_data : list of pandas.DataFrame
            Training data
        train_labels : list of pandas.DataFrame
            Training labels
        validate_data : list of pandas.DataFrame
            Validation data
        validate_labels : list of pandas.DataFrame
            Validation labels
        methods : list of str
            List of hyperparameter tuning methods

        Returns
        -------
        final_model : SARIMAX model
            Best trained model
        final_parameters : dict
            Best set of hyperparameters
        final_rmse : float
            Metric score of the best trained model (RMSE by default)
        """
        def optimize_bayesian(
            order_p, order_d, order_q, seasonal_order_p, seasonal_order_d, seasonal_order_q, seasonal_order_s
        ):
            """ Perform Bayesian optimization for SARIMAX model """
            # Set the parameter values
            self.__parameters['model'].update({
                'order_p': int(round(order_p)),
                'order_d': int(round(order_d)),
                'order_q': int(round(order_q)),
                'seasonal_order_p': int(round(seasonal_order_p)),
                'seasonal_order_d': int(round(seasonal_order_d)),
                'seasonal_order_q': int(round(seasonal_order_q)),
                'seasonal_order_s': int(round(seasonal_order_s))
            })
            print('Current parameters: (%s, %s, %s)(%s, %s, %s, %s)' % (
                self.__parameters['model']['order_p'], self.__parameters['model']['order_d'], self.__parameters['model']['order_q'],
                self.__parameters['model']['seasonal_order_p'], self.__parameters['model']['seasonal_order_d'],
                self.__parameters['model']['seasonal_order_q'], self.__parameters['model']['seasonal_order_s']
            ))
            try:
                # Train the model
                self.train(train_data, train_labels, None, None)
                if self.__data_processor.check_dataset_validity(validate_data, validate_labels):
                    # Calculate the metric score on validation data
                    score = np.mean([np.mean(
                        list(self.get_performance_metrics(data_partition, label_partition).values())
                    ) for data_partition, label_partition in zip(validate_data, validate_labels)])
                else:
                    # Calculate the metric score on training data
                    score = np.mean([np.mean(
                        list(self.get_performance_metrics(data_partition, label_partition).values())
                    ) for data_partition, label_partition in zip(train_data, train_labels)])
                # Store the trained model with the best score
                if self.__best_score is None or self.__best_score > score:
                    self.__best_models = copy.deepcopy(self.__models)
                    self.__best_score = score
                print('Score: %s' % score)
                return -score
            except:
                print('Failed to train the model using the specified parameters')
                return (-sys.maxsize - 1)

        results = pd.DataFrame(columns=['parameters', 'aic', 'rmse', 'model'])
        is_train_dataset_valid = self.__data_processor.check_dataset_validity(train_data, train_labels)
        is_validate_dataset_valid = self.__data_processor.check_dataset_validity(validate_data, validate_labels, True)
        best_bayesian_parameters = None
        if is_train_dataset_valid == True and is_validate_dataset_valid == True:
            for method in methods:
                if 'bayesian' in method.lower():
                    print('Performing the Bayesian search')
                    parameters = self.__tuner_parameters['bayesian_search']
                    optimizer = BayesianOptimization(
                        f=optimize_bayesian,
                        pbounds=parameters_list['bayesian'],
                        random_state=self.__tuner_parameters['bayesian_search']['random_state'],
                        verbose=parameters['verbose']
                    )
                    optimizer.maximize(init_points=parameters['init_points'], n_iter=parameters['n_iter'])
                    best_bayesian_parameters = optimizer.max['params']
                    best_bayesian_parameters['order_p'] = int(round(best_bayesian_parameters['order_p']))
                    best_bayesian_parameters['order_d'] = int(round(best_bayesian_parameters['order_d']))
                    best_bayesian_parameters['order_q'] = int(round(best_bayesian_parameters['order_q']))
                    best_bayesian_parameters['seasonal_order_p'] = int(round(best_bayesian_parameters['seasonal_order_p']))
                    best_bayesian_parameters['seasonal_order_d'] = int(round(best_bayesian_parameters['seasonal_order_d']))
                    best_bayesian_parameters['seasonal_order_q'] = int(round(best_bayesian_parameters['seasonal_order_q']))
                    best_bayesian_parameters['seasonal_order_s'] = int(round(best_bayesian_parameters['seasonal_order_s']))
                    results = results.append({
                        'parameters': best_bayesian_parameters,
                        'aic': np.nan,
                        'rmse': -optimizer.max['target'],
                        'model': copy.deepcopy(self.__best_models)
                    }, ignore_index=True)
                elif 'grid' in method.lower():
                    print('Performing the grid search')
                    for parameters in tqdm(list(ParameterGrid(parameters_list['grid'])), ascii=True):
                        if self.check_parameters_validity(parameters) == True:
                            print('Current parameters: (%s, %s, %s) (%s, %s, %s, %s) %s' % (
                                parameters['order_p'], parameters['order_d'], parameters['order_q'],
                                parameters['seasonal_order_p'], parameters['seasonal_order_d'],
                                parameters['seasonal_order_q'], parameters['seasonal_order_s'], parameters['trend']
                            ))
                            self.__parameters['model'].update(parameters)
                            try:
                                self.train(train_data, train_labels, None, None)
                                if self.__data_processor.check_dataset_validity(validate_data, validate_labels):
                                    rmse_score = np.mean([np.mean(
                                        list(self.get_performance_metrics(data_partition, label_partition).values())
                                    ) for data_partition, label_partition in zip(validate_data, validate_labels)])
                                else:
                                    rmse_score = np.mean([np.mean(
                                        list(self.get_performance_metrics(data_partition, label_partition).values())
                                    ) for data_partition, label_partition in zip(train_data, train_labels)])
                                train_rmse_score = np.mean([np.mean(
                                        list(self.get_performance_metrics(data_partition, label_partition).values())
                                    ) for data_partition, label_partition in zip(train_data, train_labels)])
                                if self.__aic_score != 2 and np.isnan(self.__aic_score) == False and train_rmse_score >= self.__parameters['extras']['overfit_rmse_threshold']:
                                    results = results.append({
                                        'parameters': parameters,
                                        'aic': self.__aic_score,
                                        'rmse': rmse_score,
                                        'model': copy.deepcopy(self.__models)
                                    }, ignore_index=True).sort_values(by='aic', ascending=True).head(self.__parameters['extras']['total_top_aic_scores'])
                                    print("Current aic is {}, current score is {}".format(self.__aic_score, rmse_score))
                            except:
                                print('Failed to train the model using the specified parameters')
                                continue
                elif 'random' in method.lower():
                    print('Performing the randomized search')
                    random_parameters = list(ParameterSampler(
                        parameters_list['random'],
                        n_iter=self.__tuner_parameters['randomized_search']['iterations'],
                        random_state=self.__tuner_parameters['randomized_search']['random_state']
                    ))
                    for parameters in tqdm(random_parameters, ascii=True):
                        if self.check_parameters_validity(parameters) == True:
                            print('Current parameters: (%s, %s, %s) (%s, %s, %s, %s) %s' % (
                                parameters['order_p'], parameters['order_d'], parameters['order_q'],
                                parameters['seasonal_order_p'], parameters['seasonal_order_d'],
                                parameters['seasonal_order_q'], parameters['seasonal_order_s'], parameters['trend']
                            ))
                            self.__parameters['model'].update(parameters)
                            try:
                                self.train(train_data, train_labels, None, None)
                                if self.__data_processor.check_dataset_validity(validate_data, validate_labels):
                                    rmse_score = np.mean([np.mean(
                                        list(self.get_performance_metrics(data_partition, label_partition).values())
                                    ) for data_partition, label_partition in zip(validate_data, validate_labels)])
                                else:
                                    rmse_score = np.mean([np.mean(
                                        list(self.get_performance_metrics(data_partition, label_partition).values())
                                    ) for data_partition, label_partition in zip(train_data, train_labels)])
                                if self.__aic_score != 2 and np.isnan(self.__aic_score) == False and train_rmse_score >= self.__parameters['extras']['overfit_rmse_threshold']:
                                    results = results.append({
                                        'parameters': parameters,
                                        'aic': self.__aic_score,
                                        'rmse': rmse_score,
                                        'model': copy.deepcopy(self.__models)
                                    }, ignore_index=True).sort_values(by='aic', ascending=True).head(self.__parameters['extras']['total_top_aic_scores'])
                                    print("Current aic is {}, current score is {}".format(self.__aic_score, rmse_score))
                            except:
                                print('Failed to train the model using the specified parameters')
                                continue
        if not results.empty:
            results = results.nsmallest(1, ['rmse']).reset_index(drop=True)
            final_parameters = copy.deepcopy(self.__parameters)
            final_parameters['model'].update(results['parameters'][0])
            return results['model'][0], final_parameters, results["rmse"][0]
        else:
            return None, None, None

    def train_single(self, data, labels):
        """
        Train the single regression model

        Parameters
        ----------
        data : pandas.DataFrame
            Training data
        labels : pandas.DataFrame
            Training labels

        Returns
        -------
        model : sarimax.MLEResults
            Trained regression model
        is_trained : bool
            True if the model has been trained else False if otherwise
        """
        # Fix the random seed to achieve consistent prediction results
        random.seed(self.__parameters['extras']['random_seed'])
        self.__aic_score = None
        if self.__data_processor.check_dataset_validity( data, labels) == True:
            endog_data, exog_data = self.parse_train_data(data, labels)
            if self.__parameters['extras']['include_exog_data'] == True:
                model = self.create_model(endog_data, exog_data).fit()
            else:
                model = self.create_model(endog_data, None).fit()
            self.__aic_score = model.aic
            is_trained = True
        else:
            model = None
            is_trained = False
        return model, is_trained

    def train_incremental(self, data, labels):
        """
        Train the regression models

        Parameters
        ----------
        data : pandas.DataFrame
            Training data
        labels : pandas.DataFrame
            Training labels

        Returns
        -------
        models : list of sarimax.MLEResults
            List of trained regression models
        is_trained : bool
            True if the models have been trained else False if otherwise
        """
        # Fix the random seed to achieve consistent prediction results
        random.seed(self.__parameters['extras']['random_seed'])
        aic_scores = []
        models = []
        is_trained = False
        for i in range(len(data)):
            print('Fold:', i)
            if self.__data_processor.check_dataset_validity(data, labels) == True:
                endog_data, exog_data = self.parse_train_data(data[i], labels[i])
                if self.__parameters['extras']['include_exog_data'] == True:
                    model = self.create_model(endog_data, exog_data).fit()
                else:
                    model = self.create_model(endog_data, None).fit()
                aic_scores.append(model.aic)
                models.append(model)
                is_trained = True
            else:
                is_trained = False
                break
        self.__aic_score = np.mean(aic_scores)
        return models, is_trained

    def train(self, train_data, train_labels, validate_data, validate_labels):
        """
        Train the models using both training and validation datasets

        Parameters
        ----------
        train_data : list of pandas.DataFrame
            Training data
        train_labels : list of pandas.DataFrame
            Training labels
        validation_data : list of pandas.DataFrame
            Validation data
        validation_labels : list of pandas.DataFrame
            Validation labels
        is_incremental : bool
            True if the model is incremental else False

        Returns
        -------
        None
        """
        is_validate_dataset_valid = self.__data_processor.check_dataset_validity(validate_data, validate_labels)
        data = []
        labels = []
        for i in range(len(train_data)):
            data.append(train_data[i].append(validate_data[i]) if is_validate_dataset_valid == True else train_data[i])
            labels.append(train_labels[i].append(validate_labels[i]) if is_validate_dataset_valid == True else train_labels[i])
        self.__models, self.__is_trained = self.train_incremental(data, labels)

    def predict_data(self, model, test_data):
        """
        Perform prediction on the data

        Parameters
        ----------
        model : sarimax.MLEResults
            SARIMAX fitted model

        test_data : pandas.DataFrame
            Testing data

        Returns
        -------
        outputs : NumPy array
            Prediction outputs
        """
        model_summary = model.summary().tables[0]
        test_start_date = datetime.strptime(str(model_summary[5][1])[2:], '%m-%d-%Y') + relativedelta(days=1)
        exog_data, _, end_date = self.parse_test_data(test_data, test_start_date)
        if self.__parameters['extras']['include_exog_data'] == True:
            # Store new exogenous data
            self.__exog_data = self.__exog_data.append(exog_data, ignore_index=True)
            self.__exog_data = self.__exog_data.drop_duplicates(subset='Date')
            if 'DayOfMonth' in test_data.columns:
                test_start_date = (pd.to_datetime(test_start_date)).to_period('D')
            else:
                test_start_date = (pd.to_datetime(test_start_date)).to_period('M')
            if exog_data['Date'].min() > test_start_date:
                exog_data = exog_data.append(
                    self.__exog_data[(self.__exog_data['Date'] >= test_start_date) & (self.__exog_data['Date'] < exog_data['Date'].min())],
                    ignore_index=True
                )
            exog_data = exog_data.set_index(["Date"])
            for column in ['Year', 'Month', 'DayOfMonth', 'Date']:
                if column in exog_data.columns:
                    del exog_data[column]
        else:
            exog_data = None
        outputs = model.get_prediction(
            exog=exog_data,
            start=None,
            end=pd.to_datetime(end_date.strftime("%Y-%m-%d")),
            dynamic=False
        )
        return outputs

    def predict(self, test_data):
        """
        Perform prediction on the data

        Parameters
        ----------
        test_data : pandas.DataFrame
            Testing data

        Returns
        -------
        outputs : None
        """
        if self.__is_trained == True and not test_data.empty:
            return np.mean([self.predict_data(model, test_data).predicted_mean[-(test_data.shape[0]):] for model in self.__models], axis=0)
        else:
            return None

    def predict_quantile(self, test_data):
        """
        Perform quantile prediction on the data

        Parameters
        ----------
        test_data : pandas.DataFrame
            Testing data

        Returns
        -------
        lower_quantiles : NumPy array
            Lower quantile prediction outputs
        upper_quantiles : NumPy array
            Upper quantile prediction outputs
        """
        if self.__is_trained == True and not test_data.empty:
            outputs_list = [self.predict_data(model, test_data) for model in self.__models]
            lower_quantiles = np.mean([outputs.conf_int().ix[:, 0] for outputs in outputs_list], axis=0)
            upper_quantiles = np.mean([outputs.conf_int().ix[:, 1] for outputs in outputs_list], axis=0)
            return lower_quantiles, upper_quantiles
        else:
            return None, None

    def get_performance_metrics(self, data, labels):
        """
        Get the performance metrics of the trained model

        Parameters
        ----------
        data : pandas.DataFrame
            Data
        labels : pandas.DataFrame
            Labels

        Returns
        -------
        metrics : dict
            Dictionary of metric scores (RMSE by default)
        """
        metrics = {}
        if self.__is_trained == True and self.__data_processor.check_dataset_validity(data, labels) == True:
            filled_labels = labels.copy().fillna(labels.mean())
            for index, model in enumerate(self.__models):
                metrics[(self.__parameters['model']['metric'] + '_' + str(index))] = np.sqrt(
                    mean_squared_error(self.predict_data(model, data).predicted_mean[-(data.shape[0]):].interpolate().fillna(0), filled_labels)
                )
        return metrics

    def get_feature_importance(self, train_data, train_labels, validate_data=None, validate_labels=None, method='permutation'):
        """
        Get the importance of each feature
        
        Parameters
        ----------
        train_data : list of pandas.DataFrame
            Training data
        train_labels : list of pandas.DataFrame
            Training labels
        validation_data : list of pandas.DataFrame
            Validation data
        validation_labels : list of pandas.DataFrame
            Validation labels
        method : str
            Feature importance method
        
        Returns
        -------
        features : pandas.DataFrame
            Feature names and their corresponding importance scores
        """
        # Fix the random seed to achieve consistent prediction results
        random.seed(self.__parameters['extras']['random_seed'])
        feature_importances = np.zeros(len(train_data[0].columns))
        for i in range(len(train_data)):
            model = GradientBoostingRegressor()
            # Perform data imputation as the model requires input data with non-empty values
            imputed_data = self.__data_processor.impute_data_columns(train_data[i], method='zero')
            imputer = IterativeImputer(max_iter=self.__parameters['data_imputation']['max_iteration'])
            imputed_data = pd.DataFrame(
                imputer.fit_transform(imputed_data), columns=train_data[i].columns.values
            )
            model.fit(imputed_data, train_labels[i])
            feature_importances += model.feature_importances_
        features = pd.DataFrame({
            'Feature': train_data[0].columns,
            'Score': model.feature_importances_ / len(train_data)
        }).sort_values(by='Score', ascending=False).reset_index(drop=True)
        # Set the model to include exogenous data if feature selection is enabled and called
        self.__parameters['extras']['include_exog_data'] = True
        if len(features) > len(train_data[0]):
            return features[:len(train_data[0])]
        else:
            return features
