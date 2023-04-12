import pandas as pd
import numpy as np 
from prod.engine_utils import * 


"""Main class NN Regressor
"""

import torch
from prod.nn_execution_functions_prod import main_nn_fit, forecast_loop
from prod.nn_data_prep_prod import get_loaders
from prod.nn_models_prod import ParetoBBNet
from prod.nn_loss_function_prod import  triangle_weighted_msle_loss, mse_loss, weighted_msle_loss, msle_loss
from prod.nn_utils import remove_nans_from_array, count_nons_nans_in_array, nan_padding, cohort_grouper_fn, generate_subdomain_input_data


class NeuralNetworkRegression:
    def __init__(self, config, model_class=None, model_transformer=None, registry=None, config_nn=None):
        self.config = config
        self.model_class = model_class
        self.model_transformer = model_transformer
        self.registry = registry
        self.curve_params = []
        
        if config_nn is not None:
            self.config_nn = config_nn
        else:
            self.setup_config_nn()
    
    def setup_config_nn(self):
        config_nn = {}

        config_nn['start_input_date'], config_nn['end_input_date'] = pd.to_datetime('2020-01-02'), pd.to_datetime('2021-01-02')
        config_nn['end_forecast_date'] = pd.to_datetime('2021-09-20')
        config_nn['grouper_freq'] = 'W'
        config_nn['min_training_dx'] = 7

        config_nn['features_list'] = ['norm_cohort_code','log_dnu']  
        config_nn['trend_week'] = None

        config_nn['batch_size'] = 16
        config_nn['model_name'] = 'pbb'
        config_nn['learning_rate'] = 0.01
        config_nn['decayRate'] = 0.95
        config_nn['dropout_coef'] = 0.05
        config_nn['layer_size_factor'] = 2
        config_nn['loss_function'] = triangle_weighted_msle_loss#mse_loss#weighted_msle_loss#msle_loss##

        config_nn['n_epochs'] = 100
        config_nn['min_epoch'] = 40

    def local_prep_actual_input(self, segment_data):
        active_users_df = segment_data[['calendar_date','cohort_date','dx','active_users']].reset_index(drop=True).copy()
        cohort_size_df = segment_data.query('dx==0')[['cohort_date','active_users']]\
                                   .rename({'active_users':'cohort_size'},axis=1)\
                                   .reset_index(drop=True).copy()


        full_domain_dimensions = generate_ranged_clf_dataframe(
                        start_date=segment_data['cohort_date'].min(),#config_nn['start_input_date'],
                        end_date=segment_data['cohort_date'].max()#config_nn['end_forecast_date'],
                    )

        actual_input = full_domain_dimensions[['cohort_date','dx']].merge(active_users_df, on=['cohort_date','dx'], how='left')\
        .merge(cohort_size_df, on=['cohort_date'], how='left')\
        .assign(retention=lambda x: x['active_users']/x['cohort_size'])\
        .fillna({"cohort_size": 0,"retention": 0}).copy()
        
        return actual_input
                                
    def prep_actual_input(self, actual_input, cohort_size_actual):
        return (
            actual_input
            .pipe(convert_wide_to_long)
            .pipe(calculate_activity_date)
            .pipe(correct_d1_retention)
            .query(
                "cohort_date <= @self.config.end_input_range and cohort_date >= @self.config.start_input_range"
            )
            .query("calendar_date <= @self.config.end_input_range")
            .join(cohort_size_actual
           .rename("cohort_size"), on="cohort_date")
            .fillna({"cohort_size": 0})
            .dropna()
        )
    
        
    def prep_training_and_forecast_data(
        self, actual_input
        ):
        
        grouped_actual_input = actual_input.pipe(cohort_grouper_fn, target_variables=['active_users','cohort_size'], group_freq=self.config_nn['grouper_freq'])\
                                .assign(retention=lambda x: x['active_users']/x['cohort_size'])

        entire_domain = generate_ranged_clf_dataframe(
                        start_date=self.config_nn['start_input_date'] + timedelta(days=1),
                        end_date=self.config_nn['end_forecast_date'],
                    ).pipe(cohort_grouper_fn, target_variables=[], group_freq=self.config_nn['grouper_freq'])\
                     .pipe(calculate_activity_date)


        training_domain = entire_domain.query("calendar_date <= @self.config_nn['end_input_date'] & dx >= @self.config_nn['min_training_dx']").copy()
        validation_domain = entire_domain.query("calendar_date >= @self.config_nn['end_input_date'] & cohort_date <= @self.config_nn['end_input_date'] & dx >= @self.config_nn['min_training_dx']").copy()
        forecast_domain = entire_domain.query("dx >= @self.config_nn['min_training_dx']").copy()

        training_input = generate_subdomain_input_data(grouped_actual_input, training_domain)
        validation_input = generate_subdomain_input_data(grouped_actual_input, validation_domain)
        forecast_input = generate_subdomain_input_data(grouped_actual_input, forecast_domain)
        
        # Feature engineering the input columns to the NN
        fe_transformer = FeatureEngineeringTransformer()
        fe_transformer.fit_transform(training_input)
        fe_transformer.transform(validation_input)
        fe_transformer.transform(forecast_input)
        
        return (training_input, validation_input, forecast_input, grouped_actual_input), \
               (forecast_domain, validation_domain)
    
    
    def train_step(
        self, train_loader, validation_loader
    ):
        
        model = ParetoBBNet(self.config_nn).to(torch.float64).to('cpu')
        
        if self.config_nn['optimizer_name']=='sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.config_nn['learning_rate'], momentum=0.9)
        elif self.config_nn['optimizer_name']=='adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config_nn['learning_rate'])
        else:
            raise "Unknown optimizer selected : " + self.config_nn['optimizer']
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                                              gamma=self.config_nn['decayRate'])
        
        train_losses, val_duds, all_curve_params, epoch_count = main_nn_fit(train_loader, validation_loader, model, optimizer, self.config_nn, lr_scheduler)
        
        self.curve_params = all_curve_params
        
        return model, train_losses, val_duds, epoch_count
    

    def predicting_step(
        self, model, forecast_loader, forecast_domain
    ):
        forecast_df, curve_parameters = forecast_loop(forecast_loader, model, forecast_domain)
        return forecast_df, curve_parameters
    
    
    def generate_model_curve(
        self, actual_input, cohort_size_actual, retention_model_override=None
    ):
        actual_input = self.prep_actual_input(actual_input, cohort_size_actual)
        
        inputs, domains = self.prep_training_and_forecast_data(actual_input)
        training_input, validation_input, forecast_input, grouped_actual_input = inputs
        forecast_domain, validation_domain = domains
        
        train_loader, val_loader, forecast_loader = get_loaders(training_input, validation_input, forecast_input, self.config_nn)
        
        model, train_losses, val_duds, nepochs = self.train_step(train_loader, validation_loader)
        
        model_curve, curve_parameters = self.predicting_step(model, forecast_loader, forecast_domain)
        
        return (
            model_curve,
            None,
            None
        )

    def local_generate_model_curve(
        self, segment_data
    ):
        actual_input = self.local_prep_actual_input(segment_data)
        
        inputs, domains = self.prep_training_and_forecast_data(actual_input)
        training_input, validation_input, forecast_input, grouped_actual_input = inputs
        forecast_domain, validation_domain = domains
        
        train_loader, validation_loader, forecast_loader = get_loaders(training_input, validation_input, forecast_input, self.config_nn)
        
        model, train_losses, val_duds, nepochs = self.train_step(train_loader, validation_loader)
        
        model_curve, curve_parameters = self.predicting_step(model, forecast_loader, forecast_domain)
        
        _actual_cohort_indices = grouped_actual_input.cohort_date.between(self.config_nn['start_input_date'], self.config_nn['end_forecast_date'])
        real_pivot_data = grouped_actual_input[_actual_cohort_indices]\
                            .pivot(index='cohort_date',columns='dx',values='retention')
        
        
        total_error = abs(real_pivot_data.loc[model_curve.index,model_curve.columns] - model_curve).sum().sum()
        packed = (total_error, train_losses, val_duds, nepochs)
        
        return (
            model_curve,
            packed,
            curve_parameters.mean(axis=0).detach().numpy()
        )

    
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineeringTransformer(TransformerMixin, BaseEstimator):
    """Feature Engineering transform class for ML inputs"""

    def __init__(self):
        self.encoding_variables = {}

    def fit_transform(self, df, mutiple_segment=False):
        
        # OHE segment: example of OHE countries, to be used in multi segment cases
        if mutiple_segment:
            ohe_segment_encoder = OneHotEncoder(sparse=False)
            columns_to_one_hot = ['country']
            encoded_array = ohe_segment_encoder.fit_transform(df.loc[:,columns_to_one_hot])
            df_encoded = pd.DataFrame(encoded_array, columns=ohe_segment_encoder.get_feature_names_out() )
            df = pd.concat([df,df_encoded],axis=1)
            self.encoders['ohe_segment'] = ohe_segment_encoder

        # Cohort encoding and normalization
        self._fit_transform_cohort_date(df)

        # DNU transform
        self._fit_transform_log_dnu(df)
        

    def transform(self, df, max_trend_weeks=None):
        cohort_code = (df['cohort_date'] - self.encoding_variables['cohort_min_date']).dt.days.values
        df['norm_cohort_code'] = cohort_code / self.encoding_variables['cohort_code_max']
        
        if max_trend_weeks is not None:
            self._dampen_cohort_date_horizon(df, max_trend_weeks)
            
        df['log_dnu'] = (np.log1p(df['cohort_size']) - self.encoding_variables['mean_log_dnu']) / self.encoding_variables['std_log_dnu']
    
    
    def inverse_transform(self, df):
        inverse_cohort_code = round(df['norm_cohort_code'] * self.encoding_variables['cohort_code_max'])
        df['cohort_date'] = self.encoding_variables['cohort_min_date'] + pd.Timedelta(inverse_cohort_code)
        
        df['cohort_size'] = round(np.exp(self.encoding_variables['std_log_dnu'] * df['log_dnu'] + self.encoding_variables['mean_log_dnu'] )) - 1
        
    
    def _fit_transform_cohort_date(self, df):
        cohort_min_date = df['cohort_date'].min()
        cohort_code = (df['cohort_date'] - cohort_min_date).dt.days.values
        cohort_code_max =  cohort_code.max()

        df['norm_cohort_code'] = cohort_code / cohort_code_max
        
        self.encoding_variables['cohort_min_date'] = cohort_min_date
        self.encoding_variables['cohort_code_max'] = cohort_code_max

    def _fit_transform_log_dnu(self, df):
        mean_log_dnu, std_log_dnu = np.log1p(df['cohort_size']).mean(), np.log1p(df['cohort_size']).std()
        df['log_dnu'] = (np.log1p(df['cohort_size']) - mean_log_dnu) / std_log_dnu
        self.encoding_variables['mean_log_dnu'] = mean_log_dnu
        self.encoding_variables['std_log_dnu'] = std_log_dnu
    
    
    def _dampen_cohort_date_horizon(self, df, max_trend_weeks):
         # should be equal to the encoding cohort code max from feature engineering
        training_n_days = self.encoding_variables['cohort_code_max']
        new_cohort_code = df['norm_cohort_code'].copy().values
        
        max_value_cohort_code = 1. + (max_trend_weeks * 7.) / training_n_days
        
        # by assuming smoothness in transition point cohort_code_max we can
        # compute damp_factor and xs
        damp_factor = max_value_cohort_code - 1 # this was determined for smooth transition
        xs = 1.0 + damp_factor * np.log(max_value_cohort_code-1) # determine
        
        horizon_indices = new_cohort_code[new_cohort_code>1]
        df.loc[df['norm_cohort_code']>1,'norm_cohort_code'] = max_value_cohort_code - np.exp(-(horizon_indices-xs) / damp_factor)
        
    
        