#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:53:27 2020

@author: seykia
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, concatenate, Subtract
from keras.layers import LSTM, TimeDistributed
from Utilities.utils import robust_minmax
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from Utilities.evaluation import calibration_error


def nan_mse(y_actual, y_predicted):
    y_actual = tf.where(tf.math.is_nan(y_actual),
                            y_predicted,
                            y_actual)
    per_instance = tf.square(tf.subtract(y_predicted, y_actual))
    #per_instance = tf.where(tf.math.is_nan(y_actual),
    #                        tf.zeros_like(y_actual),
    #                        tf.square(tf.subtract(y_predicted, y_actual)))
    return tf.math.reduce_mean(per_instance, axis=-1)


class multiclass_auc(keras.metrics.AUC):
    
    
    """AUC for a single class in a muliticlass problem.

    Parameters
    ----------
    pos_label : int
        Label of the positive class (the one whose AUC is being computed).

    from_logits : bool, optional (default: False)
        If True, assume predictions are not standardized to be between 0 and 1.
        In this case, predictions will be squeezed into probabilities using the
        softmax function.

    sparse : bool, optional (default: True)
        If True, ground truth labels should be encoded as integer indices in the
        range [0, n_classes-1]. Otherwise, ground truth labels should be one-hot
        encoded indicator vectors (with a 1 in the true label position and 0
        elsewhere).

    **kwargs : keyword arguments
        Keyword arguments for tf.keras.metrics.AUC.__init__(). For example, the
        curve type (curve='ROC' or curve='PR').
    """

    def __init__(self, pos_label=None, from_logits=False, sparse=False, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.from_logits = from_logits
        self.sparse = sparse

    def update_state(self, y_true, y_pred, **kwargs):
        
        """Accumulates confusion matrix statistics.
        
        :param y_true: 
            The ground truth values. Either an integer tensor of shape
            (n_examples,) (if sparse=True) or a one-hot tensor of shape
            (n_examples, n_classes) (if sparse=False).
        :type y_true: tf.Tensor
        :param y_pred: The predicted values, a tensor of shape (n_examples, n_classes).
        :type y_pred: tf.Tensor
        :param **kwargs: Extra keyword arguments for tf.keras.metrics.AUC.update_state
            (e.g., sample_weight).
        :type **kwargs: keyword arguments
        

        """

        if self.sparse:
            y_true = tf.math.equal(y_true, self.pos_label)
            y_true = tf.squeeze(y_true)
        else:
            y_true = y_true[..., self.pos_label]
            
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_pred = y_pred[..., self.pos_label]

        super().update_state(y_true, y_pred, **kwargs)


class PPP_Network:
    
    def __init__(self, X_dynamic, X_static=None, Y_classes=None,
                 lstm_neurons=10, fc_dynamic_neurons=10, fc_static_neurons=10, 
                 fc_classification_neurons=5, dropout_prob=0.0, l2_w=0.0, 
                 use_c=False, use_tlstm=False, with_nan=False, fusion='concat',
                 with_residual=True, tlstm_decay_function='original'):
        
        self.dynamic_input_modalities = list(X_dynamic[0].keys())
        self.dropout_prob = dropout_prob
        self.tlstm_decay_function = tlstm_decay_function
        self.l2_w = l2_w
        self.with_nan = with_nan
        self.calibrated = False
        
        if X_static is not None:
            self.with_static = True
            self.static_input_modalities = list(X_static[0].keys())
        else: 
            self.with_static = False
            
        if Y_classes is not None:    
            self.classes = list(Y_classes[0].keys())
            self.class_num = []
            for classification in self.classes:
                self.class_num.append(len(np.unique(Y_classes[0][classification])))
                
        if isinstance(lstm_neurons, int):
            t = lstm_neurons
            lstm_neurons = []
            for k in range(len(self.dynamic_input_modalities)):
                lstm_neurons.append(t)
        self.lstm_neurons = lstm_neurons
        
        if isinstance(fc_dynamic_neurons, int):
            t = fc_dynamic_neurons
            fc_dynamic_neurons = []
            for k in range(len(self.dynamic_input_modalities)):
                fc_dynamic_neurons.append(t)
        self.fc_dynamic_neurons = fc_dynamic_neurons
        
        if isinstance(fc_classification_neurons, int):
            t = fc_classification_neurons
            fc_classification_neurons = []
            for k in range(len(self.classes)):
                fc_classification_neurons.append(t)
        self.fc_classification_neurons = fc_classification_neurons
        
        if isinstance(fc_static_neurons, int):
            t = fc_static_neurons
            fc_static_neurons = []
            for k in range(len(self.static_input_modalities)):
                fc_static_neurons.append(t)
        self.fc_static_neurons = fc_static_neurons
                                
        l2_reg = l2(l=l2_w)
        
        
        ##### DYNAMIC MODULE #####
        lstm_inputs = list()
        lstms = list()
        
        for i, assessment in enumerate(self.dynamic_input_modalities):
            lstm_inputs.append(Input(shape=(None, X_dynamic[0][assessment].shape[2]), 
                                     name='LSTM_input_'+assessment))
            if use_tlstm :
                lstms.append(TLSTM(lstm_neurons[i], input_shape=(None, X_dynamic[0][assessment].shape[2]), 
                            return_sequences=True, dropout=dropout_prob, recurrent_dropout=dropout_prob,
                            tlstm_decay_function=tlstm_decay_function,
                            kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, 
                            name='TLSTM_'+assessment)(lstm_inputs[i], training=True))
            else:
                if with_nan:
                    lstms.append(nanLSTM(lstm_neurons[i], use_c=use_c,
                                         input_shape=(None, X_dynamic[0][assessment].shape[2]), 
                                         return_sequences=True, dropout=dropout_prob, 
                                         recurrent_dropout=dropout_prob, 
                                         kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, 
                                         name='nanLSTM_'+assessment)(lstm_inputs[i], training=True))
                else:
                    lstms.append(LSTM(lstm_neurons[i], input_shape=(None, X_dynamic[0][assessment].shape[2]), 
                            return_sequences=True, dropout=dropout_prob, recurrent_dropout=dropout_prob, 
                            kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg, 
                            name='LSTM_'+assessment)(lstm_inputs[i], training=True))
          
            
        ##### STATIC MODULE #####   
        if self.with_static:
            static_inputs = list()
            if (fc_static_neurons is None):
                for i, modality in enumerate(self.static_input_modalities):
                    static_inputs.append(Input(shape=(None, X_static[0][modality].shape[2]), 
                                               name='Static_input_'+modality))
                if len(self.static_input_modalities) > 1:
                    static_outputs = concatenate(static_inputs, axis=2, name='Static_concat')
                else:
                    static_outputs = static_inputs[0]
            else:
                static_dense = list()
                for i, modality in enumerate(self.static_input_modalities):
                    static_inputs.append(Input(shape=(None, X_static[0][modality].shape[2]), 
                                               name='Static_input_'+modality))
                    if with_nan:
                        static_dense.append(TimeDistributed(nanDense(fc_static_neurons[i], 
                                                                        activation='relu', 
                                                                        kernel_regularizer=l2_reg, 
                                                                        name='Static_Dence_'+modality, use_c=use_c))
                                                                        (static_inputs[i], training=True))
                    else:
                        static_dense.append(TimeDistributed(Dense(fc_static_neurons[i], 
                                                                        activation='relu', 
                                                                        kernel_regularizer=l2_reg, 
                                                                        name='Static_Dence_'+modality))
                                                                        (static_inputs[i], training=True))
                                
                if len(self.static_input_modalities) > 1:
                    static_merged = concatenate(static_dense, axis=2, name='Static_concat')
                else:
                    static_merged = static_dense[0]
                
                static_merged_dropout = Dropout(dropout_prob, name='Static_Dropout')(static_merged, 
                                                                                     training=True)
                            
                static_outputs = TimeDistributed(Dense(20, activation='tanh', 
                                       kernel_regularizer=l2_reg, 
                                       name='Static_interaction_Dense'))(static_merged_dropout, 
                                                                         training=True)  
        else:
            static_outputs = None
            
        
        ######## Fusion Module ####################
        
        if fusion == 'concat':
            if len(self.dynamic_input_modalities) > 1:
                lstm_outputs = concatenate(lstms, axis=2, name='LSTM_concat')
            elif len(self.dynamic_input_modalities) == 1:
                lstm_outputs = lstms[0]
            else:
                lstm_outputs = None
                
            if (lstm_outputs is not None and static_outputs is not None):
                fused_outputs = concatenate([lstm_outputs, static_outputs], 
                                           axis=2, name='Concat_all')
            elif lstm_outputs is  None:
                fused_outputs=static_outputs
            elif static_outputs is  None:
                fused_outputs=lstm_outputs
            else:
                pass
        
        else:
            pass
            
        fused_outputs = Dropout(dropout_prob, name='Fusion_dropout')(fused_outputs, training=True)           
       
        ##### REGRESSION MODULE #####   
        output_layers = list()
        fc_layers = list()
        res_layers = list()
        dropout_fc = list()
        for i, assessment in enumerate(self.dynamic_input_modalities):
            
            if use_tlstm:
                reg_out_num = X_dynamic[0][assessment].shape[2] - 1
            else: 
                reg_out_num = X_dynamic[0][assessment].shape[2]
                
            if (fc_dynamic_neurons is None or fc_dynamic_neurons[i] is None):
                if with_nan:
                    output_layers.append(TimeDistributed(nanDense(reg_out_num, 
                                                   activation='relu', kernel_regularizer=l2_reg,
                                                   use_c=use_c), 
                                                    name=assessment)(fused_outputs))
                else:
                    output_layers.append(TimeDistributed(Dense(reg_out_num, 
                                                   activation='relu', kernel_regularizer=l2_reg), 
                                                    name=assessment)(fused_outputs))
            else:
                if with_nan:
                    fc_layers.append(TimeDistributed(nanDense(fc_dynamic_neurons[i], 
                                                   activation='relu', kernel_regularizer=l2_reg,
                                                   use_c=use_c))(fused_outputs))
                else:
                    fc_layers.append(TimeDistributed(Dense(fc_dynamic_neurons[i], 
                                                   activation='relu', kernel_regularizer=l2_reg))(fused_outputs))
                if with_residual:
                    if use_tlstm:
                        res_layers.append(Subtract()([lstm_inputs[i][:,:,0:-1], fc_layers[i]]))
                    else:
                        res_layers.append(Subtract()([lstm_inputs[i], fc_layers[i]]))
                    dropout_fc.append(Dropout(dropout_prob)(res_layers[i], training=True))
                else: 
                    dropout_fc.append(Dropout(dropout_prob)(fc_layers[i], training=True))
                if with_nan:
                    output_layers.append(TimeDistributed(nanDense(reg_out_num, 
                                                   activation='relu', kernel_regularizer=l2_reg), 
                                                    name=assessment)(dropout_fc[i]))
                else:
                    output_layers.append(TimeDistributed(Dense(reg_out_num, 
                                                   activation='relu', kernel_regularizer=l2_reg), 
                                                    name=assessment)(dropout_fc[i]))
        
        ##### CLASSIFICATION MODULE #####   
        for i, classification in enumerate(self.classes):
            if (fc_classification_neurons is None or fc_classification_neurons[i] is None):    
                if with_nan:
                    output_layers.append(TimeDistributed(nanDense(self.class_num[i], 
                                                           activation='softmax', 
                                                           kernel_regularizer=l2_reg,
                                                           use_c=use_c), 
                                                     name=classification)(fused_outputs))
                else:
                    output_layers.append(TimeDistributed(Dense(self.class_num[i], 
                                                           activation='softmax', 
                                                           kernel_regularizer=l2_reg), 
                                                     name=classification)(fused_outputs))
            else:
                if with_nan:
                    fc_layers.append(TimeDistributed(nanDense(fc_classification_neurons[i], 
                                                       activation='relu', 
                                                       kernel_regularizer=l2_reg,
                                                       use_c=use_c))(fused_outputs))
                else:
                    fc_layers.append(TimeDistributed(Dense(fc_classification_neurons[i], 
                                                       activation='relu', 
                                                       kernel_regularizer=l2_reg))(fused_outputs))
                dropout_fc.append(Dropout(dropout_prob)(fc_layers[len(self.dynamic_input_modalities)+i], 
                                                        training=True))
                output_layers.append(TimeDistributed(Dense(self.class_num[i], 
                                                           activation='softmax',
                                                           kernel_regularizer=l2_reg), 
                                                    name=classification)(dropout_fc[len(self.dynamic_input_modalities)+i]))
         
        if not self.with_static:
            model = Model(inputs=lstm_inputs, outputs=output_layers)
        else:
            model = Model(inputs=lstm_inputs+[static_inputs], outputs=output_layers)
         
        self.model = model
        self.model.summary()
                
        
    def fit(self, X_train, Y_train, R_train=None, X_static_tr=None, 
            epoch_num=10, batch_size=1, validation_split=0, verbose=1, checkpoint_path=None):
        
        loss = dict()
        loss_weights = dict()
        #class_weights = dict()
        metrics = dict()
        for assessment in self.dynamic_input_modalities:
            if self.with_nan: 
                loss[assessment] = nan_mse #'mean_squared_error'
                metrics[assessment] = nan_mse
            else:
                loss[assessment] = 'mean_squared_error'
                metrics[assessment] = 'mse'
            loss_weights[assessment] = 1.0
           
        for classification in self.classes:
            loss[classification] = 'categorical_crossentropy'
            loss_weights[classification] = 5.0
            #cw = class_weight.compute_class_weight('balanced',
            #                                     np.unique(R_train[-1][classification][:,-1,1]),
            #                                     R_train[-1][classification][:,-1,1])
            #class_weights[classification] = {0:cw[0],1:cw[1]}
            metrics[classification] = multiclass_auc(pos_label=1, 
                                                     name='multiclass_auc', 
                                                     curve='PR')
            
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=3e-4,
                                                                  decay_steps=10000,
                                                                  decay_rate=0.9,
                                                                  staircase=True)
        opt = Adam(learning_rate = lr_schedule)
       
        self.model.compile(loss=loss, loss_weights=loss_weights, metrics=metrics, optimizer=opt)
        
        if checkpoint_path != None:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                save_weights_only=True,
                save_best_only=False,
                save_freq="epoch")
        else:
            model_checkpoint_callback = None
        
        for i in reversed(range(len(X_train))): 
            X = [X_train[i][key] for key in X_train[i].keys()]
            if X_static_tr is not None:
                Xs = [X_static_tr[i][key] for key in self.static_input_modalities]
                X = X + Xs
            
            Y = Y_train[i]
            for classification in self.classes:
                Y[classification] = R_train[i][classification]
            
            if validation_split != 0:
                es = EarlyStopping(monitor='val_PANNS_remission_multiclass_auc', 
                                   mode='max', verbose=verbose, patience=max(3,epoch_num/10), 
                                   restore_best_weights=True)
                self.model.fit(X, Y, validation_split=validation_split, 
                               epochs=epoch_num, batch_size=batch_size, 
                               verbose=verbose, callbacks=[i for i in [es, model_checkpoint_callback] if i], 
                               ) 
            elif model_checkpoint_callback != None:
                self.model.fit(X, Y, epochs=epoch_num, batch_size=batch_size, 
                               verbose=verbose, callbacks = [model_checkpoint_callback]) 
            else:
                self.model.fit(X, Y, epochs=epoch_num, batch_size=batch_size, 
                               verbose=verbose) 
        
        #for e in range(epoch_num):
        #    for i in range(len(X_train)): 
        #        X = [X_train[i][key] for key in X_train[i].keys()]
        #        if X_static_tr is not None:
        #            X.append(X_static_tr[i])
                
        #        Y = Y_train[i]
        #        Y['remission'] = R_train[i]
        #        self.model.fit(X, Y, epochs=1, batch_size=batch_size, 
        #                       verbose=1)

    
    def predict(self, X_test, X_static_ts=None, past=1, future=1, uncertainty=True, 
                repetition = 100, intervals=None):
        
        X = [X_test[key] for key in self.dynamic_input_modalities]
        
        if self.with_static:
            if X_static_ts is None:
                print('X_static_ts must be specified!')
                return
            else:
                Xs = [X_static_ts[key] for key in self.static_input_modalities]
                Xt = X + Xs
        else:
            Xs = None
            Xt = X
        
        p = dict()
        for assessment in self.dynamic_input_modalities:
            p[assessment] = []    
        r = dict()
        for classification in self.classes:
            r[classification] = []   
        
        if not uncertainty:
            repetition = 1
            
        for i in range(repetition):
            if future == 1:
                predictions = self.model.predict(Xt)
            else:
                predictions = self.recursive_LSTM_predict(X, Xs, past=past, 
                                                          future=future, intervals=intervals)
            for a, assessment in enumerate(self.dynamic_input_modalities):
                p[assessment].append(predictions[a][:,past-1:,:])
            for c, classification in enumerate(self.classes):
                if type(predictions)==list:
                    r[classification].append(predictions[len(self.dynamic_input_modalities)+c][:,past-1:,:])
                else:
                    r[classification].append(predictions[:,past-1:,:])
        
        p_mean = dict()
        p_std = dict()
        for assessment in self.dynamic_input_modalities:
            p[assessment] = np.stack(p[assessment], axis=3)
            p_mean[assessment] = p[assessment].mean(axis=3)
            p_std[assessment] = p[assessment].std(axis=3)
        
        r_med = dict()
        r_range = dict()
        for classification in self.classes:
            r[classification] = np.stack(r[classification], axis=3)
            r_med[classification] = np.median(r[classification], axis=3)
            r_range[classification] = robust_minmax(r[classification])
            if self.calibrated:
                for k in range(r_med[classification].shape[1]):
                    r_med[classification][:,k,1] = self.calibrator[classification].predict(r_med[classification][:,k,1])
                    r_med[classification][:,k,0] = 1 - r_med[classification][:,k,1]
                    r_range[classification][:,k,1,0] = self.calibrator[classification].predict(r_range[classification][:,k,1,0])
                    r_range[classification][:,k,0,0] = 1 - r_range[classification][:,k,1,0]
                    r_range[classification][:,k,1,1] = self.calibrator[classification].predict(r_range[classification][:,k,1,1])
                    r_range[classification][:,k,0,1] = 1 - r_range[classification][:,k,1,1]
        return p_mean, p_std, r_med, r_range
    
    
    def load_weights(self, checkpoint_filepath):
        return self.model.load_weights(checkpoint_filepath)
    
    def summary(self):
        return self.model.summary()
    
    def print_layers_weights(self):
        for layer in self.model.layers:
            weights = layer.get_weights()
            print (layer.name)
            print(weights)
    
    
    def calibrate(self, X_train, Y_train, R_train=None, X_static_tr=None,
                  uncertainty=True, repetition = 100, 
                  intervals=None):
        
        Y_pred = {key:list() for key in self.classes} 
        Y_true = {key:list() for key in self.classes} 
        
        for i in range(0, len(X_train)):
            
            if self.with_static:
                predictions, predictions_std, predicted_remissions, predicted_remissions_std = self.predict(X_train[i], 
                                                                                                 X_static_ts=X_static_tr[i],
                                                                                                  past=X_train[i]['PANNS'].shape[1], 
                                                                                                  future=1, uncertainty=uncertainty, 
                                                                                                  repetition = repetition, 
                                                                                                  intervals=intervals)
            else:
                predictions, predictions_std, predicted_remissions, predicted_remissions_std= self.predict(X_train[i], past=X_train[i]['PANNS'].shape[1],
                                                                                                        future=1, uncertainty=uncertainty, 
                                                                                                        repetition = repetition, 
                                                                                                        intervals=intervals)  
                
            for c in self.classes:
                Y_pred[c].append(predicted_remissions[c][:,0,1:2])
                Y_true[c].append(R_train[i][c][:,-1,1:2])
                 
        self.calibrator = dict()
        for c in self.classes:
            Y_p = np.concatenate(Y_pred[c], axis=0)
            Y_t = np.concatenate(Y_true[c], axis=0)
            self.calibrator[c] = IsotonicRegression(out_of_bounds="clip")
            self.calibrator[c].fit(Y_p.squeeze(), Y_t.squeeze())
        self.calibrated = True

    
    def recursive_LSTM_predict(self, X, Xs, past=1, future=1, intervals=None):
        
        l = len(self.dynamic_input_modalities)
        if future == 1:
            temp = []
            if Xs is not None:
                X = X + Xs
            for i in range(len(X)):
                temp.append(X[i][:,0:past,:])
            return self.model.predict(temp)
        else:
            temp = self.recursive_LSTM_predict(X, Xs, past=past, future=future-1)[0:l]
            for i in range(len(self.dynamic_input_modalities)): 
                if intervals is None:
                    temp[i] = np.concatenate((X[i][:,0:past,:], temp[i][:,-future+1:,:]), axis=1)
                else:
                    temp_intervals = np.zeros([temp[i].shape[0],temp[i].shape[1],1])
                    for j in range(temp[i].shape[1]):
                        temp_intervals[:,j,0] = intervals[j+1]
                    temp[i] = np.concatenate((temp[i], temp_intervals), axis=2)
                    temp[i] = np.concatenate((X[i][:,0:past,:], temp[i][:,-future+1:,:]), axis=1)
                    
            if self.with_static:
                for i in range(len(Xs)):
                    Xs[i] = np.concatenate([Xs[i][:,0:1,:] for k in range(temp[0].shape[1])], axis=1)
                temp = temp + Xs 
                
            return self.model.predict(temp)
        
    def save(self, path):
        
        self.model.save(path)
    
    def load(self, path):
        
        # Retrieve the config
        config = self.model.get_config()
        
        # At loading time, register the custom objects with a `custom_object_scope`:
        custom_objects = {'multiclass_auc':multiclass_auc}
        with keras.utils.generic_utils.custom_object_scope(custom_objects):
            self.model = keras.Model.from_config(config)
                
        self.model = load_model(path, custom_objects=custom_objects)
              
        return self
        

def prepare_network_inputs(features, visits, remissions=None, static_features=None, 
                           assessments=[], intervals=None, training_ratio=0.8,
                           fixed_features=False, training_subs=None,
                           testing_subs=None, extended_training_subjects=True, 
                           target_visit=None, reverse=False):
     
    if static_features is None:
        static_features = dict()
        
    if len(assessments) > 0:
        with_dynamic = True
    else:
        with_dynamic = False
        if target_visit is None:
            target_visit = visits[-1]
        target_visit = visits.index(target_visit)
       
    if remissions is None:
        remissions = dict()
    else:
        classes = list(remissions['2'].keys())
        class_num = []
        for classification in classes:
            class_num.append(len(np.unique(remissions['2'][classification])))
        
    if (training_subs is None and testing_subs is None):
        subjects = np.array(list(features['2']['PANNS'].index))
        rand_ind = np.random.permutation(len(subjects))
        training_subs = subjects[rand_ind[0:int(np.ceil(training_ratio * rand_ind.shape[0]))]]
        testing_subs = subjects[rand_ind[int(np.ceil(training_ratio * rand_ind.shape[0])):]]
    elif extended_training_subjects:
        subjects = list(features['2']['PANNS'].index)
        training_subs = np.array(list(set(subjects) - set(testing_subs)))
    
    if with_dynamic:
        time_length = len(visits)-1
    else:
        time_length = 1
    
    temp_tr = {assessment: list() for assessment in assessments}
    temp_ts = {assessment: list() for assessment in assessments}
    temp_rem_tr = {classification: list() for classification in classes}
    temp_rem_ts = {classification: list() for classification in classes}
    temp_static_tr = {key: list() for key in static_features.keys()}
    temp_static_ts = {key: list() for key in static_features.keys()}
    
    for v, visit in enumerate(visits):
        for assessment in assessments: 
            a = features[visit][assessment].loc[features[visit][assessment].index.intersection(training_subs)]
            #a = features[visit][assessment].loc[training_subs]
            #b = features[visit][assessment].loc[testing_subs]
            b = features[visit][assessment].loc[features[visit][assessment].index.intersection(testing_subs)]
            temp_tr[assessment].append(a.dropna())
            temp_ts[assessment].append(b.dropna())
            if intervals is not None:
                t = np.zeros([temp_tr[assessment][v].shape[0],1])
                t[:,0] = intervals[v] 
                t = pd.DataFrame(t, index=temp_tr[assessment][v].index)
                temp_tr[assessment][v] = pd.concat([temp_tr[assessment][v], t], axis=1)
                t = np.zeros([temp_ts[assessment][v].shape[0],1])
                t[:,0] = intervals[v] 
                t = pd.DataFrame(t, index=temp_ts[assessment][v].index)
                temp_ts[assessment][v] = pd.concat([temp_ts[assessment][v], t], axis=1)
        for classification in classes:
            temp_rem_tr[classification].append(remissions[visit][classification].loc[remissions[visit][classification].index.intersection(training_subs)].dropna())
            temp_rem_ts[classification].append(remissions[visit][classification].loc[remissions[visit][classification].index.intersection(testing_subs)].dropna())
        for key in static_features.keys():
            temp_static_tr[key].append(features[visit][key].loc[features[visit][key].index.intersection(training_subs)].dropna())
            temp_static_ts[key].append(features[visit][key].loc[features[visit][key].index.intersection(testing_subs)].dropna())
    
    X_train = list()
    Y_train = list()
    R_train = list()
    X_test = list()
    Y_test = list()
    R_test = list()
    X_static_tr = list()
    X_static_ts = list()

    for i in range(time_length):
        
        X_train.append(dict())
        Y_train.append(dict())
        X_test.append(dict())
        Y_test.append(dict())
        R_train.append(dict())
        R_test.append(dict())
        X_static_tr.append(dict())
        X_static_ts.append(dict())
        
        for a, assessment in enumerate(assessments):
            x_train = list()
            y_train = list()
            x_test = list()
            y_test = list()
           

            for j in range(0, time_length-i):
                try:
                    index_intersect_tr = pd.concat([temp_tr[assessment][k] for k in range(j,j+i+2)], axis=1).dropna().index
                    x_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][k].reindex(index_intersect_tr).values,axis=1) for k in range(j,j+i+1)], 
                                         axis=1))
                    
                    if intervals is None:
                        y_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][k].reindex(index_intersect_tr).values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1))
                    else:
                        x_train[j][:,0,-1] = 0
                        y_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][k].reindex(index_intersect_tr).values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1)[:,:,:-1])
                except:
                    pass
                
                index_intersect_ts = pd.concat([temp_ts[assessment][k] for k in range(j,j+i+2)], axis=1).dropna().index
                x_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][k].loc[index_intersect_ts].values,axis=1) for k in range(j,j+i+1)], 
                                         axis=1))
                if intervals is None:
                    y_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][k].loc[index_intersect_ts].values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1))
                else: 
                    y_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][k].loc[index_intersect_ts].values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1)[:,:,:-1])
                
            X_train[i][assessment] = np.concatenate(x_train, axis=0)
            Y_train[i][assessment] = np.concatenate(y_train, axis=0)
            X_test[i][assessment] = np.concatenate(x_test, axis=0)
            Y_test[i][assessment] = np.concatenate(y_test, axis=0)
            
        for c, classification in enumerate(classes):
            r_train = list()
            r_test = list()

            if with_dynamic:
                for j in range(0, time_length-i):
                    index_intersect_tr = pd.concat([temp_rem_tr[classification][k] for k in range(j,j+i+2)], axis=1).dropna().index
                    index_intersect_ts = pd.concat([temp_rem_ts[classification][k] for k in range(j,j+i+2)], axis=1).dropna().index
                    ohe = OneHotEncoder(sparse=False, categories=[range(class_num[c])] * 1)
                    r_train.append(np.concatenate([np.expand_dims(ohe.fit_transform(temp_rem_tr[classification][k].loc[index_intersect_tr].values),
                                                                  axis=1) for k in range(j+1,j+i+2)], axis=1))
                    r_test.append(np.concatenate([np.expand_dims(ohe.fit_transform(temp_rem_ts[classification][k].loc[index_intersect_ts].values), 
                                                                 axis=1) for k in range(j+1,j+i+2)], axis=1))
            else:
                index_intersect_tr = temp_rem_tr[classification][target_visit].dropna().index
                index_intersect_ts = temp_rem_ts[classification][target_visit].dropna().index
                ohe = OneHotEncoder(sparse=False, categories=[range(class_num[c])] * 1)
                r_train.append(np.expand_dims(ohe.fit_transform(temp_rem_tr[classification][target_visit].loc[index_intersect_tr].values), axis=1))
                r_test.append(np.expand_dims(ohe.fit_transform(temp_rem_ts[classification][target_visit].loc[index_intersect_ts].values), axis=1))
        
            R_train[i][classification] = np.concatenate(r_train, axis=0)
            R_test[i][classification] = np.concatenate(r_test, axis=0)
                
        for key in static_features.keys():
          
            x_static_tr = list()
            x_static_ts = list()
            
            if with_dynamic:
                for j in range(0, time_length-i):
                
                    try:
                        index_intersect_tr = pd.concat([temp_static_tr[key][k] for k in range(j,j+i+2)], axis=1).dropna().index
                        #x_static_tr.append([temp_static_tr[key][k].loc[index_intersect_tr].values for k in range(j,j+i+1)][0])
                        x_static_tr.append(np.concatenate([np.expand_dims(temp_static_tr[key][k].reindex(index_intersect_tr).values,axis=1) for k in range(j,j+i+1)], 
                                             axis=1))
                    except:
                        pass
                    
                    index_intersect_ts = pd.concat([temp_static_ts[key][k] for k in range(j,j+i+2)], axis=1).dropna().index
                    #x_static_ts.append([temp_static_ts[key][k].loc[index_intersect_ts].values for k in range(j,j+i+1)][0])
                    x_static_ts.append(np.concatenate([np.expand_dims(temp_static_ts[key][k].reindex(index_intersect_ts).values,axis=1) for k in range(j,j+i+1)], 
                                             axis=1))     
            else:
                index_intersect_tr = temp_static_tr[key][target_visit].dropna().index
                x_static_tr.append(np.expand_dims(temp_static_tr[key][target_visit].reindex(index_intersect_tr).values,axis=1)) 
                                     
                index_intersect_ts = temp_static_ts[key][target_visit].dropna().index
                x_static_ts.append(np.expand_dims(temp_static_ts[key][target_visit].reindex(index_intersect_ts).values,axis=1)) 
                
            X_static_tr[i][key] = np.concatenate(x_static_tr, axis=0) 
            X_static_ts[i][key] = np.concatenate(x_static_ts, axis=0) 
          
        
    
    if not fixed_features:   
        X_static_tr = None
        X_static_ts = None
    
    if reverse:
        return X_train[::-1], Y_train[::-1], R_train[::-1], X_static_tr[::-1],\
                X_test, Y_test, R_test, X_static_ts, training_subs, testing_subs
    else:   
        return X_train, Y_train, R_train, X_static_tr, X_test, Y_test, R_test, \
            X_static_ts, training_subs, testing_subs


def evaluate_classification(remission, R_pred, tr=0.5):
    
    results = dict()
    try:
        results['auc'] = roc_auc_score(remission, R_pred)
        cf = confusion_matrix(remission, R_pred>=tr)
        TN = cf[0][0]
        FN = cf[1][0]
        TP = cf[1][1]
        FP = cf[0][1]
        results['TNR'] = TN / (TN + FP)
        results['FNR'] = FN / (FN + TP)
        results['TPR'] = TP / (TP + FN)
        results['FPR'] = FP / (FP + TN)
        results['sensitivity'] = TP / (TP + FN)
        results['specificity'] = TN / (TN + FP)
    except:
        results['auc'] = np.nan
        results['TNR'] = np.nan
        results['FNR'] = np.nan
        results['TPR'] = np.nan
        results['FPR'] = np.nan
        results['sensitivity'] = np.nan
        results['specificity'] = np.nan
    results['bac'] = balanced_accuracy_score(remission, R_pred>=tr)
    results['brier_score_loss'] = brier_score_loss(remission, R_pred)
    results['log_loss'] = log_loss(remission, R_pred)
    results['ECE'] = calibration_error(remission, R_pred, n_bins=10)
    
    return results

def evaluate_regression(regression_results):
    
    results = dict()
    for key in regression_results.keys():
        results[key] = np.zeros([regression_results[key].shape[2], 
                                 regression_results[key].shape[3]])
        for i in range(regression_results[key].shape[2]):
            for j in range(regression_results[key].shape[3]):
                results[key][i,j] = np.corrcoef(regression_results[key][0,:,i,j], 
                                                regression_results[key][1,:,i,j])[0,1]
            
    return results


def prepare_network_inputs2(features, visits, remissions=None, static_features=None, 
                           assessments=[], intervals=None, training_ratio=0.8,
                           fixed_features=False, training_subs=None,
                           testing_subs=None, extended_training_subjects=True, 
                           target_visit=None, reverse=False):
     
    if static_features is None:
        static_features = dict()
        
    if len(assessments) > 0:
        with_dynamic = True
    else:
        with_dynamic = False
        if target_visit is None:
            target_visit = visits[-1]
        target_visit = visits.index(target_visit)
       
    if remissions is None:
        remissions = dict()
    else:
        classes = list(remissions['2'].keys())
        class_num = []
        for classification in classes:
            class_num.append(len(np.unique(remissions['2'][classification])))
        
    if (training_subs is None and testing_subs is None):
        subjects = np.array(list(features['2']['PANNS'].index))
        rand_ind = np.random.permutation(len(subjects))
        training_subs = subjects[rand_ind[0:int(np.ceil(training_ratio * rand_ind.shape[0]))]]
        testing_subs = subjects[rand_ind[int(np.ceil(training_ratio * rand_ind.shape[0])):]]
    elif extended_training_subjects:
        subjects = list(features['2']['PANNS'].index)
        training_subs = np.array(list(set(subjects) - set(testing_subs)))
    
    temp_tr = {assessment: list() for assessment in assessments}
    temp_ts = {assessment: list() for assessment in assessments}
    temp_rem_tr = {classification: list() for classification in classes}
    temp_rem_ts = {classification: list() for classification in classes}
    temp_static_tr = {key: list() for key in static_features.keys()}
    temp_static_ts = {key: list() for key in static_features.keys()}
    
    for v, visit in enumerate(visits):
        for assessment in assessments: 
            a = features[visit][assessment].loc[features[visit][assessment].index.intersection(training_subs)]
            #a = features[visit][assessment].loc[training_subs]
            #b = features[visit][assessment].loc[testing_subs]
            b = features[visit][assessment].loc[features[visit][assessment].index.intersection(testing_subs)]
            if a.shape[0]>0:
                temp_tr[assessment].append(a)
            if b.shape[0]>0:
                temp_ts[assessment].append(b)
            if intervals is not None:
                t = np.zeros([temp_tr[assessment][v].shape[0],1])
                t[:,0] = intervals[v] 
                t = pd.DataFrame(t, index=temp_tr[assessment][v].index)
                temp_tr[assessment][v] = pd.concat([temp_tr[assessment][v], t], axis=1)
                t = np.zeros([temp_ts[assessment][v].shape[0],1])
                t[:,0] = intervals[v] 
                t = pd.DataFrame(t, index=temp_ts[assessment][v].index)
                temp_ts[assessment][v] = pd.concat([temp_ts[assessment][v], t], axis=1)
        for classification in classes:
            temp = remissions[visit][classification].loc[remissions[visit][classification].index.intersection(training_subs)]
            if temp.shape[0]>0:
                temp_rem_tr[classification].append(temp)
            temp = remissions[visit][classification].loc[remissions[visit][classification].index.intersection(testing_subs)]
            if temp.shape[0]>0:
                temp_rem_ts[classification].append(temp)
        for key in static_features.keys():
            temp = features[visit][key].loc[features[visit][key].index.intersection(training_subs)]
            if temp.shape[0]>0:
                temp_static_tr[key].append(temp)
            temp = features[visit][key].loc[features[visit][key].index.intersection(testing_subs)]
            if temp.shape[0]>0:
                temp_static_ts[key].append(temp)
    
    if with_dynamic:
        time_length_train = len(temp_tr[assessments[0]]) - 1
        time_length_test = len(temp_ts[assessments[0]]) - 1
    else:
        time_length_train = 1
        time_length_test = 1
    
    X_train = list()
    Y_train = list()
    R_train = list()
    X_test = list()
    Y_test = list()
    R_test = list()
    X_static_tr = list()
    X_static_ts = list()

    for i in range(time_length_train):
        
        X_train.append(dict())
        Y_train.append(dict())
        R_train.append(dict())
        X_static_tr.append(dict())
        
        for a, assessment in enumerate(assessments):
            x_train = list()
            y_train = list()
          
            for j in range(0, time_length_train-i):
                try:
                    index_intersect_tr = set.intersection(*[set(temp_tr[assessment][k].index) for k in range(j,j+i+2)])
                    x_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][k].reindex(index_intersect_tr).values,axis=1) for k in range(j,j+i+1)], 
                                         axis=1))
                    if intervals is None:
                        y_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][k].reindex(index_intersect_tr).values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1))
                    else:
                        x_train[j][:,0,-1] = 0
                        y_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][k].reindex(index_intersect_tr).values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1)[:,:,:-1])
                except:
                    pass
                
            X_train[i][assessment] = np.concatenate(x_train, axis=0)
            Y_train[i][assessment] = np.concatenate(y_train, axis=0)
            
        for c, classification in enumerate(classes):
            r_train = list()

            if with_dynamic:
                for j in range(0, time_length_train-i):
                    index_intersect_tr = set.intersection(*[set(temp_rem_tr[classification][k].index) for k in range(j,j+i+2)])
                    ohe = OneHotEncoder(sparse=False, categories=[range(class_num[c])] * 1)
                    r_train.append(np.concatenate([np.expand_dims(ohe.fit_transform(temp_rem_tr[classification][k].reindex(index_intersect_tr).dropna().values),
                                                                  axis=1) for k in range(j+1,j+i+2)], axis=1))
            else:
                index_intersect_tr = temp_rem_tr[classification][target_visit].dropna().index
                ohe = OneHotEncoder(sparse=False, categories=[range(class_num[c])] * 1)
                r_train.append(np.expand_dims(ohe.fit_transform(temp_rem_tr[classification][target_visit].reindex(index_intersect_tr).dropna().values), axis=1))
        
            R_train[i][classification] = np.concatenate(r_train, axis=0)
                
        for key in static_features.keys():
          
            x_static_tr = list()
            
            if with_dynamic:
                for j in range(0, time_length_train-i):
                
                    try:
                        index_intersect_tr = set.intersection(*[set(temp_static_tr[key][k].index) for k in range(j,j+i+2)])
                        x_static_tr.append(np.concatenate([np.expand_dims(temp_static_tr[key][k].reindex(index_intersect_tr).values,axis=1) for k in range(j,j+i+1)], 
                                             axis=1))
                    except:
                        pass
                    
            else:
                index_intersect_tr = temp_static_tr[key][target_visit].dropna().index
                x_static_tr.append(np.expand_dims(temp_static_tr[key][target_visit].reindex(index_intersect_tr).values,axis=1)) 
                                        
            X_static_tr[i][key] = np.concatenate(x_static_tr, axis=0) 
          
        
    for i in range(time_length_test):
        
        X_test.append(dict())
        Y_test.append(dict())
        R_test.append(dict())
        X_static_ts.append(dict())
        
        for a, assessment in enumerate(assessments):
            x_test = list()
            y_test = list()
          
            for j in range(0, time_length_test-i):
                index_intersect_ts = set.intersection(*[set(temp_ts[assessment][k].index) for k in range(j,j+i+2)])
                x_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][k].loc[index_intersect_ts].values,axis=1) for k in range(j,j+i+1)], 
                                         axis=1))
                if intervals is None:
                    y_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][k].loc[index_intersect_ts].values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1))
                else: 
                    y_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][k].loc[index_intersect_ts].values,axis=1) for k in range(j+1,j+i+2)], 
                                         axis=1)[:,:,:-1])
                
            X_test[i][assessment] = np.concatenate(x_test, axis=0)
            Y_test[i][assessment] = np.concatenate(y_test, axis=0)
            
        for c, classification in enumerate(classes):
            r_test = list()

            if with_dynamic:
                for j in range(0, time_length_test-i):
                    index_intersect_ts = set.intersection(*[set(temp_rem_ts[classification][k].index) for k in range(j,j+i+2)])
                    ohe = OneHotEncoder(sparse=False, categories=[range(class_num[c])] * 1)
                    r_test.append(np.concatenate([np.expand_dims(ohe.fit_transform(temp_rem_ts[classification][k].loc[index_intersect_ts].values), 
                                                                 axis=1) for k in range(j+1,j+i+2)], axis=1))
            else:
                index_intersect_ts = temp_rem_ts[classification][target_visit].dropna().index
                ohe = OneHotEncoder(sparse=False, categories=[range(class_num[c])] * 1)
                r_test.append(np.expand_dims(ohe.fit_transform(temp_rem_ts[classification][target_visit].loc[index_intersect_ts].values), axis=1))
        
            R_test[i][classification] = np.concatenate(r_test, axis=0)
                
        for key in static_features.keys():
          
            x_static_tr = list()
            x_static_ts = list()
            
            if with_dynamic:
                for j in range(0, time_length_test-i):
            
                    index_intersect_ts = set.intersection(*[set(temp_static_ts[key][k].index) for k in range(j,j+i+2)])        
                    x_static_ts.append(np.concatenate([np.expand_dims(temp_static_ts[key][k].reindex(index_intersect_ts).values,axis=1) for k in range(j,j+i+1)], 
                                             axis=1))     
            else:
                                 
                index_intersect_ts = temp_static_ts[key][target_visit].dropna().index
                x_static_ts.append(np.expand_dims(temp_static_ts[key][target_visit].reindex(index_intersect_ts).values,axis=1)) 
                
            X_static_ts[i][key] = np.concatenate(x_static_ts, axis=0) 
          
        
    if not fixed_features:   
        X_static_tr = None
        X_static_ts = None
    
    if reverse:
        return X_train[::-1], Y_train[::-1], R_train[::-1], X_static_tr[::-1],\
                X_test, Y_test, R_test, X_static_ts, training_subs, testing_subs
    else:   
        return X_train, Y_train, R_train, X_static_tr, X_test, Y_test, R_test, \
            X_static_ts, training_subs, testing_subs
            


def prepare_TLSTM_inputs(features, visits, intervals, remissions=None, static_features=None, 
                           assessments=[], training_ratio=0.8,
                           fixed_features=False, training_subs=None,
                           testing_subs=None, extended_training_subjects=True, 
                           target_visit=None, reverse=False):
     
    if static_features is None:
        static_features = dict()
       
    if remissions is None:
        remissions = dict()
    else:
        classes = list(remissions['2'].keys())
        class_num = []
        for classification in classes:
            class_num.append(len(np.unique(remissions['2'][classification])))
        
    if (training_subs is None and testing_subs is None):
        subjects = np.array(list(features['2']['PANNS'].index))
        rand_ind = np.random.permutation(len(subjects))
        training_subs = subjects[rand_ind[0:int(np.ceil(training_ratio * rand_ind.shape[0]))]]
        testing_subs = subjects[rand_ind[int(np.ceil(training_ratio * rand_ind.shape[0])):]]
    elif extended_training_subjects:
        subjects = list(features['2']['PANNS'].index)
        training_subs = np.array(list(set(subjects) - set(testing_subs)))
    
    
    time_length = len(visits)-1
    
    
    temp_tr = {assessment: list() for assessment in assessments}
    temp_ts = {assessment: list() for assessment in assessments}
    temp_rem_tr = {classification: list() for classification in classes}
    temp_rem_ts = {classification: list() for classification in classes}
    temp_static_tr = {key: list() for key in static_features.keys()}
    temp_static_ts = {key: list() for key in static_features.keys()}
    
    for v, visit in enumerate(visits):
        for assessment in assessments: 
            a = features[visit][assessment].loc[features[visit][assessment].index.intersection(training_subs)]
            b = features[visit][assessment].loc[features[visit][assessment].index.intersection(testing_subs)]
            temp_tr[assessment].append(a.dropna())
            temp_ts[assessment].append(b.dropna())
            
            t = np.zeros([temp_tr[assessment][v].shape[0],1])
            t[:,0] = intervals[v] 
            t = pd.DataFrame(t, index=temp_tr[assessment][v].index)
            temp_tr[assessment][v] = pd.concat([temp_tr[assessment][v], t], axis=1)
            t = np.zeros([temp_ts[assessment][v].shape[0],1])
            t[:,0] = intervals[v] 
            t = pd.DataFrame(t, index=temp_ts[assessment][v].index)
            temp_ts[assessment][v] = pd.concat([temp_ts[assessment][v], t], axis=1)
        
        for classification in classes:
            temp_rem_tr[classification].append(remissions[visit][classification].loc[remissions[visit][classification].index.intersection(training_subs)].dropna())
            temp_rem_ts[classification].append(remissions[visit][classification].loc[remissions[visit][classification].index.intersection(testing_subs)].dropna())
        for key in static_features.keys():
            temp_static_tr[key].append(features[visit][key].loc[features[visit][key].index.intersection(training_subs)].dropna())
            temp_static_ts[key].append(features[visit][key].loc[features[visit][key].index.intersection(testing_subs)].dropna())
    
    X_train = list()
    Y_train = list()
    R_train = list()
    X_test = list()
    Y_test = list()
    R_test = list()
    X_static_tr = list()
    X_static_ts = list()

    for i in range(1,time_length):
        
        X_train.append(dict())
        Y_train.append(dict())
        X_test.append(dict())
        Y_test.append(dict())
        R_train.append(dict())
        R_test.append(dict())
        X_static_tr.append(dict())
        X_static_ts.append(dict())
        
        for a, assessment in enumerate(assessments):
            x_train = list()
            y_train = list()
            x_test = list()
            y_test = list()
            z = 0
            for j in range(0, time_length-i): #start
                for k in range(j+i, time_length): #end
                    v0 = list(np.unique(list(range(j,j+i+1))+[k, k+1]))
                    v1 = list(np.unique(list(range(j,j+i))+[k]))
                    v2 = list(np.unique(list(range(j+1,j+i+1))+[k+1]))
                    intv = np.array(intervals)[v1[0]:v1[-1]+1]
                    intv[0] = 0
                    index_intersect_tr = pd.concat([temp_tr[assessment][l] for l in v0], axis=1).dropna().index
                    x_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][l].reindex(index_intersect_tr).values,axis=1) for l in v1], 
                                         axis=1))
                    
                    for y,l in enumerate(v1):
                        x_train[z][:,y,-1] = np.sum(intv[0:l])
                                                
                    y_train.append(np.concatenate([np.expand_dims(temp_tr[assessment][l].reindex(index_intersect_tr).values,axis=1) for l in v2], 
                                         axis=1)[:,:,:-1])
                
                
                    index_intersect_ts = pd.concat([temp_ts[assessment][l] for l in v0], axis=1).dropna().index
                    x_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][l].loc[index_intersect_ts].values,axis=1) for l in v1], 
                                             axis=1))
                   
                    y_test.append(np.concatenate([np.expand_dims(temp_ts[assessment][l].loc[index_intersect_ts].values,axis=1) for l in v2], 
                                             axis=1)[:,:,:-1])
                    z += 1
                    
            if len(x_train)>0:
                X_train[i-1][assessment] = np.concatenate(x_train, axis=0)
                Y_train[i-1][assessment] = np.concatenate(y_train, axis=0)
                X_test[i-1][assessment] = np.concatenate(x_test, axis=0)
                Y_test[i-1][assessment] = np.concatenate(y_test, axis=0)
            
        for c, classification in enumerate(classes):
            r_train = list()
            r_test = list()

            for j in range(0, time_length-i):
                for k in range(j+i, time_length):
                    v0 = list(np.unique(list(range(j,j+i+1))+[k, k+1]))
                    v2 = list(np.unique(list(range(j+1,j+i+1))+[k+1]))
                    index_intersect_tr = pd.concat([temp_rem_tr[classification][l] for l in v0], axis=1).dropna().index
                    index_intersect_ts = pd.concat([temp_rem_ts[classification][l] for l in v0], axis=1).dropna().index
                    ohe = OneHotEncoder(sparse=False, categories=[range(class_num[c])] * 1)
                    r_train.append(np.concatenate([np.expand_dims(ohe.fit_transform(temp_rem_tr[classification][l].loc[index_intersect_tr].values),
                                                                  axis=1) for l in v2], axis=1))
                    r_test.append(np.concatenate([np.expand_dims(ohe.fit_transform(temp_rem_ts[classification][l].loc[index_intersect_ts].values), 
                                                                 axis=1) for l in v2], axis=1))
                    
            if len(r_train)>0:
                R_train[i-1][classification] = np.concatenate(r_train, axis=0)
                R_test[i-1][classification] = np.concatenate(r_test, axis=0)
                
        for key in static_features.keys():
          
            x_static_tr = list()
            x_static_ts = list()
            
            for j in range(0, time_length-i):
                for k in range(j+i, time_length):
                    v0 = list(np.unique(list(range(j,j+i+1))+[k, k+1]))
                    v1 = list(np.unique(list(range(j,j+i))+[k]))
                    try:
                        index_intersect_tr = pd.concat([temp_static_tr[key][l] for l in v0], axis=1).dropna().index
                        x_static_tr.append(np.concatenate([np.expand_dims(temp_static_tr[key][l].reindex(index_intersect_tr).values,axis=1) for l in v1], 
                                             axis=1))
                    except:
                        pass
                    
                    index_intersect_ts = pd.concat([temp_static_ts[key][l] for l in v0], axis=1).dropna().index
                    x_static_ts.append(np.concatenate([np.expand_dims(temp_static_ts[key][l].reindex(index_intersect_ts).values,axis=1) for l in v1], 
                                             axis=1))     
            if len(x_static_tr)>0:
                X_static_tr[i-1][key] = np.concatenate(x_static_tr, axis=0) 
                X_static_ts[i-1][key] = np.concatenate(x_static_ts, axis=0) 
          
        
    
    if not fixed_features:   
        X_static_tr = None
        X_static_ts = None
    
    if reverse:
        return X_train[::-1], Y_train[::-1], R_train[::-1], X_static_tr[::-1],\
                X_test, Y_test, R_test, X_static_ts, training_subs, testing_subs
    else:   
        return X_train, Y_train, R_train, X_static_tr, X_test, Y_test, R_test, \
            X_static_ts, training_subs, testing_subs
