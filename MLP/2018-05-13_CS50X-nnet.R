#' ---
#' title: "Fitting a multi layer perceptron to CS50X"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Sun May 13 15:56:01 2018

library(keras)

#' *Building the MLP*

x_train <- readRDS("MLP/cs50x_xtrain.rds")
x_test <- readRDS("MLP/cs50x_xtest.rds")

y_train <- readRDS("MLP/cs50x_ytrain.rds")
y_test <- readRDS("MLP/cs50x_ytest.rds")

FLAGS <- flags(flag_integer("dense_units1", 1000),
               flag_numeric("dropout1", 0.1),
               flag_integer("dense_units2", 400),
               flag_numeric("dropout2", 0.2),
               flag_integer("dense_units3", 200),
               flag_numeric("dropout3", 0.3),
               flag_integer("dense_units4", 50),
               flag_numeric("dropout4", 0.3),
               flag_integer("epochs", 15),
               flag_integer("batch_size", 100),
               flag_numeric("learning_rate", 10^-6))

model <- keras_model_sequential() %>% 
  layer_dense(units = FLAGS$dense_units1, 
              activation = 'relu',
              kernel_initializer = 'glorot_normal',
              input_shape = ncol(x_train)) %>% # layer 1
  
  layer_batch_normalization() %>% 
  
  layer_dropout(rate = FLAGS$dropout1) %>% 

  layer_dense(units = FLAGS$dense_units2, 
              activation = 'relu',
              kernel_initializer = 'glorot_normal') %>% # layer 2
  
  layer_batch_normalization() %>% 
  
  layer_dropout(rate = FLAGS$dropout2) %>% 
  
  layer_dense(units = FLAGS$dense_units3, 
              activation = 'relu',
              kernel_initializer = 'glorot_normal') %>% # layer 3
  
  layer_batch_normalization() %>% 
  
  layer_dropout(rate = FLAGS$dropout3) %>% 
  
  layer_dense(units = FLAGS$dense_units4, 
              activation = 'relu',
              kernel_initializer = 'glorot_normal') %>% # layer 4
  
  layer_batch_normalization() %>% 
  
  layer_dropout(rate = FLAGS$dropout4) %>% 
  
  layer_dense(units = 1, 
              activation = 'sigmoid') # output

compile(model, 
        optimizer = optimizer_adam(lr = FLAGS$learning_rate),
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history <- fit(model, x_train, y_train,
               epochs = FLAGS$epochs,
               batch_size = FLAGS$batch_size,
               verbose = 1,
               validation_split = 0.2, 
               callbacks = list(callback_early_stopping(patience = 2),
                                callback_reduce_lr_on_plateau(patience = 2)))

k_clear_session()

# tuning_run("MLP/2018-05-13_CS50X-nnet.R",
#            runs_dir = "mlp_tuning",
#            sample = 0.01,
#            flags = list(dense_units1 = c(1000, 500),
#                         dense_units2 = c(500, 400),
#                         dense_units3 = c(200, 100),
#                         dense_units4 = c(100, 50),
#                         dropout1 = c(0.1, 0.2, 0.3),
#                         dropout2 = c(0.1, 0.2, 0.3),
#                         dropout3 = c(0.1, 0.2, 0.3),
#                         dropout4 = c(0.1, 0.2, 0.3),
#                         epochs = c(3, 5),
#                         batch_size = c(100, 200),
#                         learning_rate = c(0.001, 0.005)))

# ls_runs(order = metric_val_acc, runs_dir = "mlp_tuning")
# copy_run(ls_runs(metric_val_acc >= 0.6872), to = "best-runs")