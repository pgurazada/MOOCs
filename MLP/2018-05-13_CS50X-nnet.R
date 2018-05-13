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

model <- keras_model_sequential() %>% 
  layer_dense(units = 500, 
              activation = 'relu', 
              input_shape = ncol(x_train)) %>% # layer 1
  
  layer_dropout(rate = 0.1) %>% 
  
  layer_dense(units = 250, 
              activation = 'relu') %>% # layer 2
  
  layer_dropout(rate = 0.1) %>% 
  
  layer_dense(units = 100, 
              activation = 'relu') %>% # layer 3
  
  layer_dropout(rate = 0.1) %>%
  
  layer_dense(units = 50, 
              activation = 'relu') %>% # layer 4
  
  layer_dropout(rate = 0.1) %>%
  
  layer_dense(units = 1, 
              activation = 'sigmoid') # output

compile(model, 
        optimizer = optimizer_adam(lr = 0.005),
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history <- fit(model, x_train, y_train,
               epochs = 10,
               batch_size = 100, 
               validation_split = 0.2)

