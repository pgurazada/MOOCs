#' ---
#' title: "Final neural net for CS50X"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Tue May 15 11:00:58 2018

library(keras)
library(yardstick)
library(pROC)

#' *Building the MLP*

x_train_sc <- readRDS("MLP/cs50x_xtrain_sc.rds")
x_test_sc <- readRDS("MLP/cs50x_xtest_sc.rds")

y_train <- readRDS("MLP/cs50x_ytrain.rds")
y_test <- readRDS("MLP/cs50x_ytest.rds")

model <- keras_model_sequential() %>% 
  layer_dense(units = 1000, 
              activation = 'relu',
              kernel_initializer = 'glorot_normal',
              input_shape = ncol(x_train_sc)) %>% # layer 1
  
  layer_dropout(rate = 0.1) %>% 
  
  layer_dense(units = 400, 
              activation = 'relu') %>% # layer 2
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 200, 
              activation = 'relu') %>% # layer 3
  
  layer_dropout(rate = 0.3) %>% 
  
  layer_dense(units = 50, 
              activation = 'relu') %>% # layer 4
  
  layer_dropout(rate = 0.3) %>% 
  
  layer_dense(units = 1, 
              activation = 'sigmoid') # output

compile(model, 
        optimizer = optimizer_adam(lr = 0.005),
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history <- fit(model, x_train, y_train,
               epochs = 3,
               batch_size = 100,
               verbose = 1)


#' *Computing the ROC*

predict_classes(object = model, 
                x = x_test) %>% 
  as.vector() ->
  yhat_class

predict_proba(object = model,
              x = x_test) %>% 
  as.vector() ->
  yhat_probs

estimates_test <- data.frame(truth = as.factor(y_test),
                             estimate = as.factor(yhat_class),
                             class_prob = yhat_probs)

conf_mat(estimates_test, truth, estimate)
metrics(estimates_test, truth , estimate)
roc_auc(estimates_test, truth, class_prob)

#' Plotting the ROC curve

dev.new()
plot(roc(estimates_test$truth, estimates_test$class_prob))

#' The above plot shows that the model is definitely much better than random 
#' guessing 
#' 

k_clear_session()
