#' ---
#' title: "Exploring neural net with CS50X"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Sun May 13 13:23:29 2018

#' In this script we explore if a neural net will perform better than the 
#' usual suite of classification algorithms we explored within scikit

library(rsample)
library(recipes)
library(onehot)
library(yardstick)
library(tidyverse)
library(ggthemes)

theme_set(theme_few())

#' *Preparing the data*

mooc_df <- read_csv("data/HMXPC13_DI_v2_5-14-14.csv", progress = TRUE)
glimpse(mooc_df)

mooc_df %>% 
  filter(course_id == "HarvardX/CS50x/2012") %>% 
  select(userid_DI, grade, ndays_act, registered, explored, certified, viewed, 
         start_time_DI, last_event_DI, nevents, nchapters, nforum_posts, 
         LoE_DI, gender, YoB, final_cc_cname_DI) ->
  cs50x_df

glimpse(cs50x_df)

mooc_df %>% 
  filter(course_id == "HarvardX/CS50x/2012") %>%
  
  select(registered, viewed, explored, certified, 
         gender, LoE_DI, YoB, final_cc_cname_DI, start_time_DI) %>% 
  
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         launch_date = as.Date("2012-10-15")) %>% 
  
  mutate(gender_na = ifelse(is.na(gender), 1, 0),
         male = ifelse(gender == "m" & !is.na(gender), 1, 0),
         female = ifelse(gender == "f" & !is.na(gender), 1, 0),
         gender_other = ifelse(gender == "o" & !is.na(gender), 1, 0)) %>% 
  
  mutate(education_na = ifelse(is.na(LoE_DI), 1, 0),
         education_ls = ifelse(LoE_DI == "Less than Secondary" & !is.na(LoE_DI), 1, 0),
         education_se = ifelse(LoE_DI == "Secondary" & !is.na(LoE_DI), 1, 0),
         education_ba = ifelse(LoE_DI == "Bachelor's" & !is.na(LoE_DI), 1, 0),
         education_ma = ifelse(LoE_DI == "Master's" & !is.na(LoE_DI), 1, 0),
         education_phd = ifelse(LoE_DI == "Doctorate" & !is.na(LoE_DI), 1, 0)) %>% 
  
  mutate(country = factor(final_cc_cname_DI)) %>%  
  
  mutate(age = factor(2013 - as.numeric(YoB))) %>% 
  
  mutate(registered_before_launch = if_else(as.numeric(launch_date - start_time_DI) > 0,
                                            as.numeric(launch_date - start_time_DI),
                                            0),
         registered_after_launch = if_else(as.numeric(launch_date - start_time_DI) > 0,
                                           0,
                                           -as.numeric(launch_date - start_time_DI))) %>% 
  mutate(early_by = factor(registered_before_launch),
         late_by = factor(registered_after_launch)) %>% 
  select(engaged, everything(), 
         -registered, -viewed, -explored, -certified, -gender, -LoE_DI, -YoB, 
         -final_cc_cname_DI, -start_time_DI, -launch_date, 
         -registered_before_launch, -registered_after_launch) ->
  
 cs50_tbl

oh_encoder <- onehot(cs50_tbl, max_levels = 330)

cs50_mat <- predict(oh_encoder, cs50_tbl)
glimpse(cs50_mat)

train_test_split <- initial_split(cs50_mat, prop = 0.8)
train_mat <- training(train_test_split)
test_mat <- testing(train_test_split)

glimpse(train_mat)

x_train <- train_mat[, -1]
x_test <- test_mat[, -1]

y_train <- train_mat[, 1]
y_test <- test_mat[, 1]

saveRDS(object = x_train, "MLP/cs50x_xtrain.rds")
saveRDS(object = x_test, "MLP/cs50x_xtest.rds")
saveRDS(object = y_train, "MLP/cs50x_ytrain.rds")
saveRDS(object = y_test, "MLP/cs50x_ytest.rds")


