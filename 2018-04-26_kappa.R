#' ---
#' title: "Compute the engagement Kappa across courses"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Thu Apr 26 16:28:40 2018

library(caret)
library(tidyverse)
library(ggthemes)

set.seed(20130810)

theme_set(theme_few() + theme(plot.title = element_text(face="bold")))

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

mooc_df <- read_csv("data/HMXPC13_DI_v2_5-14-14.csv", progress = TRUE)
glimpse(mooc_df)

#' The following function munges the data to extract the features in the required
#' format that can then be passed on to the model

extract_features <- function(data_df, course_str, launch_date_str) {
  data_df %>% 
    filter(course_id == course_str) %>%
    select(registered, viewed, explored, certified, 
           gender, LoE_DI, YoB, final_cc_cname_DI, start_time_DI) %>% 
    mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
           launch_date = as.Date(launch_date_str),
           registered_before_launch = if_else(as.numeric(launch_date - start_time_DI) > 0,
                                              as.numeric(launch_date - start_time_DI),
                                              0),
           registered_after_launch = if_else(as.numeric(launch_date - start_time_DI) > 0,
                                             0,
                                             -as.numeric(launch_date - start_time_DI)),
           age = lubridate::year(launch_date) - as.numeric(YoB),
           male = case_when(gender == "m" ~ 1,
                            gender == "f" ~ 0),
           country = case_when(final_cc_cname_DI == "United States" ~ "US",
                               final_cc_cname_DI %in% c("India", "Pakistan", 
                                                        "Bangladesh", "China",
                                                        "Indonesia", "Japan", 
                                                        "Other East Asia", "Other Middle East/Central Asia",
                                                        "Other South Asia", "Philippines",
                                                        "Egypt") ~ "AS",
                               final_cc_cname_DI %in% c("France", "Germany", 
                                                        "Greece", "Other Europe",
                                                        "Poland", "Portugal", 
                                                        "Russian Federation", "Spain",
                                                        "Ukraine", "United Kingdom") ~ "EU",
                               final_cc_cname_DI %in% c("Morocco", "Nigeria", 
                                                        "Other Africa") ~ "AF",
                               TRUE ~ "OT"),
           education = case_when(LoE_DI == "Less than Secondary" ~ "LS",
                                 LoE_DI == "Secondary" ~ "SE",
                                 LoE_DI == "Bachelor's" ~ "BA",
                                 LoE_DI == "Master's" ~ "MA",
                                 LoE_DI == "Doctorate" ~ "DO")) %>% 
    mutate(country = factor(country),
           education = factor(education)) %>% 
    select(engaged, everything(), -registered, -viewed, -explored, -certified, 
           -final_cc_cname_DI, -LoE_DI, -YoB, -gender, -start_time_DI, -launch_date) %>%
    filter(age > 13) %>% 
    drop_na() ->
    out_df
  
  return(out_df)
}

#' The following function creates the training and testing sets, fits a random
#' forests model and computes the kappa from cross-validation in the training
#' set and the actual performance on the test set

compute_kappa <- function(clean_df) {
  tr_rows <- caret::createDataPartition(clean_df[["engaged"]], 
                                        p = 0.8, 
                                        list = FALSE)
  
  train_df <- clean_df[tr_rows, ]
  test_df <- clean_df[-tr_rows, ]
  
  mdl_rf <- train(factor(engaged) ~ .,
                  data = train_df,
                  method = "rf",
                  metric = "Kappa",
                  tuneGrid = data.frame(mtry = 1:4),
                  trControl = trainControl(method = "repeatedcv",
                                           number = 10,
                                           repeats = 3,
                                           sampling = "smote",
                                           allowParallel = TRUE))
  
  kappa_train <- max(mdl_rf[["results"]][["Kappa"]])
  
  c <- caret::confusionMatrix(predict(mdl_rf, test_df), test_df[["engaged"]])
  
  kappa_test <- c[["overall"]][["Kappa"]]
  
  return (c(kappa_train, kappa_test))
}

launch_date <- list()
launch_date[["HarvardX/CB22x/2013_Spring"]] <- "2013-03-13"
launch_date[["HarvardX/CS50x/2012"]] <- "2012-10-15"
launch_date[["HarvardX/ER22x/2013_Spring"]] <- "2013-03-02"
launch_date[["HarvardX/PH207x/2012_Fall"]] <- "2012-10-15"
launch_date[["HarvardX/PH278x/2013_Spring"]] <- "2013-05-15"
launch_date[["MITx/6.002x/2012_Fall"]] <- "2012-09-05"
launch_date[["MITx/6.002x/2013_Spring"]] <- "2013-03-03"
launch_date[["MITx/14.73x/2013_Spring"]] <- "2013-02-12"
launch_date[["MITx/2.01x/2013_Spring"]] <- "2013-04-15"
launch_date[["MITx/3.091x/2012_Fall"]] <- "2012-10-09"
launch_date[["MITx/3.091x/2013_Spring"]] <- "2013-02-05"
launch_date[["MITx/6.00x/2012_Fall"]] <- "2012-09-26"
launch_date[["MITx/6.00x/2013_Spring"]] <- "2013-02-04"
launch_date[["MITx/7.00x/2013_Spring"]] <- "2013-03-05"
launch_date[["MITx/8.02x/2013_Spring"]] <- "2013-02-18"
launch_date[["MITx/8.MReV/2013_Summer"]] <- "2013-06-01"

kappa_list <- list()

#' Analysis of "HarvardX/CB22x/2013_Spring"

course <- "HarvardX/CB22x/2013_Spring"
cat ("Starting random forests model on", course, "at: ", format(Sys.time()), "\n")

mooc_df %>% 
  extract_features(course, launch_date[[course]]) %>% 
  compute_kappa() ->
  kappa_vec

kappa_list[[course]] <- kappa_vec

#' Analysis of "HarvardX/CS50x/2012"

course <- "HarvardX/CS50x/2012"
cat ("Starting random forests model on", course, "at: ", format(Sys.time()), "\n")

mooc_df %>% 
  extract_features(course, launch_date[[course]]) %>% 
  compute_kappa() ->
  kappa_vec

kappa_list[[course]] <- kappa_vec


stopCluster(cluster)

#' Alternative apprroach to transfer data into a dataframe and run modeling in
#' another language
for (course in unique(mooc_df$course_id)) {
  mooc_df %>% 
    extract_features(course, launch_date[[course]]) ->
    df
    write_feather(df, paste0("data/", make.names(course), ".feather"))
}
