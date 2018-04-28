#' ---
#' title: "Analysis of MITx/8.02x/2013_Spring"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Thu Apr 26 12:40:59 2018

library(caret)
library(tidyverse)
library(ggthemes)
library(gridExtra)

set.seed(20130810)

theme_set(theme_few() + theme(plot.title = element_text(face="bold")))

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

mooc_df <- read_csv("data/HMXPC13_DI_v2_5-14-14.csv", progress = TRUE)
glimpse(mooc_df)

mooc_df %>% 
  filter(course_id == "MITx/8.02x/2013_Spring") %>% 
  select(userid_DI, grade, ndays_act, registered, explored, certified, viewed, 
         start_time_DI, last_event_DI, nevents, nchapters, nforum_posts, 
         LoE_DI, gender, YoB, final_cc_cname_DI) ->
  mit802x_df

glimpse(mit802x_df)

#' MIT 8.02x is an experimental aproach to electricity and magnetism

mit802x_df %>%
  gather(Variable, Value) %>% 
  group_by(Variable) %>% 
  summarize(missing_perc = floor(sum(is.na(Value)) * 100/length(Value)))

dev.new()
mit802x_df %>% 
  select(registered, viewed, explored, certified) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>% 
  ggplot() +
  geom_bar(aes(x = engaged, y = ..prop..)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey() +
  labs(x = "Engaged?",
       y = "Proportion",
       title = "Distribution of registered participants (MIT 8.02x)",
       caption = "Note: a registered participant engaged if they watched at least one video")

#' 40% dropout

mit802x_df %>% 
  select(registered, viewed, explored, certified, gender) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>%
  drop_na() %>% 
  ggplot() +
  geom_bar(aes(x = engaged, fill = gender)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey("Gender", labels = c("Female", "Male")) +
  labs(x = "Engaged?",
       y = "Count",
       title = "Distribution of registered participants by gender (MIT 8.02x)") -> p1

mit802x_df %>% 
  select(registered, viewed, explored, certified, LoE_DI) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         LoE_DI = factor(LoE_DI, levels = c("Less than Secondary", "Secondary", "Bachelor's", "Master's", "Doctorate"))) %>%
  drop_na() %>% 
  ggplot() +
  geom_bar(aes(x = engaged, fill = LoE_DI)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey("Education") +
  labs(x = "Engaged?",
       y = "Count",
       title = "Distribution of registered participants by level of education (MIT 8.02x)") -> p2

grid.arrange(p1, p2, nrow = 2)

#' Heavily male dominant

mooc_df %>% 
  filter(course_id == "MITx/8.02x/2013_Spring") %>%
  select(registered, viewed, explored, certified, 
         gender, LoE_DI, YoB, final_cc_cname_DI, start_time_DI) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         launch_date = as.Date("2013-02-18"),
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
  mit802x_neng_df

glimpse(mit802x_neng_df)

#' We fit a couple of models to see if we can predict engagement from
#' demographics and registration times

tr_rows <- createDataPartition(mit802x_neng_df$engaged, p = 0.8, list = FALSE)

mit802x_train <- mit802x_neng_df[tr_rows, ]
mit802x_test <- mit802x_neng_df[-tr_rows, ]

mit802x_neng_logit <- train(factor(engaged) ~ .,
                            data = mit802x_train,
                            method = "glm",
                            metric = "Kappa",
                            trControl = trainControl(method = "repeatedcv",
                                                     number = 10,
                                                     repeats = 5,
                                                     allowParallel = TRUE,
                                                     sampling = "smote"),
                            family = binomial(link = "logit"))

summary(mit802x_neng_logit)

mit802x_neng_logit$results

confusionMatrix(mit802x_neng_logit)

mit802x_neng_rf <- train(factor(engaged) ~ .,
                         data = mit802x_train,
                         method = "rf",
                         tuneGrid = expand.grid(mtry = 1:4),
                         trControl = trainControl(method = "repeatedcv",
                                                  number = 10,
                                                  repeats = 3,
                                                  sampling = "smote",
                                                  allowParallel = TRUE))

mit802x_neng_rf$results
varImp(mit802x_neng_rf)

confusionMatrix(mit802x_neng_rf)
