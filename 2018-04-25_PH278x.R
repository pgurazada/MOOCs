#' ---
#' title: "Anaysis of Harvard PH278x"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Wed Apr 25 11:51:26 2018

library(tidyverse)
library(ggthemes)
library(gridExtra)
library(caret)

set.seed(20130810)

theme_set(theme_few() + theme(plot.title = element_text(face="bold")))

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

mooc_df <- read_csv("data/HMXPC13_DI_v2_5-14-14.csv", progress = TRUE)
glimpse(mooc_df)

mooc_df %>% 
  filter(course_id == "HarvardX/PH278x/2013_Spring") %>% 
  select(userid_DI, grade, ndays_act, registered, explored, certified, viewed, 
         start_time_DI, last_event_DI, nevents, nchapters, nforum_posts, 
         LoE_DI, gender, YoB, final_cc_cname_DI) ->
  ph278x_df

glimpse(ph278x_df)

#' The PH278x course focuses on global environmental changes, its causes, health 
#' consequences, and proposed solutions

ph278x_df %>%
  gather(Variable, Value) %>% 
  group_by(Variable) %>% 
  summarize(missing_perc = floor(sum(is.na(Value)) * 100/length(Value)))

dev.new()
ph278x_df %>% 
  select(registered, viewed, explored, certified) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>% 
  ggplot() +
  geom_bar(aes(x = engaged, y = ..prop..)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey() +
  labs(x = "Engaged?",
       y = "Proportion",
       title = "Distribution of registered participants (Harvard PH278x)",
       caption = "Note: a registered participant engaged if they watched at least one video")

#' The above figure shows a 60-40 split between the notengaged-engaged
#' participants. 

ph278x_df %>% 
  select(registered, viewed, explored, certified, gender) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>%
  drop_na() %>% 
  ggplot() +
  geom_bar(aes(x = engaged, fill = gender)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey("Gender", labels = c("Female", "Male")) +
  labs(x = "Engaged?",
       y = "Count",
       title = "Distribution of registered participants by gender (Harvard PH278x)") -> p1

ph278x_df %>% 
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
       title = "Distribution of registered participants by level of education (Harvard PH278x)") -> p2

grid.arrange(p1, p2, nrow = 2)

#' This course had an even split of male-female

mooc_df %>% 
  filter(course_id == "HarvardX/PH278x/2013_Spring") %>%
  select(registered, viewed, explored, certified, gender, LoE_DI, YoB, final_cc_cname_DI) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         launch_date = as.Date("2013-05-15"),
         registered_before_launch = if_else(as.numeric(launch_date - start_time_DI) > 0,
                                            as.numeric(launch_date - start_time_DI),
                                            0),
         registered_after_launch = if_else(as.numeric(launch_date - start_time_DI) > 0,
                                           0,
                                           -as.numeric(launch_date - start_time_DI)),
         age = 2013 - as.numeric(YoB),
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
         -final_cc_cname_DI, -LoE_DI, -YoB, -gender) %>% 
  drop_na() ->
  ph278x_neng_df

glimpse(ph278x_neng_df)

#' We fit a couple of models to see if we can predict engagement from demographics

tr_rows <- createDataPartition(ph278x_neng_df$engaged, p = 0.8, list = FALSE)

ph278x_train <- ph278x_neng_df[tr_rows, ]
ph278x_test <- ph278x_neng_df[-tr_rows, ]

ph278x_neng_logit <- train(factor(engaged) ~ .,
                           data = ph278x_train,
                           method = "glm",
                           trControl = trainControl(method = "repeatedcv",
                                                    number = 10,
                                                    repeats = 5,
                                                    allowParallel = TRUE),
                           family = binomial(link = "logit"))

summary(ph278x_neng_logit)

ph278x_neng_logit$results

#' The logit returns a 62% accuracy on initial dropout just based on the
#' demographics

ph278x_neng_rf <- train(factor(engaged) ~ .,
                        data = ph278x_train,
                        method = "rf",
                        tuneGrid = expand.grid(mtry = 1:4),
                        trControl = trainControl(method = "repeatedcv",
                                                 number = 10,
                                                 repeats = 3,
                                                 allowParallel = TRUE))

ph278x_neng_rf$results
varImp(ph278x_neng_rf)

#' Random forests also do no better than 62%