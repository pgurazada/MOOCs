#' ---
#' title: "Analysis of MITx/2.01x/2013_Spring"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Thu Apr 26 06:29:28 2018

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
  filter(course_id == "MITx/2.01x/2013_Spring") %>% 
  select(userid_DI, grade, ndays_act, registered, explored, certified, viewed, 
         start_time_DI, last_event_DI, nevents, nchapters, nforum_posts, 
         LoE_DI, gender, YoB, final_cc_cname_DI) ->
  mit201x_df

glimpse(mit201x_df)

#' MIT 2.01x is a first course on solid mechanics, composed of several key
#' topics on structural balance. This course equips students with the tools
#' needed to ensure that their structures perform the specified mechanical
#' function without failing

mit201x_df %>%
  gather(Variable, Value) %>% 
  group_by(Variable) %>% 
  summarize(missing_perc = floor(sum(is.na(Value)) * 100/length(Value)))

dev.new()
mit201x_df %>% 
  select(registered, viewed, explored, certified) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>% 
  ggplot() +
  geom_bar(aes(x = engaged, y = ..prop..)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey() +
  labs(x = "Engaged?",
       y = "Proportion",
       title = "Distribution of registered participants (MIT 2.01x)",
       caption = "Note: a registered participant engaged if they watched at least one video")

#' The above figure shows that unlike several other courses, the number of people
#' dropping out before the course is significantly lesser 

mit201x_df %>% 
  select(registered, viewed, explored, certified, gender) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>%
  drop_na() %>% 
  ggplot() +
  geom_bar(aes(x = engaged, fill = gender)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey("Gender", labels = c("Female", "Male")) +
  labs(x = "Engaged?",
       y = "Count",
       title = "Distribution of registered participants by gender (MIT 2.01x)") -> p1

mit201x_df %>% 
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
       title = "Distribution of registered participants by level of education (MIT 2.01x)") -> p2

grid.arrange(p1, p2, nrow = 2)

#' Dominated by male

mooc_df %>% 
  filter(course_id == "MITx/2.01x/2013_Spring") %>%
  select(registered, viewed, explored, certified, 
         gender, LoE_DI, YoB, final_cc_cname_DI, start_time_DI) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         launch_date = as.Date("2013-04-15"),
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
         -final_cc_cname_DI, -LoE_DI, -YoB, -gender, -start_time_DI, -launch_date) %>% 
  drop_na() ->
  mit201x_neng_df

glimpse(mit201x_neng_df)

#' We fit a couple of models to see if we can predict engagement from
#' demographics and registration times

tr_rows <- createDataPartition(mit201x_neng_df$engaged, p = 0.8, list = FALSE)

mit201x_train <- mit201x_neng_df[tr_rows, ]
mit201x_test <- mit201x_neng_df[-tr_rows, ]

mit201x_neng_logit <- train(factor(engaged) ~ .,
                             data = mit201x_train,
                             method = "glm",
                             metric = "Kappa",
                             trControl = trainControl(method = "repeatedcv",
                                                      number = 10,
                                                      repeats = 5,
                                                      allowParallel = TRUE),
                             family = binomial(link = "logit"))

summary(mit201x_neng_logit)

mit201x_neng_logit$results

confusionMatrix(mit201x_neng_logit)

#' The logit returns a 67% accuracy on initial dropout

mit201x_neng_rf <- train(factor(engaged) ~ .,
                          data = mit201x_train,
                          method = "rf",
                          tuneGrid = expand.grid(mtry = 1:4),
                          trControl = trainControl(method = "repeatedcv",
                                                   number = 10,
                                                   repeats = 3,
                                                   allowParallel = TRUE))

mit201x_neng_rf$results
varImp(mit201x_neng_rf)

confusionMatrix(mit201x_neng_logit)
#' Random forests also do no better than 67%
