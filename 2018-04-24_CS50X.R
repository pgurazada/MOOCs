#' ---
#' title: "Analysis of CS50X"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Tue Apr 24 10:18:44 2018

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
  filter(course_id == "HarvardX/CS50x/2012", is.na(incomplete_flag)) %>% 
  select(userid_DI, grade, ndays_act, registered, explored, certified, viewed, 
         start_time_DI, last_event_DI, nevents, nchapters, nforum_posts, 
         LoE_DI, gender, YoB, final_cc_cname_DI) ->
  cs50x_df

glimpse(cs50x_df)

#' The CS50x course is an introductory Computer Science course that deals with a 
#' broad range of topics. The emphasis is on breadth of topics rather than depth.
#' Given the nature of the course, it attracts a large audience.

cs50x_df %>%
  gather(Variable, Value) %>% 
  group_by(Variable) %>% 
  summarize(missing_perc = floor(sum(is.na(Value)) * 100/length(Value)))

dev.new()
cs50x_df %>% 
  select(registered, viewed, explored, certified) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>% 
  ggplot() +
  geom_bar(aes(x = engaged, y = ..prop..)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey() +
  labs(x = "Engaged?",
       y = "Proportion",
       title = "Distribution of registered participants (Harvard CS50X)",
       caption = "Note: a registered participant engaged if they watched at least one video")

#' The above figure shows a 60-40 split between the notengaged-engaged
#' participants. 60% drop out before a single video is watched is not a good
#' sign

cs50x_df %>% 
  select(registered, viewed, explored, certified, gender) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0)) %>%
  drop_na() %>% 
  ggplot() +
  geom_bar(aes(x = engaged, fill = gender)) +
  scale_x_continuous(breaks = c(0.0, 1.0), labels = c("0", "1")) +
  scale_fill_grey("Gender", labels = c("Female", "Male", "Other")) +
  labs(x = "Engaged?",
       y = "Count",
       title = "Distribution of registered participants by gender (Harvard CS50X)") -> p1

cs50x_df %>% 
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
       title = "Distribution of registered participants by level of education (Harvard CS50X)") -> p2

grid.arrange(p1, p2, nrow = 2)

#' The above figure shows that the course is dominated by a male audience! Also,
#' class imbalance is not severe

mooc_df %>% 
  filter(course_id == "HarvardX/CS50x/2012", is.na(incomplete_flag)) %>%
  select(registered, viewed, explored, certified, gender, LoE_DI, YoB, final_cc_cname_DI) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         age = 2012 - as.numeric(YoB),
         gender = factor(gender),
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
  select(engaged, everything(), -registered, -viewed, -explored, -certified, -final_cc_cname_DI, -LoE_DI, -YoB) %>% 
  drop_na() ->
  cs50x_neng_df

glimpse(cs50x_neng_df)

tr_rows <- createDataPartition(cs50x_neng_df$engaged, p = 0.8, list = FALSE)

cs50x_train <- cs50x_neng_df[tr_rows, ]
cs50x_test <- cs50x_neng_df[-tr_rows, ]

cs50x_neng_logit <- train(factor(engaged) ~ .,
                          data = cs50x_train,
                          method = "glm",
                          trControl = trainControl(method = "repeatedcv",
                                                   number = 10,
                                                   repeats = 5,
                                                   allowParallel = TRUE),
                          family = binomial(link = "logit"))

summary(cs50x_neng_logit)

cs50x_neng_logit$results

#' The logit returns a 68% accuracy on initial dropout just based on the
#' demographics

cs50x_neng_rf <- train(factor(engaged) ~ .,
                       data = cs50x_train,
                       method = "rf",
                       tuneGrid = expand.grid(mtry = 1:4),
                       trControl = trainControl(method = "repeatedcv",
                                                number = 10,
                                                repeats = 3,
                                                allowParallel = TRUE),
                       ntree= 1000)

cs50x_neng_rf$results
varImp(cs50x_neng_rf)

#' Random forests also do no better than 68%