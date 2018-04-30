#' ---
#' title: "Feature plots for initial drop-out"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon Apr 30 06:10:11 2018

library(caret)
library(tidyverse)
library(ggthemes)
library(gridExtra)

set.seed(20130810)

theme_set(theme_few() + theme(plot.title = element_text(face="bold")))

mooc_df <- read_csv("data/HMXPC13_DI_v2_5-14-14.csv", progress = TRUE)
glimpse(mooc_df)

mooc_df %>% 
  select(course_id, registered, viewed, explored, certified, 
         gender, LoE_DI, YoB, final_cc_cname_DI, start_time_DI) %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         age = 2012 - as.numeric(YoB),
         gender = case_when(gender == "m" ~ 1,
                            gender == "f" ~ 0),
         country = case_when(final_cc_cname_DI == "United States" ~ "USA",
                             final_cc_cname_DI %in% c("India", "Pakistan", 
                                                      "Bangladesh", "China",
                                                      "Indonesia", "Japan", 
                                                      "Other East Asia", "Other Middle East/Central Asia",
                                                      "Other South Asia", "Philippines",
                                                      "Egypt") ~ "Asia",
                             final_cc_cname_DI %in% c("France", "Germany", 
                                                      "Greece", "Other Europe",
                                                      "Poland", "Portugal", 
                                                      "Russian Federation", "Spain",
                                                      "Ukraine", "United Kingdom") ~ "Europe",
                             final_cc_cname_DI %in% c("Morocco", "Nigeria", 
                                                      "Other Africa") ~ "Africa",
                             TRUE ~ "Other"),
         education = LoE_DI) %>% 
  select(course_id, engaged, everything(), -registered, -viewed, -explored, -certified, 
         -final_cc_cname_DI, -LoE_DI, -YoB, -start_time_DI) %>% 
  drop_na() ->
  features_df

glimpse(features_df)

dev.new()
ggplot(features_df %>% filter(grepl("Harvard", course_id)) %>% filter(age > 10)) +
  geom_boxplot(aes(x = factor(engaged), y = age)) + 
  scale_x_discrete(labels = c("Not Engaged", "Engaged")) +
  labs(x = "Student status",
       y = "Age",
       title = "Distribution of outcome by age (Harvard courses)") +
  coord_flip() -> harvard_age

ggplot(features_df %>% filter(grepl("MIT", course_id)) %>% filter(age > 10)) +
  geom_boxplot(aes(x = factor(engaged), y = age)) + 
  scale_x_discrete(labels = c("Not Engaged", "Engaged")) +
  labs(x = "Student status",
       y = "Age",
       title = "Distribution of outcome by age (MIT courses)") +
  coord_flip() -> mit_age

age_plot <- grid.arrange(harvard_age, mit_age, nrow = 2)
ggsave("2018-04-30_activity-age-distribution.png", age_plot, width = 9, height = 7)

dev.new()
ggplot(features_df %>% filter(grepl("Harvard", course_id))) +
  geom_bar(aes(x = factor(engaged), fill = education), width = 0.5) + 
  scale_fill_grey("Education") +
  scale_x_discrete("Student Status", labels = c("Not Engaged", "Engaged")) +
  labs(y = "Count",
       title = "Distribution of outcome by education (Harvard courses)") +
  coord_flip() -> 
  harvard_education

ggplot(features_df %>% filter(grepl("MIT", course_id))) +
  geom_bar(aes(x = factor(engaged), fill = education), width = 0.5) + 
  scale_fill_grey("Education") +
  scale_x_discrete("Student Status", labels = c("Not Engaged", "Engaged")) +
  labs(y = "Count",
       title = "Distribution of outcome by education (MIT courses)") +
  coord_flip() -> 
  mit_education

education_plot <- grid.arrange(harvard_education, mit_education, nrow = 2)
ggsave("2018-04-30_activity-education-distribution.png", education_plot, width = 9, height = 7)
