#' ---
#' title: "Feature plots for initial drop-out"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon Apr 30 06:10:11 2018

library(caret)
library(tidyverse)
library(ggthemes)

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
ggplot(features_df %>% filter(grepl("Harvard", course_id))) +
  geom_point(aes(x = gender, y = engaged, color = course_id), position = position_jitter()) +
  scale_color_discrete("Course") +
  scale_x_continuous("Gender", 
                     breaks = c(0, 1), 
                     labels = c("Female", "Male"))

