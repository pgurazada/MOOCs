#' ---
#' title: "Plot initial drop-out per course"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon Apr 30 05:07:38 2018

library(caret)
library(tidyverse)
library(ggthemes)

set.seed(20130810)

theme_set(theme_few() + theme(plot.title = element_text(face="bold")))

mooc_df <- read_csv("data/HMXPC13_DI_v2_5-14-14.csv", progress = TRUE)
glimpse(mooc_df)

mooc_df %>% 
  mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
         status = ifelse(engaged == 1, "Engaged", "Not Engaged")) %>% 
  group_by(course_id, status) %>% 
  summarize(count = n()) ->
  initial_dropout

dev.new()
ggplot(initial_dropout) +
  geom_bar(aes(x = course_id, y = count, fill = status), stat = "identity", position = position_dodge()) +
  labs(x = "Course",
       y = "Number of students",
       title = "Activity status across all courses") +
  scale_fill_grey("Course Status") +
  coord_flip() -> p1

ggsave("2018-04-30_activity-status.png", p1, width = 9, height = 7)
