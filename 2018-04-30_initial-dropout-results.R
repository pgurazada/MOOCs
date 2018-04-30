#' ---
#' title: "Predicting initial dropout "
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon Apr 30 14:27:54 2018

library(tidyverse)
library(ggthemes)
library(gridExtra)
library(feather)

set.seed(20130810)

theme_set(theme_few() + theme(plot.title = element_text(face="bold")))

logit_kappa <- read_feather("data/course_logit_kappa.feather")
glimpse(logit_kappa)

dev.new()
ggplot(logit_kappa) +
  geom_pointrange(aes(x = course_name, y = kappa_train, ymin = min_kappa_train, ymax = max_kappa_train)) +
  labs(x = "Course",
       y = "Cohen's Kappa",
       title = "Estimated Cohen's Kappa from logistic regression on the training set",
       subtitle = "(Training set proportion was set to 0.8; Kappa was estimated from 10-fold repeated crossvalidation)") +
  coord_flip() ->
  logit_plot

ggsave("2018-04-30_logit-kappa.png", logit_plot, width = 9, height = 7)

rf_kappa <- read_feather("data/course_rf_kappa.feather")
glimpse(rf_kappa)

dev.new()
ggplot(rf_kappa) +
  geom_pointrange(aes(x = course_name, y = kappa_train, ymin = min_kappa_train, ymax = max_kappa_train)) +
  labs(x = "Course",
       y = "Cohen's Kappa",
       title = "Estimated Cohen's Kappa from random forests on the training set",
       subtitle = "(Training set proportion was set to 0.8; Kappa was estimated from 10-fold repeated crossvalidation)") +
  coord_flip() ->
  rf_plot

ggsave("2018-04-30_rf-kappa.png", rf_plot, width = 9, height = 7)

kappa_plots <- grid.arrange(logit_plot, rf_plot, nrow = 2)
ggsave("2018-04-30_kappa-plots.png", kappa_plots, width = 9, height = 9)
