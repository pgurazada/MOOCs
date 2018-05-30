
library(tidyverse)

theme_set(theme_bw())

options(repr.plot.width=9, repr.plot.height=5)

course_metrics = read_csv('course_metrics.csv')

course_metrics %>% arrange(desc(F1_test)) %>% select(course_name, everything(), -X1, -index) -> course_metrics

course_metrics

ggplot(course_metrics) +
geom_pointrange(aes(x = course_name, y = F1_test, ymin = F1_train_min, ymax = F1_train_max)) +
labs(x = 'Course',
     y = 'F1 score',
     title= 'Estimated F1 score from Random Forests on the test set',
     subtitle = '(error bars indicate variance among training scores)') + 
coord_flip() -> f1_plot

ggsave('2018-05-30_f1-score-plot.png', width = 10, height = 7)
