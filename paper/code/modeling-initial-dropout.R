#' ---
#' title: Assemble the data set
#' author: Pavan Gurazada
#' ---

#' last update:  Wed Oct  3 06:41:16 2018 

library(tidyverse)
library(here)

DATA_LOC <- here('paper', 'data', 'full-mooc-data.csv')

COURSE_START_DATES <- data.frame(course_id = c('HarvardX/CS50x/2012',
                                               'HarvardX/CB22x/2013_Spring',
                                               'HarvardX/ER22x/2013_Spring',
                                               'HarvardX/PH207x/2012_Fall',
                                               'HarvardX/PH278x/2013_Spring',
                                               'MITx/14.73x/2013_Spring',
                                               'MITx/2.01x/2013_Spring',
                                               'MITx/6.002x/2012_Fall',
                                               'MITx/6.002x/2013_Spring',
                                               'MITx/3.091x/2012_Fall',
                                               'MITx/3.091x/2013_Spring',
                                               'MITx/6.00x/2012_Fall',
                                               'MITx/6.00x/2013_Spring',
                                               'MITx/7.00x/2013_Spring',
                                               'MITx/8.02x/2013_Spring',
                                               'MITx/8.MReV/2013_Summer'),
                                 course_start_date = c(as.Date('2012-10-15'),
                                                       as.Date('2013-03-13'),
                                                       as.Date('2013-03-02'),
                                                       as.Date('2012-10-15'),
                                                       as.Date('2013-05-15'),
                                                       as.Date('2013-02-12'),
                                                       as.Date('2013-04-15'),
                                                       as.Date('2012-09-05'),
                                                       as.Date('2013-03-03'),
                                                       as.Date('2012-10-09'),
                                                       as.Date('2013-02-05'),
                                                       as.Date('2012-09-06'),
                                                       as.Date('2013-02-04'),
                                                       as.Date('2013-03-05'),
                                                       as.Date('2013-02-18'),
                                                       as.Date('2013-06-01')))

mooc_df <- read_csv(DATA_LOC)

mooc_df %>% 
    
    count(userid_DI) %>% 
    
    filter(n < 2) %>% 
    
    pull(userid_DI) ->
    
    single_course_participants_vec

mooc_df %>% 
    
    filter(userid_DI %in% single_course_participants_vec) %>% 
    
    left_join(COURSE_START_DATES, by = 'course_id') %>% 
    
    mutate(engaged = ifelse(viewed == 1 | explored == 1 | certified == 1, 1, 0),
           male = case_when(gender == "m" ~ 1,
                            gender == "f" ~ 0),
           age = 2013 - as.numeric(YoB),
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
                                 LoE_DI == "Doctorate" ~ "DO"),
           joined_early_or_late = course_start_date - start_time_DI) %>% # Negative values of this variable indicate that the participant joined after the course began
    
    select(engaged, male, age, country, education, course_id, joined_early_or_late) %>% 
    
    drop_na() ->
    
    processed_mooc_df
    
write_feather(processed_mooc_df, here('paper', 'data', 'processed-mooc-data.feather'))
