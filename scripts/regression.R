library(lme4)

df <- read.csv("../csvs/RT_finalcsv.csv")
model <- lmer(mean_logRT ~ n + n_squared + (1 | pid) + (1 | feature), data = df)
summary(model)