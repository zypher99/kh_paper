
################################################################################
# R Environment Container
################################################################################

# renv::init()
# renv::snapshot()


################################################################################
# R Project
################################################################################

library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cores = detectCores())
# machine learning code goes in here
# stopCluster(cl)

library(dplyr)
library(ggplot2)
library(tseries)
library(forecast)
library(tidyquant)
library(tibbletime)
library(RemixAutoML)
library(data.table)
library(magrittr)
library(scales)
library(magick)
library(grid)
library(forecast)
library(changepoint)
library(rstan)
library(prophet)

library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read interest data

#int_kor <- read.csv("/Users/eugene041105gmail.com/Documents/git/kh_paper/int_kor.csv")
int_kor <- read.csv("int_kor.csv")
int_kor$Time <- as.Date(int_kor$Time)

int_kor_train <- int_kor[1:153, ]
int_kor_test <- int_kor[153:164, ]

# AutoTS by RemixAutoML

int_kor_train_dt <- data.table(int_kor_train)

autots_int_kor = RemixAutoML::AutoTS(
  data = int_kor_train,
  TargetName = "Value",
  DateName = "Time",
  FCPeriods = 12,
  HoldOutPeriods = 12,
  TimeUnit = "Month",
  ModelFreq = FALSE
)

AutoTS_fcst_1 <- data.frame(as.data.frame(autots_int_kor$Forecast[,c(1,2)]), int_kor_test$Value)
AutoTS_fcst_1 <- data.frame(AutoTS_fcst_1, 1-(abs(AutoTS_fcst_1[,2] - AutoTS_fcst_1[,3])/AutoTS_fcst_1[,3]))
colnames(AutoTS_fcst_1) <- c("Time", "AutoTS", "Value", "1-MAPE")



# transform data into time series object in r

int_kor_ts <- ts(int_kor_train$Value, freq=12)

################################################################################
# Holt-Winters 
################################################################################

library(forecast)

int_kor_hw <- HoltWinters(int_kor_ts)
int_kor_hw_fcst <- forecast(int_kor_hw, h=12)

# 1-MAPE

hw_fcst_1 <- data.frame(int_kor_test$Time, as.numeric(int_kor_hw_fcst$mean), int_kor_test$Value)
hw_fcst_1 <- data.frame(hw_fcst_1, 1-(abs(hw_fcst_1[,2] - hw_fcst_1[,3])/hw_fcst_1[,3]))
colnames(hw_fcst_1) <- c("Time", "hw", "Value", "1-MAPE")

################################################################################
# Tbats
################################################################################

library(forecast)

int_kor_tbats <- tbats(int_kor_ts)
int_kor_tbats_fcst <- forecast(int_kor_tbats, h=12)

# 1-MAPE

tbats_fcst_1 <- data.frame(int_kor_test$Time, as.numeric(int_kor_tbats_fcst$mean), int_kor_test$Value)
tbats_fcst_1 <- data.frame(tbats_fcst_1, 1-(abs(tbats_fcst_1[,2] - tbats_fcst_1[,3])/tbats_fcst_1[,3]))
colnames(tbats_fcst_1) <- c("Time", "tbats", "Value", "1-MAPE")

################################################################################
# ARIMA 
################################################################################

library(forecast)

int_kor_arima <- auto.arima(ts(int_kor_train$Value, freq=12))
int_kor_arima_fcst <- forecast(int_kor_arima, h=12)

# 1-MAPE

ARIMA_fcst_1 <- data.frame(int_kor_test$Time, as.numeric(int_kor_arima_fcst$mean), int_kor_test$Value)
ARIMA_fcst_1 <- data.frame(ARIMA_fcst_1, 1-(abs(ARIMA_fcst_1[,2] - ARIMA_fcst_1[,3])/ARIMA_fcst_1[,3]))
colnames(ARIMA_fcst_1) <- c("Time", "ARIMA", "Value", "1-MAPE")

################################################################################
# Neural network AR 
################################################################################

library(forecast)

int_kor_nnet <- nnetar(ts(int_kor_train$Value, freq=12))
int_kor_nnet_fcst <- forecast(int_kor_nnet, h=12)

# 1-MAPE

NNET_fcst_1 <- data.frame(int_kor_test$Time, as.numeric(int_kor_nnet_fcst$mean), int_kor_test$Value)
NNET_fcst_1 <- data.frame(NNET_fcst_1, 1-(abs(NNET_fcst_1[,2] - NNET_fcst_1[,3])/NNET_fcst_1[,3]))
colnames(NNET_fcst_1) <- c("Time", "NNET", "Value", "1-MAPE")

################################################################################
# summary of results 
################################################################################

Time <- int_kor_test$Time
AutoTS_FCST <- AutoTS_fcst_1$AutoTS
HW_FCsT <- as.numeric(int_kor_hw_fcst$mean)
TBATS_FCAT <- as.numeric(int_kor_tbats_fcst$mean)
ARIMA_FCST <- as.numeric(int_kor_arima_fcst$mean)
NNET_FCST <- as.numeric(int_kor_nnet_fcst$mean)
Value <- int_kor_test$Value

FCST_result <- data.frame(Time, round(AutoTS_FCST, digits = 3), round(HW_FCsT, digits = 3), round(TBATS_FCAT, digits = 3), round(ARIMA_FCST, digits = 3), round(NNET_FCST, digits = 3), round(Value, digits = 3))

colnames(FCST_result) <- c( "Time", "AutoTS", "HW", "TBATS", "ARIMA", "Neuralnet_AR", "Value")

################################################################################
# Prophet
################################################################################

df <- int_kor[, c("Time", "Value")]
colnames(df) <- c("ds", "y")

# df <- df %>% mutate(y = log(y))
# training & test set

df_ms <- df[1:152, ]
df_ts <- df[153:165, ]

# fitting model
# Only 50% of first data considered for change point detection
# changepoint.range = 0.5

m <- prophet(df_ms, seasonality.mode = 'multiplicative', mcmc.samples = 300, changepoint.range = 0.5)
# m <- prophet(df_ms, seasonality.mode = 'multiplicative', mcmc.samples = 300, changepoint.prior.scale = 0.5)

# forecasting

future <- make_future_dataframe(m, periods = 12, freq = 'month')
forecast <- predict(m, future)

# plot change point detection

plot(m, forecast) + add_changepoints_to_plot(m)

# plot trend & seasonality

plot(m, forecast)
prophet_plot_components(m, forecast)

# forecasting result

fcst_prophet <- forecast %>% select(ds, trend, yearly, yhat) 

fcst_prophet <- cbind(fcst_prophet, as.data.frame(int_kor$Value))
colnames(fcst_prophet) <- c("Time", "Trend", "Yearly Effect", "Forecast", "Interest")

fcst_prophet <- fcst_prophet %>% mutate( Forecast = round(Forecast, digits = 3), Interest = round(Interest, digits = 3), MAPE = 1-abs(Interest-Forecast)/Interest) %>% mutate( MAPE = round(MAPE, digits = 3)) 
fcst_prophet <- fcst_prophet %>% mutate( Time = as.Date(Time))
tail(fcst_prophet, 12)

#saveRDS(m, file = "model_prophet.rds")
#saveRDS(fcst_prophet, file = "fcst_prophet.rds")
