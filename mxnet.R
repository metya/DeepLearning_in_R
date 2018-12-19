# installation 
if (require(mxnet)!=TRUE) {
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU/cu92"
options(repos = cran)
install.packages("mxnet")
}





# load data and split 
df <- read_rds('data.rds')

set.seed(100)


#transform and split train on x and y
train_ind <- sample(1:77, 60)
x_train <- as.matrix(df[train_ind, 2:8])
y_train <- unlist(df[train_ind, 9])
x_val <- as.matrix(df[-train_ind, 2:8])
y_val <- unlist(df[-train_ind, 9])

iter_train_data <- mx.io.arrayiter(t(x_train), 
                                   y_train, 
                                   batch.size = 15, 
                                   shuffle = TRUE)


# define model
data <- mx.symbol.Variable("data")

fc1 <- mx.symbol.FullyConnected(data, 
                                num_hidden = 1)
linreg <- mx.symbol.LinearRegressionOutput(fc1)

# define initializer
initializer <- mx.init.normal(sd = 0.1)

# define optimizer algorythm to update weigths
optimizer <- mx.opt.create("sgd", 
                           learning.rate = 1e-6,
                           momentum = 0.9)

# define logger where we will keep updates
logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(
  period = 4, # число батчей, после которого оценивается метрика
  logger = logger)

n_epoch <- 20

# plot our model
graph.viz(linreg)

# train model with parameters
model <- mx.model.FeedForward.create(
  symbol = linreg, 
  X = x_train, 
  y = y_train,
  ctx = mx.cpu(), 
  num.round = n_epoch, 
  initializer = initializer, 
  optimizer = optimizer,
  eval.data = list(data = x_val, label = y_val),
  eval.metric = mx.metric.rmse,
  array.batch.size = 15,
  epoch.end.callback = epoch.end.callback)

# plot train loss curve and eval
rmse_log <- data.frame(RMSE = c(logger$train, logger$eval),
                       dataset = c(rep("train", 
                                       length(logger$train)), 
                                   rep("val", 
                                       length(logger$eval))),
                       epoch = 1:n_epoch)

library(ggplot2)
ggplot(rmse_log, aes(epoch, RMSE, 
                     group = dataset,
                     colour = dataset)) +
  geom_point() +
  geom_line()
