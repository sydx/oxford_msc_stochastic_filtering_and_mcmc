library(ks)

# Make sure you've set the working directory to that containing this script
setwd('C:/Users/Paul/Documents/dev/alexandria/bilokon-msc/dissertation/code/winbugs')

data_dir <- 'svl2/results-for-dataset-2'
coda_index <- read.table(file.path(data_dir, 'coda-index.txt'),
    header=FALSE, col.names=c('param', 'firstindex', 'lastindex'))
coda_index <- coda_index[!coda_index$param=="beta",]

coda_data <- read.table(file.path(data_dir, 'coda-chain-1.txt'),
    header=FALSE, col.names=c('index', 'value'))

data_matrix <- matrix(NA, nrow=nrow(coda_index),
    ncol=coda_index$lastindex[1] - coda_index$firstindex[1] + 1)

for (i in 1:nrow(coda_index)) {
    data_matrix[i,] <-
        coda_data$value[coda_index$firstindex[i]:coda_index$lastindex[i]]
}

# Produce the kernel density plot of the parameter. Useful for comparing with
# plots produced by OpenBUGS and setting xmin and xmax below.
param_index <- 3
x <- data_matrix[param_index,]
univariate_density_estimate <- kde(x=x)
plot(univariate_density_estimate, col=3)
title(coda_index$param[param_index])

x <- t(data_matrix)
# mu, phi, rho, sigmav
xmin <- c(-2.0, 0.92, -0.7, 0.05)
xmax <- c(1.0, 1.0, 0.3, 0.3)
bgridsize <- c(50, 50, 50, 50)
multivariate_density_estimate <- kde(x=x, binned=TRUE, compute.cont=TRUE,
    xmin=xmin, xmax=xmax, bgridsize=bgridsize)

# Output the multivariate density estimate computed at a given point:
pt <- c(-0.2076, 0.9745, -0.275, 0.1492)
predict(multivariate_density_estimate, x=pt)

# data_dir <- 'svl/results-for-dataset-1'
# pt <- c(-0.5486, 0.9861, -0.1969, 0.149)
# 8902.443

# data_dir <- 'svl/results-for-dataset-2'
# pt <- c(-0.1706, 0.9755, -0.2699, 0.1464)
# 348708.5

# data_dir <- 'svl2/results-for-dataset-1'
# pt <- c(-0.5883, 0.9853, -0.1472, 0.1456)
# 9302.415

# data_dir <- 'svl2/results-for-dataset-2'
# pt <- c(-0.2076, 0.9745, -0.275, 0.1492)
# 335933.4
