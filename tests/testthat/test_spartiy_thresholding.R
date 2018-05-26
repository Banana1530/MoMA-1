context("Thresolding Tests")

test_that("Logging controls work", {
    expect_error(moma_logger_level("BAD LEVEL"))

    moma_logger_level("INFO")
    expect_equal("INFO", moma_logger_level())
    test.points = c(0,0.5, 1, 1.2, 1.5, 2.5,3,3.5)

    # # plot the thresholding function
    x <- seq(-4,4,0.5)

    plot(x,x,type="l")
    lines(x,prox_scad(x,1,3),type="l")
    lines(x,prox_scadvec(x,1,3),type="l")
    lines(x,prox_mcp(x,1,3),type="l")
    lines(x,prox_lasso(x,1),type="l",col=1)
    lines(x,prox_nnlasso(x,1),type="l",col="red")

    moma_logger_level("MESSAGE")
})

# benchmark vectorized version of proximal operators
novec <- function(p){
    x <- 4*runif(p)
    prox_scad(x,1,3)
}
vec <- function(p){
    x <- 4*runif(p)
    prox_scadvec(x,1,3)
}
library(rbenchmark)
rep <- 200
pset = exp(seq(15,17,0.1))
np = length(pset)
store <- matrix(nrow = 2, ncol = np)
for(i in 1:np){
    print(i)
    res <- benchmark(vec(pset[i]),novec(pset[i]), replications=rep,order="test")
    store[,i] = res$elapsed
}
plot(log(store[,1]),type='l')
lines(log(store[,2]),col="red")
