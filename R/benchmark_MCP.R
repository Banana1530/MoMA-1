library(MoMA)
devtools::load_all("MoMA")
# benchmark vectorized version of proximal operators
scad <- function(p){
    x <- 4*runif(p)
    prox_mcp(x,1,3)
}
scad.vec <- function(p){
    x <- 4*runif(p)
    prox_mcpvec(x,1,3)
}
library(rbenchmark)
bm.range <- function(st,end,num,rep){
    rep <- 100000
    st <- 12
    end <- 17
    num <-20
    pset = exp(seq(st,end,(end-st)/num))  # a set of `p`s
    np = length(pset)
    store <- matrix(nrow = 2, ncol = np)
    for(i in 1:np){
        print(paste("Dimenstion is ",i))
        res <- benchmark(scad(pset[i]),scad.vec(pset[i]),
                         replications=rep,order="test")
        store[,i] = res$elapsed
        print(res)
    }

    plot(pset,log(store[1,]),type='l',
         main="SCAD benchmark (black=vectorized)",
         xlab="dimension of the vec",
         ylab="time")
    lines(pset,log(store[2,]),col="red")
}

# I find that vectorization using sparse matrix is slow
# It also causes error is sp_umat is used. In conclusion, in Armadillo umat cannot multiply with vec<double>
# For small matrix(<10000), vecotrization with mat<double> is comparable
# but simple `if-else` is faster for large matrix.

# Not fully vectorized,
