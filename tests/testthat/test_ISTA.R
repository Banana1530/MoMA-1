context("ISTA Tests")
#-------------------
# Generate data
#-------------------
n <- 200 # set n != p to test bugs
p <- 199
O_v <- diag(p)
O_u <- diag(n)
set.seed(32)
X = matrix(runif(n*p),n)


#-------------------
# test_that
#-------------------

test_that("Equal to SVD when no penalty", {

    sfpca <- sfpca(X,
                   O_u,O_v, 0,0,
                   lambda_u=0,lambda_v=0,"LASSO","LASSO",
                   gamma=3.7,EPS=1e-9,MAX_ITER = 1e+5)
    svdd <- svd(X)
    expect_equal(norm(svdd$v[,1]-sfpca$v),0)
    expect_equal(norm(svdd$u[,1]-sfpca$u),0)
    expect_equal(svdd$d[1],sfpca$d);
    expect_error(moma_logger_level("BAD LEVEL"))
})


test_that("Closed form solution when no sparsity",{
    # TODO
})

# simple test for group lasso
x <- c(3,4,5,12,3,4,12)
gp <- as.factor(c(1,1,2,2,3,3,3))
prox_grplasso(x,gp,8)
test()


# group PCA

# util function
show_vec <- function(v,n1,n2){
    image(matrix(v,nrow=n1,ncol=n2,byrow=FALSE),
          col=grey(seq(0, 1, length = 256)))
}
# generate data
n1 <- 20
n2 <- 19
p <- n1*n2
n <- 150
a <- matrix(30,ncol=n2,nrow = n1)
for(i in 1:n1){
    for(j in 1:n2/2){
        a[i,j] = 1

    }
}
image(a)
v = as.vector(a)
v = v/sqrt(sum(v^2))
plot(v,type="l")

group <- matrix(1,ncol=n2,nrow = n1)
for(i in seq(1,n1,4)){
    if(i+4<n2)
    group[,i:(i+4)] <- i
}
image(group)
group = as.vector(group)
group = factor(group)
# another side
u = runif(n)
u = u/sqrt(sum(u^2))
# noise and X
eps <- matrix(rnorm(n*p),n,p)/100
X <- u %*% t(v) + eps
norm(X)/norm(eps)

# svd as benchmark
res <- svd(X)
plot(-res$v[,1],type="l")
show_vec(res$v[,1],n1,n2)

# group lasso
O_v=matrix(0,p,p)
O_u=matrix(0,n,n)
spset=seq(0,0.8,0.05)
for(p in spset){
    res1 <- sfpca(X=X,
                  Omega_u=O_u,Omega_v=O_v,
                  group_v = group,
                  lambda_v=p,
                  P_v="GRPLASSO",
                  solver='FISTA')
    par(mfrow=c(1,2))
    plot(-res1$v[,1],type="l",main=paste(p),ylim=c(-.06,.09))
    show_vec(res1$v[,1],n1,n2)

}




