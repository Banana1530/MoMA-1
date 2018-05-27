context("ISTA Tests")
#-------------------
# Generate data
#-------------------
n <- 3 # set n != p to test bugs
p <- 5
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
show_vec <- function(v,n1,n2){
    image(matrix(v,nrow=n1,ncol=n2,byrow=FALSE),
          col = grey(seq(0, 1, length = 256)))
}
n1 <- 19
n2 <- 20
p <- n1*n2
n <- 150
a <- matrix(1,ncol=n2,nrow = n1)
for(i in 1:n1/2){
    for(j in 1:n2/2)
        a[j,i] = 30
}
image(a)
v = as.vector(a)
v = v/sqrt(sum(v^2))
show_vec(v,n1,n2)
u = runif(n)
u = u/sqrt(sum(u^2))
eps <- matrix(rnorm(n*p),n,p)/20
X <- u %*% t(v) + eps
norm(X)/norm(eps)
image(X)
res <- svd(X)
show_vec(res$v[,1],n1,n2)
res1 <- sfpca(X=X,
              "GRPLASSO","GRPLASSO",
              solver='ISTA')
