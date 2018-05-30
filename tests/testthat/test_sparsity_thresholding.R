context("Thresolding Tests")


test_that("Group Lasso test",{
    x <- c(3,4,5,12,3,4,12)
    gp <- as.factor(c(1,1,2,2,3,3,3))
    gpl <- prox_grplasso(x,gp,8)
    expect_equal(norm(gpl - c(0,0,5,5,5,5,5)),0)

})
