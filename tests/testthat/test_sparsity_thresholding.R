context("Thresolding Tests")


test_that("Group Lasso test",{
    x <- c(-3,-5,4,12,3,4,12)
    gp <- as.factor(c(1,2,1,2,3,3,3))
    gpl <- prox_grplasso(x,gp,0)
    gpl
    expect_equal(norm(gpl - c(-3,-5,4,12,3,4,12)),0)

})
