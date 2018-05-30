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


test_that("Group Lasso test",{
    x <- c(3,4,5,12,3,4,12)
    gp <- as.factor(c(1,1,2,2,3,3,3))
    gpl <- prox_grplasso(x,gp,8)
    expect_equal(norm(gpl - c(0,0,5,5,5,5,5)),0)

})
