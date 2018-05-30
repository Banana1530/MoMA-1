# include "moma_prox.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_lasso(const arma::vec &x, double l)
{
    Lasso a;
    return a.prox(x,l);
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_scad(const arma::vec &x, double l, double g=3.7)
{
    Scad a(g);
    return a.prox(x,l);
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_scadvec(const arma::vec &x, double l, double g=3.7)
{

    Scad a(g);
    return a.vec_prox(x,l);
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_mcp(const arma::vec &x, double l, double g=4)
{
    Mcp a(g);
    return a.prox(x,l);
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_mcpvec(const arma::vec &x, double l, double g=4)
{
    Mcp a(g);
    return a.vec_prox(x,l);
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_nnlasso(const arma::vec &x, double l)
{
    NNLasso a;
    return a.prox(x,l);
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_grplasso(const arma::vec &x, const arma::vec &g,double l)
{
    GrpLasso a(g);
    
    return a.prox(x,l);

};


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
void test(){
    arma::uvec x(100,1);
    arma::vec y(100,1);
    Rcpp::Rcout << x%y;
}