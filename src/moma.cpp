// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include "RcppArmadillo.h"
#include "moma.h"
using namespace Rcpp;
using namespace arma;
#define MAX(a,b) (a)>(b)?(a):(b)


class Prox{
public:
    virtual arma::vec prox(const arma::vec &x, double l)=0;
};

class Lasso: public Prox{
public:
    arma::vec prox(const arma::vec &x, double l){
        return sign(x) % arma::max(abs(x) - l, zeros(size(x)));
    }
};

class Scad: public Prox{
private:
    double gamma; // >= 2
    
public:
    Scad(double g=3.7){
        if(g<2) 
        Rcpp::stop("Gamma for MCP should be larger than 2!\n");
        gamma=g;
    }
    arma::vec prox(const arma::vec &x, double l){
        int n = x.n_elem;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgn = sign(x);
        for (int i = 0; i < n; i++) // Probably need vectorization
        {
            // the implementation follows Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties
            // Jianqing Fan and Runze Li, formula(2.8)
         
            z(i) = absx(i) > gamma * l ? x(i) : (
                                                    absx(i) > 2 * l ? (gamma - 1) * x(i) - sgn(i) * gamma * l / (gamma - 2)
                                                    : sgn(i) * MAX(double(absx(i) - l),0.0)
                                                );
        }
        return z;    
    }
};


class Mcp: public Prox{
private:
    double gamma; // >= 1
public:
    Mcp(double g=4){
        if(g<1) stop("Gamma for MCP should be larger than 1!\n");
        gamma=g;
    }
    arma::vec prox(const arma::vec &x, double l){
        int n = x.n_elem;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgn = arma::sign(x);
        for (int i = 0; i < n; i++) // Probably need vectorization
        {
            // implementation follows lecture notes of Patrick Breheny
            // http://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf
            // slide 19
            z(i) = absx(i) > gamma * l ? x(i)
                                    : sgn(i)*(gamma/(gamma-1)) * MAX(double(absx(i) - l),0.0);
            if(absx(i) <= gamma*l){
                Rcout << i << "\t";
                Rcout << double(gamma/(gamma-1)) <<"\t";
                Rcout << double(sgn(i) * MAX(absx(i) - l,0.0)) << "\t";
                Rcout << double(MAX(absx(i) - l,0.0))<< endl;
            }
           
        }
        return z;    
    }
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_lasso(arma::vec x, double l)
{
    Lasso a;
    return a.prox(x,l);
};


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_scad(arma::vec x, double l, double g=3.7)
{
    Scad a(g);
    return a.prox(x,l);
};


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec prox_mcp(arma::vec x, double l, double g=4)
{
  
    Mcp a(g);
    return a.prox(x,l);
};
