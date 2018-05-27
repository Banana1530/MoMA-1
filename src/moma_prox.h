// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "moma.h"
using namespace Rcpp;
using namespace arma;
using namespace std;
#define MAX(a,b) (a)>(b)?(a):(b)
#define THRES_P(x,l) (MAX(x-l,0.0)) // shrink a positive value by `l`

/////////////////
// Section 1: Prox operators
/////////////////
inline arma::vec soft_thres(const arma::vec &x, double l){
    return sign(x) % arma::max(abs(x) - l, zeros<vec>(x.n_elem));
}

class Prox{
public:
    virtual arma::vec prox(const arma::vec &x, double l)=0;
    virtual ~Prox() = default;
};

class Lasso: public Prox{
public:
    Lasso(){
        MoMALogger::debug("A Lasso prox\n");
    }
    arma::vec prox(const arma::vec &x, double l){
        return soft_thres(x,l);
    }
};

class NNLasso: public Prox{
public:
    NNLasso(){
        MoMALogger::debug("A Non-negative Lasso prox\n");
    }
    arma::vec prox(const arma::vec &x, double l){
        return arma::max(x - l, zeros<vec>(x.n_elem));
    }
};

class Scad_vec: public Prox{
private:
    double gamma; // gamma_SCAD >= 2
public:
    Scad_vec(double g=3.7){
        MoMALogger::debug("A vectorized Scad prox\n");
        if(g<2) 
        Rcpp::stop("Gamma for MCP should be larger than 2!\n");
        gamma=g;
    }
    arma::vec prox(const arma::vec &x, double l){
        int n = x.n_elem;
        double gl = gamma*l;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgnx = sign(x);
        
        arma::umat D(x.n_elem,3);    // if we use sp_mat, it becomes slower; it also errors if we use sp_umat
        // MoMALogger::debug("D is constructed as\n") << mat(D);
        // arma::vec x0 = arma::max(absx-l,zeros<vec>(n));
        // MoMALogger::debug("Pass x0\n") << x0;
        // arma::vec x1 = ((gamma-1)*absx - gl)/(gamma-2);
        // MoMALogger::debug("Pass x1\n") << x1;
        // Rcpp::Rcout << D.col(0);
        for(int i = 0; i < n; i++){
            uword flag = absx(i) > gl ? 2 : (absx(i) > 2 * l ? 1: 0);   
            D(i,flag) = 1;
        }
        z = D.col(0) % arma::max(absx-l,arma::zeros<vec>(n)) + D.col(1) % ((gamma-1)*absx - gl)/(gamma-2) + D.col(2)%absx;    
        return sgnx%z;
    }
};


class Scad: public Prox{
private:
    double gamma; // gamma_SCAD >= 2
public:
    Scad(double g=3.7){
        MoMALogger::debug("A Scad prox\n");
        if(g<2) 
        Rcpp::stop("Gamma for MCP should be larger than 2!\n");
        gamma=g;
    }
    arma::vec prox(const arma::vec &x, double l){
        int n = x.n_elem;
        double gl = gamma*l;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgnx = sign(x);
        for (int i = 0; i < n; i++) // Probably need vectorization
        {
            // the implementation follows Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties
            // Jianqing Fan and Runze Li, formula(2.8)
            z(i) = absx(i) > gamma * l ? absx(i) : (absx(i) > 2 * l ? //(gamma-1)/(gamma-2) * THRES_P(absx(i),gamma*l/(gamma-1)) 
                                                    ((gamma - 1) * absx(i) - gl)/ (gamma - 2)
                                                    : THRES_P(absx(i),l)
                                                    );
        }
        return z%sgnx;    
    }

    
};

class Mcp: public Prox{
private:
    double gamma; // gamma_MCP >= 1
public:
    Mcp(double g=4){
        MoMALogger::debug("A MC+ prox\n");

        if(g<1) Rcpp::stop("Gamma for MCP should be larger than 1!\n");
        gamma=g;
    }
    arma::vec prox(const arma::vec &x, double l){
        int n = x.n_elem;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgnx = arma::sign(x);

        //// Try vectorization
        // arma::vec thr = sgnx % arma::max(absx - l, zeros(size(x)));
        // arma::vec flag = ones<vec>(n) * gamma*l;
        // arma::vec large = x>flag;
        // arma::vec small = ones(gamma*l)-large;
        for (int i = 0; i < n; i++) // Probably need vectorization
        {
            // implementation follows lecture notes of Patrick Breheny
            // http://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf
            // slide 19
            z(i) = absx(i) > gamma * l ? absx(i)
                                    : (gamma/(gamma-1)) * THRES_P(absx(i),l);         
        }
        return z%sgnx;    
    }
};

class GrpLasso: public Prox{
private:
    arma::sp_mat D;  // Probably not using sparsity would be faster
                    // a boolean matrix, D \in R^{g \times p}, g is the number of groups, 
                    // D_ji = 1 means \beta_i in group j.
                    // should be integer, probably use arma::sp_umat; it will cause error though, when it multipies a vec
public:
    GrpLasso(const arma::vec &x){   // takes in a factor
        D = sp_mat(int(x.max()),x.n_elem);  // density will be 1/p = 1/x.n_elem
        for(int i = 0; i < x.n_elem; i++){
            uword g = x(i) - 1; // the i-th parameter is in g-th group. Note factor in R starts from 1
            D(g,i) = 1;
        }
    }

    arma::vec prox(const arma::vec &x, double l){

        MoMALogger::debug("D is initialized as ") << D;
        arma::vec to_be_thres = D.t() * arma::sqrt(D * arma::square(x));
        MoMALogger::debug("norm for each group is\n") << to_be_thres;
        return sign(x) % arma::max(to_be_thres - l,zeros<vec>(x.n_elem));
    }
    
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

    Scad_vec a(g);
    return a.prox(x,l);
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
arma::vec prox_nnlasso(const arma::vec &x, double l)
{
    NNLasso a;
    return a.prox(x,l);
};

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
void test()
{
    arma::umat a(5,5);
    arma::vec x(5);
 
    return;
};