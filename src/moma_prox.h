// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "moma.h"

#define MAX(a,b) (a)>(b)?(a):(b)
#define THRES_P(x,l) (MAX(x-l,0.0)) // shrink a positive value by `l`

/////////////////
// Section 1: Prox operators
/////////////////

// soft-thresholding a non-negative vector
inline arma::vec soft_thres_p(const arma::vec &x, double l){
    return arma::max(x - l, arma::zeros<arma::vec>(x.n_elem));
}

class Prox{
public:
    Prox(){
        MoMALogger::debug("A Prox!\n");
    }
    virtual arma::vec prox(const arma::vec &x, double l){
        return x;   // to be tested, return a reference might cause extra copying.
    };
    virtual ~Prox() {
        MoMALogger::debug("Releasing Prox\n");
    };
};

class Lasso: public Prox{
public:
    Lasso(){
        MoMALogger::debug("A Lasso prox\n");
    }
    arma::vec prox(const arma::vec &x, double l){
        arma::vec absx = arma::abs(x);
        return arma::sign(x) % soft_thres_p(absx,l);
    }
    ~Lasso(){
        MoMALogger::debug("Releasing Lasso\n");
    }
};

class NNLasso: public Prox{
public:
    NNLasso(){
        MoMALogger::debug("A Non-negative Lasso prox\n");
    }
    arma::vec prox(const arma::vec &x, double l){
        return soft_thres_p(x,l);
    }
    ~NNLasso(){
        MoMALogger::debug("Releasing NNLasso\n");
    }
};


class Scad: public Prox{
private:
    double gamma; // gamma_SCAD >= 2
public:
    Scad(double g=3.7){
        MoMALogger::debug("A Scad prox\n");
        if(g<2) 
            MoMALogger::error("Gamma for SCAD should be larger than 2!\n");
        gamma=g;
    }
     ~Scad(){
        MoMALogger::debug("Releasing Scad\n");
    }

    arma::vec prox(const arma::vec &x, double l){
        int n = x.n_elem;
        double gl = gamma*l;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgnx = arma::sign(x);
        for (int i = 0; i < n; i++) // Probably need vectorization
        {
            // The implementation follows 
            // Variable Selection via Nonconcave Penalized Likelihood and its Oracle Properties
            // Jianqing Fan nd Runze Li
            // formula(2.8).
            z(i) = absx(i) > gamma * l ? absx(i) : (absx(i) > 2 * l ? //(gamma-1)/(gamma-2) * THRES_P(absx(i),gamma*l/(gamma-1)) 
                                                    ((gamma - 1) * absx(i) - gl)/ (gamma - 2)
                                                    : THRES_P(absx(i),l)
                                                    );
        }
        return z%sgnx;    
    }

    arma::vec vec_prox(const arma::vec &x, double l){
        int n = x.n_elem;
        double gl = gamma*l;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgnx = arma::sign(x);
        
        arma::umat D(x.n_elem,3,arma::fill::zeros);    
        // If we use sp_mat, it becomes slower; 
        // it also errors if we use sp_umat; 
        // it gives wrong result if umat is used. // need to initialize!!!!

        // MoMALogger::debug("D is constructed as\n") << mat(D);
        // arma::vec x0 = arma::max(absx-l,arma::zeros<arma::vec>(n));
        // MoMALogger::debug("Pass x0\n") << x0;
        // arma::vec x1 = ((gamma-1)*absx - gl)/(gamma-2);
        // MoMALogger::debug("Pass x1\n") << x1;
        // Rcpp::Rcout << D.col(0);

        for(int i = 0; i < n; i++){
            arma::uword flag = absx(i) > gl ? 2 : (absx(i) > 2 * l ? 1: 0);   
            D(i,flag) = 1;
        }

        z = D.col(0) % soft_thres_p(absx,l) + D.col(1) % ((gamma-1)*absx - gl)/(gamma-2) + D.col(2) % absx;    
        return sgnx%z;
    }
    
};

class Mcp: public Prox{
private:
    double gamma; // gamma_MCP >= 1
public:
    Mcp(double g=4){
        MoMALogger::debug("A MC+ prox\n");
        if(g<1) 
            MoMALogger::error("Gamma for MCP should be larger than 1!\n");
        gamma=g;
    }
    ~Mcp(){
        MoMALogger::debug("Releasing Mcp\n");
    }
    arma::vec prox(const arma::vec &x, double l){
        int n = x.n_elem;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgnx = arma::sign(x);
        for (int i = 0; i < n; i++) 
        {
            // implementation follows lecture notes of Patrick Breheny
            // http://myweb.uiowa.edu/pbreheny/7600/s16/notes/2-29.pdf
            // slide 19
            z(i) = absx(i) > gamma * l ? absx(i) : (gamma / (gamma - 1)) * THRES_P(absx(i),l);         
        }
        return z%sgnx;    
    }
    arma::vec vec_prox(const arma::vec &x, double l){
        int n = x.n_elem;
        double gl = gamma*l;
        arma::vec z(n);
        arma::vec absx = arma::abs(x);
        arma::vec sgnx = arma::sign(x);
        
        arma::umat D(x.n_elem,2,arma::fill::zeros);    

        // MoMALogger::debug("D is constructed as\n") << mat(D);
        // arma::vec x0 = arma::max(absx-l,arma::zeros<arma::vec>(n));
        // MoMALogger::debug("Pass x0\n") << x0;
        // arma::vec x1 = ((gamma-1)*absx - gl)/(gamma-2);
        // MoMALogger::debug("Pass x1\n") << x1;
        // Rcpp::Rcout << D.col(0);

        for(int i = 0; i < n; i++){
            arma::uword flag = 0;
            if(absx(i) <= gl)
                flag = 1;   
            D(i,flag) = 1;
        }
        z = (gamma / (gamma - 1)) * D.col(1) % soft_thres_p(absx,l) + D.col(0) % absx;
        return sgnx%z;
    }
};

class GrpLasso: public Prox{
private:
    arma::umat D;  // Probably not using sparse matrix would be faster, TODO
                    // a boolean matrix, D \in R^{g \times p}, g is the number of groups, p the number of parameters.
                    // D_ji = 1 means \beta_i in group j.
                    // should be integer, probably use arma::sp_umat; it will cause error though, when it multipies a vec
public:
    GrpLasso(const arma::vec &grp){   // takes in a factor
        MoMALogger::debug("A Group Lasso prox\n");
        arma::uword num_grp = grp.max();
        D = arma::zeros<arma::umat>(num_grp,grp.n_elem);  // density will be 1/p = 1/x.n_elem
        for(int i = 0; i < grp.n_elem; i++){
            arma::uword g = grp(i) - 1; // the i-th parameter is in g-th group. Note factor in R starts from 1
            D(g,i) = 1;
        }
    }
    ~GrpLasso(){
        MoMALogger::debug("Releasing GrpLasso\n");
    }
    arma::vec prox(const arma::vec &x, double l){
       // MoMALogger::debug("D is initialized as ") << D;
        arma::vec grp_norms = D.t() * arma::sqrt(D * arma::square(x));    // to_be_thres is of dimension p.
    //  MoMALogger::debug("lambda is ") << l;
       // MoMALogger::debug("norm for each group is\n") << to_be_thres;
        return (x / grp_norms) % soft_thres_p(grp_norms,l);
    }
       
};
