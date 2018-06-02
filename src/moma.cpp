// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "moma.h"
#include "moma_prox.h"
#include <algorithm>

enum class Solver{
    APP_ISTA,
    ISTA,
    FISTA
};



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double mat_norm(const arma::vec &u, const arma::mat &S_u)   // TODO: special case when S_u = I, i.e., alpha_u = 0.
{
    return sqrt(as_scalar(u.t() * S_u * u));
}


/////////////////
// Section 2: MoMA class
/////////////////

class MoMA{

private:
    /* matrix size */
    int n;
    int p;
    double alpha_u;
    double alpha_v;
    double prox_u_step; 
    double prox_v_step;
    double grad_u_step;
    double grad_v_step;
    
    Solver solver_type;
    long MAX_ITER;
    double EPS;

    const arma::mat &X; 
    //  careful about reference, if it refenrences 
    //  something that will be released in the 
    //  constructor, things go wrong
    
    // final results
    arma::vec u; 
    arma::vec v;
    // sparse penalty
    Prox *prox_u; // careful about memory leak and destructor stuff, can be replaced by Prox &prox_u;
    Prox *prox_v;
    // S = I + alpha*Omeg
    arma::mat S_u;  // to be special-cased
    arma::mat S_v;
    
    
public:
    ~MoMA(){
        delete prox_u;
        delete prox_v;
    }
    void check_valid();
    Solver string_to_SolverT(const std::string &s); // String to solver type {ISTA,FISTA}
    Prox* string_to_Proxptr(const std::string &s,double gamma,const arma::vec &group);
   
    // turn user input into what we need to run the algorithm
    MoMA(const arma::mat &X_,   // note it is a reference
        /* sparsity*/
        std::string P_v,std::string P_u, 
        double lambda_v,double lambda_u,
        double gamma,
        /* smoothness */
        arma::mat Omega_u,arma::mat Omega_v,
        double i_alpha_u,double i_alpha_v,
        /* grouping */
        const arma::vec &group_u,const arma::vec &group_v,
        /* training para. */
        double i_EPS,long i_MAX_ITER,std::string i_solver):X(X_) // X has to be written in the initialization list
    {
        check_valid();
        MoMALogger::info("Setting up the model\n");

   
        n = X.n_rows;
        p = X.n_cols;

        // Step 0: training para. setup
        MAX_ITER = i_MAX_ITER;
        EPS = i_EPS;
        solver_type = string_to_SolverT(i_solver);

        // Step 1: find Su,Sv
        arma::mat U;
        arma::vec s;
        arma::mat V;
        arma::svd(U, s, V, X);
        S_u.eye(n,n);
        S_v.eye(p,p);
        alpha_u = i_alpha_u;
        alpha_v = i_alpha_v;
        if(i_alpha_u != 0.0)
       {     MoMALogger::debug("Here construct Su");
            S_u += n * i_alpha_u * Omega_u;}
        if(i_alpha_v != 0.0){
            MoMALogger::debug("Here construct Sv");
            S_v += p * i_alpha_v * Omega_v;}

        // Step 1.2: find Lu,Lv
        double Lu = arma::eig_sym(S_u).max() + 0.01; // +0.01 for convergence
        double Lv = arma::eig_sym(S_v).max() + 0.01;
        
        // Step 1.3: all kinds of stepsize
        grad_u_step = 1 / Lu;
        grad_v_step = 1 / Lv;
        prox_u_step = lambda_u / Lu;
        prox_v_step = lambda_v / Lv;

        // Step 2: initialize with SVD
        v = V.col(0);
        u = U.col(0);

        // Step 3: match proximal operator
        prox_u = string_to_Proxptr(P_u,gamma,group_u);
        prox_v = string_to_Proxptr(P_v,gamma,group_v);

        // Step 4: match gradient operator

    };

    void fit();
    Rcpp::List wrap(){ 
        if(norm(u)!=0){
            u = u / norm(u);
        }
        if(norm(v)!=0){
            v = v / norm(v);
        }
        double d = as_scalar(u.t() * X * v);
            return Rcpp::List::create(
            Rcpp::Named("u") = u,
            Rcpp::Named("v") = v,
            Rcpp::Named("d") = d,
            Rcpp::Named("DeflatedX") = X - d * u * v.t());
    }
};



void MoMA::check_valid(){
    MoMALogger::info("Checking input validity\n");

    // grouping vector should be a factor!
}

Solver MoMA::string_to_SolverT(const std::string &s){
    Solver res = Solver::ISTA;
    // we can first make s to upper case and provide more flexibility
    if (s.compare("ISTA") == 0)
        res = Solver::ISTA;
    else if (s.compare("FISTA") == 0)
        res = Solver::FISTA;
    else if (s.compare("APP_FISTA"))
        res = Solver::APP_ISTA;
    else{
        MoMALogger::error("Your choice of algorithm is not provided!") << s;
    }
    return res;
}

Prox* MoMA::string_to_Proxptr(const std::string &s,double gamma,const arma::vec &group){   // free it!
    Prox* res = new Prox();
    if (s.compare("LASSO") == 0)
       res = new Lasso();
    else if (s.compare("NONNEGLASSO") == 0)
        MoMALogger::error("Nonnegative Lasso not implemented yet!\n");
    else if (s.compare("SCAD") == 0)
        res = new Scad(gamma);
    else if (s.compare("MCP") == 0)
        res = new Mcp(gamma);
    else if(s.compare("NNLASSO") == 0){
        res = new NNLasso();
    }
    else if(s.compare("GRPLASSO") == 0){
        res = new GrpLasso(group);
    }
    else
        MoMALogger::warning("Your sparse penalty is not provided by us/specified by you! Use `Prox` by default\n");
    return res;
}


// [[Rcpp::depends(Matrix,RcppArmadillo)]]
// [[Rcpp::export]]
Rcpp::List sfpca(
    const arma::mat &X ,

    arma::mat Omega_u,
    arma::mat Omega_v,  /* any idea to set up default values for these matrices? */ 
                                               
    double alpha_u = 0,double alpha_v = 0,

    std::string P_u = "none",std::string P_v = "none",
    double lambda_u = 0,double lambda_v = 0,
    double gamma = 3.7,

    arma::vec group_u=Rcpp::IntegerVector::create(0), arma::vec group_v=Rcpp::IntegerVector::create(0),
    double EPS = 1e-6,  
    long MAX_ITER = 1e+3,
    std::string solver = "ISTA"
)
                                                  
{
   // MoMALogger::debug("Omega_u is:") << Omega_u;
    MoMA model(X,  
        /* sparsity*/
         P_v,P_u, 
        lambda_v,lambda_u,
        gamma,
        /* smoothness */
          /* About setting default value: Handle it on the R side
                                                     This way is safe. When alpha != 0, we will ask user to specifiy the matrices.
                                                     When alpha=0, we will not encounter dimenstion discompatibility, 
                                                     because we special case it in the `MoMA.fit()` function, thus avoiding multiplying them with vectors.
                                                     */
        Omega_u,Omega_v, 
        alpha_u,alpha_v,
        /* grouping */
        group_u,group_v,
        /* optimization parameters */
        EPS,
        MAX_ITER,
        solver);
    model.fit();
    return model.wrap();
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double test_norm(arma::vec x){
    return norm(x);
}

void MoMA::fit(){

        MoMALogger::info("Model info=========\n")<<"n:"<<n<<"\n"
            <<"p:" << p << "\n";
        MoMALogger::info("Start fitting.\n");

        // keep the value of u at the start of outer loop, hence call it oldu1
        arma::vec oldu1 = arma::zeros<arma::vec>(n);
        arma::vec oldv1 = arma::zeros<arma::vec>(p);
        // keep the value of u at the start of inner loop
        arma::vec oldu2 = arma::zeros<arma::vec>(n);
        arma::vec oldv2 = arma::zeros<arma::vec>(p);

        // stopping tolerance
        int iter = 0;
        int iter_u = 0;
        int iter_v = 0;

        double in_u_tol = 1;   // tolerance for inner loop of u updates
        double in_v_tol = 1;   // tolerance for inner loop of v updates
        double out_tol = 1;    // that of outer loop

        if (solver_type == Solver::APP_ISTA)
        {
            MoMALogger::info("Running APP_ISTA!\n");
            MoMALogger::debug("==Before the loop: training setup==\n") 
                    << "\titer" << iter
                    << "\tEPS:" << EPS 
                    << "\tMAX_ITER:" << MAX_ITER << "\n";
            while (out_tol > EPS && iter < MAX_ITER)
            {
                
                oldu1 = u;  
                oldv1 = v;
                in_u_tol = 1;
                in_v_tol = 1;
                iter_u = 0;
                iter_v = 0;
                while (in_u_tol > EPS && iter_u < MAX_ITER)
                {
                    iter_u++;
                    oldu2 = u;  
                    // gradient step
                    if(alpha_u == 0.0){
                        u = u + grad_u_step * (X*v - u);  
                    }else{
                        u = u + grad_u_step * (X*v - S_u*u);  
                    }
                    // proxiaml step
                    u = prox_u->prox(u,prox_u_step);
                    // nomalize w.r.t S_u
                    double mn = mat_norm(u, S_u);
                    mn > 0 ? u /= mn : u.zeros();    
                    // find torlerance
                    in_u_tol = norm(u - oldu2) / norm(oldu2);
                    MoMALogger::debug("u ") << iter_u << "--" << "% of change " << in_u_tol;
                }

                while (in_v_tol > EPS && iter_v < MAX_ITER)
                {
                    iter_v++;
                    oldv2 = v;
                    // gradient step
                    if(alpha_v == 0.0){
                        v = v + grad_v_step * (X.t()*u - v);   
                    }else{
                       v = v + grad_v_step * (X.t()*u - S_v*v);  
                    }
                    // proximal step
                    v = prox_v->prox(v,prox_v_step);
                    // nomalize w.r.t S_v
                    double mn = mat_norm(v, S_v);
                    mn > 0 ? v /= mn : v.zeros();
                    // find torlerance
                    in_v_tol = norm(v - oldv2) / norm(oldv2);
                    MoMALogger::debug("v ") << iter_v << "---" << "% of change " << in_v_tol;
                }

                out_tol = norm(oldu1 - u) / norm(oldu1) + norm(oldv1 - v) / norm(oldv1);
                iter++;
                MoMALogger::message("--Finish iter:") << iter << "---\n";
            }
        }
        else if (solver_type == Solver::FISTA){
            MoMALogger::info("Running FISTA!\n");
            MoMALogger::debug("==Before the loop: training setup==\n") 
                    << "\titer" << iter
                    << "\tEPS:" << EPS 
                    << "\tMAX_ITER:" << MAX_ITER << "\n";
            while (out_tol > EPS && iter < MAX_ITER)
            {
                
                oldu1 = u;  
                oldv1 = v;
                in_u_tol = 1;
                in_v_tol = 1;
                iter_u = 0;
                iter_v = 0;

                double t = 1;
                while (in_u_tol > EPS && iter_u < MAX_ITER)
                {
                    iter_u++;
                    oldu2 = u;  
                    double oldt = t;
                    t = 0.5 * (1 + sqrt(1 + 4 * oldt*oldt));
                    // gradient step
                    if(alpha_u == 0.0){
                        u = u + grad_u_step * (X*v - u);  

                    }else{
                        u = u + grad_u_step * (X*v - S_u*u);  
                    }
                    // proxiaml step
                    u = prox_u->prox(u,prox_u_step);
                    // momemtum step
                    u = u + (oldt-1)/t * (u-oldu2);

                    in_u_tol = norm(u - oldu2) / norm(oldu2);
                    MoMALogger::debug("u ") << iter_u << "--" << "% of change " << in_u_tol;
                }
                // nomalize w.r.t S_u
                double mn = mat_norm(u, S_u);
                mn > 0 ? u /= mn : u.zeros();    // 
                MoMALogger::debug("mat_norm is")  << mn;
                // restore
                t = 1;
                while (in_v_tol > EPS && iter_v < MAX_ITER)
                {
                    iter_v++;
                    oldv2 = v;
                    double oldt = t;
                    t = 0.5 * (1 + sqrt(1 + 4 * oldt*oldt));
                    // gradient step
                    if(alpha_v == 0.0){
                        v = v + grad_v_step * (X.t()*u - v);   
                    }else{
                       v = v + grad_v_step * (X.t()*u - S_v*v);  
                    }
                    // proximal step
                    v = prox_v->prox(v,prox_v_step);
                    // momemtum step
                    v = v + (oldt-1)/t * (v-oldv2);
                    // find tolerance
                    in_v_tol = norm(v - oldv2) / norm(oldv2);
                    MoMALogger::debug("v ") << iter_v << "---"<< "% of change " << in_v_tol;
                }
                // normalize w.r.t. S_v
                mn = mat_norm(v, S_v);
                mn > 0 ? v /= mn : v.zeros();
                MoMALogger::debug("mat_norm is") << "\tmat_norm:" << mn;


                out_tol = norm(oldu1 - u) / norm(oldu1) + norm(oldv1 - v) / norm(oldv1);
                iter++;
                MoMALogger::message("--Finish iter:") << iter << "---\n";
               //MoMALogger::error("FISTA is not provided yet!\n");
            }
        }
        else if (solver_type == Solver::ISTA) {
            MoMALogger::info("Running ISTA!\n");
            MoMALogger::debug("==Before the loop: training setup==\n") 
                    << "\titer" << iter
                    << "\tEPS:" << EPS 
                    << "\tMAX_ITER:" << MAX_ITER << "\n";
            while (out_tol > EPS && iter < MAX_ITER)
            {
                oldu1 = u;  
                oldv1 = v;
                in_u_tol = 1;
                in_v_tol = 1;
                iter_u = 0;
                iter_v = 0;
                while (in_u_tol > EPS && iter_u < MAX_ITER)
                {
                    iter_u++; 
                    oldu2 = u;  
                    // gradient step
                    if(alpha_u == 0.0){
                        u = u + grad_u_step * (X*v - u);  
                    }else{
                        u = u + grad_u_step * (X*v - S_u*u);  
                    }
                    // proxiaml step
                    u = prox_u->prox(u,prox_u_step);

                    // find tolerance
                    in_u_tol = norm(u - oldu2) / norm(oldu2);
                    MoMALogger::debug("u ") << iter_u << "--"<< "% of change " << in_u_tol;
                }
                // nomalize w.r.t S_u
                double mn = mat_norm(u, S_u);
                mn > 0 ? u /= mn : u.zeros();    
                MoMALogger::debug("mat_norm is") << mn;

                while (in_v_tol > EPS && iter_v < MAX_ITER)
                {
                    iter_v++;
                    oldv2 = v;
                    // gradient step
                    if(alpha_v == 0.0){
                        v = v + grad_v_step * (X.t()*u - v);   
                    }else{
                       v = v + grad_v_step * (X.t()*u - S_v*v);  
                    }
                    // proximal step
                    v = prox_v->prox(v,prox_v_step);
                    // find tolerance
                    in_v_tol = norm(v - oldv2) / norm(oldv2);   
                    MoMALogger::debug("v ") << iter_v << "---" << "% of change " << in_v_tol;
                }
                // nomalize w.r.t S_v
                mn = mat_norm(v, S_v);
                mn > 0 ? v /= mn : v.zeros();
                MoMALogger::debug("mat_norm is") << mn;

                out_tol = norm(oldu1 - u) / norm(oldu1) + norm(oldv1 - v) / norm(oldv1);
                iter++;
                MoMALogger::message("--Finish iter:") << iter << "---\n";
            }
        }
     
        else{
            MoMALogger::error("Your choice of solver is not provided yet!");
        }
        MoMALogger::debug("==After the outer loop!==\n") 
                   << "out_tol:" << out_tol << "\t iter" << iter;
}
