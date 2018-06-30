
// a non-existing child of the node of a heap is far away
#include "moma_prox_fusion_util.h"


int sgn(double val) {
    return (double(0) < val) - (val < double(0));
}

FusionGroups::FusionGroups(const arma::vec &x){
    int n = x.n_elem;
    if(n <= 1){
        MoMALogger::error("TODO deal with scalar");
    }
    g.resize(n);
    pq.resize(n-1);

    // Group g
    // Easy initalize
    for(int i = 0; i < g.size(); i++){
        g[i] = Group(i,i,i,0,x(i));
    }
    // slope
    g[0].slope = - sgn(double(x(0) - x(1)));
    for(int i = 1; i < g.size()-1; i++){
        double s = - (sgn(x(i)-x(i-1)) + sgn(x(i) - x(i+1)));
        g[i].slope = s;
    }
    g[n-1].slope = - (sgn(x(n-1) - x(n-2)));
    // lambda

    // Heap lambda;
    for(int i = 0; i < pq.size(); i++){
        // next merge point of group i and i+1
        double h = 0;
        if(abs(g[i+1].slope - g[i].slope) > 1e-10)  // not parallel
            h =  (g[i].beta - g[i+1].beta) / (g[i+1].slope - g[i].slope);
        else 
            h = INFTY;
        MoMALogger::info("see h")<<double(h) << "=" <<  (g[i].beta - g[i+1].beta) << "//" <<  (g[i+1].slope - g[i].slope);
        pq[i] = HeapNode(i,h);
    }
    std::make_heap(pq.begin(),pq.end(),gt);
    heap_print(pq);
    return;
}

void FusionGroups::print(){
    Rcpp::Rcout<<"Grouping now is\n";
    for(auto i:g)
        i.print();
    Rcpp::Rcout<<"\n";
}

bool FusionGroups::is_valid(int this_node){
    return g[this_node].parent == this_node;
}

int FusionGroups::pre_group(int this_group){
    if(!is_valid(this_group)){
        MoMALogger::error("Only valid groups can be accessed");
    }
    if(this_group == 0)
        return NO_PRE;
    return g[this_group-1].parent;
}

int FusionGroups::next_group(int this_group){
    if(!is_valid(this_group)){
        MoMALogger::error("Only valid groups be accessed");
    }
    if(g[this_group].tail == g.size()-1){
        return NO_NEXT;
    }
    else
        return g[this_group].tail + 1;
}

int FusionGroups::group_size(int this_group){
    if(!is_valid(this_group)){
        MoMALogger::error("Only valid groups be accessed");
    }
    return g[this_group].tail - g[this_group].head + 1;
}

double FusionGroups::line_value_at(double x,double y,double k,double x_){
    return y + k * (x_ - x);
}

arma::vec FusionGroups::print_vec(double target_lam){
    int n = (this->g).size();
    arma::vec x = arma::zeros<arma::vec>(n);
    for(int i = 0; i != NO_NEXT;){
        for(int j = g[i].head; j <= g[i].tail; j++){
            x(j) = line_value_at(g[i].lambda,g[i].beta,g[i].slope,target_lam);
            Rcpp::Rcout << "target_lam" << target_lam << "current_lam" << g[i].lambda << "\n";
        }
        i = next_group(i);
    }
    Rcpp::Rcout << x;
    return x;
}

double FusionGroups::meet(double x1,double x2,double k1,double k2,double y1,double y2){
    return ((y1 - y2) - (k1 * x1 - k2 * x2)) / (-k1 + k2);
}

void FusionGroups::merge(int dst, double new_lambda){
    int src = this->next_group(dst);
    
    if(!is_valid(dst) || src == NO_NEXT){
        MoMALogger::error("Only valid groups can be merged: merge point is not valid");
    }
    if(dst >= src)
        MoMALogger::error("dst_grp should be in front of src_grp");

    // update beta
    g[dst].beta = g[dst].beta + g[dst].slope * (new_lambda - g[dst].lambda);

    // update lambda
    g[dst].lambda = new_lambda;
    
    // update slope 
    int pre_group = this->pre_group(dst);
    int next_group = this->next_group(src);
    int sgn1 = 0;
    int sgn2 = 0;
    if(next_group != NO_NEXT) 
        sgn2 = sgn(g[dst].beta - g[next_group].beta);
    if(pre_group != NO_PRE) 
        sgn1 = sgn(g[dst].beta - g[pre_group].beta);
    g[dst].slope = -1 / double(this->group_size(dst) + this->group_size(src)) * (sgn1 + sgn2);
    
    // set up pointers
    int last_node = g[src].tail;
    g[dst].tail = last_node;
    g[src].parent = dst;
    g[last_node].parent = dst;
    this->print();
    
    // update heap
    if(pre_group != NO_PRE){
        // double lambda_pre = ((g[pre_group].beta - g[dst].beta) - (g[pre_group].slope*g[pre_group].lambda - g[dst].slope*g[dst].lambda)) / (-g[pre_group].slope + g[dst].slope);
        double lambda_pre = meet(g[pre_group].lambda,g[dst].lambda,g[pre_group].slope,g[dst].slope,g[pre_group].beta,g[dst].beta);
        MoMALogger::info("Update lambda of pre group\n");
        heap_change_lambda(this->pq,pre_group,lambda_pre);
        heap_print(this->pq);
    }
    if(next_group != NO_NEXT){
        double lambda_next = ((g[next_group].beta - g[dst].beta) - (g[next_group].slope*g[next_group].lambda - g[dst].slope*g[dst].lambda)) / (-g[next_group].slope + g[dst].slope);
        MoMALogger::info("Update lambda of next group\n");
        heap_change_lambda(this->pq,dst,lambda_next);
        heap_print(this->pq);
        heap_delete(this->pq,src);
    }else{
        heap_delete(this->pq,dst);
    }
}



// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
void fusino_test(const arma::vec &x){
    FusionGroups a(x);

    int cnt = 5;

    HeapNode node1;
    while(cnt >= 0){
        MoMALogger::info("=================");
        node1 = heap_peek_min(a.pq);
        heap_print(a.pq);
        a.merge(node1.id,node1.lambda);
        a.print_vec(node1.lambda);

    }


    // a.merge(0,2);
    // a.merge(0,3);

    // Rcpp::Rcout << a.pre_group(0);
    // Rcpp::Rcout << a.next_group(0);

    // a.merge(4,5);
    // a.merge(4,6);
    // a.merge(4,7);

    // Rcpp::Rcout << a.pre_group(4);
    // Rcpp::Rcout << a.next_group(4);
    // a.merge(0,4);
}

// [[Rcpp::export]]
void queue_test(){

    HeapNode a(1,2.3);
    HeapNode b(2,5.2);
    HeapNode c(3,0.1);
    HeapNode d(4,0);
    HeapNode e(5,3.2);
    HeapNode h(6,2);
    HeapNode g(7,-3);
    HeapNode i(8,2.3);
    HeapNode j(9,23);
    HeapNode k(10,9.23);
    std::vector<HeapNode> q;
    q.push_back(a);
    q.push_back(b);
    q.push_back(c);
    q.push_back(d);
    q.push_back(e);
    q.push_back(h);
    q.push_back(g);
    q.push_back(i);
    q.push_back(j);
    q.push_back(k);

    std::make_heap(q.begin(), q.end(),gt);
    heap_print(q);

    for(int i=1; i<11; i++){
        heap_delete(q,i);
        heap_print(q);
        if(!is_minheap(q)){
            MoMALogger::error("Not min heap!");
        }
    }
}

