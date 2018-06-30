#include "moma_heap.h"
Heap::Heap(int n){
    heap.resize(n);   
};

void Heap::heapify(){
    std::make_heap(heap.begin(),heap.end(),gt);
}
/*
* Find the smaller child of an element in a heap. Used in siftdown
*/
int Heap::min_child(int i) {
    int cur_size = heap.size();
	int child = i * 2 + 1;
	if (child >= cur_size) {
		// no children
		return NO_CHILED;
	} else if (child+1 >= cur_size || !gt(heap[child],heap[child+1])){
		// only child or first child is biggest child
		return child;
	} else {
		// second child exists and is smallest child
		return child+1;
	}
}

/*
* TODO: extra copy can be avoided in siftdown
*/
void Heap::swap(int i, int j){
    HeapNode tmp = heap[i];
    heap[i] = heap[j];
    heap[j] = tmp;
}

/*
* In a min-heap, if the key (lambda in our case) decreases, sift it up
*/
void Heap::siftup(int i) {
    int parent = (i - 1) / 2;
    while (i != 0 && !gt(heap[i],heap[parent])) {
        swap(parent, i);
        parent = (i - 1) / 2;
    }
}

/*
* In a min-heap, if the key (lambda in our case) increases, sift it down
*/
void Heap::siftdown(int current_node) {
    int cur_size = heap.size();
	int child = min_child(current_node);
	while (child < cur_size && gt(heap[current_node], heap[child])){
		swap(child, current_node);
        current_node = child;
		child = min_child(child);
	}
}

/*
* Change the key of any nodes; TODO: not use trasversal
*/
int Heap::heap_change_lambda_by_id(int id, double new_lambda){
    for(int i = 0; i < heap.size(); i++){
        if(heap[i].id == id){
            double old_lambda = heap[i].lambda;
            heap[i].lambda = new_lambda;
            if(old_lambda < new_lambda)
                siftdown(i);
            else
                siftup(i);
            return i;
        }
    }
    MoMALogger::error("No such id in current heap: ")<<id;
    return -1;
}

/*
* To delete an element, move it to the tail, pop it out, and then sift down 
* the node that replaces it
*/
void Heap::heap_delete(int id){
    for(int i = 0; i < heap.size(); i++){
        if(heap[i].id == id){
            swap(i,heap.size()-1);
            heap.pop_back();
            siftdown(i);
            return;
        }
    }
    MoMALogger::error("No such id in current heap: ")<<id;
}

/*
* Check if an array is a min heap
*/
bool Heap::is_minheap(){
    int i = 0;
    while(2 * i + 1 < heap.size()){
        if(gt(heap[i],heap[2 * i + 1])){
            MoMALogger::error("") << "Not a min-heap" << heap[i].lambda << "and"<< heap[2*i+1].lambda;
            return 0;
        }
        if(2 * i + 2 < heap.size()){
            if(gt(heap[i],heap[2 * i + 2])){
                MoMALogger::error("") << "Not a min-heap" << heap[i].lambda << "and"<< heap[2*i+2].lambda;
                return 0;
            }
        }
        i++;
    }
    return 1;
}

bool Heap::is_empty(){
    return heap.size() == 0;
}
/*
* Get the currently minimun value without deleting the node
*/
HeapNode Heap::heap_peek_min(){
    if(heap.size() == 0){
        MoMALogger::error("Empty heap!");
    }
    HeapNode n = heap.front();
    return n;
}

/*
* Print the heap
*/
void Heap::heap_print(){
    MoMALogger::debug("") << "Heap lambda is\n";
    for (auto i : heap) MoMALogger::debug("") << i.lambda << "\t";
    MoMALogger::debug("") << "\n";
    MoMALogger::debug("") << "Heap id is\n";
    for (auto i : heap) MoMALogger::debug("") << i.id << "\t";
    MoMALogger::debug("") << "\n";
}
