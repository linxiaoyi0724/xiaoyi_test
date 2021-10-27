# xiaoyi_test
example

// 链表
#include <iostream>
#include <vector>

struct listNode {
    int data;
    listNode *next;
    listNode():data(0),next(nullptr){}
};


void creatList(listNode* list, std::vector<int> a){
    listNode* p = list;
    for(int i = 0; i < a.size();i++){
        listNode* node = new listNode;
        node->data = a[i];
        node->next = nullptr;
        
        p->next = node;
        p = p->next;
    }
}

void insertList(listNode* list, int index, int num){
    listNode* p = list;
    for(int i = 1; i < index-1; i++){
        p= p->next;
    }
    listNode* node = new listNode;
    node->data = num;
    node->next = p->next;
    p->next = node;
}

void deleteList(listNode* list, int index){
    listNode* p = list;
    for(int i = 1; i < index-1; i++){
        p = p->next;
    }
    listNode* q = p->next;
    p->next = p->next->next;
    delete q;
    
}

listNode* reverseList(listNode* list){
    listNode* p = list;
    listNode* pre = nullptr;
    listNode* temp = nullptr;
    while (p) {
        temp = p->next;
        p->next = pre;
        pre = p;
        p = temp;
    }
    return pre;
}

void printList(listNode* list){
    listNode* p = list;
    while (p) {
        std::cout<< p->data<< "   ";
        p=p->next;
        
    }
}

void getMiddleAndTailList(listNode* list, listNode* &middle, listNode* &tail){
    listNode* slow = list;
    listNode* fast = list;
    while (1) {
        slow = slow->next;
        for(int i = 0; i != 2; i++){
            if(fast->next == nullptr){
                middle = slow;
                tail = fast;
                return;
            }
            fast = fast->next;
        }
    }
}

listNode* reverseList1(listNode* list){
    listNode * p = list;
    listNode * mhead = list;
    listNode * q = list->next;
    while(p->next!=list){
        listNode * r = q->next;
        p->next = r;
        q->next = mhead;
        mhead   = q;
        q = p->next;
    }
    p->next = NULL;
    return mhead;
}

listNode* swapPair(listNode* list){
    listNode* middleList = nullptr;
    listNode* tailList = nullptr;
    
    getMiddleAndTailList(list, middleList, tailList);
    tailList->next = list;
    
    return reverseList1(middleList);
    
}



int main(){
    std::vector<int> arr = {1,2,3,4,5,6};
    listNode* list = new listNode;
    creatList(list,arr);
    printList(list->next);
    std::cout << std::endl;
    
//    链表的插入
//    insertList(list->next, 2, 100);
//    printList(list->next);
    
//    链表的删除
//    deleteList(list->next, 2);
//    printList(list->next);
    
//    链表的反转
//    printList(reverseList(list->next));
    
    
//    链表中间反转
    printList(swapPair(list->next));
    
}
