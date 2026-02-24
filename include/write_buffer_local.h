#ifndef WRITE_BUFFER_LOCAL_H
#define WRITE_BUFFER_LOCAL_H



struct slot_local
{
    uint64_t next;
    uint64_t key;
    uint64_t val;
};

struct list_node_local
{
    uint64_t key;
    uint64_t val;
    uint64_t next_offset;
};

struct write_buffer_local
{
    uint64_t *head_ptr;
    uint64_t *tail_ptr;
    int *queue_ptr;
    list_node_local *list_node_local_ptr;
    uint64_t buffer_size;//2^24
    uint64_t one;
    int one_exp=0;
public:
    write_buffer_local(uint64_t head_off,uint64_t buffer_size_):buffer_size(buffer_size_)
    {
        std::cout<<"buffer size:"<<buffer_size<<std::endl;
        one=((uint64_t)1<<63)/buffer_size<<1;
        uint64_t temp=one;
        while(temp!=1)
        {
            temp=temp>>1;
            one_exp++;
        }
        uint64_t off_head=head_off;
        uint64_t off_tail=off_head+sizeof(uint64_t);
        uint64_t off_queue=off_tail+sizeof(uint64_t);
        uint64_t off_write_buffer=off_queue+sizeof(int)*buffer_size;
        head_ptr=(uint64_t*)off_head;
        tail_ptr=(uint64_t*)off_tail;
        queue_ptr=(int*)off_queue;
        list_node_local_ptr=(list_node_local*)off_write_buffer;
    }
    void write_slot_local(uint64_t key,uint64_t val,uint64_t &node_index)
    {
        std::cout<<"buffer size"<<buffer_size<<std::endl;
        uint64_t head_before_add=__sync_fetch_and_add(head_ptr,one);
        int queue_index=head_before_add>>one_exp;
        std::cout<<head_ptr<<"queue index:"<<queue_index<<std::endl;
        node_index=queue_ptr[queue_index];
        list_node_local_ptr[node_index].key=key;
        list_node_local_ptr[node_index].val=val;
        list_node_local_ptr[node_index].next_offset=null_flag;
    }
    void free_slot_local(list_node_local *node_ptr)
    {
        uint64_t tail_before_add=__sync_fetch_and_add(tail_ptr,one);
        int queue_index=tail_before_add>>one_exp;
        node_ptr->next_offset=null_flag;
        queue_ptr[queue_index]=((uint64_t)node_ptr-(uint64_t)list_node_local_ptr)/sizeof(list_node_local);
    }
};



#endif