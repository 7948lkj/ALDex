#include "Common.h"
#include "DSM.h"

struct list_node
{
    uint64_t key;
    uint64_t val;
    uint64_t next_offset;
};

class write_buffer
{
private:
    DSM *dsm;
    int index;
    int off_head;
    int off_tail;
    int off_queue;
    int off_write_buffer;
    uint64_t buffer_size;//2^24
    uint64_t one;
    int one_exp=0;
    int thread_num;
    //std::atomic_int app_id;
    //int pre_slot_num=pre_alloc_size;
    GlobalAddress pre_alloc_slot[define::thread_num][pre_alloc_size * define::kCoroCnt];
    int now_slots_index[define::thread_num];
public:
    write_buffer(int index,int write_buffer_off,uint64_t buffer_size,int thread_num,DSM *dsm);
    write_buffer(int index,DSM *dsm,uint64_t buffer_size,int thread_num);
    write_buffer();
    bool alloc_buffers(uint64_t node_id, int thread_id, CoroContext *ctx);
    bool get_slot(int thread_id, uint64_t node_id, GlobalAddress &slot_gaddr, CoroContext *ctx);
    bool fill_slot(uint64_t key,uint64_t val,uint64_t next_offset,GlobalAddress &slot_gaddr);
    bool write_slot(int thread_id,uint64_t node_id,uint64_t key,uint64_t val,uint64_t next_offset,GlobalAddress &slot_gaddr);
    bool free_slot(GlobalAddress slot_gaddr);
    bool free_slot_local(int thread_id,uint64_t node_id,GlobalAddress slot_gaddr);
    uint64_t get_off_write_buffer();
    uint64_t get_off_start();
};




