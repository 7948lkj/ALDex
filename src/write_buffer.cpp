#include "write_buffer.h"

write_buffer::write_buffer()
{

}


write_buffer::write_buffer(int index,int write_buffer_off,uint64_t buffer_size,int thread_num,DSM *dsm):thread_num(thread_num),dsm(dsm),index(index),buffer_size(buffer_size)
{
    one=((uint64_t)1<<63)/buffer_size<<1;
    uint64_t temp=one;
    while(temp!=1)
    {
        temp=temp>>1;
        one_exp++;
    }
    
    off_head=write_buffer_off;
    off_tail=off_head+sizeof(uint64_t);
    off_queue=off_tail+sizeof(uint64_t);
    off_write_buffer=off_queue+sizeof(int)*buffer_size;
    for(int i=0;i<define::thread_num;i++)
    {
        now_slots_index[i]=define::kCoroCnt * pre_alloc_size;
    }
}

write_buffer::write_buffer(int index,DSM *dsm,uint64_t buffer_size,int thread_num):dsm(dsm),buffer_size(buffer_size),thread_num(thread_num),index(index)
{
    std::cout<<"real buffer size"<<buffer_size<<std::endl;
    one=((uint64_t)1<<63)/buffer_size<<1;
    uint64_t temp=one;
    while(temp!=1)
    {
        temp=temp>>1;
        one_exp++;
    }
    GlobalAddress head_gaddr=dsm->alloc_local_size(sizeof(uint64_t));
    off_head=head_gaddr.offset;
    GlobalAddress tail_gaddr=dsm->alloc_local_size(sizeof(uint64_t));
    off_tail=off_head+sizeof(uint64_t);
    GlobalAddress queue_gaddr=dsm->alloc_local_size(sizeof(int)*buffer_size);
    off_queue=off_tail+sizeof(uint64_t);
    GlobalAddress write_buffer_gaddr=dsm->alloc_local_size(sizeof(list_node)*buffer_size);
    off_write_buffer=off_queue+sizeof(int)*buffer_size;
    char *cache_ptr=dsm->get_cache();
    uint64_t *head_ptr=(uint64_t*)cache_ptr;
    int *queue_ptr=(int*)cache_ptr;
    head_ptr[0]=0;
    head_ptr[1]=(buffer_size-1)<<one_exp;
    dsm->write_sync(cache_ptr,head_gaddr,sizeof(uint64_t)*2);
    for(int i=0;i<buffer_size-1;i++)
    {
        queue_ptr[i]=i;
    }
    dsm->write_sync(cache_ptr,queue_gaddr,sizeof(int)*buffer_size);
    // list_node *list_node_ptr=(list_node*)cache_ptr;
    // for(int i=0;i<buffer_size;i++)
    // {
    //     list_node_ptr[i].next_offset=null_flag;
    // }
    // dsm->write_sync(cache_ptr,write_buffer_gaddr,sizeof(list_node)*buffer_size);
    for(int i=0;i<define::thread_num;i++)
    {
        now_slots_index[i]=define::kCoroCnt * pre_alloc_size;
    }
    std::cout<<"list_node_size:"<<sizeof(list_node)<<std::endl;
    std::cout<<"off_head: "<<off_head<<std::endl;
    std::cout<<"off_tail: "<<off_tail<<std::endl;
    std::cout<<"off_queue: "<<off_queue<<std::endl;
    std::cout<<"off_write_buffer: "<<off_write_buffer<<std::endl;
    std::cout<<std::hex<<"one: "<<one<<std::endl;
    std::cout<<std::dec<<"one_exp: "<<one_exp<<std::endl;
}


bool write_buffer::alloc_buffers(uint64_t node_id, int thread_id, CoroContext *ctx)
{
    // std::cout<<"alloc buffer"<<std::endl;
    char *cache_ptr = dsm->get_coro_buf(ctx ? ctx->coro_id : 0);
    GlobalAddress base;
    base.nodeID=node_id;
    base.offset=0;
    GlobalAddress head_gaddr=base;
    head_gaddr.offset+=off_head;
    uint64_t *head_ptr=(uint64_t*)cache_ptr;
    //std::cout<<head_gaddr.nodeID<<":"<<head_gaddr.offset<<std::endl;
    dsm->read_sync(cache_ptr,head_gaddr,sizeof(uint64_t)*2, ctx);
    // std::cout<<"head gaddr:"<<head_gaddr.nodeID<<":"<<head_gaddr.offset<<std::endl;
    volatile int head=(int)(head_ptr[0]>>one_exp);
    volatile int tail=(int)(head_ptr[1]>>one_exp);
    // std::cout<<"head_pre: "<<head_ptr[0]<<" tail_pre:"<<head_ptr[1]<<std::endl;
    // std::cout<<"head_aft: "<<head<<" tail_aft:"<<tail<<std::endl;
    volatile int occ_num=0;
    if(tail<head)
    {
        occ_num=(tail-head);
        occ_num+=(buffer_size-1);
    }
    else
    {
        occ_num=(tail-head);
    }
    if(occ_num<2*thread_num)
    {
        sleep(5);
        assert(1 == 0);
        return false;
    }
    uint64_t alloc_size=pre_alloc_size;
    dsm->faa_sync(head_gaddr,one*pre_alloc_size,head_ptr,ctx);
    head=(int)(head_ptr[0]>>one_exp);
    //std::cout<<"head_read: "<<head<<std::endl;
    GlobalAddress queue_gaddr=base;
    queue_gaddr.offset+=(sizeof(int)*head+off_queue);
    int *queue_ptr=(int*)cache_ptr;
    dsm->read_sync(cache_ptr,queue_gaddr,sizeof(int)*pre_alloc_size,ctx);
    int start_off = now_slots_index[thread_id] - 1;
    for(int i=0;i<pre_alloc_size;i++)
    {
        GlobalAddress temp;
        temp.nodeID=node_id;
        int slot_offset=queue_ptr[i];
        temp.offset=(off_write_buffer+slot_offset*sizeof(list_node));
        pre_alloc_slot[thread_id][start_off - i]=temp;
    }
    now_slots_index[thread_id] -= pre_alloc_size;
    return true;
}

bool write_buffer::get_slot(int thread_id,uint64_t node_id,GlobalAddress &slot_gaddr, CoroContext *ctx)
{
    if(now_slots_index[thread_id] == define::kCoroCnt * pre_alloc_size)
    {
        bool ret=alloc_buffers(node_id, thread_id, ctx);
        if(!ret)
        {
            return ret;
        }
    }
    slot_gaddr=pre_alloc_slot[thread_id][now_slots_index[thread_id]];
    now_slots_index[thread_id]++;
    return true;
}

bool write_buffer::free_slot_local(int thread_id,uint64_t node_id,GlobalAddress slot_gaddr)
{
    now_slots_index[thread_id]--;
    pre_alloc_slot[thread_id][now_slots_index[thread_id]]=slot_gaddr;
    return true;
}

bool write_buffer::fill_slot(uint64_t key,uint64_t val,uint64_t next_offset,GlobalAddress &slot_gaddr)
{
    char *cache_ptr=dsm->get_rdma_buffer();
    list_node *node_ptr=(list_node*)cache_ptr;
    node_ptr[0].key=key;
    node_ptr[0].val=val;
    node_ptr[0].next_offset=next_offset;
    dsm->write_sync(cache_ptr,slot_gaddr,sizeof(list_node));
    return true;
}


bool write_buffer::free_slot(GlobalAddress slot_gaddr)
{
    int slot_offset=(slot_gaddr.offset-off_write_buffer)/sizeof(list_node);
    if(slot_offset<0)
    {
        return false;
    }
    char *cache_ptr=dsm->get_rdma_buffer();
    GlobalAddress base;
    base.nodeID=slot_gaddr.nodeID;
    base.offset=0;
    GlobalAddress tail_gaddr=base;
    tail_gaddr.offset+=(off_tail);
    uint64_t *tail_ptr=(uint64_t *)cache_ptr;
    dsm->faa_sync(tail_gaddr,one,tail_ptr);
    int tail=(int)(tail_ptr[0]>>one_exp);
    //std::cout<<"tail_read: "<<tail<<std::endl;
    GlobalAddress queue_gaddr=base;
    queue_gaddr.offset+=(tail*sizeof(int)+off_queue);
    int *queue_ptr=(int*)cache_ptr;
    queue_ptr[0]=slot_offset;
    //std::cout<<"free slot: "<<slot_offset<<std::endl;
    dsm->write_sync(cache_ptr,queue_gaddr,sizeof(int));
    return true;
}

uint64_t write_buffer::get_off_write_buffer()
{
    return off_write_buffer;
}

uint64_t write_buffer::get_off_start()
{
    return off_head;
} 