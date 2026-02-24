#include "Directory.h"
#include "Common.h"

#include "Connection.h"
#include "leaf.h"

GlobalAddress g_root_ptr = GlobalAddress::Null();
int g_root_level = -1;
//bool enable_cache;

bool comparebykey(const kv &a,const kv &b)
{
    return a.key<b.key;
}

bool uniquebykey(const kv &a,const kv &b)
{
  return a.key==b.key;
}

Directory::Directory(DirectoryConnection *dCon, RemoteConnection *remoteInfo,
                     uint32_t machineNR, uint16_t dirID, uint16_t nodeID)
    : dCon(dCon), remoteInfo(remoteInfo), machineNR(machineNR), dirID(dirID),
      nodeID(nodeID), dirTh(nullptr) {

  { // chunck alloctor
    GlobalAddress dsm_start;
    uint64_t per_directory_dsm_size = dCon->dsmSize / NR_DIRECTORY;
    dsm_start.nodeID = nodeID;
    dsm_start.offset = per_directory_dsm_size * dirID;
    chunckAlloc = new GlobalAllocator(dsm_start, per_directory_dsm_size);
  }

  dirTh = new std::thread(&Directory::dirThread, this);
  //new std::thread(&Directory::retrain_thread,this);
}

Directory::~Directory() { delete chunckAlloc; }

void Directory::dirThread() {

  bindCore((CPU_PHYSICAL_CORE_NUM - 1 - dirID) * 2 + 1);
  Debug::notifyInfo("thread %d in memory nodes runs...\n", dirID);

  while (true) {
    struct ibv_wc wc;
    pollWithCQ(dCon->cq, 1, &wc);

    switch (int(wc.opcode)) {
    case IBV_WC_RECV: // control message
    {

      auto *m = (RawMessage *)dCon->message->getMessage();

      process_message(m);

      break;
    }
    case IBV_WC_RDMA_WRITE: {
      break;
    }
    case IBV_WC_RECV_RDMA_WITH_IMM: {

      break;
    }
    default:
      assert(false);
    }
  }
}


void Directory::process_message(const RawMessage *m) {

  RawMessage *send = nullptr;
  switch (m->type) {
  case RpcType::MALLOC: {

    send = (RawMessage *)dCon->message->getSendPool();

    send->addr = chunckAlloc->alloc_chunck();
    break;
  }
  case RpcType::MALLOC_SIZE: {

    send = (RawMessage *)dCon->message->getSendPool();

    send->addr = chunckAlloc->alloc_size(m->alloc_size);
    //std::cout<<"alloc" << m->alloc_size << " bytes, gaddr: " << send->addr.nodeID << ":" << send->addr.offset << std::endl;
    break;
  }

  case RpcType::RETRAIN_CHAIN: {

    send = (RawMessage *)dCon->message->getSendPool();
    //std::cout<<"retrain chain buffer index: "<<m->alloc_size<<std::endl;
    //std::cout<<"retrain chain gaddr "<<m->addr.nodeID<<":"<<m->addr.offset<<std::endl;
    retrain_chain_queue[m->alloc_size]->enqueue(m->addr);
    break;
  }

  case RpcType::RETRAIN_BUFFER: {
    send = (RawMessage *)dCon->message->getSendPool();
    //std::cout<<"retrain buffer index"<<m->alloc_size<<std::endl;
    retrain_buffer_flag[m->alloc_size]++;
    break;
  }
  

  case RpcType::NEW_ROOT: {

    if (g_root_level < m->level) {
      g_root_ptr = m->addr;
      g_root_level = m->level;
      if (g_root_level >= 3) {
        //enable_cache = true;
      }
    }

    break;
  }

  default:
    assert(false);
  }

  if (send) {
    dCon->sendMessage2App(send, m->node_id, m->app_id);
  }
}

// bool model_binary_search(std::vector<std::vector<model_global>>&level_models,int level,int model,uint64_t key,int &next_model,uint64_t Epsilon)
// {
//     if(level<1||level>level_models.size()-1||level_models[level][model].is_leaf)
//     {
//         return false;
//     }
//     if(level_models[level][model].child_length==1)
//     {
//         next_model=0;
//         return true;
//     }
//     long double predict=level_models[level][model].slope*key+level_models[level][model].intercept;
//     if(predict<0)
//     {
//         predict=0;
//     }
//     else if(predict>level_models[level][model].child_length-1)
//     {
//         predict=level_models[level][model].child_length-1;
//     }
//     int predict_in=(int)predict+level_models[level][model].pos_start;
//     int low=std::max<int>(0,(predict_in-Epsilon));
//     int high=std::min<int>((predict_in+Epsilon),(level_models[level][model].pos_start+level_models[level][model].child_length-1));
//     if((key>=level_models[level-1][predict_in].key_start&&predict_in==level_models[level][model].child_length-1+level_models[level][model].pos_start)||
//         (key>=level_models[level-1][predict_in].key_start&&key<level_models[level-1][predict_in+1].key_start))
//     {
//         next_model=predict_in;
//         return true;
//     }
//     else if(key<level_models[level-1][predict_in].key_start)
//     {
//         high=predict_in-1;
//     }
//     else if(key>=level_models[level-1][predict_in+1].key_start)
//     {
//         low=predict_in+1;
//     }
//     while(low<=high)
//     {
//         int mid=(high+low)>>1;
//         if((key>=level_models[level-1][mid].key_start&&mid==level_models[level][model].child_length-1+level_models[level][model].pos_start)||
//         (key>=level_models[level-1][mid].key_start&&key<level_models[level-1][mid+1].key_start))
//         {
//             next_model=mid;
//             return true;
//         }
//         else if(key<level_models[level-1][mid].key_start)
//         {
//             high=mid-1;
//         }
//         else if(key>=level_models[level-1][mid+1].key_start)
//         {
//             low=mid+1;
//         }
//     }
//     next_model=low;
//     return true;
// }


// bool model_search(std::vector<std::vector<model_global>>&level_models,uint64_t &key,model_global &target_model)
// {
//     if(level_models.size()==0)
//     {
//         return false;
//     }
//     int level_now=level_models.size()-1;
//     int model_now=0;
//     while(!level_models[level_now][model_now].is_leaf)
//     {
//         bool res=model_binary_search(level_models,level_now,model_now,key,model_now,define::Epsilon);
//         //std::cout<<"level_now: "<<level_now<<"model_now"<<model_now<<std::endl;
//         if(!res)
//         {
//             return false;
//         }
//         level_now-=1;
//     }
//     target_model=level_models[level_now][model_now];
//     //debug
//     if((key>=level_models[level_now][model_now].key_start&&model_now==level_models[level_now].size()-1)
//     ||(key>=level_models[level_now][model_now].key_start&&key<level_models[level_now][model_now+1].key_start))
//     {
//         return true;
//     }
//     else
//     {
//         return false;
//     }
// }

// bool slot_binary_search(slot_local* kvs,int size,uint64_t key,int &pos)
// {
//     if(kvs[0].key>key)
//     {
//         return false;
//     }
//     int left=0;
//     int right=size-1;
//     while(left<=right)
//     {
//         int mid=(left+right)/2;
//         if(kvs[mid].key>key)
//         {
//             right=mid-1;
//         }
//         else
//         {
//             left=mid+1;
//         }
//     }
//     if(right<0)
//     {
//         right=0;
//     }
//     pos=right; 
//     return true;
// }

// void Directory::handle_new_insert(std::vector<std::vector<model_global>>&level_models_global, list_node_local *old_head,list_node_local *new_head,int write_buffer_index,std::vector<GlobalAddress> &addr_to_free)
// {
//   list_node_local *temp_ptr=new_head;
//   write_buffer_local write_buffer_local_=write_buffer_local_infos[write_buffer_index];
//   //insert_num=0;
//   while(temp_ptr!=old_head)
//   {
//     //std::cout<<"handle a new insert"<<temp_ptr->key<<std::endl;
//     GlobalAddress now_gaddr;
//     now_gaddr.nodeID=nodeID;
//     now_gaddr.offset=(uint64_t)temp_ptr-(uint64_t)dCon->dsmPool;
//     addr_to_free.push_back(now_gaddr);
//     kv temp_kv;
//     temp_kv.key=temp_ptr->key;
//     temp_kv.val=temp_ptr->val;
//     model_global seg_model;
//     model_search(level_models_global,temp_kv.key,seg_model);
//     long double predict=seg_model.slope*temp_kv.key+seg_model.intercept;
//     int start=std::max<int>(0,(int)predict-(int)define::Epsilon);
//     int end=std::min<int>(seg_model.child_length-1,(int)predict+define::Epsilon);
//     int kv_size=end-start+1;
//     slot_local *target=(slot_local*)(seg_model.child_start.offset+(uint64_t)dCon->dsmPool+start*sizeof(slot_local));
//     int pos=0;
//     slot_binary_search(target,kv_size,temp_kv.key,pos);
//     //slot_local slot_now=target[pos];
//     uint64_t *ptr=(uint64_t*)(&target[pos]);
//     slot_local *slot_=&target[pos];
//     list_node_local *next_ptr=(list_node_local*)((uint64_t)write_buffer_local_.list_node_local_ptr+(temp_ptr->next_offset&mask_)*sizeof(list_node_local));
//     if(slot_->key==temp_kv.key)
//     {
//       slot_->val=temp_kv.val;
//       temp_ptr=next_ptr;
//       continue;
//     }
//     //std::cout<<"retrain slot key:"<<slot_now.key<<std::endl;
//     //uint64_t old=slot_now.next;
//     GlobalAddress slot_gaddr=seg_model.child_start;
//     GlobalAddress new_slot_next=seg_model.child_start;
//     slot_gaddr.offset=(uint64_t)ptr-(uint64_t)dCon->dsmPool;
//     uint64_t node_index;
//     write_buffer_local_.write_slot_local(temp_kv.key,temp_kv.val,node_index);
//     list_node_local *new_node=&(write_buffer_local_.list_node_local_ptr[node_index]);
//     new_slot_next.offset=(uint64_t)new_node-(uint64_t)dCon->dsmPool;
//     uint64_t new_=new_slot_next.val;
//     //uint64_t *new_next=(uint64_t*)new_node;
//     while(1)
//     {
//       uint64_t old=*ptr;
//       if((old&null_flag)!=0)
//       {
//         uint64_t next=slot_gaddr.val|tail_flag;
//         new_node->next_offset=next;
//         if(__sync_bool_compare_and_swap(ptr,old,(new_|chain_flag)))
//         {
//           break;
//         }
//         else
//         {
//           new_node->next_offset=null_flag;
//           old=*ptr;
//           continue;
//         }
//       }
//       else if((old&chain_flag)!=0)
//       {
//         GlobalAddress next_gaddr;
//         next_gaddr.val=(old&mask_);
//         uint64_t next=(next_gaddr.offset+(uint64_t)dCon->dsmPool-(uint64_t)(write_buffer_local_.list_node_local_ptr))/sizeof(list_node_local);
//         new_node->next_offset=next;
//         if(__sync_bool_compare_and_swap(ptr,old,new_|chain_flag))
//         {
//           break;
//         }
//         else
//         {
//           old=*ptr;
//           continue;
//         }
//       }
//     }
//     //write_buffer_local_.free_slot_local(temp_ptr);
//     temp_ptr=next_ptr;
//   }
// }

// void Directory::build_retrain_model(int write_buffer_index,slot_local *slot_,list_node_local *old_head,std::vector<kv> &kvs,GlobalAddress &root,std::vector<GlobalAddress> &addr_to_free)
// {
//   write_buffer_local write_buffer_local_=write_buffer_local_infos[write_buffer_index];
//   //init
//   std::vector<model_local> local_models;
//   make_segment_for_kvs(kvs,define::Epsilon,local_models);
//   std::vector<std::vector<model_local>> level_models_local;
//   std::vector<std::vector<model_global>> level_models_global;
//   level_models_local.push_back(local_models);
//   int level_num=0;
//   //train the local model
//   while(level_models_local[level_num].size()>1)
//   {
//     std::vector<uint64_t> temp_key;
//     get_start_key_from_model(temp_key,level_models_local[level_num]);
//     std::vector<model_local> next_level;
//     make_segment_par(temp_key,define::Epsilon,next_level);
//     level_models_local.push_back(next_level);
//     level_num++;
//   }

//   //train and write the global seg model
//   std::vector<model_global> global_models_0;
//   for(int i=0;i<local_models.size();i++)
//   {
//     size_t kvs_size=local_models[i].data_num*sizeof(slot_local);
//     GlobalAddress seg_gaddr=chunckAlloc->alloc_size(kvs_size);
//     //std::cout<<"alloc off:"<<seg_gaddr.offset<<std::endl;
//     slot_local *start_ptr=(slot_local*)(seg_gaddr.offset+(uint64_t)dCon->dsmPool);
//     for(int j=0;j<local_models[i].data_num;j++)
//     {
//       start_ptr[j].next=null_flag;
//       start_ptr[j].key=kvs[local_models[i].pos_start+j].key;
//       start_ptr[j].val=kvs[local_models[i].pos_start+j].val;
//     }
//     model_global temp;
//     temp.slope=local_models[i].slope;
//     temp.intercept=local_models[i].intercept;
//     temp.key_start=local_models[i].key_start;
//     temp.is_leaf=true;
//     temp.child_start=seg_gaddr;
//     temp.child_length=local_models[i].data_num;
//     temp.write_buffer_index=write_buffer_index;
//     global_models_0.push_back(temp);
//   }
//   level_models_global.push_back(global_models_0);

//   //train and write the upper global model
//   //std::cout<<"train and write the upper global model"<<std::endl;
//   int model_sum=0;
//   for(int i=0;i<level_models_local.size();i++)
//   {
//     model_sum+=level_models_local[i].size();
//   }
//   GlobalAddress models_gaddr=chunckAlloc->alloc_size(sizeof(model_global)*model_sum);
//   //std::cout<<"alloc off"<<models_gaddr.offset<<std::endl;
//   std::vector<std::vector<GlobalAddress>> level_addrs;
//   uint64_t cur_off=0;
//   //std::cout<<"model sum"<<model_sum<<std::endl;
//   for(int i=0;i<level_models_local.size();i++)
//   {
//     std::vector<GlobalAddress> temp_addrs;
//     for(int j=0;j<level_models_local[i].size();j++)
//     {
//       GlobalAddress temp=models_gaddr;
//       temp.offset+=(cur_off);
//       cur_off+=sizeof(model_global);
//       temp_addrs.push_back(temp);
//     }
//     level_addrs.push_back(temp_addrs);
//   }
//   //std::cout<<"11111"<<std::endl;
//   for(int i=1;i<level_models_local.size();i++)
//   {
//     std::vector<model_global> temp_global_models;
//     for(int j=0;j<level_models_local[i].size();j++)
//     {
//       model_global temp_global_model;
//       temp_global_model.slope=level_models_local[i][j].slope;
//       temp_global_model.intercept=level_models_local[i][j].intercept;
//       temp_global_model.pos_start=level_models_local[i][j].pos_start;
//       temp_global_model.key_start=level_models_local[i][j].key_start;
//       temp_global_model.child_start=level_addrs[i-1][level_models_local[i][j].pos_start];
//       if(j<level_models_local[i].size()-1)
//       {
//         temp_global_model.sibling=level_addrs[i][j+1];
//       }
//       else
//       {
//         temp_global_model.sibling.val=0xffffffffffffffff;
//       }
//       temp_global_model.child_length=level_models_local[i][j].data_num;
//       temp_global_model.is_leaf=false;
//       temp_global_models.push_back(temp_global_model);
//     }
//     level_models_global.push_back(temp_global_models);
//   }
//   //std::cout<<"22222"<<std::endl;
//   for(int i=0;i<level_models_global.size();i++)
//   {
//     for(int j=0;j<level_models_global[i].size();j++)
//     {
//       //std::cout<<"level addrs:"<<level_addrs[i][j].offset<<std::endl;
//       model_global *now=(model_global*)(level_addrs[i][j].offset+(uint64_t)dCon->dsmPool);
//       //std::cout<<i<<"   "<<j<<std::endl;
//       now->child_length=level_models_global[i][j].child_length;
//       now->child_start=level_models_global[i][j].child_start;
//       now->intercept=level_models_global[i][j].intercept;
//       now->is_leaf=level_models_global[i][j].is_leaf;
//       now->key_start=level_models_global[i][j].key_start;
//       now->pos_start=level_models_global[i][j].pos_start;
//       now->sibling=level_models_global[i][j].sibling;
//       now->slope=level_models_global[i][j].slope;
//       now->threshold=level_models_global[i][j].threshold;
//       now->write_buffer_index=level_models_global[i][j].write_buffer_index;
//     }
//   }
//   //std::cout<<"333333"<<std::endl;
//   //sleep(1);
//   root=level_addrs[level_addrs.size()-1][0];
//   model_global *root_model_ptr=(model_global*)(level_addrs[level_addrs.size()-1][0].offset+(uint64_t)dCon->dsmPool);
//   root_model_ptr->full_model_num=model_sum; 

//   //link the model to the slot
//   //sleep(5);
//   //std::cout<<"link the model to the slot"<<std::endl;
//   uint64_t *ptr = (uint64_t *)slot_;
//   uint64_t old=slot_->next;
//   uint64_t new_=root.val|model_flag;
//   //std::cout<<"root link:"<<gaddr2str(root)<<std::endl;
//   GlobalAddress old_slot_gaddr;
//   old_slot_gaddr.val=(old&mask_);
//   list_node_local *old_head_now=old_head;
//   list_node_local *new_head_now=(list_node_local*)(old_slot_gaddr.offset+(uint64_t)dCon->dsmPool);
//   handle_new_insert(level_models_global,old_head_now,new_head_now,write_buffer_index,addr_to_free);
//   old_head_now=new_head_now;
//   while(!__sync_bool_compare_and_swap(ptr,old,new_))
//   {
//     old=slot_->next;
//     GlobalAddress new_head_gaddr;
//     new_head_gaddr.val=old;
//     new_head_now=(list_node_local*)(new_head_gaddr.offset+(uint64_t)dCon->dsmPool);
//     handle_new_insert(level_models_global,old_head_now,new_head_now,write_buffer_index,addr_to_free);
//     old_head_now=new_head_now;
//   }
//   uint64_t *tail_ptr=write_buffer_local_.tail_ptr;
//   //std::cout<<"to free slot num "<<kvs.size()<<std::endl;
//   uint64_t to_add=write_buffer_local_.one*(addr_to_free.size());
//   int queue_index=(*tail_ptr)>>write_buffer_local_.one_exp;
//   for(int i=0;i<addr_to_free.size();i++)
//   {
//     uint64_t slot_index=(addr_to_free[i].offset+(uint64_t)dCon->dsmPool-(uint64_t)write_buffer_local_.list_node_local_ptr)/sizeof(list_node_local);
//     write_buffer_local_.queue_ptr[queue_index]=slot_index;
//     queue_index=(queue_index+1)%write_buffer_local_.buffer_size;
//   }
//   __sync_fetch_and_add(tail_ptr,to_add);
//   return;
// }

// void Directory::build_node_set(int write_buffer_index,slot_local *slot_,list_node_local *old_head,std::vector<GlobalAddress> &nodes_gaddr)
// {
//   //std::cout<<"build node set"<<std::endl;
//   write_buffer_local write_buffer_local_=write_buffer_local_infos[write_buffer_index];
//   GlobalAddress set_gaddr=chunckAlloc->alloc_size(sizeof(node_set));
//   node_set *node_set_=(node_set*)(set_gaddr.offset+(uint64_t)dCon->dsmPool);
//   assert(nodes_gaddr.size()<=define::min_model_size);
//   node_set_->node_num=nodes_gaddr.size();
//   for(int i=nodes_gaddr.size()-1;i>=0;i--)
//   {
//     node_set_->nodes_gaddr[nodes_gaddr.size()-1-i]=nodes_gaddr[i];
//   }
//   node_set_->next_offset=tail_flag|((uint64_t)slot_-(uint64_t)dCon->dsmPool);
//   uint64_t *ptr = (uint64_t *)slot_;
//   uint64_t old=slot_->next;
//   GlobalAddress list_node_gaddr;
//   list_node_gaddr.val=(slot_->next)&mask_;
//   list_node_local *head=(list_node_local*)(list_node_gaddr.offset+(uint64_t)dCon->dsmPool);
//   if(head==old_head)
//   {
//     uint64_t new_=set_gaddr.val|set_flag;
//     bool ret=__sync_bool_compare_and_swap(ptr,old,new_);
//     if(ret)
//     {
//       return;
//     }
//   }
//   else
//   {
//     list_node_local *now=head;
//     list_node_local *pre=nullptr;
//     while(now!=old_head)
//     {
//       list_node_local *temp=(list_node_local*)((uint64_t)write_buffer_local_.list_node_local_ptr+(now->next_offset&mask_)*sizeof(list_node_local));
//       pre=now;
//       now=temp;
//     }
//     pre->next_offset=set_gaddr.val|set_flag;
//   }
//   return;
// }


// void Directory::add_node_set(int write_buffer_index,slot_local *slot_,list_node_local *old_head,std::vector<GlobalAddress> &nodes_gaddr,node_set *node_set_)
// {
//   //std::cout<<"add node set"<<std::endl;
//   write_buffer_local write_buffer_local_=write_buffer_local_infos[write_buffer_index];
//   std::vector<GlobalAddress> addr_to_free;

//   //GlobalAddress set_gaddr=chunckAlloc->alloc_size(sizeof(node_set));
//   //node_set *node_set_=(node_set*)(set_gaddr.offset+(uint64_t)dCon->dsmPool);
//   if(node_set_->node_num+nodes_gaddr.size()>=define::min_model_size)
//   {
//     std::vector<kv> kvs;
//     kv start_kv;
//     start_kv.key=slot_->key;
//     start_kv.val=slot_->val;
//     kvs.push_back(start_kv);
//     for(int i=0;i<nodes_gaddr.size();i++)
//     {
//       list_node_local *now=(list_node_local*)(nodes_gaddr[i].offset+(uint64_t)dCon->dsmPool);
//       kv temp_kv;
//       temp_kv.key=now->key;
//       temp_kv.val=now->val;
//       temp_kv.gaddr=nodes_gaddr[i];
//       temp_kv.index=(-i-1);
//       addr_to_free.push_back(nodes_gaddr[i]);
//       kvs.push_back(temp_kv);
//     }
//     for(int i=node_set_->node_num-1;i>=0;i--)
//     {
//       GlobalAddress now_node_gaddr;
//       now_node_gaddr.val=node_set_->nodes_gaddr[i];
//       list_node_local *now=(list_node_local*)(now_node_gaddr.offset+(uint64_t)dCon->dsmPool);
//       kv temp_kv;
//       temp_kv.key=now->key;
//       temp_kv.val=now->val;
//       temp_kv.gaddr=now_node_gaddr;
//       temp_kv.index=i;
//       addr_to_free.push_back(now_node_gaddr);
//       kvs.push_back(temp_kv);
//     }
//     std::stable_sort(kvs.begin(),kvs.end(),comparebykey);
//     auto tail=std::unique(kvs.begin(),kvs.end(),uniquebykey);
//     int len=tail-kvs.begin();
//     if(len>=define::min_model_size)
//     {
//       //std::cout<<"tick"<<std::endl;
//       GlobalAddress root;
//       kvs.erase(tail,kvs.end());
//       build_retrain_model(write_buffer_index,slot_,old_head,kvs,root,addr_to_free);
//     }
//     else
//     {
//       //std::cout<<"tick"<<std::endl;
//       kvs.erase(kvs.begin());
//       len=len-1;
//       std::vector<GlobalAddress> new_addr_to_free;
//       std::vector<int> free_index;
//       std::vector<int> len_flag(kvs.size(),1);
//       std::vector<int> len_flag_1(kvs.size(),1);
//       for(int i=0;i<len;i++)
//       {
//         if(kvs[i].index>=0)
//         {
//           len_flag[kvs[i].index]=0;
//         }
//         if(kvs[i].index<0)
//         {
//           len_flag_1[(-kvs[i].index)-1]=0;
//         }
//       }
//       for(int i=0;i<node_set_->node_num;i++)
//       {
//         //std::cout<<"free"<<
//         if(len_flag[i]==1)
//         {
//           GlobalAddress  temp;
//           temp.val=node_set_->nodes_gaddr[i];
//           new_addr_to_free.push_back(temp);
//           node_set_->nodes_gaddr[i]=null_next;
//         }
//       }
//       for(int i=0;i<nodes_gaddr.size();i++)
//       {
//         if(len_flag_1[i]==1)
//         {
//           new_addr_to_free.push_back(nodes_gaddr[i]);
//         }
//       }
//       for(int i=0;i<len;i++)
//       {
//         if(len_flag[i]==1)
//         {
//           //std::cout<<"free"<<i<<std::endl;
//           free_index.push_back(i);
//         }
//       }
//       int cur_free=0;
//       for(int i=0;i<len;i++)
//       {
//         if(kvs[i].index>=len||kvs[i].index<0)
//         {
//           node_set_->nodes_gaddr[free_index[cur_free]]=kvs[i].gaddr;
//           cur_free++;
//           //std::cout<<"fill free"<<free_index[cur_free]<<std::endl;
//         }
//       }
//       node_set_->node_num=len;
//       GlobalAddress set_gaddr;
//       set_gaddr.nodeID=nodeID;
//       set_gaddr.offset=(uint64_t(node_set_)-(uint64_t)dCon->dsmPool);
//       uint64_t *ptr = (uint64_t *)slot_;
//       uint64_t old=slot_->next;
//       GlobalAddress list_node_gaddr;
//       list_node_gaddr.val=(slot_->next)&mask_;
//       list_node_local *head=(list_node_local*)(list_node_gaddr.offset+(uint64_t)dCon->dsmPool);
//       if(head==old_head)
//       {
//         uint64_t new_=set_gaddr.val|set_flag;
//         bool ret=__sync_bool_compare_and_swap(ptr,old,new_);
//         if(ret)
//         {
//           return;
//         }
//       }
//       else
//       {
//         list_node_local *now=head;
//         list_node_local *pre=nullptr;
//         while(now!=old_head)
//         {
//           list_node_local *temp=(list_node_local*)((uint64_t)write_buffer_local_.list_node_local_ptr+(now->next_offset&mask_)*sizeof(list_node_local));
//           pre=now;
//           now=temp;
//         }
//         pre->next_offset=set_gaddr.val|set_flag;
//       }
//       uint64_t *tail_ptr=write_buffer_local_.tail_ptr;
//       uint64_t to_add=write_buffer_local_.one*(new_addr_to_free.size());
//       int queue_index=(*tail_ptr)>>write_buffer_local_.one_exp;
//       for(int i=0;i<new_addr_to_free.size();i++)
//       {
//         uint64_t slot_index=(new_addr_to_free[i].offset+(uint64_t)dCon->dsmPool-(uint64_t)write_buffer_local_.list_node_local_ptr)/sizeof(list_node_local);
//         write_buffer_local_.queue_ptr[queue_index]=slot_index;
//         queue_index=(queue_index+1)%write_buffer_local_.buffer_size;
//       }
//       __sync_fetch_and_add(tail_ptr,to_add);
//     }
//     return;
//   }
//   else
//   {
//     GlobalAddress set_gaddr;
//     set_gaddr.nodeID=nodeID;
//     set_gaddr.offset=(uint64_t(node_set_)-(uint64_t)dCon->dsmPool);
//     int start=node_set_->node_num;
//     //node_set_->node_num+=nodes_gaddr.size();
//     for(int i=nodes_gaddr.size()-1;i>=0;i--)
//     {
//       node_set_->nodes_gaddr[start+nodes_gaddr.size()-1-i]=nodes_gaddr[i];
//     }
//     node_set_->node_num+=nodes_gaddr.size();
//     node_set_->next_offset=tail_flag|((uint64_t)slot_-(uint64_t)dCon->dsmPool);
//     uint64_t *ptr = (uint64_t *)slot_;
//     uint64_t old=slot_->next;
//     GlobalAddress list_node_gaddr;
//     list_node_gaddr.val=(slot_->next)&mask_;
//     list_node_local *head=(list_node_local*)(list_node_gaddr.offset+(uint64_t)dCon->dsmPool);
//     if(head==old_head)
//     {
//       uint64_t new_=set_gaddr.val|set_flag;
//       bool ret=__sync_bool_compare_and_swap(ptr,old,new_);
//       if(ret)
//       {
//         return;
//       }
//     }
//     else
//     {
//       list_node_local *now=head;
//       list_node_local *pre=nullptr;
//       while(now!=old_head)
//       {
//         list_node_local *temp=(list_node_local*)((uint64_t)write_buffer_local_.list_node_local_ptr+(now->next_offset&mask_)*sizeof(list_node_local));
//         pre=now;
//         now=temp;
//       }
//       pre->next_offset=set_gaddr.val|set_flag;
//     }
//     return;
//   }
// }

// void Directory::retrain_buffer(int write_buffer_index)
// {
//   write_buffer_local write_buffer_local_=write_buffer_local_infos[write_buffer_index];
//   uint64_t head=(*write_buffer_local_.head_ptr)>>write_buffer_local_.one_exp;
//   uint64_t tail=(*write_buffer_local_.tail_ptr)>>write_buffer_local_.one_exp;
//   list_node_local *list_node_start=write_buffer_local_.list_node_local_ptr;
//   int64_t occ_num=0;
//   std::vector<uint64_t> head_set;
//   //std::vector<GlobalAddress> addr_to_free;
//   std::vector<int> occ_flag(buffer_size,0);
//   if(tail<head)
//   {
//     occ_num=(tail-head);
//     occ_num+=(buffer_size-1);
//   }
//   else
//   {
//     occ_num=(tail-head);
//   }
//   uint64_t now=head;
//   while(now!=tail)
//   {
//     int now_off=write_buffer_local_.queue_ptr[now];
//     occ_flag[now_off]=1;
//     now=(now+1)%buffer_size;
//   }
//   for(int i=0;i<buffer_size;i++)
//   {
//     uint64_t t_flag=list_node_start[i].next_offset&tail_flag;
//     if(t_flag!=0&&occ_flag[i]==0)
//     {
//       //std::cout<<"tail key: "<<list_node_start[i].key<<"tail val:"<<list_node_start[i].val<<"tail next: "<<std::hex<<list_node_start[i].next_offset<<std::endl;
//       GlobalAddress slot_;
//       slot_.val=(list_node_start[i].next_offset&mask_);
//       head_set.push_back(slot_.offset+(uint64_t)dCon->dsmPool);
//     }
//   }
//   //std::cout<<"head set size:"<<head_set.size()<<std::endl;
//   for(int i=0;i<head_set.size();i++)
//   {
//     slot_local *slot_ptr=(slot_local*)head_set[i];
//     if((slot_ptr->next&set_flag)!=0)
//     {
//       continue;
//     }
//     //std::cout<<"slot offset: "<<head_set[i]<<std::endl;
//     //std::cout<<"slot key:"<<slot_ptr->key<<"slot val: "<<slot_ptr->val<<"slot next: "<<std::hex<<slot_ptr->next<<std::endl;
//     GlobalAddress list_node_gaddr;
//     list_node_gaddr.val=(slot_ptr->next)&mask_;
//     list_node_local *head=(list_node_local*)(list_node_gaddr.offset+(uint64_t)dCon->dsmPool);
//     list_node_local *now=head;
//     std::vector<GlobalAddress> nodes_gaddr;
//     std::vector<kv> kvs;
//     int length=0;
//     bool is_set=false;
//     node_set *node_set_=nullptr;
//     //std::cout<<"chain length before: "<<length<<std::endl;
//     //std::cout<<"head key: "<<head->key<<" head val: "<<head->val<<" head next: "<<std::hex<<head->next_offset<<std::endl;
//     while((now->next_offset&tail_flag)==0)
//     {
//       //std::cout<<"next off"<<(now->next_offset&mask_)<<std::endl;
//       length++;
//       GlobalAddress now_node_gaddr;
//       now_node_gaddr.nodeID=nodeID;
//       now_node_gaddr.offset=((uint64_t)now-(uint64_t)dCon->dsmPool);
//       nodes_gaddr.push_back(now_node_gaddr);
//       kv temp_kv;
//       temp_kv.key=now->key;
//       temp_kv.val=now->val;
//       temp_kv.gaddr=now_node_gaddr;
//       kvs.push_back(temp_kv);
//       if((now->next_offset&set_flag)!=0)
//       {
//         is_set=true;
//         break;
//       }
//       list_node_local *temp=(list_node_local*)((uint64_t)write_buffer_local_.list_node_local_ptr+(now->next_offset&mask_)*sizeof(list_node_local));
//       now=temp;
//     }

//     if(is_set)
//     {
//       GlobalAddress seg_gaddr;
//       seg_gaddr.val=now->next_offset&mask_;
//       node_set_=(node_set*)(seg_gaddr.offset+(uint64_t)dCon->dsmPool);
//       add_node_set(write_buffer_index,slot_ptr,head,nodes_gaddr,node_set_);
//       continue;
//     }
//     //std::cout<<"chain length: "<<length<<std::endl;
//     GlobalAddress now_node_gaddr;
//     now_node_gaddr.nodeID=nodeID;
//     now_node_gaddr.offset=((uint64_t)now-(uint64_t)dCon->dsmPool);
//     nodes_gaddr.push_back(now_node_gaddr);
//     kv temp_kv;
//     temp_kv.key=now->key;
//     temp_kv.val=now->val;
//     temp_kv.gaddr=now_node_gaddr;
//     kvs.push_back(temp_kv);
//     length++;
//     if(length>=define::max_chain_length&&length<define::min_model_size)
//     {
//       build_node_set(write_buffer_index,slot_ptr,head,nodes_gaddr);
//       continue;
//     }
//     else if(length>=define::min_model_size)
//     {
//       //std::cout<<"tick model"<<std::endl;
//       kv start_kv;
//       start_kv.key=slot_ptr->key;
//       start_kv.val=slot_ptr->val;
//       kvs.push_back(start_kv);
//       std::stable_sort(kvs.begin(),kvs.end(),comparebykey);
//       kvs.erase(std::unique(kvs.begin(),kvs.end(),uniquebykey),kvs.end());
//       GlobalAddress root;
//       build_retrain_model(write_buffer_index,slot_ptr,head,kvs,root,nodes_gaddr);
//       continue;
//     }
//   }
// }

/*void Directory::retrain_chain(slot_local *slot,int write_buffer_index)
{
  if((slot->next&model_flag)!=0)
  {
    return;
  }
  int i=slot->key;
  write_buffer_local write_buffer_local_=write_buffer_local_infos[write_buffer_index];
  GlobalAddress first_node;
  first_node.val=(slot->next)&mask_;
  //std::cout<<"retrain: "<<slot->key<<" "<<slot->next<<std::endl;
  list_node_local *head=(list_node_local*)(first_node.offset+(uint64_t)dCon->dsmPool);
  list_node_local *now=head;
  std::vector<kv> kvs;
  while((now->next_offset&tail_flag)==0)
  {
    kv temp_kv;
    temp_kv.key=now->key;
    temp_kv.val=now->val;
    kvs.push_back(temp_kv);
    list_node_local *temp=(list_node_local*)((uint64_t)write_buffer_local_.list_node_local_ptr+(now->next_offset&mask_)*sizeof(list_node_local));
    now=temp;
  }
  kv temp_kv;
  temp_kv.key=now->key;
  temp_kv.val=now->val;
  kvs.push_back(temp_kv);
  std::stable_sort(kvs.begin(),kvs.end(),comparebykey);
  kvs.erase(std::unique(kvs.begin(),kvs.end(),uniquebykey),kvs.end());
  GlobalAddress root;
  build_retrain_model(write_buffer_index,slot,head,kvs,root);
}*/

//modify the insert opration, no need to write the ori kv in the first write


// void Directory::retrain_thread()
// {
//   bindCore(23 - dirID-1);
//   sleep(5);
//   while(1)
//   {
//     //std::cout<<"in"<<std::endl;
//     //std::cout<<"buffer num:"<<buffer_num<<std::endl;
//     for(int i=0;i<buffer_num;i++)
//     {

//       int app_size=retrain_chain_queue[i]->size_approx();
//       for(int j=0;j<app_size;j++)
//       {
//         //std::cout<<"chain retrain"<<std::endl;
//         GlobalAddress slot_gaddr;
//         retrain_chain_queue[i]->try_dequeue(slot_gaddr);
//         slot_local *slot_=(slot_local*)(slot_gaddr.offset+(uint64_t)dCon->dsmPool);
//         //retrain_chain(slot_,i);
//       }
//       retrain_buffer(i);
//       //std::cout<<"buffer num:"<<buffer_num<<std::endl;
//       /*if(retrain_buffer_flag[i]!=0)
//       {
//         //std::cout<<"buffer retrain"<<std::endl;
//         //std::cout<<"retrain request: "<<retrain_buffer_flag[i]<<std::endl;
//         int app_size=retrain_chain_queue[i]->size_approx();
//         retrain_buffer(i);
//         GlobalAddress slot_gaddr;
//         for(int j=0;j<app_size;j++)
//         {
//           retrain_chain_queue[i]->try_dequeue(slot_gaddr);
//         }
//         retrain_buffer_flag[i]=0;
//       }
//       else
//       {
//         int app_size=retrain_chain_queue[i]->size_approx();
//         for(int j=0;j<app_size;j++)
//         {
//           std::cout<<"chain retrain"<<std::endl;
//           GlobalAddress slot_gaddr;
//           retrain_chain_queue[i]->try_dequeue(slot_gaddr);
//           slot_local *slot_=(slot_local*)(slot_gaddr.offset+(uint64_t)dCon->dsmPool);
//           retrain_chain(slot_,i);
//         }
//       }*/
//     }
//   }
// }