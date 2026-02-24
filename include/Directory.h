#ifndef __DIRECTORY_H__
#define __DIRECTORY_H__

#include <thread>

#include <unordered_map>

#include "Common.h"
#include "concurrentqueue.h"
#include "Connection.h"
#include "GlobalAllocator.h"
#include "write_buffer_local.h"
#include "leaf.h"



class Directory {
public:
  Directory(DirectoryConnection *dCon, RemoteConnection *remoteInfo,
            uint32_t machineNR, uint16_t dirID, uint16_t nodeID);

  ~Directory();

private:
  DirectoryConnection *dCon;
  RemoteConnection *remoteInfo;

  uint32_t machineNR;
  uint16_t dirID;
  uint16_t nodeID;
  uint64_t base_addr;

  std::thread *dirTh;

  GlobalAllocator *chunckAlloc;
  int buffer_num;
  int buffer_size;
  
  std::vector<write_buffer_local> write_buffer_local_infos;
  std::vector<int> retrain_buffer_flag;
  //moodycamel::ConcurrentQueue<GlobalAddress> retrain_chain_queue;
  std::vector<moodycamel::ConcurrentQueue<GlobalAddress>*> retrain_chain_queue;

  void dirThread();

  void handle_new_insert(std::vector<std::vector<model_global>>&level_models_global, list_node_local *old_head,list_node_local *new_head,int write_buffer_index,std::vector<GlobalAddress> &addr_to_free);

  void build_retrain_model(int write_buffer_index,slot_local *slot_,list_node_local *old_head,std::vector<kv> &kvs,GlobalAddress &root,std::vector<GlobalAddress> &addr_to_free);

  void build_node_set(int write_buffer_index,slot_local *slot_,list_node_local *old_head,std::vector<GlobalAddress> &nodes_gaddr);

  void add_node_set(int write_buffer_index,slot_local *slot_,list_node_local *old_head,std::vector<GlobalAddress> &nodes_gaddr,node_set *node_set_);

  void retrain_buffer(int write_buffer_index);

  void retrain_chain(slot_local *slot,int write_buffer_index);

  void retrain_thread();

  void sendData2App(const RawMessage *m);

  void process_message(const RawMessage *m);

  

public:
  void set_buffer_num(int buffer_num_)
  {
    buffer_num=buffer_num_;
    for(int i=0;i<buffer_num_;i++)
    {
      std::cout<<"buffer num: "<<buffer_num<<std::endl;
      retrain_buffer_flag.push_back(0);
      moodycamel::ConcurrentQueue<GlobalAddress> *temp=new moodycamel::ConcurrentQueue<GlobalAddress>();
      retrain_chain_queue.push_back(temp);
    }
    std::cout<<"retrain chain queue init already"<<std::endl;
  }

  void set_write_buffer_info(std::vector<uint64_t> write_buffer_info_,int buffer_size_)
  {
    buffer_size=buffer_size_;
    
    for(int i=0;i<write_buffer_info_.size();i++)
    {
      write_buffer_local temp(write_buffer_info_[i]+(uint64_t)dCon->dsmPool,buffer_size_);
      write_buffer_local_infos.push_back(temp);
    }
    std::cout<<"retrain buffer init already"<<std::endl;
    return;
  }

};

#endif /* __DIRECTORY_H__ */
