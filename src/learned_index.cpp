#include "leaf.h"
#include <random>

uint64_t lock_fail[MAX_APP_THREAD] = {0};
uint64_t try_lock_op[MAX_APP_THREAD] = {0};
uint64_t write_handover_num[MAX_APP_THREAD] = {0};
uint64_t read_handover_num[MAX_APP_THREAD] = {0};
uint64_t latency[MAX_APP_THREAD][MAX_CORO_NUM][LATENCY_WINDOWS] = {0};
volatile bool need_stop = false;



//write the data segment to remote, return a vector of incomplelte level 0 global_model
using OptimalPLR = PLR<u_int64_t, u_int64_t>;
void make_segment(const std::vector<uint64_t> &keys, const std::vector<uint64_t> &vals, uint64_t Epsilon, std::vector<model_local> &models)
{
    assert(keys.size() == vals.size());
    if(keys.size()==0) return;
    //std::cout << "Training data: "<<keys[0]<<", Epsilon: "<<Epsilon;
    OptimalPLR* opt = new OptimalPLR(Epsilon-1);
    uint64_t p = keys[0];
    uint64_t data_num = 0;
    size_t pos=0;
    size_t pos_start=0;
    opt->add_point(p, pos);
    data_num++;
    auto k_iter = keys.begin();
    auto v_iter = vals.begin();
    for(int i=1; i<keys.size(); i++) {
      uint64_t next_p = keys[i];
      if (next_p == p){
        std::cout<<"DUPLICATE keys";
        exit(0);
      }
      p = next_p;
      pos++;
      if(!opt->add_point(p, pos)) {
        auto cs = opt->get_segment();
        auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
        model_local model_temp;
        model_temp.slope=cs_slope;
        model_temp.intercept=cs_intercept;
        model_temp.key_start=*k_iter;
        model_temp.data_num=data_num;
        model_temp.pos_start=pos_start;
        models.push_back(model_temp);
        k_iter += pos;
        v_iter += pos;
        pos=0;
        pos_start=i;
        opt = new OptimalPLR(Epsilon-1);
        opt->add_point(p, pos);
        data_num=0;
      }
      data_num++;
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    model_local model_temp;
    model_temp.slope=cs_slope;
    model_temp.intercept=cs_intercept;
    model_temp.key_start=*k_iter;
    model_temp.data_num=data_num;
    model_temp.pos_start=pos_start;
    models.push_back(model_temp);
    for(int i=0;i<models.size()-1;i++)
    {
      models[i].sibling=&models[i+1];
    }
    models[models.size()-1].sibling=nullptr;
    uint64_t total_size = models.size();
    //std::cout << "Training models: "<<total_size<<std::endl;
}

void make_segment_for_kvs(const std::vector<kv> &kvs, uint64_t Epsilon, std::vector<model_local> &models)
{
    if(kvs.size()==0) return;
    //std::cout << "Training data: "<<kvs.size()<<", Epsilon: "<<Epsilon;
    OptimalPLR* opt = new OptimalPLR(Epsilon-1);
    uint64_t p = kvs[0].key;
    uint64_t data_num = 0;
    size_t pos=0;
    size_t pos_start=0;
    opt->add_point(p, pos);
    data_num++;
    auto kv_iter = kvs.begin();
    for(int i=1; i<kvs.size(); i++) {
      uint64_t next_p = kvs[i].key;
      if (next_p == p){
        std::cout<<"DUPLICATE keys";
        exit(0);
      }
      p = next_p;
      pos++;
      if(!opt->add_point(p, pos)) {
        auto cs = opt->get_segment();
        auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
        model_local model_temp;
        model_temp.slope=cs_slope;
        model_temp.intercept=cs_intercept;
        model_temp.key_start=(*kv_iter).key;
        model_temp.data_num=data_num;
        model_temp.pos_start=pos_start;
        models.push_back(model_temp);
        kv_iter += pos;
        pos=0;
        pos_start=i;
        opt = new OptimalPLR(Epsilon-1);
        opt->add_point(p, pos);
        data_num=0;
      }
      data_num++;
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    model_local model_temp;
    model_temp.slope=cs_slope;
    model_temp.intercept=cs_intercept;
    model_temp.key_start=(*kv_iter).key;
    model_temp.data_num=data_num;
    model_temp.pos_start=pos_start;
    models.push_back(model_temp);
    for(int i=0;i<models.size()-1;i++)
    {
      models[i].sibling=&models[i+1];
    }
    models[models.size()-1].sibling=nullptr;
    uint64_t total_size = models.size();
    //std::cout << "Training models: "<<total_size<<std::endl;
}

void make_segment_par(const std::vector<uint64_t> &keys, uint64_t Epsilon, std::vector<model_local>& models)
{
    if(keys.size()==0) return;
    //std::cout << "Training data: "<<keys.size()<<", Epsilon: "<<Epsilon;

    OptimalPLR* opt = new OptimalPLR(Epsilon-1);
    uint64_t p = keys[0];
    uint64_t data_num=0;
    size_t pos=0;
    size_t pos_start=0;
    opt->add_point(p, pos);
    data_num++;
    auto k_iter = keys.begin();
    for(int i=1; i<keys.size(); i++) {
      uint64_t next_p = keys[i];
      if (next_p == p){
        std::cout<<"DUPLICATE keys";
        exit(0);
      }
      p = next_p;
      pos++;
      if(!opt->add_point(p, pos)) {
        auto cs = opt->get_segment();
        auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
        model_local model_temp;
        model_temp.slope=cs_slope;
        model_temp.intercept=cs_intercept;
        model_temp.key_start=*k_iter;
        model_temp.data_num=data_num;
        model_temp.pos_start=pos_start;
        models.push_back(model_temp);
        k_iter += pos;
        pos=0;
        pos_start=i;
        opt = new OptimalPLR(Epsilon-1);
        opt->add_point(p, pos);
        data_num=0;
      }
      data_num++;
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    model_local model_temp;
    model_temp.slope=cs_slope;
    model_temp.intercept=cs_intercept;
    model_temp.key_start=*k_iter;
    model_temp.data_num=data_num;
    model_temp.pos_start=pos_start;
    models.push_back(model_temp);
    for(int i=0;i<models.size()-1;i++)
    {
      models[i].sibling=&models[i+1];
    }
    models[models.size()-1].sibling=nullptr;
    uint64_t total_size = models.size();
    //std::cout << "Training models: "<<total_size<<std::endl;
}

void make_segment_with_empty(const std::vector<uint64_t> &keys, const std::vector<uint64_t> &vals, 
                            uint64_t Epsilon, std::vector<model_local> &models, 
                            std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals)
{
    uint64_t nega_split_num = 0;
    uint64_t posi_split_num = 0;
    assert(keys.size() == vals.size());
    if(keys.size()==0) return;
    //std::cout << "Training data: "<<keys[0]<<", Epsilon: "<<Epsilon<<std::endl;
    OptimalPLR* opt = new OptimalPLR(Epsilon-1);
    std::vector<uint64_t> temp_keys;
    std::vector<uint64_t> temp_vals;
    uint64_t empty_slot_sum = 0;
    uint64_t p = keys[0];
    uint64_t data_num = 0;
    uint64_t used_empty_slot_num = 0;
    size_t out_limit_error = 2;
    size_t try_empty_dif = Epsilon * 0.5;
    size_t untry_empty_dif = Epsilon * 1.5;
    //size_t retry_limit = 1;
    size_t pos=0;
    //size_t pos_start=0;
    opt->add_point(p, pos);
    temp_keys.push_back(p);
    temp_vals.push_back(p);
    data_num++;
    auto k_iter = keys.begin();
    auto v_iter = vals.begin();
    for(int i=1; i<keys.size(); i++) {
      uint64_t next_p = keys[i];
      if (next_p == p){
        std::cout<<"DUPLICATE keys";
        exit(0);
      }
      p = next_p;
      pos++;
      auto cs = opt->get_segment();
      auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
      long double predict_pos = p * cs_slope + cs_intercept;
      long double pos_dif = -(pos - predict_pos);
      int pos_dif_int = (int)pos_dif;
      if(pos_dif < untry_empty_dif && pos_dif > try_empty_dif && (used_empty_slot_num + pos_dif_int / 2) < data_num * 0.3)
      {
        used_empty_slot_num += pos_dif_int / 2;
        empty_slot_sum += pos_dif_int / 2;
        for(int i = 0; i < pos_dif_int / 2; i++)
        {
          temp_keys.push_back(null_flag);
          temp_vals.push_back(null_flag);
        }
        pos += pos_dif_int / 2;
      }
      bool ret = opt->add_point(p, pos);
      if(ret == false) {
        auto cs = opt->get_segment();
        auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
        if(pos_dif > 0)
        {
          nega_split_num++;
        }
        else
        {
          posi_split_num++;
        }
        model_local model_temp;
        model_temp.slope=cs_slope;
        model_temp.intercept=cs_intercept;
        model_temp.key_start=*k_iter;
        model_temp.data_num=temp_keys.size();
        models.push_back(model_temp);
        std::vector<uint64_t> keys_pushback(temp_keys.begin(), temp_keys.end());
        std::vector<uint64_t> vals_pushback(temp_vals.begin(), temp_vals.end());
        seg_keys.push_back(keys_pushback);
        seg_vals.push_back(vals_pushback);
        temp_keys.clear();
        temp_vals.clear();
        k_iter += data_num;
        v_iter += data_num;
        pos=0;
        opt = new OptimalPLR(Epsilon-1);
        opt->add_point(p, pos);
        temp_keys.push_back(p);
        temp_vals.push_back(p);
        data_num=0;
        used_empty_slot_num = 0;
      }
      else
      {
        temp_keys.push_back(p);
        temp_vals.push_back(p);
      }
      data_num++;
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    model_local model_temp;
    model_temp.slope=cs_slope;
    model_temp.intercept=cs_intercept;
    model_temp.key_start=*k_iter;
    model_temp.data_num=temp_keys.size();
    models.push_back(model_temp);
    std::vector<uint64_t> keys_pushback(temp_keys.begin(), temp_keys.end());
    std::vector<uint64_t> vals_pushback(temp_vals.begin(), temp_vals.end());
    seg_keys.push_back(keys_pushback);
    seg_vals.push_back(vals_pushback);
    for(int i=0;i<models.size()-1;i++)
    {
      models[i].sibling=&models[i+1];
    }
    models[models.size()-1].sibling=nullptr;
    // std::cout<<"empty slot num:"<<empty_slot_sum<<std::endl;
    // std::cout<<"nega split:"<<nega_split_num<<"posi split:"<<posi_split_num<<std::endl;
    //std::cout << "Training models: "<<total_size<<std::endl;
}

void make_segment_emptyPLR_oneloop(std::vector<uint64_t> &keys, std::vector<uint64_t> &vals, uint64_t Epsilon, 
                                    int loop, uint64_t empty_slot_upper, uint64_t &empty_slot_used)
{
  std::vector<uint64_t> temp_keys;
  std::vector<uint64_t> temp_vals;
  temp_keys.reserve(keys.size() * 1.5); 
  temp_vals.reserve(vals.size() * 1.5);
  uint64_t nega_split_num = 0;
  uint64_t posi_split_num = 0;
  if(keys.size()==0) return;
  OptimalPLR* opt = new OptimalPLR(Epsilon-1);
  uint64_t empty_slot_sum = 0;
  uint64_t data_num = 0;
  uint64_t used_empty_slot_num = 0;
  size_t try_empty_dif = Epsilon * 0.5;
  try_empty_dif = 1;
  //if(loop != 0 && empty_slot_used < empty_slot_upper * 0.5) try_empty_dif = 1;
  // if(loop == 0) try_empty_dif = 1;
  // else if(loop == 1) try_empty_dif = Epsilon * 0.5;
  // std::cout<<"try empty dif in loop:"<<loop<<" is "<<try_empty_dif<<std::endl;
  size_t untry_empty_dif = Epsilon * 4;


  uint64_t p = keys[0];
  uint64_t v = vals[0];
  size_t pos=0;
  assert(p != null_flag);
  opt->add_point(p, pos);
  temp_keys.push_back(p);
  temp_vals.push_back(v);
  data_num++;

  for(int i=1; i<keys.size(); i++) {
    uint64_t next_p = keys[i];
    uint64_t next_v = vals[i];
    if (next_p == p && p != null_flag){
      std::cout<<"DUPLICATE keys";
      exit(0);
    }
    p = next_p;
    v = next_v;
    pos++;
    if(p == null_flag)
    {
      temp_keys.push_back(p);
      temp_vals.push_back(v);
      data_num++;
      continue;
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    long double predict_pos = p * cs_slope + cs_intercept;
    long double pos_dif = -(pos - predict_pos);
    int pos_dif_int = (int)pos_dif;
    if(pos_dif <= untry_empty_dif && pos_dif >= try_empty_dif 
      && (used_empty_slot_num + pos_dif_int) < data_num * 0.3 
      && empty_slot_sum + empty_slot_used < empty_slot_upper)
    {
      used_empty_slot_num += pos_dif_int;
      empty_slot_sum += pos_dif_int;
      for(int i = 0; i < pos_dif_int; i++)
      {
        temp_keys.push_back(null_flag);
        temp_vals.push_back(null_flag);
      }
      pos += pos_dif_int;
      data_num += pos_dif_int;
    }
    bool ret = opt->add_point(p, pos);
    if(ret == false) {
      if(pos_dif > 0)
      {
        nega_split_num++;
      }
      else
      {
        posi_split_num++;
      }
      pos=0;
      delete opt;
      opt = new OptimalPLR(Epsilon-1);
      opt->add_point(p, pos);
      temp_keys.push_back(p);
      temp_vals.push_back(v);
      data_num=0;
      used_empty_slot_num = 0;
    }
    else
    {
      temp_keys.push_back(p);
      temp_vals.push_back(v);
    }
    data_num++;
  }

//   std::cout<<"pre key size: "<<keys.size()<<std::endl;
//   std::cout<<"pre empty slot use: "<<empty_slot_used<<std::endl;

//   keys.clear();
//   vals.clear();
//   keys = temp_keys;
//   vals = temp_vals;
  keys.swap(temp_keys);
  vals.swap(temp_vals);
  empty_slot_used += empty_slot_sum;

  std::cout<<"key size: "<<keys.size()<<std::endl;
//   std::cout<<"after empty slot use: "<<empty_slot_used<<std::endl;

//   std::cout<<"empty slot add:"<<empty_slot_sum<<std::endl;
  std::cout<<"nega split:"<<nega_split_num<<"posi split:"<<posi_split_num<<std::endl;
  std::cout<<"model num: "<<nega_split_num + posi_split_num + 1<<std::endl;
  std::cout<<"------------------------------------"<<std::endl;
}

void make_segment_emptyPLR_multiloop(std::vector<uint64_t> &keys, std::vector<uint64_t> &vals, 
                                        uint64_t Epsilon, std::vector<model_local> &models,
                                        std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals)
{
    // auto print_memory_usage = []() {
    //     std::ifstream status("/proc/self/status");
    //     std::string line;
    //     while (std::getline(status, line)) {
    //         if (line.find("VmRSS") != std::string::npos || 
    //             line.find("VmSize") != std::string::npos) {
    //             std::cout << line << std::endl;
    //         }
    //     }
    // };
  uint64_t empty_slot_upper = keys.size() * 0.42;
  uint64_t empty_slot_used = 0;
  int loop = 0;
  std::vector<uint64_t> keys_with_empty(keys.begin(), keys.end());
  std::vector<uint64_t> vals_with_empty(vals.begin(), vals.end());
  float try_factor = 0.5;
  while(empty_slot_used < empty_slot_upper && loop < 5)
  {
    std::cout<<"------------MAF-PLR loop: "<<loop<<"-------------"<<std::endl;
    make_segment_emptyPLR_oneloop(keys_with_empty, vals_with_empty, Epsilon, loop, empty_slot_upper, empty_slot_used);
    loop++;
    // std::cout<<"keys size: "<<keys_with_empty.size()<<std::endl;
    // print_memory_usage();
  }

  uint64_t nega_split_num = 0;
  uint64_t posi_split_num = 0;
  assert(keys_with_empty.size() == vals_with_empty.size());
  if(keys_with_empty.size()==0) return;
  OptimalPLR* opt = new OptimalPLR(Epsilon-1);
  std::vector<uint64_t> temp_keys;
  std::vector<uint64_t> temp_vals;
  uint64_t p = keys_with_empty[0];
  uint64_t v = vals_with_empty[0];
  uint64_t data_num = 0;
  size_t pos=0;
  opt->add_point(p, pos);
  temp_keys.push_back(p);
  temp_vals.push_back(v);
  data_num++;
  auto k_iter = keys_with_empty.begin();
  for(int i=1; i<keys_with_empty.size(); i++) {
    uint64_t next_p = keys_with_empty[i];
    uint64_t next_v = vals_with_empty[i];
    if (next_p == p && p != null_flag){
      std::cout<<"DUPLICATE keys";
      exit(0);
    }
    p = next_p;
    v = next_v;
    pos++;
    if(p == null_flag)
    {
      temp_keys.push_back(p);
      temp_vals.push_back(v);
      data_num++;
      continue;
    }
    auto cs = opt->get_segment();
    auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
    long double predict_pos = p * cs_slope + cs_intercept;
    long double pos_dif = -(pos - predict_pos);
    int pos_dif_int = (int)pos_dif;
    bool ret = opt->add_point(p, pos);
    if(ret == false || temp_keys.size() > keys.size() * 0.05) {
      auto cs = opt->get_segment();
      auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
      if(pos_dif > 0)
      {
        nega_split_num++;
      }
      else
      {
        posi_split_num++;
      }
      model_local model_temp;
      model_temp.slope=cs_slope;
      model_temp.intercept=cs_intercept;
      model_temp.key_start=*k_iter;
      model_temp.data_num=temp_keys.size();
      models.push_back(model_temp);
      std::vector<uint64_t> keys_to_store;
      std::vector<uint64_t> vals_to_store;
      keys_to_store.swap(temp_keys);  // 交换，O(1)
      vals_to_store.swap(temp_vals);
      seg_keys.push_back(std::move(keys_to_store));
      seg_vals.push_back(std::move(vals_to_store));
      temp_keys.clear();
      temp_vals.clear();
      k_iter += data_num;
      pos=0;
      delete opt;
      opt = new OptimalPLR(Epsilon-1);
      opt->add_point(p, pos);
      temp_keys.push_back(p);
      temp_vals.push_back(v);
      data_num=0;
    }
    else
    {
      temp_keys.push_back(p);
      temp_vals.push_back(v);
    }
    data_num++;
  }
  auto cs = opt->get_segment();
  auto[cs_slope, cs_intercept] = cs.get_slope_intercept();
  model_local model_temp;
  model_temp.slope=cs_slope;
  model_temp.intercept=cs_intercept;
  model_temp.key_start=*k_iter;
  model_temp.data_num=temp_keys.size();
  models.push_back(model_temp);
  std::vector<uint64_t> keys_pushback(temp_keys.begin(), temp_keys.end());
  std::vector<uint64_t> vals_pushback(temp_vals.begin(), temp_vals.end());
  seg_keys.push_back(keys_pushback);
  seg_vals.push_back(vals_pushback);
  for(int i=0;i<models.size()-1;i++)
  {
    models[i].sibling=&models[i+1];
  }
  models[models.size()-1].sibling=nullptr;
  delete opt;
  std::cout<<"------------MAF-PLR final model info-------------"<<std::endl;
  std::cout<<"nega split:"<<nega_split_num<<"posi split:"<<posi_split_num<<std::endl;
  std::cout<<"model num: "<<nega_split_num + posi_split_num + 1<<std::endl;
  std::cout<<"-------------------------------------------------"<<std::endl;
  return;
}

void get_start_key_from_model(std::vector<uint64_t>&start_keys,std::vector<model_local>&models)
{
  for(int i=0;i<models.size();i++)
  {
    start_keys.push_back(models[i].key_start);
  }
  return;
}

void build_local(std::vector<std::vector<model_local>> &level_model,std::vector<model_local>leafs,uint64_t Epsilon)
{
  level_model.push_back(leafs);
  int level_num=0;
  while(level_model[level_num].size()>1)
  {
    std::vector<uint64_t> temp_key;
    get_start_key_from_model(temp_key,level_model[level_num]);
    std::vector<model_local> next_level;
    make_segment_par(temp_key,Epsilon,next_level);
    level_model.push_back(next_level);
    level_num++;
  }
  return;
}

void learned_index_local::build_local_with_empty(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals,
                                      std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals)
{
    std::vector<model_local> local_models;
    //make_segment_with_empty(keys, vals, Epsilon, local_models, seg_keys, seg_vals);
    make_segment_emptyPLR_multiloop(keys, vals, Epsilon, local_models, seg_keys, seg_vals);
    level_models.push_back(local_models);
    int level_num=0;
    while(level_models[level_num].size()>1)
    {
        std::vector<uint64_t> temp_key;
        get_start_key_from_model(temp_key,level_models[level_num]);
        std::vector<model_local> next_level;
        make_segment_par(temp_key,Epsilon,next_level);
        level_models.push_back(next_level);
        level_num++;
    }
    return;
}

std::string gaddr2str(GlobalAddress gaddr)
{
    std::string gaddr_str;
    int node_id=(int)gaddr.nodeID;
    uint64_t offset=gaddr.offset;
    gaddr_str=std::to_string(node_id);
    gaddr_str.append(":");
    gaddr_str.append(std::to_string(offset));
    return gaddr_str;
}

void print_model_info(std::vector<model_local>&models)
{
  int sum=0;
  for(int i=0;i<models.size();i++)
  {
    std::cout<<"k: "<<models[i].slope<<"b: "<<models[i].intercept<<" start key: "<<models[i].key_start<<" pre_pos: "<<models[i].slope*models[i].key_start+models[i].intercept
    <<" pos_start: "<<models[i].pos_start<<" data_num: "<<models[i].data_num<<std::endl;
    sum+=models[i].data_num;
  }
  std::cout<<"data_num: "<<sum<<std::endl;
}

void print_local_model(model_local model)
{ 
    std::cout<<"k: "<<model.slope<<"b: "<<model.intercept<<" start key: "<<model.key_start<<" pre_pos: "<<model.slope*model.key_start+model.intercept
    <<" pos_start: "<<model.pos_start<<" data_num: "<<model.data_num<<std::endl;
}

void print_global_model(model_global model)
{
    std::cout<<"k: "<<model.slope<<"b: "<<model.intercept<<" start key: "<<model.key_start<<" pre_pos: "<<model.slope*model.key_start+model.intercept
    <<" pos_start: "<<model.pos_start<<" data_num: "<<model.child_length<<" child_gaddr: "<<gaddr2str(model.child_start)<<std::endl;
}

void print_load_model(model_mincost model)
{
    std::cout<<"k: "<<model.slope<<"b: "<<model.intercept<<" start key: "<<model.key_start<<" pre_pos: "<<model.slope*model.key_start+model.intercept
    <<" data_num: "<<model.child_length<<" child_gaddr: "<<model.pos_start<<std::endl;
}


learned_index_local::learned_index_local(uint64_t Epsilon):Epsilon(Epsilon)
{
    
}

void learned_index_local::build_local(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals)
{
    std::vector<model_local> local_models;
    make_segment(keys,vals,Epsilon,local_models);
    level_models.push_back(local_models);
    int level_num=0;
    while(level_models[level_num].size()>1)
    {
        std::vector<uint64_t> temp_key;
        get_start_key_from_model(temp_key,level_models[level_num]);
        std::vector<model_local> next_level;
        make_segment_par(temp_key,Epsilon,next_level);
        level_models.push_back(next_level);
        level_num++;
    }
    return;
}

void learned_index_local::print_level_model()
{
    for(int i=0;i<level_models.size();i++)
    {
        std::cout<<"level: "<<i<<"model num:"<<level_models[i].size()<<std::endl;
        for(int j=0;j<level_models[i].size();j++)
        {
            print_local_model(level_models[i][j]);
        }
    }
    return;
}

learned_index_global::learned_index_global(DSM *dsm,uint64_t Epsilon,write_buffer_conf write_buffer_conf_):
    dsm(dsm),Epsilon(Epsilon),write_buffer_conf_(write_buffer_conf_)
{
    write_buffers=(write_buffer**)malloc(sizeof(write_buffer*)*write_buffer_conf_.buffer_num);
    uint64_t single_buffer_size=write_buffer_conf_.buffer_size/write_buffer_conf_.buffer_num;
    int write_buffer_off=0;
    for(int i=0;i<write_buffer_conf_.buffer_num;i++)
    {
        write_buffers[i]=new write_buffer(i,write_buffer_off,single_buffer_size,write_buffer_conf_.thread_num,dsm);
        write_buffer_off+=(sizeof(uint64_t)*2+sizeof(int)*single_buffer_size+sizeof(list_node)*single_buffer_size);
    }
    return;
}

learned_index_global::learned_index_global(){}

void learned_index_global::print_level_model()
{
    for(int i=0;i<level_models.size();i++)
    {
        std::cout<<"level: "<<i<<"model num:"<<level_models[i].size()<<std::endl;
        // for(int j=0;j<level_models[i].size();j++)
        // {
        //     print_global_model(level_models[i][j]);
        // }
    }
    
    int max_seg_length = 0;
    for(int i = 0; i < level_models[0].size(); i++)
    {
        max_seg_length = std::max(max_seg_length, (int)level_models[0][i].child_length);
    }

    std::cout<<"max data num"<<max_seg_length<<std::endl;
    
    return;
}

void learned_index_global::write_seg_remote(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals,std::vector<model_local> &local_models)
{
    char *cache_ptr=dsm->get_cache();
    slot *write_buf=(slot*)cache_ptr;
    std::vector<model_global> global_models_0;
    for(int i=0;i<local_models.size();i++)
    {
        for(int j=0;j<local_models[i].data_num;j++)
        {
            write_buf[j].next=null_flag;
            write_buf[j].key=keys[local_models[i].pos_start+j];
            write_buf[j].val=keys[local_models[i].pos_start+j];
            //write_buf[2*j]=keys[local_models[i].pos_start+j];
            //write_buf[2*j+1]=keys[local_models[i].pos_start+j];
        }
        size_t kvs_size=local_models[i].data_num*sizeof(slot);
        GlobalAddress target=dsm->alloc_size(kvs_size);
        dsm->write_sync(cache_ptr,target,kvs_size);
        model_global temp;
        temp.slope=local_models[i].slope;
        temp.intercept=local_models[i].intercept;
        temp.key_start=local_models[i].key_start;
        temp.is_leaf=true;
        temp.child_start=target;
        temp.child_length=local_models[i].data_num;
        temp.write_buffer_index=i%write_buffer_conf_.buffer_num;
        global_models_0.push_back(temp);
    }
    level_models.push_back(global_models_0);
    return;
}

void learned_index_global::write_seg_remote_with_empty(std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals,std::vector<model_local> &local_models)
{
    char *cache_ptr=dsm->get_cache();
    std::vector<model_global> global_models_0;
    for(int i=0;i<local_models.size();i++)
    {
        //slot *write_buf=(slot*)cache_ptr;
        kv_pair *kv_ptr = (kv_pair*)cache_ptr;
        uint64_t *next_ptr = (uint64_t*)((uint64_t)cache_ptr + sizeof(kv_pair) * seg_keys[i].size());
        for(int j = 0; j < seg_keys[i].size(); j++)
        {
            kv_ptr[j].key = seg_keys[i][j];
            kv_ptr[j].val = seg_vals[i][j];
            next_ptr[j] = null_flag;
        }
        size_t alloc_size = sizeof(kv_pair) * seg_keys[i].size() + sizeof(uint64_t) * seg_keys[i].size();
        GlobalAddress target=dsm->alloc_size(alloc_size);
        dsm->write_sync(cache_ptr,target,alloc_size);
        model_global temp;
        temp.slope=local_models[i].slope;
        temp.intercept=local_models[i].intercept;
        temp.key_start=local_models[i].key_start;
        temp.is_leaf=true;
        temp.child_start=target;
        temp.child_length=local_models[i].data_num;
        temp.write_buffer_index=i%write_buffer_conf_.buffer_num;
        global_models_0.push_back(temp);
    }
    level_models.push_back(global_models_0);
    return;
}

void learned_index_global::build_remote(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals,learned_index_local &local_index,GlobalAddress &root_model_addr)
{    
    write_seg_remote(keys,vals,local_index.level_models[0]);
    int model_sum=0;
    int cur=0;
    char *cache_ptr=dsm->get_cache();
    model_global *write_buf=(model_global*)cache_ptr;
    for(int i=0;i<local_index.level_models.size();i++)
    {
        model_sum+=2*local_index.level_models[i].size();
    }
    GlobalAddress models_addr=dsm->alloc_size(sizeof(model_global)*model_sum*2);
    std::vector<std::vector<GlobalAddress>> child_addrs;
    std::vector<std::vector<GlobalAddress>> level_addrs;
    for(int i=0;i<local_index.level_models.size();i++)
    {
        std::vector<GlobalAddress> temp_addrs;
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            GlobalAddress temp=models_addr;
            temp_addrs.push_back(temp);
        }
        level_addrs.push_back(temp_addrs);
    }
    std::vector<GlobalAddress> occ;
    child_addrs.push_back(occ);
    //addr of all models's child start
    for(int i=1;i<local_index.level_models.size();i++)
    {
        std::vector<GlobalAddress> temp_addrs;
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            GlobalAddress temp=models_addr;
            temp.offset+=sizeof(model_global)*cur;
            cur+=local_index.level_models[i][j].data_num*2;
            temp_addrs.push_back(temp);
        }
        child_addrs.push_back(temp_addrs);
    }



    //addr of root
    GlobalAddress root_addr=models_addr;
    root_addr.offset+=sizeof(model_global)*cur;
    cur+=2;
    level_addrs[level_addrs.size()-1][0]=root_addr;

    //addr for all model
    for(int i=1;i<local_index.level_models.size();i++)
    {
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            for(int k=0;k<local_index.level_models[i][j].data_num;k++)
            {
                level_addrs[i-1][local_index.level_models[i][j].pos_start+k]=child_addrs[i][j];
                level_addrs[i-1][local_index.level_models[i][j].pos_start+k].offset+=k*sizeof(model_global);
            }   
        }
    }


    for(int i=0;i<level_models[0].size();i++)
    {
        if(i<level_models[0].size()-1)
        {
            level_models[0][i].sibling=level_addrs[0][i+1];
        }
        else
        {
            level_models[0][i].sibling.val=0xffffffffffffffff;
        }
    }
    for(int i=1;i<local_index.level_models.size();i++)
    {
        std::vector<model_global> temp_global_models;
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            model_global temp_global_model;
            temp_global_model.slope=local_index.level_models[i][j].slope;
            temp_global_model.intercept=local_index.level_models[i][j].intercept;
            temp_global_model.pos_start=local_index.level_models[i][j].pos_start;
            temp_global_model.key_start=local_index.level_models[i][j].key_start;
            temp_global_model.child_start=level_addrs[i-1][local_index.level_models[i][j].pos_start];
            if(j<local_index.level_models[i].size()-1)
            {
                temp_global_model.sibling=level_addrs[i][j+1];
            }
            else
            {
                temp_global_model.sibling.val=0xffffffffffffffff;
            }
            temp_global_model.child_length=local_index.level_models[i][j].data_num;
            temp_global_model.threshold=temp_global_model.child_length*2;
            temp_global_model.is_leaf=false;
            temp_global_models.push_back(temp_global_model);
        }
        level_models.push_back(temp_global_models);
    }
    for(int i=0;i<level_models.size();i++)
    {
        for(int j=0;j<level_models[i].size();j++)
        {
            int offset=(level_addrs[i][j].offset-models_addr.offset)/sizeof(model_global);
            write_buf[offset]=level_models[i][j];
            cur++;
        }
    }
    /*for(int i=0;i<level_models[0].size();i++)
    {
        boost::icl::interval_map<uint64_t,int> temp_ca;
        std::vector<model_global> temp_content;
        std::shared_mutex temp_lc;
        cache.push_back(temp_ca);
        cache_content.push_back(temp_content);
        cache_lock.push_back(temp_lc);
    }*/
    cache.resize(level_models[0].size());
    cached_num.resize(level_models[0].size());
    for(int i=0;i<cached_num.size();i++)
    {
        cached_num[i]=0;
    }
    cache_content.resize(level_models[0].size());
    std::vector<std::shared_mutex> list(level_models[0].size());
    cache_lock.swap(list);
    dsm->write_sync(cache_ptr,models_addr,sizeof(model_global)*model_sum);
    root_model_addr=level_addrs[level_addrs.size()-1][0];
    return;
}

void learned_index_global::build_remote_with_empty(std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals,learned_index_local &local_index,GlobalAddress &root_model_addr)
{    
    //std::cout<<"build without 2"<<std::endl;
    write_seg_remote_with_empty(seg_keys,seg_vals,local_index.level_models[0]);
    int model_sum=0;
    int cur=0;
    char *cache_ptr=dsm->get_cache();
    model_global *write_buf=(model_global*)cache_ptr;
    for(int i=0;i<local_index.level_models.size();i++)
    {
        model_sum+=local_index.level_models[i].size();
    }
    GlobalAddress models_addr=dsm->alloc_size(sizeof(model_global)*model_sum);
    std::vector<std::vector<GlobalAddress>> child_addrs;
    std::vector<std::vector<GlobalAddress>> level_addrs;
    for(int i=0;i<local_index.level_models.size();i++)
    {
        std::vector<GlobalAddress> temp_addrs;
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            GlobalAddress temp=models_addr;
            temp_addrs.push_back(temp);
        }
        level_addrs.push_back(temp_addrs);
    }
    std::vector<GlobalAddress> occ;
    child_addrs.push_back(occ);
    //addr of all models's child start
    for(int i=1;i<local_index.level_models.size();i++)
    {
        std::vector<GlobalAddress> temp_addrs;
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            GlobalAddress temp=models_addr;
            temp.offset+=sizeof(model_global)*cur;
            cur+=local_index.level_models[i][j].data_num;
            temp_addrs.push_back(temp);
        }
        child_addrs.push_back(temp_addrs);
    }



    //addr of root
    GlobalAddress root_addr=models_addr;
    root_addr.offset+=sizeof(model_global)*cur;
    cur+=1;
    level_addrs[level_addrs.size()-1][0]=root_addr;

    //addr for all model
    for(int i=1;i<local_index.level_models.size();i++)
    {
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            for(int k=0;k<local_index.level_models[i][j].data_num;k++)
            {
                level_addrs[i-1][local_index.level_models[i][j].pos_start+k]=child_addrs[i][j];
                level_addrs[i-1][local_index.level_models[i][j].pos_start+k].offset+=k*sizeof(model_global);
            }   
        }
    }


    for(int i=0;i<level_models[0].size();i++)
    {
        if(i<level_models[0].size()-1)
        {
            level_models[0][i].sibling=level_addrs[0][i+1];
        }
        else
        {
            level_models[0][i].sibling.val=0xffffffffffffffff;
        }
    }
    for(int i=1;i<local_index.level_models.size();i++)
    {
        std::vector<model_global> temp_global_models;
        for(int j=0;j<local_index.level_models[i].size();j++)
        {
            model_global temp_global_model;
            temp_global_model.slope=local_index.level_models[i][j].slope;
            temp_global_model.intercept=local_index.level_models[i][j].intercept;
            temp_global_model.pos_start=local_index.level_models[i][j].pos_start;
            temp_global_model.key_start=local_index.level_models[i][j].key_start;
            temp_global_model.child_start=level_addrs[i-1][local_index.level_models[i][j].pos_start];
            if(j<local_index.level_models[i].size()-1)
            {
                temp_global_model.sibling=level_addrs[i][j+1];
            }
            else
            {
                temp_global_model.sibling.val=0xffffffffffffffff;
            }
            temp_global_model.child_length=local_index.level_models[i][j].data_num;
            temp_global_model.threshold=temp_global_model.child_length;
            temp_global_model.is_leaf=false;
            temp_global_models.push_back(temp_global_model);
        }
        level_models.push_back(temp_global_models);
    }

    level_models[level_models.size()-1][0].full_model_num = model_sum;

    
    for(int i=0;i<level_models.size();i++)
    {
        for(int j=0;j<level_models[i].size();j++)
        {
            int offset=(level_addrs[i][j].offset-models_addr.offset)/sizeof(model_global);
            write_buf[offset]=level_models[i][j];
            cur++;
        }
    }
    cache.resize(level_models[0].size());
    cached_num.resize(level_models[0].size());
    for(int i=0;i<cached_num.size();i++)
    {
        cached_num[i]=0;
    }
    cache_content.resize(level_models[0].size());
    std::vector<std::shared_mutex> list(level_models[0].size());
    cache_lock.swap(list);
    dsm->write_sync(cache_ptr,models_addr,sizeof(model_global)*model_sum);
    root_model_addr=level_addrs[level_addrs.size()-1][0];
    return;
}

void learned_index_global::read_model_from_remote(GlobalAddress root_addr)
{
    char *cache_ptr=dsm->get_rdma_buffer();
    model_global*read_buf=(model_global*)cache_ptr;
    dsm->read_sync(cache_ptr,root_addr,sizeof(model_global));
    std::vector<std::vector<model_global>> temp_level_models;
    std::vector<model_global> models;
    models.push_back(*read_buf);
    temp_level_models.push_back(models);
    int level=0;
    while(!temp_level_models[level][0].is_leaf)
    {
        std::vector<model_global> temp_models;
        for(int i=0;i<temp_level_models[level].size();i++)
        {
            dsm->read_sync(cache_ptr,temp_level_models[level][i].child_start,temp_level_models[level][i].child_length*sizeof(model_global));
            int data_num=temp_level_models[level][i].child_length;
            for(int i=0;i<data_num;i++)
            {
                temp_models.push_back(read_buf[i]);
            }
        }
        temp_level_models.push_back(temp_models);
        level++;
    }
    for(int i=temp_level_models.size()-1;i>=0;i--)
    {
        level_models.push_back(temp_level_models[i]);
    }
    cache.resize(level_models[0].size());
    cached_num.resize(level_models[0].size());
    for(int i=0;i<cached_num.size();i++)
    {
        cached_num[i]=0;
    }
    cache_content.resize(level_models[0].size());
    std::vector<std::shared_mutex> list(level_models[0].size());
    cache_lock.swap(list);
    //cache_lock.resize(level_models[0].size());
    return;
}

void learned_index_global::read_model_from_remote_oneoff(GlobalAddress root_addr,int model_num)
{
    char *cache_ptr=dsm->get_rdma_buffer();
    GlobalAddress start_addr=root_addr;
    start_addr.offset-=(model_num-1)*sizeof(model_global);
    model_global*read_buf=(model_global*)cache_ptr;
    dsm->read_sync(cache_ptr,start_addr,sizeof(model_global)*model_num);
    std::vector<model_global> models;
    int i=0;
    int now_length=0;
    while(read_buf[i].is_leaf)
    {
        models.push_back(read_buf[i]);
        i++;
    }
    level_models.push_back(models);
    models.clear();
    while(i<model_num)
    {
        models.push_back(read_buf[i]);
        now_length+=read_buf[i].child_length;
        if(now_length!=level_models[level_models.size()-1].size())
        {
            i++;
            continue;
        }
        else
        {
            level_models.push_back(models);
            models.clear();
            i++;
        }
    }
    return;
}



void learned_index_global::cache_model(int model_now,int slot_size,int slot_pos,slot *read_buf,learned_index_global *temp_model,int &cache_index)
{
    int range_start=read_buf[slot_pos].key+1;
    int range_end=0;
    if(slot_pos!=slot_size-1)
    {
        std::cout<<1<<std::endl;
        range_end=read_buf[slot_pos+1].key-1;
    }
    else
    {
        if(model_now==level_models[0].size()-1)
        {
            std::cout<<2<<std::endl;
            range_end=0xffffffffffffffff-1;
        }
        else
        {
            std::cout<<3<<std::endl;
            range_end=level_models[0][model_now+1].key_start-1;
        }
    }
    cache_lock[model_now].lock();
    cache_index=cached_num[model_now];
    cache_content[model_now].push_back(temp_model);
    cached_num[model_now]++;
    std::cout<<"range start:"<<range_start<<"range end"<<range_end<<std::endl;
    cache[model_now].add(std::make_pair(ival(range_start,range_end),cache_index));
    cache_lock[model_now].unlock();
}

void learned_index_global::cache_model_range(int model_now,int range_start,int range_end,learned_index_global *temp_model,int &cache_index)
{  
    cache_lock[model_now].lock();
    cache_index=cached_num[model_now];
    cache_content[model_now].push_back(temp_model);
    cached_num[model_now]++;
    //cache[model_now].erase(boost::icl::interval<uint64_t>::closed(range_start,range_end));
    cache[model_now].add(std::make_pair(ival(range_start,range_end),cache_index));
    cache_lock[model_now].unlock();
}

void learned_index_global::cache_set(int model_now,int slot_size,int slot_pos,slot *read_buf,GlobalAddress slot_gaddr)
{
    int range_start=read_buf[slot_pos].key+1;
    int range_end=0;
    if(slot_pos!=slot_size-1)
    {
        range_end=read_buf[slot_pos+1].key-1;
    }
    else
    {
        if(model_now==level_models[0].size()-1)
        {
            range_end=0xffffffffffffffff-1;
        }
        else
        {
            range_end=level_models[0][model_now+1].key_start-1;
        }
    }
    learned_index_global *temp_learned_g=new learned_index_global();
    std::vector<model_global> temp_models_g;
    model_global temp_model_g;
    temp_learned_g->Epsilon=0;
    GlobalAddress set_gaddr;
    set_gaddr.val=(read_buf[slot_pos].next&mask_);
    temp_model_g.child_start=set_gaddr;
    temp_model_g.sibling=slot_gaddr;
    temp_models_g.push_back(temp_model_g);
    temp_learned_g->level_models.push_back(temp_models_g);
    cache_lock[model_now].lock();
    int cache_index=cached_num[model_now];
    cache_content[model_now].push_back(temp_learned_g);
    cached_num[model_now]++;
    //cache[model_now].erase(boost::icl::interval<uint64_t>::closed(range_start,range_end));
    cache[model_now].add(std::make_pair(ival(range_start,range_end),cache_index));
    cache_lock[model_now].unlock();
    //std::cout<<"cache range"<<range_start<<" "<<range_end<<"in "<<model_now<<" "<<cache_index<<" ptr:"<<cache_content[model_now][cache_index]<<std::endl;
}

void learned_index_global::get_cache(int &model_now,int content_index,learned_index_global **cache_content_)
{
    cache_lock[model_now].lock_shared();
    *cache_content_=(cache_content[model_now][content_index]);
    cache_lock[model_now].unlock_shared();
}

bool learned_index_global::search_cache(int &model_now,uint64_t key,learned_index_global **cache_content_,uint64_t &range_start,uint64_t &range_end)
{
    cache_lock[model_now].lock();
    auto it=cache[model_now].find(key);
    if(it==cache[model_now].end())
    {
        cache_lock[model_now].unlock();
        return false;
    }
    else
    {
        volatile int cache_index=it->second;
        range_start=it->first.lower();
        range_end=it->first.upper();
        *cache_content_=cache_content[model_now][cache_index];
        cache_lock[model_now].unlock();
        assert(*cache_content_!=nullptr);
        return true;
    }
    return false;
}



/*int vec_exponential_search(slot* data_key, int size, uint64_t target, int pos){
    pos = (pos >= size ? (size - 1) : pos);
    pos = (pos < 0 ? 0 : pos);
    int begin_i = 0, end_i = size;
    int step = 1;
    if (data_key[pos].key < target) {
        begin_i = pos;
        end_i = begin_i + step;
        while (end_i < size && data_key[end_i].key <= target) {
            step = step << 1;
            begin_i = end_i;
            end_i = begin_i + step;

            // std::cout  <<"\r\n size = "<< size <<"  begin_i = "<< begin_i <<"  end_i = "<< end_i << std::endl;
        }
        if (end_i >= size) {
            end_i = size - 1;
        }
    }
    else if (data_key[pos].key > target) {
        end_i = pos;
        begin_i = end_i - step;
        while (begin_i >= 0 && data_key[begin_i].key > target) {
            step = step << 1;
            end_i = begin_i;
            begin_i = end_i - step;
        }
        if (begin_i < 0) {
            begin_i = 0;
        }
    }
    else{
        return pos;
    };

    // std::cout  <<"\r\n size = "<< size <<"  begin_i = "<< begin_i <<"  end_i = "<< end_i << std::endl;
    if(begin_i >= 0 && end_i < size && begin_i <= end_i){
        while (begin_i <= end_i) {
            int mid = (begin_i + end_i) >> 1;
            if (data_key[mid].key >= target) {
                end_i=mid-1;
            } else {
                begin_i = mid + 1;
            }
        }
        return end_i;
    }
    //return -1;
}*/


bool learned_index_global::model_binary_search(int level,int model,uint64_t key,int &next_model)
{
    if(level<1||level>level_models.size()-1||level_models[level][model].is_leaf)
    {
        return false;
    }
    if(level_models[level][model].child_length==1)
    {
        next_model=0;
        return true;
    }
    long double predict=level_models[level][model].slope*key+level_models[level][model].intercept;
    if(predict<0)
    {
        predict=0;
    }
    else if(predict>level_models[level][model].child_length-1)
    {
        predict=level_models[level][model].child_length-1;
    }
    int64_t predict_in=(int64_t)predict+level_models[level][model].pos_start;
    int64_t low=std::max<int64_t>(0,(predict_in-Epsilon));
    int64_t high=std::min<int64_t>((predict_in+Epsilon),(level_models[level][model].pos_start+level_models[level][model].child_length-1));
    if((key>=level_models[level-1][predict_in].key_start&&predict_in==level_models[level][model].child_length-1+level_models[level][model].pos_start)||
        (key>=level_models[level-1][predict_in].key_start&&key<level_models[level-1][predict_in+1].key_start))
    {
        next_model=predict_in;
        return true;
    }
    else if(key<level_models[level-1][predict_in].key_start)
    {
        high=predict_in-1;
    }
    else if(key>=level_models[level-1][predict_in+1].key_start)
    {
        low=predict_in+1;
    }
    while(low<=high)
    {
        int mid=(high+low)>>1;
        if((key>=level_models[level-1][mid].key_start&&mid==level_models[level][model].child_length-1+level_models[level][model].pos_start)||
        (key>=level_models[level-1][mid].key_start&&key<level_models[level-1][mid+1].key_start))
        {
            next_model=mid;
            return true;
        }
        else if(key<level_models[level-1][mid].key_start)
        {
            high=mid-1;
        }
        else if(key>=level_models[level-1][mid+1].key_start)
        {
            low=mid+1;
        }
    }
    next_model=low;
    return true;
}

bool learned_index_global::model_search(uint64_t &key,model_global &target_model,int &model_now)
{
    if(level_models.size()==0)
    {
        return false;
    }
    int level_now=level_models.size()-1;
    model_now=0;
    while(!level_models[level_now][model_now].is_leaf)
    {
        bool res=model_binary_search(level_now,model_now,key,model_now);
        //std::cout<<"level_now: "<<level_now<<"model_now"<<model_now<<std::endl;
        if(!res)
        {
            std::cout<<"error in binary"<<std::endl;
            return false;
        }
        level_now-=1;
    }
    target_model=level_models[level_now][model_now];
    if((key>=level_models[level_now][model_now].key_start&&model_now==level_models[level_now].size()-1)
    ||(key>=level_models[level_now][model_now].key_start&&key<level_models[level_now][model_now+1].key_start))
    {
        return true;
    }
    else
    {
        std::cout<<"key: "<<key<<"level_models[level_now][model_now].key_start: "<<level_models[level_now][model_now].key_start<<std::endl;
        std::cout<<"error in judge"<<std::endl;
        return false;
    }
}

bool learned_index_global::model_scan(uint64_t &key_start,uint64_t &key_end,std::vector<model_global>& target_models)
{
    if(level_models.size()==0)
    {
        return false;
    }
    int level_now=level_models.size()-1;
    int model_start_now=0;
    int model_end_now=0;
    while(!level_models[level_now][model_start_now].is_leaf)
    {
        bool res_start=model_binary_search(level_now,model_start_now,key_start,model_start_now);
        bool res_end=model_binary_search(level_now,model_end_now,key_end,model_end_now);
        //std::cout<<"level_now: "<<level_now<<"model_now"<<model_now<<std::endl;
        if(!res_start||!res_end)
        {
            std::cout<<"error in binary"<<std::endl;
            return false;
        }
        level_now-=1;
    }
    //target_model=level_models[level_now][model_now];
    if((key_start>=level_models[level_now][model_start_now].key_start&&model_start_now==level_models[level_now].size()-1)
    ||(key_start>=level_models[level_now][model_start_now].key_start&&key_start<level_models[level_now][model_start_now+1].key_start))
    {
        if((key_end>=level_models[level_now][model_end_now].key_start&&model_end_now==level_models[level_now].size()-1)
        ||(key_end>=level_models[level_now][model_end_now].key_start&&key_end<level_models[level_now][model_end_now+1].key_start))
        {
            for(int i=model_start_now;i<=model_end_now;i++)
            {
                target_models.push_back(level_models[level_now][i]);
            }
        }
        return true;
    }
    else
    {
        std::cout<<"key: "<<key_start<<"level_models[level_now][model_now].key_start: "<<level_models[level_now][model_start_now].key_start<<std::endl;
        std::cout<<"error in judge"<<std::endl;
        return false;
    }
}

bool learned_index_global::slot_binary_search(slot* kvs,int size,uint64_t key,int &pos)
{
    if(kvs[0].key>key)
    {
        return false;
    }
    int left=0;
    int right=size-1;
    while(left<=right)
    {
        int mid=(left+right)/2;
        if(kvs[mid].key>key)
        {
            right=mid-1;
        }
        else
        {
            left=mid+1;
        }
    }
    if(right<0)
    {
        right=0;
    }
    pos=right; 
    return true;
}

bool learned_index_global::write_buffer_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t key, uint64_t val, uint64_t next)
{
    RdmaOpRegion write_ror;
    RdmaOpRegion cas_ror;
    list_node *list_node_ptr = (list_node*)write_source;
    list_node_ptr->key = key;
    list_node_ptr->val = val;
    list_node_ptr->next_offset = next;
    write_ror.source = (uint64_t)(write_source);
    write_ror.dest = write_gaddr;
    write_ror.size = sizeof(list_node);
    write_ror.is_on_chip = false;
    cas_ror.source = (uint64_t)(cas_source);
    cas_ror.dest = cas_gaddr;
    cas_ror.size = sizeof(uint64_t);
    cas_ror.is_on_chip = false;
    bool ret = dsm -> write_cas_sync(write_ror, cas_ror, equal, swap_val);
    return ret;
}

bool learned_index_global::write_next_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t next)
{
    RdmaOpRegion write_ror;
    RdmaOpRegion cas_ror;
    //list_node *list_node_ptr = (list_node*)write_source;
    *write_source = next;
    write_ror.source = (uint64_t)(write_source);
    write_ror.dest = write_gaddr;
    write_ror.size = sizeof(uint64_t);
    write_ror.is_on_chip = false;
    cas_ror.source = (uint64_t)(cas_source);
    cas_ror.dest = cas_gaddr;
    cas_ror.size = sizeof(uint64_t);
    cas_ror.is_on_chip = false;
    bool ret = dsm -> write_cas_sync(write_ror, cas_ror, equal, swap_val);
    return ret;
}

bool learned_index_global::search(uint64_t &key,uint64_t &val)
{
    bool ret=false;
    char *cache_ptr=dsm->get_rdma_buffer();
    slot *read_buf=(slot*)cache_ptr;
    node_set *node_set_buf=(node_set*)cache_ptr;
    list_node *write_buf=(list_node*)cache_ptr;
    uint64_t dest_start=(uint64_t)cache_ptr;
    model_global seg_model;
    int model_now; 
    bool res=model_search(key,seg_model,model_now);
    write_buffer *write_buffer_=write_buffers[seg_model.write_buffer_index];
    if(!res)
    {
        std::cout<<"search : error in search model"<<std::endl;
        std::cout<<"key:"<<key<<" val: "<<val<<std::endl;
        return res;
    }
    if(enable_cache)
    {
        learned_index_global *model_in_cache=nullptr;
        uint64_t range_start;
        uint64_t range_end;
        bool find=search_cache(model_now,key,&model_in_cache,range_start,range_end);
        if(find)
        {
            if(model_in_cache->Epsilon==0)
            {
                //std::cout<<"read in cache set "<<key<<std::endl;
                std::vector<RdmaOpRegion> cache_set_op;
                GlobalAddress set_gaddr=model_in_cache->level_models[0][0].child_start;
                GlobalAddress slot_gaddr=model_in_cache->level_models[0][0].sibling;
                RdmaOpRegion r_set;
                RdmaOpRegion r_slot;
                r_set.source=dest_start;
                r_set.dest=set_gaddr;
                r_set.size=sizeof(node_set);
                r_set.is_on_chip=false;
                r_slot.source=dest_start+sizeof(node_set);
                r_slot.dest=slot_gaddr;
                r_slot.size=sizeof(slot);
                r_slot.is_on_chip=false;
                cache_set_op.push_back(r_set);
                cache_set_op.push_back(r_slot);
                dsm->read_batches_sync(cache_set_op);
                slot *cache_slot_ptr=(slot*)((uint64_t)cache_ptr+sizeof(node_set));
                slot slot_in_cache=cache_slot_ptr[0];
                reread_cache:
                if(((slot_in_cache.next)&model_flag)!=0)
                {
                    GlobalAddress root;
                    root.val=(slot_in_cache.next&mask_);
                    learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                    temp->read_model_from_remote(root);
                    int cache_index;
                    cache_model_range(model_now,range_start,range_end,temp,cache_index);
                    learned_index_global *cached_model=nullptr;
                    get_cache(model_now,cache_index,&cached_model);
                    bool ret=cached_model->search(key,val);
                    return ret;
                }
                else if(((slot_in_cache.next)&set_flag)!=0)
                {
                    std::vector<RdmaOpRegion> read_batch;
                    int cur_off=0;
                    for(int i=0;i<node_set_buf[0].node_num;i++)
                    {
                        if(node_set_buf[0].nodes_gaddr[i]!=null_next)
                        {
                            RdmaOpRegion r;
                            r.source=dest_start+cur_off;
                            r.dest=node_set_buf[0].nodes_gaddr[i];
                            r.size=sizeof(list_node);
                            r.is_on_chip=false;
                            cur_off+=sizeof(list_node);
                            read_batch.push_back(r);
                        }
                    }
                    dsm->read_batches_sync(read_batch);
                    for(int i=read_batch.size()-1;i>=0;i--)
                    {
                        if(write_buf[i].key==key)
                        {
                            val=write_buf[i].val;
                            return true;
                        }
                    }
                }
                else
                {
                    GlobalAddress write_slot_gaddr;
                    write_slot_gaddr.val=(slot_in_cache.next&mask_);
                    dsm->read_sync(cache_ptr,write_slot_gaddr,sizeof(list_node));
                    list_node now=write_buf[0];
                    do
                    {
                        if(now.key==key)
                        {
                            val=now.val;
                            ret=true;
                            break;
                        }
                        if((now.next_offset&set_flag)!=0)
                        {
                            GlobalAddress set_gaddr;
                            set_gaddr.val=(now.next_offset&mask_);
                            dsm->read_sync(cache_ptr,set_gaddr,sizeof(node_set));
                            int cur_off=0;
                            std::vector<RdmaOpRegion> read_batch;
                            for(int i=0;i<node_set_buf[0].node_num;i++)
                            {
                                RdmaOpRegion r;
                                r.source=dest_start+cur_off;
                                r.dest=node_set_buf[0].nodes_gaddr[i];
                                r.size=sizeof(list_node);
                                r.is_on_chip=false;
                                cur_off+=sizeof(list_node);
                                read_batch.push_back(r);
                            }
                            dsm->read_batches_sync(read_batch);
                            for(int i=read_batch.size()-1;i>=0;i--)
                            {
                                if(write_buf[i].key==key)
                                {
                                    val=write_buf[i].val;
                                    //std::cout<<"batch ret"<<std::endl;
                                    return true;
                                }
                            }
                            return false;
                        }
                        if(now.next_offset==(slot_gaddr.val|tail_flag))
                        {
                            ret=false;
                            break;
                        }
                        else if(now.next_offset==null_flag)
                        {
                            dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                            slot_in_cache=read_buf[0];
                            std::cout<<"goto error"<<key<<" "<<now.key<<std::endl;
                            //return false;
                            goto reread_cache;
                        }
                        GlobalAddress next_slot_gaddr;
                        next_slot_gaddr.nodeID=write_slot_gaddr.nodeID;
                        next_slot_gaddr.offset=(write_buffer_->get_off_write_buffer()+(now.next_offset&mask_)*sizeof(list_node));
                        //std::cout<<"2"<<std::endl;
                        dsm->read_sync(cache_ptr,next_slot_gaddr,sizeof(list_node));
                        now=write_buf[0];
                    }while(now.next_offset!=(slot_gaddr.val|tail_flag));
                    if(ret)
                    {
                        return ret;
                    }
                    else if(now.key==key)
                    {
                        val=now.val;
                        ret=true;
                    }
                    //std::cout<<"chain ret"<<std::endl;
                    return ret;
                }
            }
            else
            {
                //std::cout<<"read in cache model "<<key<<std::endl;
                bool ret=model_in_cache->search(key,val);
                return ret;
            }
        }
    }
    long double predict=seg_model.slope*key+seg_model.intercept;
    int start=std::max<int>(0,(int)predict-(int)Epsilon);
    //read one more slot for cache
    int end=std::min<int>(seg_model.child_length-1,(int)predict+Epsilon+1);
    if(start>end)
    {
        start=end;
    }
    int kv_size=end-start+1;
    GlobalAddress target=seg_model.child_start;
    target.offset+=start*sizeof(slot);
    //int end=std::min<int>(seg_model.child_length-1,(int)predict+(int)Epsilon);
    dsm->read_sync(cache_ptr,target,kv_size*sizeof(slot));
    int pos=0;
    slot_binary_search(read_buf,kv_size,key,pos);
    target.offset+=(sizeof(slot)*pos);
    slot slot_=read_buf[pos];
    int chain_length=0;
    reread:
    if(slot_.next==null_flag&&slot_.key==key)
    {
        val=slot_.val;
        return true;
    }
    if((slot_.next&set_flag)!=0)
    {
        if(slot_.key==key)
        {
            val=slot_.val;
            return true;
        }
        GlobalAddress set_gaddr;
        set_gaddr.val=(slot_.next&mask_);
        //model_global set;
        //set.child_start=set_gaddr;
        //set.sibling=target;
        //set.is_leaf=false;
        //std::cout<<"cache set "<<key<<std::endl;
        cache_set(model_now,kv_size,pos,read_buf,target);
        dsm->read_sync(cache_ptr,set_gaddr,sizeof(node_set));
        int cur_off=0;
        std::vector<RdmaOpRegion> read_batch;
        //std::cout<<"node num"<<node_set_buf[0].node_num<<std::endl;
        for(int i=0;i<node_set_buf[0].node_num;i++)
        {
            if(node_set_buf[0].nodes_gaddr[i]!=null_next)
            {
                RdmaOpRegion r;
                r.source=dest_start+cur_off;
                r.dest=node_set_buf[0].nodes_gaddr[i];
                r.size=sizeof(list_node);
                r.is_on_chip=false;
                cur_off+=sizeof(list_node);
                read_batch.push_back(r);
            }
        }
        dsm->read_batches_sync(read_batch);
        for(int i=read_batch.size()-1;i>=0;i--)
        {
            if(write_buf[i].key==key)
            {
                val=write_buf[i].val;
                //std::cout<<"batch ret"<<std::endl;
                return true;
            }
        }
        return false;
    }
    //std::cout<<"slot key:"<<slot_.key<<"slot next:"<<slot_.next<<std::endl;
    if(((slot_.next)&chain_flag)!=0)
    {
        if(slot_.key==key)
        {
            val=slot_.val;
            return true;
        }
        GlobalAddress write_slot_gaddr;
        write_slot_gaddr.val=(slot_.next&mask_);
        //std::cout<<"1"<<std::endl;
        dsm->read_sync(cache_ptr,write_slot_gaddr,sizeof(list_node));
        list_node now=write_buf[0];
        do
        {
            //std::cout<<"now key:"<<now.key<<"now off"<<now.next_offset<<std::endl;
            chain_length++;
            if(now.key==key)
            {
                val=now.val;
                ret=true;
                break;
            }
            if((now.next_offset&set_flag)!=0)
            {
                GlobalAddress set_gaddr;
                set_gaddr.val=(now.next_offset&mask_);
                dsm->read_sync(cache_ptr,set_gaddr,sizeof(node_set));
                int cur_off=0;
                std::vector<RdmaOpRegion> read_batch;
                for(int i=0;i<node_set_buf[0].node_num;i++)
                {
                    RdmaOpRegion r;
                    r.source=dest_start+cur_off;
                    r.dest=node_set_buf[0].nodes_gaddr[i];
                    r.size=sizeof(list_node);
                    r.is_on_chip=false;
                    cur_off+=sizeof(list_node);
                    read_batch.push_back(r);
                }
                dsm->read_batches_sync(read_batch);
                for(int i=read_batch.size()-1;i>=0;i--)
                {
                    if(write_buf[i].key==key)
                    {
                        val=write_buf[i].val;
                        //std::cout<<"batch ret"<<std::endl;
                        return true;
                    }
                }
                return false;
            }
            if(now.next_offset==(target.val|tail_flag))
            {
                //std::cout<<"false key: "<<now.key<<" value: "<<now.val<<std::endl;
                ret=false;
                break;
            }
            else if(now.next_offset==null_flag)
            {
                dsm->read_sync(cache_ptr,target,sizeof(slot));
                slot_=read_buf[0];
                std::cout<<"goto error"<<key<<" "<<now.key<<std::endl;
                //return false;
                goto reread;
            }
            GlobalAddress next_slot_gaddr;
            next_slot_gaddr.nodeID=write_slot_gaddr.nodeID;
            next_slot_gaddr.offset=(write_buffer_->get_off_write_buffer()+(now.next_offset&mask_)*sizeof(list_node));
            //std::cout<<"2"<<std::endl;
            dsm->read_sync(cache_ptr,next_slot_gaddr,sizeof(list_node));
            now=write_buf[0];
        }while(now.next_offset!=(target.val|tail_flag));
        if(chain_length>=define::max_chain_length)
        {
            dsm->retrain_chain(target,seg_model.write_buffer_index);
        }
        if(ret)
        {
            return ret;
        }
        else if(now.key==key)
        {
            val=now.val;
            ret=true;
        }
        //std::cout<<"chain ret"<<std::endl;
        return ret;
    }
    uint64_t test=(slot_.next&model_flag);
    //std::cout<<"test:"<<test<<std::endl;
    if(test!=0)
    {
        
        if(slot_.key==key)
        {
            val=slot_.val;
            return true;
        }
        GlobalAddress root;
        root.val=(slot_.next&mask_);
        //std::cout<<"root gaddr "<<gaddr2str(root)<<std::endl;
        int range_start=read_buf[pos].key+1;
        int range_end=0;
        if(pos!=kv_size-1)
        {
            range_end=read_buf[pos+1].key-1;
        }
        else
        {
            if(model_now==level_models[0].size()-1)
            {
                range_end=0xffffffffffffffff-1;
            }
            else
            {
                range_end=level_models[0][model_now+1].key_start-1;
            }
        }
        learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
        temp->read_model_from_remote(root);
        int cache_index;
        //std::cout<<"cache model"<<std::endl;
        cache_model_range(model_now,range_start,range_end,temp,cache_index);
        //cache_model(model_now,kv_size,pos,read_buf,temp,cache_index);
        //learned_index_global *cached_model=nullptr;
        //get_cache(model_now,cache_index,&cached_model);
        bool ret=temp->search(key,val);
        
        //std::cout<<"root ret "<<std::endl;
        return ret;
    }
    std::cout<<"return false end"<<std::endl;
    assert(1==0);
    return false;
}


bool learned_index_global::insert(uint64_t &key,uint64_t &val,int thread_id)
{
    char *cache_ptr=dsm->get_rdma_buffer();
    uint64_t *rdam_buf_ptr=(uint64_t*)cache_ptr;
    slot *read_buf=(slot*)cache_ptr;
    //uint64_t *next_off_buf=(uint64_t*)cache_ptr;
    model_global seg_model;
    int model_now;
    bool res=model_search(key,seg_model,model_now);
    int write_buffer_index=seg_model.write_buffer_index;
    write_buffer *write_buffer_=write_buffers[write_buffer_index];
    if(enable_cache)
    {
        learned_index_global *model_in_cache=nullptr;
        uint64_t range_start;
        uint64_t range_end;
        bool find=search_cache(model_now,key,&model_in_cache,range_start,range_end);
        if(find)
        {
            if(model_in_cache->Epsilon==0)
            {
                std::cout<<"insert to cache set"<<std::endl;
                GlobalAddress set_gaddr=model_in_cache->level_models[0][0].child_start;
                GlobalAddress slot_gaddr=model_in_cache->level_models[0][0].sibling;
                GlobalAddress write_slot_gaddr;
                GlobalAddress addr_to_cas=slot_gaddr;
                int write_buffer_flag=0; 
                dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                slot slot_in_cache=read_buf[0];
                if(((read_buf[0].next)&model_flag)==0)
                {
                    while(1)
                    {
                        if((slot_in_cache.next&null_flag)!=0)
                        {
                            uint64_t next=slot_gaddr.val|tail_flag;
                            //GlobalAddress new_slot;
                            bool get_slot_ret = write_buffer_ -> get_slot(thread_id, slot_gaddr.nodeID, write_slot_gaddr, nullptr);
                            if(!get_slot_ret)
                            {
                                std::cout<<"alloc error"<<std::endl;
                                return false;
                            }
                            uint64_t *cas_source = rdam_buf_ptr;
                            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                            uint64_t equal = slot_in_cache.next;
                            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                            bool ret_2 = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                            /*if(!write_buffer_->write_slot(thread_id,slot_gaddr.nodeID,key,val,next,write_slot_gaddr))
                            {
                                //write_buffer_->free_slot(ori_write_slot_gaddr);
                                return false;
                            }*/
                            write_buffer_flag=1;
                            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                            if(ret_2)
                            {
                                return true;
                            }
                            else
                            {
                                //std::cout<<"cas after ori fail"<<std::endl;
                                //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                slot_in_cache.next = *cas_source;
                                //slot_in_cache=read_buf[0];
                                continue;
                            }
                        }
                        else if((slot_in_cache.next&chain_flag)!=0||(slot_in_cache.next&set_flag)!=0)
                        {
                            GlobalAddress next_gaddr;
                            next_gaddr.val=(slot_in_cache.next&mask_);
                            uint64_t next;
                            if((slot_in_cache.next&chain_flag)!=0)
                            {
                                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
                            }
                            else
                            {
                                next=slot_in_cache.next;
                            }
                            if(write_buffer_flag==0)
                            {
                                //GlobalAddress new_slot;
                                bool get_slot_ret = write_buffer_->get_slot(thread_id, slot_gaddr.nodeID, write_slot_gaddr, nullptr);
                                if(!get_slot_ret)
                                {
                                    std::cout<<"alloc error"<<std::endl;
                                    return false;
                                }
                                uint64_t *cas_source = rdam_buf_ptr;
                                uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                                uint64_t equal = slot_in_cache.next;
                                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                                bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                                /*if(!write_buffer_->write_slot(thread_id,slot_gaddr.nodeID,key,val,next,write_slot_gaddr))
                                {
                                    return false;
                                }*/
                                write_buffer_flag=1;
                                //bool ret=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                                if(ret)
                                {
                                    return true;
                                }
                                else
                                {   
                                    //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                    //slot_in_cache=read_buf[0];
                                    slot_in_cache.next = *cas_source;
                                    continue;
                                }
                            }
                            else
                            {
                                GlobalAddress write_slot_next_gaddr=write_slot_gaddr;
                                write_slot_next_gaddr.offset+=(2*sizeof(uint64_t));
                                //next_off_buf[0]=next;
                                uint64_t *cas_source = rdam_buf_ptr;
                                uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                                uint64_t equal = slot_in_cache.next;
                                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                                bool ret = write_next_and_cas(write_slot_next_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, next);
                                //dsm->write_sync(cache_ptr,write_slot_next_gaddr,sizeof(uint64_t));
                                //bool ret=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                                if(ret)
                                {
                                    return true;
                                }
                                else
                                {
                                    //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                    //slot_in_cache=read_buf[0];
                                    slot_in_cache.next = *cas_source;
                                    continue;
                                }
                            }
                        }
                        else if((slot_in_cache.next&model_flag)!=0)
                        {
                            GlobalAddress root;
                            root.val=(slot_in_cache.next&mask_);
                            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                            temp->read_model_from_remote(root);
                            int cache_index;
                            cache_model_range(model_now,range_start,range_end,temp,cache_index);
                            GlobalAddress new_slot;
                            write_buffer_->get_slot(thread_id,slot_gaddr.nodeID,new_slot,nullptr);
                            bool ret=temp->sub_insert(new_slot,key,val,thread_id);
                            return ret;
                        }
                    }
                }
                else
                {
                    GlobalAddress root;
                    root.val=(slot_in_cache.next&mask_);
                    learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                    temp->read_model_from_remote(root);
                    int cache_index;
                    std::cout<<"set to model"<<std::endl;
                    cache_model_range(model_now,range_start,range_end,temp,cache_index);
                    GlobalAddress new_slot;
                    write_buffer_->get_slot(thread_id,slot_gaddr.nodeID,new_slot,nullptr);
                    bool ret=temp->sub_insert(new_slot,key,val,thread_id);
                    return ret;
                }
            }
            else
            {
                GlobalAddress new_slot;
                write_buffer_->get_slot(thread_id,model_in_cache->level_models[0][0].child_start.nodeID,new_slot,nullptr);
                std::cout<<"insert to cache model"<<std::endl;
                bool ret=model_in_cache->sub_insert(new_slot,key,val,thread_id);
                return ret;
            }
        }
    }
    long double predict=seg_model.slope*key+seg_model.intercept;
    int start=std::max<int>(0,(int)predict-(int)Epsilon);
    if(start>seg_model.child_length-1)
    {
        start=seg_model.child_length-1;
    }
    int end=std::min<int>(seg_model.child_length-1,(int)predict+Epsilon+1);
    int kv_size=end-start+1;
    GlobalAddress target=seg_model.child_start;
    target.offset+=start*sizeof(slot);
    dsm->read_sync(cache_ptr,target,kv_size*sizeof(slot));
    int pos=0;
    slot_binary_search(read_buf,kv_size,key,pos);
    //std::cout<<"size of slot: "<<sizeof(slot)<<std::endl;
    //std::cout<<"pos: "<<pos<<std::endl;
    target.offset+=pos*sizeof(slot);
    slot slot_=read_buf[pos];
    int write_buffer_flag=0;
    GlobalAddress write_slot_gaddr;
    GlobalAddress addr_to_cas=target;
    //addr_to_cas.offset+=2*sizeof(uint64_t);
    //std::cout<<"intsert to buffer "<<write_buffer_index<<std::endl;
    
    //std::cout<<"write once"<<std::endl;
    //std::cout<<"gaddr: "<<gaddr2str(target)<<std::endl;
    while(1)
    {
        if((slot_.next&null_flag)!=0)
        {
            uint64_t next=target.val|tail_flag;
            bool get_slot_ret = write_buffer_ -> get_slot(thread_id, target.nodeID, write_slot_gaddr,nullptr);
            if(!get_slot_ret)
            {
                std::cout<<"alloc error"<<std::endl;
            }
            uint64_t *cas_source = rdam_buf_ptr;
            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
            uint64_t equal = slot_.next;
            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
            /*if(!write_buffer_->write_slot(thread_id,target.nodeID,key,val,next,write_slot_gaddr))
            {
                //write_buffer_->free_slot(ori_write_slot_gaddr);
                return false;
            }*/
            write_buffer_flag=1;
            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
            if(ret)
            {
                return true;
            }
            else
            {
                //std::cout<<"cas after ori fail"<<std::endl;
                //dsm->read_sync(cache_ptr,target,sizeof(slot));
                //slot_=read_buf[0];
                slot_.next = *cas_source;
                continue;
            }
        }
        else if((slot_.next&chain_flag)!=0||(slot_.next&set_flag)!=0)
        {
            //std::cout<<"chain flag"<<std::endl;
            //std::cout<<"flag=1"<<std::endl;
            GlobalAddress next_gaddr;
            next_gaddr.val=(slot_.next&mask_);
            uint64_t next;
            if((slot_.next&chain_flag)!=0)
            {
                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
            }
            else
            {
                next=slot_.next;
            }
            if(write_buffer_flag==0)
            {
                bool get_slot_ret = write_buffer_->get_slot(thread_id, target.nodeID, write_slot_gaddr,nullptr);
                uint64_t* cas_source = rdam_buf_ptr;
                uint64_t* write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                uint64_t equal = slot_.next;
                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                /*if(!write_buffer_->write_slot(thread_id,target.nodeID,key,val,next,write_slot_gaddr))
                {
                    return false;
                }*/
                write_buffer_flag=1;
                //bool ret=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                if(ret)
                {
                    return true;
                }
                else
                {   
                    //dsm->read_sync(cache_ptr,target,sizeof(slot));
                    //slot_=read_buf[0];
                    slot_.next = *cas_source;
                    continue;
                }
            }
            else
            {
                GlobalAddress write_slot_next_gaddr=write_slot_gaddr;
                write_slot_next_gaddr.offset+=(2*sizeof(uint64_t));
                uint64_t *cas_source = rdam_buf_ptr;
                uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                uint64_t equal = slot_.next;
                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                bool ret = write_next_and_cas(write_slot_next_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, next);
                //next_off_buf[0]=next;
                //dsm->write_sync(cache_ptr,write_slot_next_gaddr,sizeof(uint64_t));
                //bool ret=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                if(ret)
                {
                    return true;
                }
                else
                {
                    //dsm->read_sync(cache_ptr,target,sizeof(slot));
                    //slot_=read_buf[0];
                    slot_.next = *cas_source;
                    continue;
                }
            }
        }
        else if((slot_.next&model_flag)!=0)
        {
            //std::cout<<"read model"<<std::endl;
            GlobalAddress root;
            root.val=(slot_.next&mask_);
            int range_start=read_buf[pos].key+1;
            int range_end=0;
            if(pos!=kv_size-1)
            {
                range_end=read_buf[pos+1].key-1;
            }
            else
            {
                if(model_now==level_models[0].size()-1)
                {
                    range_end=0xffffffffffffffff-1;
                }
                else
                {
                    range_end=level_models[0][model_now+1].key_start-1;
                }
            }
            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
            temp->read_model_from_remote(root);
            int cache_index;
            cache_model_range(model_now,range_start,range_end,temp,cache_index);
            std::cout<<"cache model in insert"<<std::endl;
            GlobalAddress new_slot;
            write_buffer_->get_slot(thread_id,target.nodeID,new_slot,nullptr);
            bool ret=temp->sub_insert(new_slot,key,val,thread_id);
            return ret;
        }
    }
}

bool learned_index_global::sub_insert(GlobalAddress new_slot,uint64_t key,uint64_t val,int thread_id)
{
    char *cache_ptr=dsm->get_rdma_buffer();
    uint64_t *rdam_buf_ptr=(uint64_t*)cache_ptr;
    slot *read_buf=(slot*)cache_ptr;
    uint64_t *next_off_buf=(uint64_t*)cache_ptr;
    model_global seg_model;
    int model_now;
    bool res=model_search(key,seg_model,model_now);
    int write_buffer_index=seg_model.write_buffer_index;
    write_buffer *write_buffer_=write_buffers[write_buffer_index];
    if(enable_cache)
    {
        learned_index_global *model_in_cache=nullptr;
        uint64_t range_start=0;
        uint64_t range_end=0;
        bool find=search_cache(model_now,key,&model_in_cache,range_start,range_end);
        if(find)
        {
            if(model_in_cache->Epsilon==0)
            {
                GlobalAddress set_gaddr=model_in_cache->level_models[0][0].child_start;
                GlobalAddress slot_gaddr=model_in_cache->level_models[0][0].sibling;
                GlobalAddress write_slot_gaddr=new_slot;
                GlobalAddress addr_to_cas=slot_gaddr; 
                dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                slot slot_in_cache=read_buf[0];
                if(((read_buf[0].next)&model_flag)==0)
                {
                    while(1)
                    {
                        if((slot_in_cache.next&null_flag)!=0)
                        {
                            uint64_t next=slot_gaddr.val|tail_flag;
                            uint64_t *cas_source = rdam_buf_ptr;
                            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                            uint64_t equal = slot_in_cache.next;
                            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                            //write_buffer_->fill_slot(key,val,next,new_slot);
                            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                            if(ret)
                            {
                                return true;
                            }
                            else
                            {
                                //std::cout<<"cas after ori fail"<<std::endl;
                                //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                //slot_in_cache=read_buf[0];
                                slot_in_cache.next = *cas_source;
                                continue;
                            }
                        }
                        else if((slot_in_cache.next&chain_flag)!=0||(slot_in_cache.next&set_flag)!=0)
                        {
                            //std::cout<<"chain flag"<<std::endl;
                            //std::cout<<"flag=1"<<std::endl;
                            GlobalAddress next_gaddr;
                            next_gaddr.val=(slot_in_cache.next&mask_);
                            uint64_t next;
                            if((slot_in_cache.next&chain_flag)!=0)
                            {
                                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
                            }
                            else
                            {
                                next=slot_in_cache.next;
                            }
                            uint64_t *cas_source = rdam_buf_ptr;
                            uint64_t *write_source = (uint64_t*)(uint64_t(rdam_buf_ptr) + sizeof(uint64_t));
                            uint64_t equal = slot_in_cache.next;
                            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                            //write_buffer_->fill_slot(key,val,next,new_slot);
                            //bool ret=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                            if(ret)
                            {
                                return true;
                            }
                            else
                            {
                                //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                //slot_in_cache=read_buf[0];
                                slot_in_cache.next = *cas_source;
                                continue;
                            }
                        }
                        else if((slot_in_cache.next&model_flag)!=0)
                        {
                            //std::cout<<"read model"<<std::endl;
                            GlobalAddress root;
                            root.val=(slot_in_cache.next&mask_);
                            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                            temp->read_model_from_remote(root);
                            int cache_index;
                            cache_model_range(model_now,range_start,range_end,temp,cache_index);
                            //GlobalAddress new_slot;
                            //write_buffer_->get_slot(thread_id,slot_gaddr.nodeID,new_slot);
                            //learned_index_global *cached_model=nullptr;
                            //get_cache(model_now,cache_index,cached_model);
                            bool ret=temp->sub_insert(new_slot,key,val,thread_id);
                            return ret;
                        }
                    }
                }
                else
                {
                    GlobalAddress root;
                    root.val=(slot_in_cache.next&mask_);
                    learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                    temp->read_model_from_remote(root);
                    int cache_index;
                    cache_model_range(model_now,range_start,range_end,temp,cache_index);
                    bool ret=temp->sub_insert(new_slot,key,val,thread_id);
                    return ret;
                }
            }
            else
            {
                //GlobalAddress new_slot;
                //write_buffer_->get_slot(thread_id,model_in_cache->level_models[0][0].child_start.nodeID,new_slot);
                bool ret=model_in_cache->sub_insert(new_slot,key,val,thread_id);
                return ret;
            }
        }
    }
    long double predict=seg_model.slope*key+seg_model.intercept;
    int start=std::max<int>(0,(int)predict-(int)Epsilon);
    if(start>seg_model.child_length-1)
    {
        start=seg_model.child_length-1;
    }
    int end=std::min<int>(seg_model.child_length-1,(int)predict+Epsilon+1);
    int kv_size=end-start+1;
    GlobalAddress target=seg_model.child_start;
    target.offset+=start*sizeof(slot);
    dsm->read_sync(cache_ptr,target,kv_size*sizeof(slot));
    int pos=0;
    slot_binary_search(read_buf,kv_size,key,pos);
    //std::cout<<"size of slot: "<<sizeof(slot)<<std::endl;
    //std::cout<<"pos: "<<pos<<std::endl;
    target.offset+=pos*sizeof(slot);
    slot slot_=read_buf[pos];
    GlobalAddress write_slot_gaddr=new_slot;
    GlobalAddress addr_to_cas=target;
    //addr_to_cas.offset+=2*sizeof(uint64_t);
    //std::cout<<"intsert to buffer "<<write_buffer_index<<std::endl;
    
    //std::cout<<"write once"<<std::endl;
    //std::cout<<"gaddr: "<<gaddr2str(target)<<std::endl;
    while(1)
    {
        if((slot_.next&null_flag)!=0)
        {
            uint64_t next=target.val|tail_flag;
            uint64_t *cas_source = rdam_buf_ptr;
            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
            uint64_t equal = slot_.next;
            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
            //write_buffer_->fill_slot(key,val,next,new_slot);
            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
            if(ret)
            {
                return true;
            }
            else
            {
                //std::cout<<"cas after ori fail"<<std::endl;
                //dsm->read_sync(cache_ptr,target,sizeof(slot));
                //slot_=read_buf[0];
                slot_.next = *cas_source;
                continue;
            }
        }
        else if((slot_.next&chain_flag)!=0||(slot_.next&set_flag)!=0)
        {
            //std::cout<<"chain flag"<<std::endl;
            //std::cout<<"flag=1"<<std::endl;
            GlobalAddress next_gaddr;
            next_gaddr.val=(slot_.next&mask_);
            uint64_t next;
            if((slot_.next&chain_flag)!=0)
            {
                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
            }
            else
            {
                next=slot_.next;
            }
            uint64_t* cas_source = rdam_buf_ptr;
            uint64_t* write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
            uint64_t equal = slot_.next;
            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
            //write_buffer_->fill_slot(key,val,next,new_slot);
            //bool ret=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
            if(ret)
            {
                return true;
            }
            else
            {   
                //dsm->read_sync(cache_ptr,target,sizeof(slot));
                //slot_=read_buf[0];
                slot_.next = *cas_source;
                continue;
            }
        }
        else if((slot_.next&model_flag)!=0)
        {
            //std::cout<<"read model"<<std::endl;
            GlobalAddress root;
            root.val=(slot_.next&mask_);
            int range_start=read_buf[pos].key+1;
            int range_end=0;
            if(pos!=kv_size-1)
            {
                range_end=read_buf[pos+1].key-1;
            }
            else
            {
                if(model_now==level_models[0].size()-1)
                {
                    range_end=0xffffffffffffffff-1;
                }
                else
                {
                    range_end=level_models[0][model_now+1].key_start-1;
                }
            }
            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
            temp->read_model_from_remote(root);
            int cache_index;
            cache_model_range(model_now,range_start,range_end,temp,cache_index);
            bool ret=temp->sub_insert(new_slot,key,val,thread_id);
            return ret;
        }
    }
}

bool learned_index_global::scan(uint64_t &key_start,uint64_t &key_end,std::vector<kv> &kvs)
{
    bool ret=false;
    char *cache_ptr=dsm->get_rdma_buffer();
    slot *read_buf=(slot*)cache_ptr;
    uint64_t dest_start=(uint64_t)cache_ptr;
    list_node *write_buf=(list_node*)cache_ptr;
    std::vector<model_global> target_models;
    std::vector<kv> chain_kvs;
    std::vector<kv> slot_kvs;
    bool res=model_scan(key_start,key_end,target_models);
    if(!res)
    {
        std::cout<<"search : error in search model"<<std::endl;
        return res;
    }
    for(int i=0;i<target_models.size();i++)
    {
        //print_global_model(target_models[i]);
    }
    std::vector<RdmaOpRegion> read_batch;
    std::vector<batch_info> batch_infos;
    int cur_off=0;
    if(target_models.size()==1)
    {
        long double start_predict=target_models[0].slope*key_start+target_models[0].intercept;
        long double end_predict=target_models[0].slope*key_end+target_models[0].intercept;
        write_buffer *write_buffer_=write_buffers[target_models[0].write_buffer_index];
        int start=std::max<int>(0,(int)start_predict-(int)Epsilon);
        int end=std::min<int>(target_models[0].child_length-1,(int)end_predict+Epsilon);
        if(start>target_models[0].child_length-1)
        {
            start=target_models[0].child_length-1;
        }
        int kv_size=end-start+1;
        //std::cout<<"kv num"<<kv_size<<std::endl;
        GlobalAddress target=target_models[0].child_start;
        target.offset+=start*sizeof(slot);
        RdmaOpRegion r;
        r.source=dest_start;
        r.dest=target;
        r.size=kv_size*sizeof(slot);
        r.is_on_chip=false;
        read_batch.push_back(r);
        batch_info temp_info;
        temp_info.flag=slots_flag;
        temp_info.offset=cur_off;
        temp_info.node_id=target.nodeID;
        temp_info.length=kv_size;
        temp_info.gaddr=target;
        temp_info.write_buffer_index=target_models[0].write_buffer_index;
        read_batch.push_back(r);
        batch_infos.push_back(temp_info);
    }
    else
    {
        model_global start_model=target_models[0];
        model_global end_model=target_models[target_models.size()-1];
        long double start_predict=start_model.slope*key_start+start_model.intercept;
        long double end_predict=end_model.slope*key_end+end_model.intercept;
        int start_model_start_off=std::max<int>(0,(int)start_predict-(int)Epsilon);
        int start_model_end_off=start_model.child_length-1;
        if(start_model_start_off>start_model.child_length-1)
        {
            start_model_start_off=start_model.child_length-1;
        }
        int end_model_start_off=0;
        int end_model_end_off=std::min<int>(end_model.child_length-1,(int)end_predict+Epsilon);
        GlobalAddress target_start=start_model.child_start;
        target_start.offset+=start_model_start_off*sizeof(slot);
        GlobalAddress target_end=end_model.child_start;
        RdmaOpRegion r_start;
        r_start.source=dest_start;
        r_start.dest=target_start;
        r_start.size=(start_model_end_off-start_model_start_off+1)*sizeof(slot);
        r_start.is_on_chip=false;
        batch_info temp_info;
        temp_info.flag=slots_flag;
        temp_info.length=start_model_end_off-start_model_start_off+1;
        temp_info.node_id=target_start.nodeID;
        temp_info.offset=cur_off;
        temp_info.gaddr=target_start;
        temp_info.write_buffer_index=start_model.write_buffer_index;
        read_batch.push_back(r_start);
        batch_infos.push_back(temp_info);
        cur_off+=(sizeof(slot)*(start_model_end_off-start_model_start_off+1));
        for(int i=1;i<target_models.size()-1;i++)
        {
            RdmaOpRegion r_temp;
            r_temp.source=dest_start+cur_off;
            r_temp.dest=target_models[i].child_start;
            r_temp.size=target_models[i].child_length*sizeof(slot);
            r_temp.is_on_chip=false;
            temp_info.flag=slots_flag;
            temp_info.length=target_models[i].child_length;
            temp_info.node_id=target_models[i].child_start.nodeID;
            temp_info.offset=cur_off;
            temp_info.gaddr=target_models[i].child_start;
            temp_info.write_buffer_index=target_models[i].write_buffer_index;
            read_batch.push_back(r_temp);
            batch_infos.push_back(temp_info);
            cur_off+=(sizeof(slot)*target_models[i].child_length);
        }
        RdmaOpRegion r_end;
        r_end.source=dest_start+cur_off;
        r_end.dest=target_end;
        r_end.size=((end_model_end_off+1)*sizeof(slot));
        r_end.is_on_chip=false;
        temp_info.flag=slots_flag;
        temp_info.length=end_model_end_off+1;
        temp_info.node_id=target_end.nodeID;
        temp_info.offset=cur_off;
        temp_info.gaddr=target_end;
        temp_info.write_buffer_index=end_model.write_buffer_index;
        read_batch.push_back(r_end);
        batch_infos.push_back(temp_info);
    }
    dsm->read_batches_sync(read_batch);
    int loop=0;
    while (read_batch.size()!=0)
    {
        read_batch.clear();
        cur_off=0;
        std::vector<batch_info> new_batch;
        for(int i=0;i<batch_infos.size();i++)
        {
            if(batch_infos[i].flag==chain_flag)
            {
                list_node *now=(list_node*)(dest_start+batch_infos[i].offset);
                if(now->key>=key_start&&now->key<=key_end)
                {
                    kv temp_kv;
                    temp_kv.key=now->key;
                    temp_kv.val=now->val;
                    temp_kv.index=loop+10;
                    chain_kvs.push_back(temp_kv);
                    //std::cout<<"chain kv:"<<temp_kv.key<<" "<<temp_kv.val<<std::endl;
                }
                if((now->next_offset&tail_flag)==0)
                {
                    if((now->next_offset&null_flag)!=0)
                    {
                        //std::cout<<"null reread "<<now->next_offset<<std::endl;
                        RdmaOpRegion r;
                        r.source=dest_start+cur_off;
                        r.dest=batch_infos[i].gaddr;
                        r.size=sizeof(slot);
                        r.is_on_chip=false;
                        batch_info temp_info;
                        temp_info.flag=slots_flag;
                        temp_info.offset=cur_off;
                        temp_info.node_id=batch_infos[i].node_id;
                        temp_info.length=1;
                        new_batch.push_back(temp_info);
                        read_batch.push_back(r);
                        cur_off+=(sizeof(slot));
                    }
                    else if((now->next_offset&set_flag)!=0)
                    {
                        //std::cout<<"chain set in"<<std::endl;
                        GlobalAddress set_gaddr;
                        set_gaddr.val=now->next_offset&mask_;
                        RdmaOpRegion r;
                        r.source=dest_start+cur_off;
                        r.dest=set_gaddr;
                        r.size=sizeof(node_set);
                        r.is_on_chip=false;
                        batch_info temp_info;
                        temp_info.flag=set_flag;
                        temp_info.offset=cur_off;
                        temp_info.node_id=batch_infos[i].node_id;
                        temp_info.length=1;
                        new_batch.push_back(temp_info);
                        read_batch.push_back(r);
                        cur_off+=(sizeof(node_set));
                    }
                    else
                    {
                        GlobalAddress list_node_gaddr;
                        list_node_gaddr.nodeID=batch_infos[i].node_id;
                        uint64_t write_buffer_start_off=write_buffers[batch_infos[i].write_buffer_index]->get_off_write_buffer();
                        list_node_gaddr.offset=write_buffer_start_off+(now->next_offset&mask_)*sizeof(list_node);
                        RdmaOpRegion r;
                        r.source=dest_start+cur_off;
                        r.dest=list_node_gaddr;
                        r.size=sizeof(list_node);
                        r.is_on_chip=false;
                        batch_info temp_info;
                        temp_info.flag=chain_flag;
                        temp_info.offset=cur_off;
                        temp_info.node_id=batch_infos[i].node_id;
                        temp_info.write_buffer_index=batch_infos[i].write_buffer_index;
                        temp_info.gaddr=batch_infos[i].gaddr;
                        new_batch.push_back(temp_info);
                        read_batch.push_back(r);
                        cur_off+=(sizeof(list_node));
                        //std::cout<<"next node"<<std::endl;
                    }
                }
            }
            else if(batch_infos[i].flag==model_flag)
            {
                model_global *now=(model_global*)(dest_start+batch_infos[i].offset);
                if(!batch_infos[i].is_leaf&&!now->is_leaf)
                {
                    //print_global_model(*now);
                    int model_num=now->full_model_num;
                    //std::cout<<"model num"<<model_num<<std::endl;
                    //std::cout<<"model_num"<<model_num<<std::endl;
                    GlobalAddress models_start_gaddr=batch_infos[i].gaddr;
                    models_start_gaddr.offset-=(model_num-1)*sizeof(model_global);
                    RdmaOpRegion r;
                    r.source=dest_start+cur_off;
                    r.dest=models_start_gaddr;
                    r.size=sizeof(model_global)*model_num;
                    r.is_on_chip=false;
                    batch_info temp_info;
                    temp_info.flag=model_flag;
                    temp_info.offset=cur_off;
                    temp_info.node_id=models_start_gaddr.nodeID;
                    //temp_info.is_leaf=true;
                    temp_info.length=model_num;
                    new_batch.push_back(temp_info);
                    read_batch.push_back(r);
                    cur_off+=sizeof(model_global)*model_num;
                }
                else
                {
                    model_global *model_buf=(model_global*)(dest_start+batch_infos[i].offset);
                    int j=0;
                    while(j<batch_infos[i].length&&model_buf[j].is_leaf)
                    {
                        //std::cout<<"in add new slot"<<std::endl;
                        RdmaOpRegion r;
                        r.source=dest_start+cur_off;
                        r.dest=model_buf[j].child_start;
                        r.size=model_buf[j].child_length*sizeof(slot);
                        r.is_on_chip=false;
                        batch_info temp_info;
                        temp_info.flag=slots_flag;
                        temp_info.offset=cur_off;
                        temp_info.node_id=batch_infos[i].node_id;
                        temp_info.length=model_buf[j].child_length;
                        temp_info.write_buffer_index=model_buf[j].write_buffer_index;
                        new_batch.push_back(temp_info);
                        read_batch.push_back(r);
                        cur_off+=(sizeof(slot)*model_buf[j].child_length);
                        //std::cout<<"new slots"<<std::endl;
                        j++;
                    }
                }
            }
            else if(batch_infos[i].flag==slots_flag)
            {
                slot *slot_buf=(slot*)(dest_start+batch_infos[i].offset);
                //std::cout<<"slot length:"<<batch_infos[i].length<<std::endl;
                for(int j=0;j<batch_infos[i].length;j++)
                {
                    if(slot_buf[j].key>=key_start&&slot_buf[j].key<=key_end)
                    {
                        kv temp_kv;
                        temp_kv.key=slot_buf[j].key;
                        temp_kv.val=slot_buf[j].val;
                        temp_kv.index=loop+20;
                        slot_kvs.push_back(temp_kv);
                        //std::cout<<"slot kv:"<<temp_kv.key<<" "<<temp_kv.val<<std::endl;
                    }
                    if(j<batch_infos[i].length-1&&(slot_buf[j+1].key-1)<key_start)
                    {
                        continue;
                    }
                    if(slot_buf[j].key>key_end)
                    {
                        break;
                    }
                    if((slot_buf[j].next&chain_flag)!=0)
                    {
                        GlobalAddress list_node_gaadr;
                        list_node_gaadr.val=(slot_buf[j].next&mask_);
                        RdmaOpRegion r;
                        r.source=dest_start+cur_off;
                        r.dest=list_node_gaadr;
                        r.size=sizeof(list_node);
                        r.is_on_chip=false;
                        batch_info temp_info;
                        temp_info.node_id=list_node_gaadr.nodeID;
                        temp_info.flag=chain_flag;
                        temp_info.offset=cur_off;
                        temp_info.write_buffer_index=batch_infos[i].write_buffer_index;
                        temp_info.gaddr=batch_infos[i].gaddr;
                        temp_info.gaddr.offset+=(j*sizeof(slot));
                        read_batch.push_back(r);
                        new_batch.push_back(temp_info);
                        //std::cout<<"new chain"<<std::endl;
                        cur_off+=(sizeof(list_node));
                        continue;
                    }
                    uint64_t test=(slot_buf[j].next&model_flag);
                    if(test!=0)
                    {
                        //std::cout<<"model key:"<<slot_buf[j].key<<"model off:"<<slot_buf[j].next<<std::endl;
                        GlobalAddress model_root_gaddr;
                        model_root_gaddr.val=(slot_buf[j].next&mask_);
                        RdmaOpRegion r;
                        r.source=dest_start+cur_off;
                        r.dest=model_root_gaddr;
                        r.size=sizeof(model_global);
                        r.is_on_chip=false;
                        batch_info temp_info;
                        temp_info.node_id=model_root_gaddr.nodeID;
                        temp_info.flag=model_flag;
                        temp_info.offset=cur_off;
                        temp_info.length=1;
                        temp_info.is_leaf=false;
                        temp_info.gaddr=model_root_gaddr;
                        cur_off+=sizeof(model_global);
                        //std::cout<<"new root model"<<std::endl;
                        read_batch.push_back(r);
                        new_batch.push_back(temp_info);
                        continue;
                    }
                    if((slot_buf[j].next&set_flag)!=0)
                    {
                        //std::cout<<"slot set in"<<std::endl;
                        GlobalAddress set_gaddr;
                        set_gaddr.val=(slot_buf[j].next&mask_);
                        RdmaOpRegion r;
                        r.source=dest_start+cur_off;
                        r.dest=set_gaddr;
                        r.size=sizeof(node_set);
                        r.is_on_chip=false;
                        batch_info temp_info;
                        temp_info.flag=set_flag;
                        temp_info.offset=cur_off;
                        temp_info.node_id=batch_infos[i].node_id;
                        temp_info.length=1;
                        new_batch.push_back(temp_info);
                        read_batch.push_back(r);
                        cur_off+=(sizeof(node_set));
                    }
                }
            }
            else if(batch_infos[i].flag==set_flag)
            {
                if(batch_infos[i].length==1)
                {
                    node_set *node_set_buf=(node_set*)(dest_start+batch_infos[i].offset);
                    uint64_t start_off=cur_off;
                    //std::cout<<"set num:"<<node_set_buf[0].node_num<<std::endl;
                    int length=0;
                    for(int j=0;j<node_set_buf[0].node_num;j++)
                    {
                        if(node_set_buf[0].nodes_gaddr[j]!=null_next)
                        {
                            RdmaOpRegion r;
                            r.source=dest_start+cur_off;
                            r.dest=node_set_buf[0].nodes_gaddr[j];
                            r.size=sizeof(list_node);
                            r.is_on_chip=false;
                            cur_off+=(sizeof(list_node));
                            read_batch.push_back(r);
                            length++;
                        }
                    }
                    batch_info temp_info;
                    temp_info.flag=set_flag;
                    temp_info.offset=start_off;
                    temp_info.node_id=batch_infos[i].node_id;
                    temp_info.length=length;
                    new_batch.push_back(temp_info);
                }
                else
                {
                    list_node *nodes=(list_node*)(dest_start+batch_infos[i].offset);
                    //std::cout<<"set node num:"<<batch_infos[i].length<<std::endl;
                    for(int j=batch_infos[i].length-1;j>=0;j--)
                    {
                        if(nodes[j].key>=key_start&&nodes[j].key<=key_end)
                        {
                            kv temp_kv;
                            temp_kv.key=nodes[j].key;
                            temp_kv.val=nodes[j].val;
                            temp_kv.index=loop+30;
                            chain_kvs.push_back(temp_kv);
                        }
                        //std::cout<<"loop in j"<<std::endl;
                    }
                }
            }
            //std::cout<<"loop upper"<<std::endl;
        }
        //std::cout<<"loop"<<std::endl;
        loop++;
        dsm->read_batches_sync(read_batch);
        batch_infos=new_batch;
    }
    std::unordered_map<std::uint64_t,kv> hash;
    int flag=0;
    for(int i=chain_kvs.size()-1;i>=0;i--)
    {
        hash[chain_kvs[i].key]=chain_kvs[i];
    }
    for(int i=0;i<slot_kvs.size();i++)
    {
        hash[slot_kvs[i].key]=slot_kvs[i];
    }
    for(auto kv_now:hash)
    {
        //kv temp_kv;
        //temp_kv.key=kv_now.first;
        //temp_kv.val=kv_now.second;
        /*if(kv_now.second.val!=100)
        {
            flag=2;
        }*/
        kvs.push_back(kv_now.second);
    }
    /*if(flag==2)
    {
        for(int i=0;i<chain_kvs.size();i++)
        {
            std::cout<<chain_kvs[i].key<<" "<<chain_kvs[i].val<<" "<<chain_kvs[i].index<<std::endl;
        }
        std::cout<<std::endl;
        for(int i=0;i<slot_kvs.size();i++)
        {
            std::cout<<slot_kvs[i].key<<" "<<slot_kvs[i].val<<slot_kvs[i].index<<std::endl;
        }
    }*/
    return true;
}

bool learned_index_global::update(uint64_t &key,uint64_t &val,int thread_id)
{
    char *cache_ptr=dsm->get_rdma_buffer();
    uint64_t *rdam_buf_ptr=(uint64_t*)cache_ptr;
    slot *read_buf=(slot*)cache_ptr;
    uint64_t *next_off_buf=(uint64_t*)cache_ptr;
    model_global seg_model;
    int model_now;
    bool res=model_search(key,seg_model,model_now);
    int write_buffer_index=seg_model.write_buffer_index;
    write_buffer *write_buffer_=write_buffers[write_buffer_index];
    if(enable_cache)
    {
        learned_index_global *model_in_cache=nullptr;
        uint64_t range_start;
        uint64_t range_end;
        bool find=search_cache(model_now,key,&model_in_cache,range_start,range_end);
        if(find)//if find, then it can't be a slot, only do insert
        {
            if(model_in_cache->Epsilon==0)
            {
                std::cout<<"update in cache set"<<std::endl;
                GlobalAddress set_gaddr=model_in_cache->level_models[0][0].child_start;
                GlobalAddress slot_gaddr=model_in_cache->level_models[0][0].sibling;
                GlobalAddress write_slot_gaddr;
                GlobalAddress addr_to_cas=slot_gaddr;
                int write_buffer_flag=0; 
                dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                slot slot_in_cache=read_buf[0];
                if(((read_buf[0].next)&model_flag)==0)
                {
                    while(1)
                    {
                        if((slot_in_cache.next&null_flag)!=0)
                        {
                            uint64_t next=slot_gaddr.val|tail_flag;
                            bool get_slot_ret = write_buffer_ -> get_slot(thread_id, slot_gaddr.nodeID, write_slot_gaddr,nullptr);
                            if(!get_slot_ret)
                            {
                                std::cout<<"alloc error"<<std::endl;
                                return false;
                            }
                            uint64_t *cas_source = rdam_buf_ptr;
                            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                            uint64_t equal = slot_in_cache.next;
                            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                            /*if(!write_buffer_->write_slot(thread_id,slot_gaddr.nodeID,key,val,next,write_slot_gaddr))
                            {
                                //write_buffer_->free_slot(ori_write_slot_gaddr);
                                return false;
                            }*/
                            write_buffer_flag=1;
                            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                            if(ret)
                            {
                                return true;
                            }
                            else
                            {
                                //std::cout<<"cas after ori fail"<<std::endl;
                                //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                //slot_in_cache=read_buf[0];
                                slot_in_cache.next = *cas_source;
                                continue;
                            }
                        }
                        else if((slot_in_cache.next&chain_flag)!=0||(slot_in_cache.next&set_flag)!=0)
                        {
                            //std::cout<<"chain flag"<<std::endl;
                            //std::cout<<"flag=1"<<std::endl;
                            GlobalAddress next_gaddr;
                            next_gaddr.val=(slot_in_cache.next&mask_);
                            uint64_t next;
                            if((slot_in_cache.next&chain_flag)!=0)
                            {
                                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
                            }
                            else
                            {
                                next=slot_in_cache.next;
                            }
                            if(write_buffer_flag==0)
                            {
                                bool get_slot_ret = write_buffer_->get_slot(thread_id, slot_gaddr.nodeID, write_slot_gaddr,nullptr);
                                if(!get_slot_ret)
                                {
                                    std::cout<<"alloc error"<<std::endl;
                                    return false;
                                }
                                uint64_t *cas_source = rdam_buf_ptr;
                                uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                                uint64_t equal = slot_in_cache.next;
                                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                                bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                                /*if(!write_buffer_->write_slot(thread_id,slot_gaddr.nodeID,key,val,next,write_slot_gaddr))
                                {
                                    return false;
                                }*/
                                write_buffer_flag=1;
                                //bool ret=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                                if(ret)
                                {
                                    return true;
                                }
                                else
                                {   
                                    //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                    //slot_in_cache=read_buf[0];
                                    slot_in_cache.next = *cas_source;
                                    continue;
                                }
                            }
                            else
                            {
                                GlobalAddress write_slot_next_gaddr=write_slot_gaddr;
                                write_slot_next_gaddr.offset+=(2*sizeof(uint64_t));
                                //next_off_buf[0]=next;
                                uint64_t *cas_source = rdam_buf_ptr;
                                uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                                uint64_t equal = slot_in_cache.next;
                                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                                bool ret = write_next_and_cas(write_slot_next_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, next);
                                //dsm->write_sync(cache_ptr,write_slot_next_gaddr,sizeof(uint64_t));
                                //bool ret=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                                if(ret)
                                {
                                    return true;
                                }
                                else
                                {
                                    //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                    slot_in_cache.next = *cas_source;
                                    //slot_in_cache=read_buf[0];
                                    continue;
                                }
                            }
                        }
                        else if((slot_in_cache.next&model_flag)!=0)
                        {
                            //std::cout<<"read model"<<std::endl;
                            std::cout<<"update in cache model"<<std::endl;
                            GlobalAddress root;
                            root.val=(slot_in_cache.next&mask_);
                            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                            temp->read_model_from_remote(root);
                            int cache_index;
                            cache_model_range(model_now,range_start,range_end,temp,cache_index);
                            GlobalAddress new_slot;
                            write_buffer_->get_slot(thread_id,slot_gaddr.nodeID,new_slot,nullptr);
                            bool ret=temp->sub_update(new_slot,key,val,thread_id);
                            return ret;
                        }
                    }
                }
                else
                {
                    GlobalAddress root;
                    root.val=(slot_in_cache.next&mask_);
                    learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                    temp->read_model_from_remote(root);
                    int cache_index;
                    cache_model_range(model_now,range_start,range_end,temp,cache_index);
                    std::cout<<"set to model in update"<<std::endl;
                    GlobalAddress new_slot;
                    write_buffer_->get_slot(thread_id,slot_gaddr.nodeID,new_slot,nullptr);
                    bool ret=temp->sub_update(new_slot,key,val,thread_id);
                    return ret;
                }
            }
            else
            {
                GlobalAddress new_slot;
                write_buffer_->get_slot(thread_id,model_in_cache->level_models[0][0].child_start.nodeID,new_slot,nullptr);
                std::cout<<"update in model"<<std::endl;
                bool ret=model_in_cache->sub_update(new_slot,key,val,thread_id);
                return ret;
            }
        }
    }
    long double predict=seg_model.slope*key+seg_model.intercept;
    int start=std::max<int>(0,(int)predict-(int)Epsilon);
    if(start>seg_model.child_length-1)
    {
        start=seg_model.child_length-1;
    }
    int end=std::min<int>(seg_model.child_length-1,(int)predict+Epsilon+1);
    int kv_size=end-start+1;
    GlobalAddress target=seg_model.child_start;
    target.offset+=start*sizeof(slot);
    dsm->read_sync(cache_ptr,target,kv_size*sizeof(slot));
    int pos=0;
    slot_binary_search(read_buf,kv_size,key,pos);
    //std::cout<<"size of slot: "<<sizeof(slot)<<std::endl;
    //std::cout<<"pos: "<<pos<<std::endl;
    target.offset+=pos*sizeof(slot);
    slot slot_=read_buf[pos];
    int write_buffer_flag=0;
    GlobalAddress write_slot_gaddr;
    GlobalAddress val_addr_to_cas=target;
    val_addr_to_cas.offset+=(sizeof(uint64_t)*2);
    GlobalAddress addr_to_cas=target;
    //addr_to_cas.offset+=2*sizeof(uint64_t);
    //std::cout<<"intsert to buffer "<<write_buffer_index<<std::endl;
    
    //std::cout<<"write once"<<std::endl;
    //std::cout<<"gaddr: "<<gaddr2str(target)<<std::endl;
    if(slot_.key==key)
    {
        while (1)
        {
            bool ret=dsm->cas_sync(val_addr_to_cas,slot_.val,val,rdam_buf_ptr);
            if(ret)
            {
                return true;
            }
            else
            {
                dsm->read_sync(cache_ptr,target,sizeof(slot));
                slot_=read_buf[0];
            }
        }
    }
    while(1)
    {
        if((slot_.next&null_flag)!=0)
        {
            uint64_t next=target.val|tail_flag;
            bool get_slot_ret = write_buffer_ -> get_slot(thread_id, target.nodeID, write_slot_gaddr,nullptr);
            if(!get_slot_ret)
            {
                std::cout<<"alloc error"<<std::endl;
                return false;
            }
            uint64_t* cas_source = rdam_buf_ptr;
            uint64_t* write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
            uint64_t equal = slot_.next;
            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
            /*if(!write_buffer_->write_slot(thread_id,target.nodeID,key,val,next,write_slot_gaddr))
            {
                //write_buffer_->free_slot(ori_write_slot_gaddr);
                return false;
            }*/
            write_buffer_flag=1;
            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
            if(ret)
            {
                return true;
            }
            else
            {
                //std::cout<<"cas after ori fail"<<std::endl;
                //dsm->read_sync(cache_ptr,target,sizeof(slot));
                //slot_=read_buf[0];
                slot_.next = *cas_source;
                continue;
            }
        }
        else if((slot_.next&chain_flag)!=0||(slot_.next&set_flag)!=0)
        {
            //std::cout<<"chain flag"<<std::endl;
            //std::cout<<"flag=1"<<std::endl;
            GlobalAddress next_gaddr;
            next_gaddr.val=(slot_.next&mask_);
            uint64_t next;
            if((slot_.next&chain_flag)!=0)
            {
                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
            }
            else
            {
                next=slot_.next;
            }
            if(write_buffer_flag==0)
            {
                bool get_slot_ret = write_buffer_ -> get_slot(thread_id, target.nodeID, write_slot_gaddr,nullptr);
                if(!get_slot_ret)
                {
                    std::cout<<"alloc error"<<std::endl;
                    return false;
                }
                uint64_t *cas_source = rdam_buf_ptr;
                uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                uint64_t equal = slot_.next;
                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                /*if(!write_buffer_->write_slot(thread_id,target.nodeID,key,val,next,write_slot_gaddr))
                {
                    return false;
                }*/
                write_buffer_flag=1;
                //bool ret=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                if(ret)
                {
                    return true;
                }
                else
                {   
                    //dsm->read_sync(cache_ptr,target,sizeof(slot));
                    //slot_=read_buf[0];
                    slot_.next = *cas_source;
                    continue;
                }
            }
            else
            {
                GlobalAddress write_slot_next_gaddr=write_slot_gaddr;
                write_slot_next_gaddr.offset+=(2*sizeof(uint64_t));
                //next_off_buf[0]=next;
                uint64_t *cas_source = rdam_buf_ptr;
                uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                uint64_t equal = slot_.next;
                uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                bool ret = write_next_and_cas(write_slot_next_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, next);
                //dsm->write_sync(cache_ptr,write_slot_next_gaddr,sizeof(uint64_t));
                //bool ret=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                if(ret)
                {
                    return true;
                }
                else
                {
                    //dsm->read_sync(cache_ptr,target,sizeof(slot));
                    //slot_=read_buf[0];
                    slot_.next = *cas_source;
                    continue;
                }
            }
        }
        else if((slot_.next&model_flag)!=0)
        {
            //std::cout<<"read model"<<std::endl;
            GlobalAddress root;
            root.val=(slot_.next&mask_);
            int range_start=read_buf[pos].key+1;
            int range_end=0;
            if(pos!=kv_size-1)
            {
                range_end=read_buf[pos+1].key-1;
            }
            else
            {
                if(model_now==level_models[0].size()-1)
                {
                    range_end=0xffffffffffffffff-1;
                }
                else
                {
                    range_end=level_models[0][model_now+1].key_start-1;
                }
            }
            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
            temp->read_model_from_remote(root);
            int cache_index;
            cache_model_range(model_now,range_start,range_end,temp,cache_index);
            GlobalAddress new_slot;
            write_buffer_->get_slot(thread_id,target.nodeID,new_slot,nullptr);
            //learned_index_global *cached_model=nullptr;
            //get_cache(model_now,cache_index,cached_model);
            bool ret=temp->sub_update(new_slot,key,val,thread_id);
            return ret;
        }
    }
}

bool learned_index_global::sub_update(GlobalAddress new_slot,uint64_t key,uint64_t val,int thread_id)
{
    char *cache_ptr=dsm->get_rdma_buffer();
    uint64_t *rdam_buf_ptr=(uint64_t*)cache_ptr;
    slot *read_buf=(slot*)cache_ptr;
    uint64_t *next_off_buf=(uint64_t*)cache_ptr;
    model_global seg_model;
    int model_now;
    bool res=model_search(key,seg_model,model_now);
    int write_buffer_index=seg_model.write_buffer_index;
    write_buffer *write_buffer_=write_buffers[write_buffer_index];
    if(enable_cache)
    {
        learned_index_global *model_in_cache=nullptr;
        uint64_t range_start;
        uint64_t range_end;
        bool find=search_cache(model_now,key,&model_in_cache,range_start,range_end);
        if(find)
        {
            if(model_in_cache->Epsilon==0)
            {
                GlobalAddress set_gaddr=model_in_cache->level_models[0][0].child_start;
                GlobalAddress slot_gaddr=model_in_cache->level_models[0][0].sibling;
                GlobalAddress write_slot_gaddr=new_slot;
                GlobalAddress addr_to_cas=slot_gaddr; 
                dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                slot slot_in_cache=read_buf[0];
                if(((read_buf[0].next)&model_flag)==0)
                {
                    while(1)
                    {
                        if((slot_in_cache.next&null_flag)!=0)
                        {
                            uint64_t next=slot_gaddr.val|tail_flag;
                            uint64_t *cas_source = rdam_buf_ptr;
                            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                            uint64_t equal = slot_in_cache.next;
                            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                            //write_buffer_->fill_slot(key,val,next,new_slot);
                            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                            if(ret)
                            {
                                return true;
                            }
                            else
                            {
                                //std::cout<<"cas after ori fail"<<std::endl;
                                //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                //slot_in_cache=read_buf[0];
                                slot_in_cache.next = *cas_source;
                                continue;
                            }
                        }
                        else if((slot_in_cache.next&chain_flag)!=0||(slot_in_cache.next&set_flag)!=0)
                        {
                            //std::cout<<"chain flag"<<std::endl;
                            //std::cout<<"flag=1"<<std::endl;
                            GlobalAddress next_gaddr;
                            next_gaddr.val=(slot_in_cache.next&mask_);
                            uint64_t next;
                            if((slot_in_cache.next&chain_flag)!=0)
                            {
                                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
                            }
                            else
                            {
                                next=slot_in_cache.next;
                            }
                            uint64_t *cas_source = rdam_buf_ptr;
                            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
                            uint64_t equal = slot_in_cache.next;
                            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
                            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
                            //write_buffer_->fill_slot(key,val,next,new_slot);
                            //bool ret=dsm->cas_sync(addr_to_cas,slot_in_cache.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
                            if(ret)
                            {
                                return true;
                            }
                            else
                            {
                                //dsm->read_sync(cache_ptr,slot_gaddr,sizeof(slot));
                                //slot_in_cache=read_buf[0];
                                slot_in_cache.next = *cas_source;
                                continue;
                            }
                        }
                        else if((slot_in_cache.next&model_flag)!=0)
                        {
                            //std::cout<<"read model"<<std::endl;
                            GlobalAddress root;
                            root.val=(slot_in_cache.next&mask_);
                            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                            temp->read_model_from_remote(root);
                            int cache_index;
                            cache_model_range(model_now,range_start,range_end,temp,cache_index);
                            //GlobalAddress new_slot;
                            //write_buffer_->get_slot(thread_id,slot_gaddr.nodeID,new_slot);
                            //learned_index_global *cached_model=nullptr;
                            //get_cache(model_now,cache_index,cached_model);
                            bool ret=temp->sub_update(new_slot,key,val,thread_id);
                            return ret;
                        }
                    }
                }
                else
                {
                    GlobalAddress root;
                    root.val=(slot_in_cache.next&mask_);
                    learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
                    temp->read_model_from_remote(root);
                    int cache_index;
                    cache_model_range(model_now,range_start,range_end,temp,cache_index);
                    //GlobalAddress new_slot;
                    //write_buffer_->get_slot(thread_id,slot_gaddr.nodeID,new_slot);
                    //learned_index_global *cached_model=nullptr;
                    //get_cache(model_now,cache_index,cached_model);
                    bool ret=temp->sub_update(new_slot,key,val,thread_id);
                    return ret;
                }
            }
            else
            {
                //GlobalAddress new_slot;
                //write_buffer_->get_slot(thread_id,model_in_cache->level_models[0][0].child_start.nodeID,new_slot);
                bool ret=model_in_cache->sub_update(new_slot,key,val,thread_id);
                return ret;
            }
        }
    }
    long double predict=seg_model.slope*key+seg_model.intercept;
    int start=std::max<int>(0,(int)predict-(int)Epsilon);
    if(start>seg_model.child_length-1)
    {
        start=seg_model.child_length-1;
    }
    int end=std::min<int>(seg_model.child_length-1,(int)predict+Epsilon+1);
    int kv_size=end-start+1;
    GlobalAddress target=seg_model.child_start;
    target.offset+=start*sizeof(slot);
    dsm->read_sync(cache_ptr,target,kv_size*sizeof(slot));
    int pos=0;
    slot_binary_search(read_buf,kv_size,key,pos);
    //std::cout<<"size of slot: "<<sizeof(slot)<<std::endl;
    //std::cout<<"pos: "<<pos<<std::endl;
    target.offset+=pos*sizeof(slot);
    slot slot_=read_buf[pos];
    GlobalAddress write_slot_gaddr=new_slot;
    GlobalAddress val_addr_to_cas=target;
    val_addr_to_cas.offset+=(sizeof(uint64_t)*2);
    GlobalAddress addr_to_cas=target;
    //addr_to_cas.offset+=2*sizeof(uint64_t);
    //std::cout<<"intsert to buffer "<<write_buffer_index<<std::endl;
    
    //std::cout<<"write once"<<std::endl;
    //std::cout<<"gaddr: "<<gaddr2str(target)<<std::endl;
    if(slot_.key==key)
    {
        while (1)
        {
            bool ret=dsm->cas_sync(val_addr_to_cas,slot_.val,val,rdam_buf_ptr);
            if(ret)
            {
                return true;
            }
            else
            {
                dsm->read_sync(cache_ptr,target,sizeof(slot));
                slot_=read_buf[0];
            }
        }
    }
    while(1)
    {
        if((slot_.next&null_flag)!=0)
        {
            uint64_t next=target.val|tail_flag;
            uint64_t *cas_source = rdam_buf_ptr;
            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
            uint64_t equal = slot_.next;
            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
            //write_buffer_->fill_slot(key,val,next,new_slot);
            //bool ret_2=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
            if(ret)
            {
                return true;
            }
            else
            {
                //std::cout<<"cas after ori fail"<<std::endl;
                //dsm->read_sync(cache_ptr,target,sizeof(slot));
                //slot_=read_buf[0];
                slot_.next = *cas_source;
                continue;
            }
        }
        else if((slot_.next&chain_flag)!=0||(slot_.next&set_flag)!=0)
        {
            //std::cout<<"chain flag"<<std::endl;
            //std::cout<<"flag=1"<<std::endl;
            GlobalAddress next_gaddr;
            next_gaddr.val=(slot_.next&mask_);
            uint64_t next;
            if((slot_.next&chain_flag)!=0)
            {
                next=(next_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
            }
            else
            {
                next=slot_.next;
            }
            uint64_t *cas_source = rdam_buf_ptr;
            uint64_t *write_source = (uint64_t*)((uint64_t)rdam_buf_ptr + sizeof(uint64_t));
            uint64_t equal = slot_.next;
            uint64_t swap_val = write_slot_gaddr.val|chain_flag;
            bool ret = write_buffer_and_cas(write_slot_gaddr, addr_to_cas, write_source, cas_source, equal, swap_val, key, val, next);
            //write_buffer_->fill_slot(key,val,next,new_slot);
            //bool ret=dsm->cas_sync(addr_to_cas,slot_.next,write_slot_gaddr.val|chain_flag,rdam_buf_ptr);
            if(ret)
            {
                return true;
            }
            else
            {   
                //dsm->read_sync(cache_ptr,target,sizeof(slot));
                //slot_=read_buf[0];
                slot_.next = *cas_source;
                continue;
            }
        }
        else if((slot_.next&model_flag)!=0)
        {
            //std::cout<<"read model"<<std::endl;
            GlobalAddress root;
            root.val=(slot_.next&mask_);
            int range_start=read_buf[pos].key+1;
            int range_end=0;
            if(pos!=kv_size-1)
            {
                range_end=read_buf[pos+1].key-1;
            }
            else
            {
                if(model_now==level_models[0].size()-1)
                {
                    range_end=0xffffffffffffffff-1;
                }
                else
                {
                    range_end=level_models[0][model_now+1].key_start-1;
                }
            }
            learned_index_global *temp=new learned_index_global(dsm,define::Epsilon,write_buffer_conf_);
            temp->read_model_from_remote(root);
            int cache_index;
            cache_model_range(model_now,range_start,range_end,temp,cache_index);
            //learned_index_global *cached_model=nullptr;
            //get_cache(model_now,cache_index,cached_model);
            bool ret=temp->sub_update(new_slot,key,val,thread_id);
            return ret;
        }
    }
}

thread_local CoroCall LLDex::worker[define::kMaxCoro];
thread_local CoroCall LLDex::master;
thread_local uint64_t LLDex::coro_ops_total;
thread_local uint64_t LLDex::coro_ops_cnt_start;
thread_local uint64_t LLDex::coro_ops_cnt_finish;
thread_local CoroQueue LLDex::busy_waiting_queue;
// uint64_t latency[MAX_APP_THREAD][LATENCY_WINDOWS];

LLDex::LLDex(DSM *dsm, uint64_t Epsilon, write_buffer_conf write_buffer_conf_):
    dsm(dsm),Epsilon(Epsilon),write_buffer_conf_(write_buffer_conf_)
{
    write_buffers=(write_buffer**)malloc(sizeof(write_buffer*)*write_buffer_conf_.buffer_num);
    uint64_t single_buffer_size=write_buffer_conf_.buffer_size/write_buffer_conf_.buffer_num;
    int write_buffer_off=0;
    for(int i=0;i<write_buffer_conf_.buffer_num;i++)
    {
        write_buffers[i]=new write_buffer(i,write_buffer_off,single_buffer_size,write_buffer_conf_.thread_num,dsm);
        write_buffer_off+=(sizeof(uint64_t)*2+sizeof(int)*single_buffer_size+sizeof(list_node)*single_buffer_size);
    }

    local_lock_table = new LocalLockTable();
    // retrain_thread = std::thread(&LLDex::retrain_worker, this);
    return;
}

LLDex::~LLDex()
{
    // {
    //     std::unique_lock<std::mutex> lock(retrain_mutex);
    //     retrain_running = false;
    // }
    // retrain_cv.notify_all();
    // if (retrain_thread.joinable()) {
    //     retrain_thread.join();
    // }
}

void LLDex::read_model_from_remote(GlobalAddress root, CoroContext *ctx)
{
    // char *cache_ptr = dsm -> get_rdma_buffer();
    char *cache_ptr = dsm->get_coro_buf(ctx ? ctx->coro_id : 0);
    model_global *read_buf = (model_global*)cache_ptr;
    dsm -> read_sync(cache_ptr,root,sizeof(model_global), ctx);
    std::vector<std::vector<model_global>> temp_level_models;
    std::vector<model_global> models;
    models.push_back(*read_buf);
    temp_level_models.push_back(models);
    volatile int level = 0;
    while(!temp_level_models[level][0].is_leaf)
    {
        std::vector<model_global> temp_models;
        for(int i = 0; i < temp_level_models[level].size(); i++)
        {
            dsm->read_sync(cache_ptr,temp_level_models[level][i].child_start,sizeof(model_global)*temp_level_models[level][i].child_length, ctx);
            int data_num = temp_level_models[level][i].child_length;
            for(int j = 0; j < data_num; j++)
            {
                temp_models.push_back(read_buf[j]);
            }
        }
        temp_level_models.push_back(temp_models);
        level++;
    }
    //std::cout<<"full model num: "<<temp_level_models[0][0].full_model_num<<std::endl;
    std::vector<model_mincost> leaf_models;
    for(int i = 0; i < temp_level_models[temp_level_models.size() - 1].size(); i++)
    {
        model_mincost temp_model;
        temp_model.intercept = temp_level_models[temp_level_models.size() - 1][i].intercept;
        temp_model.slope = temp_level_models[temp_level_models.size() - 1][i].slope;
        temp_model.pos_start = temp_level_models[temp_level_models.size() - 1][i].child_start;
        temp_model.key_start = temp_level_models[temp_level_models.size() - 1][i].key_start;
        temp_model.child_length = temp_level_models[temp_level_models.size() - 1][i].child_length;
        leaf_models.push_back(temp_model);
    }
    level_models.push_back(leaf_models);
    for(int i = temp_level_models.size() - 2; i >= 0; i--)
    {
        std::vector<model_mincost> inner_models;
        for(int j = 0; j < temp_level_models[i].size(); j++)
        {
            model_mincost temp_model;
            temp_model.slope = temp_level_models[i][j].slope;
            temp_model.intercept = temp_level_models[i][j].intercept;
            temp_model.pos_start = temp_level_models[i][j].pos_start;
            temp_model.key_start = temp_level_models[i][j].key_start;
            temp_model.child_length = temp_level_models[i][j].child_length;
            inner_models.push_back(temp_model);
        }
        level_models.push_back(inner_models);
    }
    // cache.resize(level_models[0].size());
    // cached_num.resize(level_models[0].size());
    // for(int i = 0; i < cached_num.size(); i++)
    // {
    //     cached_num[i] = 0;
    // }
    // cache_content.resize(level_models[0].size());
    // std::vector<std::shared_mutex> list(level_models[0].size());
    // cache_lock.swap(list);

    cache.resize(level_models[0].size());
    cached_num.resize(level_models[0].size());
    for(int i=0;i<cached_num.size();i++)
    {
        cached_num[i]=0;
    }
    cache_content.resize(level_models[0].size());
    std::vector<std::shared_mutex> list(level_models[0].size());
    cache_lock.swap(list);
    return;
}

void LLDex::print_level_model()
{
    for(int i=0;i<level_models.size();i++)
    {
        std::cout<<"level: "<<i<<"model num:"<<level_models[i].size()<<std::endl;
        // for(int j=0;j<level_models[i].size();j++)
        // {
        //     print_load_model(level_models[i][j]);
        // }
    }
    // std::cout<<"8 model data_num:"<<level_models[0][8].child_length<<std::endl;
    return;
}

bool LLDex::is_set(LLDex *model_in_cache)
{
    if(model_in_cache->Epsilon == 0)
    {
        return true;
    }
    else 
    {
        return false;
    }
}

//[2，length - 2]范围内的预测一定被精准预测，不存在被偏移的情况
void LLDex::get_range_start_end(model_mincost &seg_model, uint64_t predict_int, int model_now, uint64_t &range_start, uint64_t &range_end)
{
    //std::cout<<"range start predict int: "<<predict_int<<std::endl;
    if(predict_int == 0)
    {
        range_start = seg_model.key_start;
        long double predict_next_pos = predict_int + 1;
        long double range_end_double = (predict_next_pos - seg_model.intercept) / seg_model.slope;  
        range_end = std::floor(range_end_double);
        //std::cout<<"seg model start:"<<seg_model.key_start<<std::endl;
        return;
    }
    if(predict_int == seg_model.child_length - 1)
    {
        long double predict_pre_pos = predict_int - 1;
        long double range_start_double = (predict_pre_pos - seg_model.intercept) / seg_model.slope;
        range_start = std::ceil(range_start_double);
        if(model_now == level_models[0].size() - 1)
        {
            range_end = 0xffffffffffffffff - 1;
        }
        else
        {
            range_end = level_models[0][model_now + 1].key_start - 1;
        }
        return;
    }
    long double predict_pre_pos = predict_int;
    long double range_start_double = (predict_pre_pos - seg_model.intercept) / seg_model.slope;
    //std::cout<<"range start double"<<range_start_double<<std::endl;
    range_start = std::ceil(range_start_double) > 0 ? std::ceil(range_start_double) : 0;
    //std::cout<<"range start"<<range_start<<std::endl;
    long double predict_next_pos = predict_int + 1;
    long double range_end_double = (predict_next_pos - seg_model.intercept) / seg_model.slope;  
    range_end = std::floor(range_end_double);
    return;
}

//逻辑重写
void LLDex::cache_model(int model_now, uint64_t range_start, uint64_t range_end, LLDex *temp_model, int &cache_index)
{
    cache_lock[model_now].lock();
    cache_index=cached_num[model_now];
    cache_content[model_now].push_back(temp_model);
    cached_num[model_now]++;
    cache[model_now].add(std::make_pair(ival(range_start,range_end),cache_index));
    cache_lock[model_now].unlock();
}

bool LLDex::search_cache(int &model_now,uint64_t key,LLDex **cache_content_,uint64_t &range_start,uint64_t &range_end)
{
    cache_lock[model_now].lock();
    auto it=cache[model_now].find(key);
    if(it==cache[model_now].end())
    {
        cache_lock[model_now].unlock();
        return false;
    }
    else
    {
        volatile int cache_index=it->second;
        range_start=it->first.lower();
        range_end=it->first.upper();
        *cache_content_=cache_content[model_now][cache_index];
        cache_lock[model_now].unlock();
        assert(*cache_content_!=nullptr);
        return true;
    }
    return false;
}

void LLDex::get_cache(int &model_now,int content_index,LLDex **cache_content_)
{
    cache_lock[model_now].lock_shared();
    *cache_content_=(cache_content[model_now][content_index]);
    cache_lock[model_now].unlock_shared();
}

RdmaOpRegion LLDex::fill_RdmaOpRegion(uint64_t source, uint64_t dest, uint64_t size, bool is_on_chip)
{
    RdmaOpRegion ret;
    ret.source = source;
    ret.dest = dest;
    ret.size = size;
    ret.is_on_chip  = is_on_chip;
    return ret;
}

batch_info LLDex::fill_BatchInfo(GlobalAddress gaddr, uint64_t flag, int write_buffer_index, int offset, int node_id, bool is_leaf, int length)
{
    batch_info ret;
    ret.gaddr = gaddr;
    ret.flag = flag;
    ret.write_buffer_index = write_buffer_index;
    ret.offset = offset;
    ret.node_id = node_id;
    ret.is_leaf = is_leaf;
    ret.length = length;
    return ret;
}

void LLDex::fill_cacheset_op(uint64_t dest_start, LLDex *model_in_cache, std::vector<RdmaOpRegion> &cache_set_op)
{
    GlobalAddress set_gaddr;
    set_gaddr.val = model_in_cache->level_models[0][0].pos_start;
    GlobalAddress slot_gaddr;
    slot_gaddr.val = model_in_cache->level_models[0][0].key_start;
    RdmaOpRegion r_set;
    RdmaOpRegion r_slot;
    r_set.source=dest_start;
    r_set.dest=set_gaddr;
    r_set.size=sizeof(node_set);
    r_set.is_on_chip=false;
    r_slot.source=dest_start+sizeof(node_set);
    r_slot.dest=slot_gaddr;
    r_slot.size=sizeof(slot);
    r_slot.is_on_chip=false;
    cache_set_op.push_back(r_set);
    cache_set_op.push_back(r_slot);
}

bool LLDex::model_binary_search(int level,int model,uint64_t key,int &next_model)
{
    if(level<1||level>level_models.size()-1)
    {
        return false;
    }
    if(level_models[level][model].child_length==1)
    {
        next_model=0;
        return true;
    }
    long double predict=level_models[level][model].slope*key+level_models[level][model].intercept;
    if(predict<0)
    {
        predict=0;
    }
    else if(predict>level_models[level][model].child_length-1)
    {
        predict=level_models[level][model].child_length-1;
    }
    int64_t predict_in=(int64_t)predict+level_models[level][model].pos_start;
    int64_t low=std::max<int64_t>(0,(predict_in-Epsilon));
    int64_t high=std::min<int64_t>((predict_in+Epsilon),(level_models[level][model].pos_start+level_models[level][model].child_length-1));
    if((key>=level_models[level-1][predict_in].key_start&&predict_in==level_models[level][model].child_length-1+level_models[level][model].pos_start)||
        (key>=level_models[level-1][predict_in].key_start&&key<level_models[level-1][predict_in+1].key_start))
    {
        next_model=predict_in;
        return true;
    }
    else if(key<level_models[level-1][predict_in].key_start)
    {
        high=predict_in-1;
    }
    else if(key>=level_models[level-1][predict_in+1].key_start)
    {
        low=predict_in+1;
    }
    while(low<=high)
    {
        int mid=(high+low)>>1;
        if((key>=level_models[level-1][mid].key_start&&mid==level_models[level][model].child_length-1+level_models[level][model].pos_start)||
        (key>=level_models[level-1][mid].key_start&&key<level_models[level-1][mid+1].key_start))
        {
            next_model=mid;
            return true;
        }
        else if(key<level_models[level-1][mid].key_start)
        {
            high=mid-1;
        }
        else if(key>=level_models[level-1][mid+1].key_start)
        {
            low=mid+1;
        }
    }
    next_model=low;
    return true;
}

bool LLDex::model_search(uint64_t &key, model_mincost &target_model, int &model_now)
{
    if(level_models.size()==0)
    {
        return false;
    }
    int level_now=level_models.size()-1;
    model_now=0;
    //while(!level_models[level_now][model_now].is_leaf)
    while(level_now != 0)
    {
        bool res=model_binary_search(level_now,model_now,key,model_now);
        //std::cout<<"level_now: "<<level_now<<"model_now"<<model_now<<std::endl;
        if(!res)
        {
            std::cout<<"error in binary"<<std::endl;
            return false;
        }
        level_now-=1;
    }
    target_model=level_models[level_now][model_now];
    if((key>=level_models[level_now][model_now].key_start&&model_now==level_models[level_now].size()-1)
    ||(key>=level_models[level_now][model_now].key_start&&key<level_models[level_now][model_now+1].key_start))
    {
        return true;
    }
    else
    {
        std::cout<<"key: "<<key<<"level_models[level_now][model_now].key_start: "<<level_models[level_now][model_now].key_start<<std::endl;
        std::cout<<"error in judge"<<std::endl;
        return false;
    }
}

bool LLDex::model_scan(uint64_t &key_start,uint64_t &key_end,std::vector<model_mincost>& target_models)
{
    if(level_models.size()==0)
    {
        return false;
    }
    int level_now=level_models.size()-1;
    int model_start_now=0;
    int model_end_now=0;
    while(level_now != 0)
    {
        bool res_start=model_binary_search(level_now,model_start_now,key_start,model_start_now);
        bool res_end=model_binary_search(level_now,model_end_now,key_end,model_end_now);
        //std::cout<<"level_now: "<<level_now<<"model_now"<<model_now<<std::endl;
        if(!res_start||!res_end)
        {
            std::cout<<"error in binary"<<std::endl;
            return false;
        }
        level_now-=1;
    }
    //target_model=level_models[level_now][model_now];
    if((key_start>=level_models[level_now][model_start_now].key_start&&model_start_now==level_models[level_now].size()-1)
    ||(key_start>=level_models[level_now][model_start_now].key_start&&key_start<level_models[level_now][model_start_now+1].key_start))
    {
        if((key_end>=level_models[level_now][model_end_now].key_start&&model_end_now==level_models[level_now].size()-1)
        ||(key_end>=level_models[level_now][model_end_now].key_start&&key_end<level_models[level_now][model_end_now+1].key_start))
        {
            for(int i=model_start_now;i<=model_end_now;i++)
            {
                target_models.push_back(level_models[level_now][i]);
            }
        }
        return true;
    }
    else
    {
        std::cout<<"key: "<<key_start<<"level_models[level_now][model_now].key_start: "<<level_models[level_now][model_start_now].key_start<<std::endl;
        std::cout<<"error in judge"<<std::endl;
        return false;
    }
}

bool LLDex::kvs_binary_search(kv_pair* kvs, int size, uint64_t key, int &pos)
{
    if(kvs[0].key>key)
    {
        return false;
    }
    int left=0;
    int right=size-1;
    while(left<=right)
    {
        int mid=(left+right)/2;
        if(kvs[mid].key>key)
        {
            right=mid-1;
        }
        else
        {
            left=mid+1;
        }
    }
    if(right<0)
    {
        right=0;
    }
    pos=right; 
    return true;
}

bool LLDex::kvs_search(kv_pair* kvs, int size, uint64_t key, int &pos, bool &exit_empty)
{
    for(int i = 0; i < size; i++)
    {
        if(kvs[i].key == key)
        {
            pos = i;
            return true;
        }
        if(kvs[i].key == null_flag){
            exit_empty = true;
        }
    }
    return false;
}

void LLDex::find_key_or_empty(kv_pair* kvs, int size, uint64_t key, int &pos, int &empty_pos)
{
    pos = -1;
    empty_pos = -1;
    int temp_empty_pos = -1;
    bool empty_found = false;
    for(int i = 0; i < size; i++)
    {
        if(!empty_found)
        {
            if(kvs[i].key == null_flag)
            {
                empty_found = true;
                temp_empty_pos = i;
            }
        }
        if(kvs[i].key == key)
        {
            pos = i;
            return;
        }
    }
    empty_pos = temp_empty_pos;
    return;
}

bool LLDex::write_buffer_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t key, uint64_t val, uint64_t next, CoroContext *ctx)
{
    RdmaOpRegion write_ror;
    RdmaOpRegion cas_ror;
    list_node *list_node_ptr = (list_node*)write_source;
    list_node_ptr->key = key;
    list_node_ptr->val = val;
    list_node_ptr->next_offset = next;
    write_ror.source = (uint64_t)(write_source);
    write_ror.dest = write_gaddr;
    write_ror.size = sizeof(list_node);
    write_ror.is_on_chip = false;
    cas_ror.source = (uint64_t)(cas_source);
    cas_ror.dest = cas_gaddr;
    cas_ror.size = sizeof(uint64_t);
    cas_ror.is_on_chip = false;
    bool ret = dsm->write_cas_parallel_sync(write_ror, cas_ror, equal, swap_val, ctx);
    return ret;
}

bool LLDex::write_next_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t next, CoroContext *ctx)
{
    RdmaOpRegion write_ror;
    RdmaOpRegion cas_ror;
    *write_source = next;
    write_ror.source = (uint64_t)(write_source);
    write_ror.dest = write_gaddr;
    write_ror.size = sizeof(uint64_t);
    write_ror.is_on_chip = false;
    cas_ror.source = (uint64_t)(cas_source);
    cas_ror.dest = cas_gaddr;
    cas_ror.size = sizeof(uint64_t);
    cas_ror.is_on_chip = false;
    bool ret = dsm -> write_cas_parallel_sync(write_ror, cas_ror, equal, swap_val, ctx);
    return ret;
}

bool LLDex::search(uint64_t &key, uint64_t &val, CoroContext *ctx)
{
    bool search_ret = false;
    // char *cache_ptr=dsm->get_rdma_buffer();
    char *cache_ptr = dsm->get_coro_buf(ctx ? ctx->coro_id : 0);

    kv_pair *read_buf = (kv_pair*)cache_ptr;
    node_set *node_set_buf=(node_set*)cache_ptr;
    list_node *write_buf=(list_node*)cache_ptr;
    uint64_t dest_start=(uint64_t)cache_ptr;
    model_mincost seg_model; 
    int model_now;
    std::pair<bool, bool> lock_res = std::make_pair(false, false);
    bool read_handover = false;
    bool res = false;
    GlobalAddress model_gaddr;
    write_buffer *write_buffer_ = nullptr;
    long double predict = 0;
    int start = 0;
    int end = 0;
    int predict_int = 0;
    int kv_size = 0;
    GlobalAddress kv_target;
    int pos = 0;
    GlobalAddress target_next;
    std::vector<kv_pair> kvs;
    uint64_t range_start = 0;
    uint64_t range_end = 0;
    uint64_t next = 0;
    uint64_t test = 0;
    bool exit_empty = false;


    if(enable_read_delegation){
        lock_res = local_lock_table->acquire_local_read_lock(key, &busy_waiting_queue, ctx);
        read_handover = (lock_res.first && !lock_res.second);
    }
    if(read_handover){
        read_handover_num[dsm->getMyThreadID()]++;
        goto search_over;
    }
    

    res=model_search(key,seg_model,model_now);
    if(!res)
    {
        std::cout<<"search : error in search model"<<std::endl;
        std::cout<<"key:"<<key<<" val: "<<val<<std::endl;
        search_ret = res;
        goto search_over;
        // return res;
    }


    model_gaddr.val = seg_model.pos_start;
    write_buffer_ = write_buffers[model_gaddr.nodeID];
    predict=seg_model.slope*key+seg_model.intercept;
    start=std::max<int>(0,(int)predict-(int)Epsilon+1);
    end=std::min<int>(seg_model.child_length-1,(int)predict+Epsilon-1);
    
    if(predict<0) predict_int = 0;
    else if(predict > seg_model.child_length - 1) predict_int = seg_model.child_length - 1;
    else predict_int = (int)predict;
    if(start>end)
    {
        start=end;
    }
    kv_size = end - start + 1;
    //GlobalAddress next_target;
    kv_target.val = seg_model.pos_start;
    kv_target.offset += sizeof(kv_pair) * start;
    dsm->read_sync(cache_ptr, kv_target, sizeof(kv_pair) * kv_size, ctx);
    
    res = kvs_search(read_buf, kv_size, key, pos, exit_empty);
    // for(int i = 0; i < kv_size; i++)
    // {
    //     std::cout<<read_buf[i].key<<" "<<read_buf[i].val<<std::endl;
    // }
    // std::cout<<std::endl;
    if(res)
    {
        val = read_buf[pos].val;
        search_ret = true;
        goto search_over;
        // return true;
    }

    if(!res && exit_empty)
    {
        search_ret = false;
        goto search_over;
        // return false;
    }

    if(enable_cache)
    {
        LLDex *model_in_cache = nullptr;
        uint64_t range_start;
        uint64_t range_end;
        bool find = search_cache(model_now, key, &model_in_cache, range_start, range_end);
        if(find)
        {
        //    std::cout<<"find"<<std::endl;
           bool ret = model_in_cache->search(key, val, ctx);
           search_ret = ret;
           goto search_over;
        //    return ret;
        }
    }


    //kv_pair kv_pair_ = read_buf[pos];
    reread:
        target_next.val = seg_model.pos_start;
        target_next.offset += (sizeof(kv_pair) * seg_model.child_length + sizeof(uint64_t) * predict_int);
        kvs.clear();
        range_start = 0;
        range_end = 0;
        get_range_start_end(seg_model, predict_int, model_now, range_start, range_end);
        dsm->read_sync(cache_ptr, target_next, sizeof(uint64_t), ctx);
        next = *(uint64_t*)cache_ptr;
        if(((next)&chain_flag)!=0)
        {
            GlobalAddress list_node_gaddr;
            list_node_gaddr.val = next&mask_;
            dsm->read_sync(cache_ptr, list_node_gaddr, sizeof(list_node), ctx);
            list_node now = *(list_node*)cache_ptr;
            kvs.push_back({now.key, now.val});
            int chain_length = 0;
            do
            {
                chain_length++;
                if(now.key == key)
                {
                    val = now.val;
                    search_ret = true;
                    goto search_over;
                    // break;
                }
                if(now.next_offset==(target_next.val|tail_flag))
                {
                    search_ret=false;
                    goto search_over;
                    // break;
                }
                else if(now.next_offset == null_flag)
                {
                    dsm->read_sync(cache_ptr, target_next, sizeof(uint64_t), ctx);
                    next = *(uint64_t*)cache_ptr;
                    std::cout<<"goto in"<<key<<" "<<now.key<<std::endl;
                    //return false;
                    goto reread;
                }
                GlobalAddress next_list_node_gaddr;
                next_list_node_gaddr.nodeID = list_node_gaddr.nodeID;
                next_list_node_gaddr.offset = (write_buffer_->get_off_write_buffer() + sizeof(list_node) * (now.next_offset&mask_));
                dsm->read_sync(cache_ptr, next_list_node_gaddr, sizeof(list_node), ctx);
                now = *(list_node*)cache_ptr;
                kvs.push_back({now.key, now.val});
            }while(now.next_offset!=(target_next.val|tail_flag));
            if(now.key == key)
            {
                val = now.val;
                search_ret = true;
                goto search_over;
            }
            if(chain_length >= define::max_chain_length);
            {
                // if(!ret) 
                // {
                //     std::cout<<"errro return"<<std::endl;
                //     return false;
                // }
                while(now.next_offset != (target_next.val|tail_flag))
                {
                    GlobalAddress next_list_node_gaddr;
                    next_list_node_gaddr.nodeID = list_node_gaddr.nodeID;
                    next_list_node_gaddr.offset = (write_buffer_->get_off_write_buffer() + sizeof(list_node) * (now.next_offset&mask_));
                    dsm->read_sync(cache_ptr, next_list_node_gaddr, sizeof(list_node), ctx);
                    now = *(list_node*)cache_ptr;
                    kvs.push_back({now.key, now.val});
                }
                // std::cout<<"push range start: "<<range_start<<std::endl;
                // std::cout<<"retrain predict int: "<<predict_int<<std::endl;
                kvs.push_back({range_start, 0});
                std::vector<uint64_t> retrain_addr;
                retrain_addr.push_back(target_next.val);
                retrain_addr.push_back(next);
                RetrainTask task;
                task.next_addr_content = retrain_addr;
                task.chain_kvs = kvs;
                RetrainManager::getInstance().addTask(task); 
            }
            if(search_ret)
            {
                goto search_over;
            }
        }
        test = (next&model_flag);
        if(test != 0)
        {
            GlobalAddress root;
            root.val = next&mask_;
            LLDex *temp = new LLDex(dsm, Epsilon, write_buffer_conf_);
            temp->read_model_from_remote(root, ctx);
            if(enable_cache)
            {
                int cache_index;
                cache_model(model_now, range_start, range_end, temp, cache_index);
            }
            //std::cout<<"cache model and model search"<<std::endl;
            search_ret = temp->search(key, val, ctx);
            if(!search_ret)
            {
                std::cout<<"model error"<<std::endl;
            }
            goto search_over;
        }

search_over:
    if(enable_read_delegation){
        local_lock_table->release_local_read_lock(key, lock_res, search_ret, val);
    }
    // if(search_ret){
    //     return search_ret;
    // }
    // std::cout<<"return false"<<std::endl;
    // assert(1==0);

    // in ycsb-d, there are some searches for non-existing keys for the multi thread insert and search, return false.
    return search_ret;
}

int base_delay_us = 1; // 基础延迟，单位微秒
int max_delay_us = 1000; // 最大延迟，单位微秒
std::random_device rd;
std::mt19937 gen;
std::uniform_int_distribution<> dis;

int calculate_backoff(int retry_count) {
    // 指数增长：2^retry_count * base_delay
    int exponential = base_delay_us * (1 << retry_count);
    
    // 添加随机抖动（±10%）
    int jitter = dis(gen) - 50;  // -50 到 +50
    int jitter_percent = jitter;  // -50% 到 +50%
    
    int delay = exponential + (exponential * jitter_percent / 100);
    
    // 限制最大延迟
    return std::min(delay, max_delay_us);
}

bool LLDex::insert(uint64_t &key, uint64_t &val, int thread_id, CoroContext *ctx)
{
    bool insert_ret = false;
    try_lock_op[thread_id]++;

    char *cache_ptr = nullptr;
    kv_pair *read_buf = nullptr;
    model_mincost seg_model;
    GlobalAddress model_gaddr;
    write_buffer *write_buffer_ = nullptr;
    int model_now = 0;
    bool res = false;
    long double predict = 0;
    int start = 0;
    int end = 0;
    int kv_size = 0;
    int predict_int = 0;
    GlobalAddress kv_target;
    int pos = -1;
    int empty_pos = -1;
    GlobalAddress nextptr_target;
    uint64_t *nextptr_buf = nullptr;
    uint64_t next;
    int write_buffer_flag=0;

    bool write_handover = false;
    std::pair<bool, bool> lock_res = std::make_pair(false, false);
    if(enable_local_lock){
        lock_res = local_lock_table->acquire_local_write_lock(key, val, &busy_waiting_queue, ctx);
        write_handover = (lock_res.first && !lock_res.second);
    }
    
    if(write_handover){
        insert_ret = true;
        write_handover_num[thread_id]++;
        goto insert_finish;
    }

    cache_ptr = dsm->get_coro_buf(ctx ? ctx->coro_id : 0);
    read_buf = (kv_pair*)cache_ptr;
    res = model_search(key, seg_model, model_now);
    model_gaddr.val = seg_model.pos_start;
    write_buffer_ = write_buffers[model_gaddr.nodeID];

    predict=seg_model.slope*key+seg_model.intercept;
    start=std::max<int>(0,(int)predict-(int)Epsilon+1);
    if(start>seg_model.child_length-1)
    {
        start=seg_model.child_length-1;
    }
    end = std::min<int>(seg_model.child_length-1,(int)predict+Epsilon-1);
    kv_size = end - start +  1;
    if(predict<0) predict_int = 0;
    else if(predict > seg_model.child_length - 1) predict_int = seg_model.child_length - 1;
    else predict_int = (int)predict;
    kv_target.val = seg_model.pos_start;
    kv_target.offset += sizeof(kv_pair) * start;
    dsm -> read_sync(cache_ptr, kv_target, sizeof(kv_pair) * kv_size, ctx);
    find_key_or_empty(read_buf, kv_size, key, pos, empty_pos);
    nextptr_buf = (uint64_t*)((uint64_t)cache_ptr + sizeof(kv_pair) * kv_size);
    
    if(pos != -1)//key exit, need to update, not insert
    {
        kv_target.offset += pos*sizeof(kv_pair);
        GlobalAddress val_gaddr = kv_target;
        val_gaddr.offset += sizeof(uint64_t);
        uint64_t ori_val = read_buf[pos].val;
        int fail_count = 0;
        while(1)
        {
            bool ret = dsm->cas_sync(val_gaddr, ori_val, val, (uint64_t*)cache_ptr, ctx);
            if(ret)
            {
                insert_ret = true;
                goto insert_finish;
                // return true;
            }
            else
            {
                lock_fail[thread_id]++;
                if(enable_backoff){
                    std::this_thread::sleep_for(std::chrono::microseconds(calculate_backoff(fail_count)));
                }
                if(ctx != nullptr){
                    busy_waiting_queue.push(ctx->coro_id);
                    (*ctx->yield)(*ctx->master);
                }
                fail_count++;
                ori_val = *(uint64_t*)cache_ptr;
                continue;
            }
        }
    }
    else if(empty_pos != -1)
    {
        GlobalAddress empty_kv_gaddr = kv_target;
        empty_kv_gaddr.offset += sizeof(kv_pair) * empty_pos;
        GlobalAddress empty_val_gaddr = empty_kv_gaddr;
        empty_val_gaddr.offset += sizeof(uint64_t);
        bool ret = dsm -> cas_sync(empty_kv_gaddr, null_flag, key, (uint64_t*)cache_ptr, ctx);
        //std::cout<<"false val: "<<*(uint64_t*)cache_ptr<<std::endl;
        if(ret)
        {
            *(uint64_t*)cache_ptr = val;
            dsm -> write_sync((char*)cache_ptr, empty_val_gaddr, sizeof(uint64_t), ctx);
            insert_ret = true;
            goto insert_finish;
            // return true;
        }
        lock_fail[thread_id]++;
    }

    
    nextptr_target.val = seg_model.pos_start;
    nextptr_target.offset += (sizeof(kv_pair) * seg_model.child_length + sizeof(uint64_t) * predict_int);
    
    dsm -> read_sync((char*)nextptr_buf, nextptr_target, sizeof(uint64_t), ctx);
    next = *nextptr_buf;

    if(enable_cache && ((next&model_flag)!=0))
    {
        LLDex *model_in_cache = nullptr;
        uint64_t range_start;
        uint64_t range_end;
        bool find = search_cache(model_now, key, &model_in_cache, range_start, range_end);
        if(find)
        {
            GlobalAddress new_list_node;
            write_buffer_ -> get_slot(thread_id, model_gaddr.nodeID, new_list_node, ctx);
            bool ret = model_in_cache->sub_insert(new_list_node, key, val, thread_id, ctx);
            insert_ret = ret;
            goto insert_finish;
            // return ret;
        }
    }
    
    while(1)
    {
        GlobalAddress  list_node_gaddr;
        if((next&null_flag) != 0)
        {
            uint64_t next_content = nextptr_target.val | tail_flag;
            bool get_slot_ret = write_buffer_ -> get_slot(thread_id, nextptr_target.nodeID, list_node_gaddr, ctx);
            if(!get_slot_ret)
            {
                std::cout<<"alloc error"<<std::endl;
                assert(1 == 0);
                insert_ret = false;
                goto insert_finish;
                // return false;
            }
            uint64_t* cas_source = (uint64_t*)cache_ptr;
            uint64_t* write_source = (uint64_t*)((uint64_t)cache_ptr + sizeof(uint64_t));
            uint64_t equal = next;
            uint64_t swap_val = list_node_gaddr.val|chain_flag;
            //std::cout<<"list_node_gaddr: "<<list_node_gaddr.nodeID<<":"<<list_node_gaddr.offset<<std::endl;
            bool ret = write_buffer_and_cas(list_node_gaddr, nextptr_target, write_source, cas_source, equal, swap_val, key, val, next_content, ctx);
            write_buffer_flag = 1;
            if(ret)
            {
                insert_ret = true;
                goto insert_finish;
                // return true;
            }
            else
            {
                lock_fail[thread_id]++;
                next = *cas_source;
                continue;
            }
        }
        else if((next&chain_flag)!=0)
        {
            GlobalAddress next_list_node_gaddr;
            next_list_node_gaddr.val = (next&mask_);
            uint64_t next_content;
            if((next&chain_flag)!=0)
            {
                next_content=(next_list_node_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
            }
            else
            {
                next_content=next;
            }
            if(write_buffer_flag == 0)
            {
                bool get_slot_ret = write_buffer_ -> get_slot(thread_id, kv_target.nodeID, list_node_gaddr, ctx);
                if(!get_slot_ret)
                {
                    std::cout<<"alloc error"<<std::endl;
                    insert_ret = false;
                    goto insert_finish;
                    // return false;
                }
                uint64_t* cas_source = (uint64_t*)cache_ptr;
                uint64_t* write_source = (uint64_t*)((uint64_t)cache_ptr + sizeof(uint64_t));
                uint64_t equal = next;
                uint64_t swap_val = list_node_gaddr.val | chain_flag;
                bool ret = write_buffer_and_cas(list_node_gaddr, nextptr_target, write_source, cas_source, equal, swap_val, key, val, next_content, ctx);
                write_buffer_flag = 1;
                if(ret)
                {
                    insert_ret = true;
                    goto insert_finish;
                    // return true;
                }
                else
                {
                    lock_fail[thread_id]++;
                    next = *cas_source;
                    continue;
                }
            }
            else
            {
                GlobalAddress list_node_next_gaddr = list_node_gaddr;
                list_node_gaddr.offset += (2 * sizeof(uint64_t));
                uint64_t *cas_source = (uint64_t*)cache_ptr;
                uint64_t *write_source = (uint64_t*)((uint64_t)cache_ptr + sizeof(uint64_t));
                uint64_t equal = next;
                uint64_t swap_val = list_node_gaddr.val|chain_flag;
                bool ret = write_next_and_cas(list_node_next_gaddr, nextptr_target, write_source, cas_source, equal, swap_val, next_content, ctx);
                if(ret)
                {
                    insert_ret = true;
                    goto insert_finish;
                    // return true;
                }
                else
                {
                    lock_fail[thread_id]++;
                    next = *cas_source;
                    continue;
                }
            }
        }
        else if((next&model_flag)!=0)
        {
            GlobalAddress root;
            root.val = (next&mask_);
            LLDex *temp = new LLDex(dsm,define::Epsilon,write_buffer_conf_);
            temp->read_model_from_remote(root, ctx);
            if(enable_cache)
            {
                int cache_index;
                uint64_t range_start;
                uint64_t range_end;
                get_range_start_end(seg_model, predict_int, model_now, range_start, range_end);
                cache_model(model_now, range_start, range_end, temp, cache_index);
            }
            write_buffer_->get_slot(thread_id, kv_target.nodeID, list_node_gaddr, ctx);
            bool ret = temp->sub_insert(list_node_gaddr, key, val, thread_id, ctx);

            insert_ret = ret;
            goto insert_finish;
            // return ret;
        }
        else
        {
            std::cout<<"error in while"<<std::endl;
            assert(1 == 0);
        }
    }

insert_finish:
    if(enable_local_lock){
        local_lock_table->release_local_write_lock(key, lock_res);
    }
    return insert_ret;
}

bool LLDex::sub_insert(GlobalAddress new_slot, uint64_t key, uint64_t val, int thread_id, CoroContext *ctx)
{
    // char *cache_ptr = dsm->get_rdma_buffer();
    char *cache_ptr = dsm->get_coro_buf(ctx ? ctx->coro_id : 0);
    kv_pair *read_buf = (kv_pair*)cache_ptr;
    model_mincost seg_model;
    int model_now = 0;
    bool res = model_search(key, seg_model, model_now);
    GlobalAddress model_gaddr;
    model_gaddr.val = seg_model.pos_start;
    write_buffer *write_buffer_ = write_buffers[model_gaddr.nodeID];

    long double predict = seg_model.slope*key+seg_model.intercept;
    int start=std::max<int>(0,(int)predict-(int)Epsilon+1);
    if(start>seg_model.child_length-1)
    {
        start=seg_model.child_length-1;
    }
    int end = std::min<int>(seg_model.child_length-1,( int)predict+Epsilon-1);
    int kv_size = end - start + 1;
    int predict_int = 0;
    if(predict<0) predict_int = 0;
    else if(predict > seg_model.child_length - 1) predict_int = seg_model.child_length - 1;
    else predict_int = (int)predict;
    GlobalAddress kv_target;
    kv_target.val = seg_model.pos_start;
    kv_target.offset += sizeof(kv_pair) * start;
    GlobalAddress nextptr_target;
    nextptr_target.val = seg_model.pos_start;
    nextptr_target.offset += (sizeof(kv_pair) * seg_model.child_length + sizeof(uint64_t) * predict_int);
    dsm -> read(cache_ptr, kv_target, sizeof(kv_pair) * kv_size, false);
    uint64_t *nextptr_buf = (uint64_t*)((uint64_t)cache_ptr + sizeof(kv_pair) * kv_size);
    dsm -> read_sync((char*)nextptr_buf, nextptr_target, sizeof(uint64_t), ctx);
    int pos = -1;
    int empty_pos = -1;
    find_key_or_empty(read_buf, kv_size, key, pos, empty_pos);
    if(pos != -1)
    {
        kv_target.offset += pos*sizeof(kv_pair);
        GlobalAddress val_gaddr = kv_target;
        val_gaddr.offset += sizeof(uint64_t);
        uint64_t ori_val = read_buf[pos].val;
        while(1)
        {
            bool ret = dsm->cas_sync(val_gaddr, ori_val, val, (uint64_t*)cache_ptr, ctx);
            if(ret)
            {
                return true;
            }
            else
            {
                lock_fail[thread_id]++;
                ori_val = *(uint64_t*)cache_ptr;
                continue;
            }
        }
    }
    else if(empty_pos != -1)
    {
        GlobalAddress empty_kv_gaddr = kv_target;
        empty_kv_gaddr.offset += sizeof(kv_pair) * empty_pos;
        GlobalAddress empty_val_gaddr = empty_kv_gaddr;
        empty_val_gaddr.offset += sizeof(uint64_t);
        while(1)
        {
            bool ret = dsm->cas_sync(empty_kv_gaddr, null_flag, key, (uint64_t*)cache_ptr, ctx);
            if(ret)
            {
                *(uint64_t*)cache_ptr = val;
                dsm->write_sync((char*)cache_ptr, empty_val_gaddr, sizeof(uint64_t), ctx);
                return true;
            }
            else
            {
                lock_fail[thread_id]++;
                break; //empty kv_pari used, go to list insert
            }
        }
    }

    if(enable_cache)
    {
        LLDex *model_in_cache = nullptr;
        uint64_t range_start;
        uint64_t range_end;
        bool find = search_cache(model_now, key, &model_in_cache, range_start, range_end);
        if(find)
        {
            bool ret = model_in_cache->sub_insert(new_slot, key, val, thread_id, ctx);
        }
    }

    uint64_t next = *nextptr_buf;
    while(1)
    {
        if((next&null_flag) != 0)
        {
            uint64_t next_content = nextptr_target.val | tail_flag;
            uint64_t *cas_source = (uint64_t*)cache_ptr;
            uint64_t *write_source = (uint64_t*)((uint64_t)cache_ptr + sizeof(uint64_t));
            uint64_t equal = next;
            uint64_t swap_val = new_slot.val | chain_flag;
            bool ret = write_buffer_and_cas(new_slot, nextptr_target, write_source, cas_source, equal, swap_val, key, val, next_content, ctx);
            if(ret)
            {
                return true;
            }
            else
            {
                lock_fail[thread_id]++;
                next = *cas_source;
                continue;
            }
        }
        else if((next&chain_flag) != 0)
        {
            GlobalAddress next_list_node_gaddr;
            next_list_node_gaddr.val = (next&mask_);
            uint64_t next_content = (next_list_node_gaddr.offset-write_buffer_->get_off_write_buffer())/sizeof(list_node);
            GlobalAddress list_node_next_gaddr = new_slot;
            list_node_next_gaddr.offset += (2 * sizeof(uint64_t));
            uint64_t *cas_source = (uint64_t*)cache_ptr;
            uint64_t *write_source = (uint64_t*)((uint64_t)cache_ptr + sizeof(uint64_t));
            uint64_t equal = next;
            uint64_t swap_val = new_slot.val | chain_flag;
            bool ret = write_next_and_cas(list_node_next_gaddr, nextptr_target, write_source, cas_source, equal, swap_val, next_content, ctx);
            if(ret)
            {
                return true;
            }
            else
            {
                lock_fail[thread_id]++;
                next = *cas_source;
                continue;
            }
        }
        else if((next&model_flag) != 0)
        {
            GlobalAddress  root;
            root.val = (next&mask_);
            LLDex *temp = new LLDex(dsm, define::Epsilon, write_buffer_conf_);
            temp->read_model_from_remote(root, ctx);
            if(enable_cache)
            {
                int cache_index;
                uint64_t range_start;
                uint64_t range_end;
                get_range_start_end(seg_model, predict_int, model_now, range_start, range_end);
                cache_model(model_now, range_start, range_end, temp, cache_index);
            }
            bool ret = temp->sub_insert(new_slot, key, val, thread_id, ctx);
            return ret;
        }
    }
}

bool LLDex::scan(uint64_t &key_start,uint64_t &key_end,std::vector<kv> &kvs)
{
    bool ret = false;
    char *cache_ptr = dsm->get_rdma_buffer();
    uint64_t source_start = (uint64_t)cache_ptr;
    kv_pair *read_buf = (kv_pair*)cache_ptr;
    std::vector<model_mincost> target_models;
    std::vector<kv> chain_kvs;
    std::vector<kv> slot_kvs;
    bool res = model_scan(key_start, key_end, target_models);
    std::vector<RdmaOpRegion> read_batch;
    std::vector<batch_info> batch_infos;
    int cur_off = 0;
    if(!res)
    {
        std::cout<<"search : error in search model"<<std::endl;
        return res;
    }
    if(target_models.size() == 1)
    {
        long double start_predict=target_models[0].slope*key_start+target_models[0].intercept;
        long double end_predict=target_models[0].slope*key_end+target_models[0].intercept;
        int start_predict_int = 0;
        int end_predict_int = 0;
        if(start_predict < 0) start_predict_int = 0;
        else if(start_predict > target_models[0].child_length - 1) start_predict_int = target_models[0].child_length - 1;
        else start_predict_int = (int)start_predict;
        if(end_predict < 0) end_predict_int = 0;
        else if(end_predict > target_models[0].child_length - 1) end_predict_int = target_models[0].child_length - 1;
        else end_predict_int = (int)end_predict;
        GlobalAddress target_model_gaddr;
        target_model_gaddr.val = target_models[0].pos_start;
        write_buffer *write_buffer_ = write_buffers[target_model_gaddr.nodeID];
        int start=std::max<int>(0,(int)start_predict-(int)Epsilon + 1);
        int end=std::min<int>(target_models[0].child_length-1,(int)end_predict+Epsilon-1);
        if(start > target_models[0].child_length - 1)
        {
            start=target_models[0].child_length-1;
        }
        int kv_size = end - start + 1;
        int next_size = (end_predict_int - start_predict_int + 1);
        // std::cout<<"start predict: "<<start_predict_int<<" end predict: "<<end_predict_int<<std::endl;
        GlobalAddress target_kv = target_model_gaddr;
        target_kv.offset += sizeof(kv_pair) * start;
        GlobalAddress target_next = target_model_gaddr;
        target_next.offset += (sizeof(kv_pair) * target_models[0].child_length + sizeof(uint64_t) * start_predict_int);
        RdmaOpRegion r_kvs = fill_RdmaOpRegion(source_start, target_kv, kv_size * sizeof(kv_pair), false);
        batch_info temp_info_kvs = fill_BatchInfo(target_kv, kvs_flag, target_kv.nodeID, cur_off, target_kv.nodeID, false, kv_size);
        cur_off += sizeof(kv_pair) * kv_size;
        read_batch.push_back(r_kvs);
        batch_infos.push_back(temp_info_kvs);
        RdmaOpRegion r_next = fill_RdmaOpRegion(source_start + cur_off, target_next, sizeof(uint64_t) * next_size, false);
        batch_info temp_info_next = fill_BatchInfo(target_next, next_flag, target_next.nodeID, cur_off, target_next.nodeID, false, next_size);
        read_batch.push_back(r_next);
        batch_infos.push_back(temp_info_next);
    }
    else
    {
        //do the pridict in start model
        model_mincost start_model=target_models[0];
        long double start_predict=start_model.slope*key_start+start_model.intercept;
        int start_predict_int;
        if(start_predict < 0) start_predict_int = 0;
        else if(start_predict > start_model.child_length - 1) start_predict_int = start_model.child_length - 1;
        else start_predict_int = (int)start_predict;
        int start_model_start_off=std::max<int>(0,(int)start_predict-(int)Epsilon+1);
        int start_model_end_off = start_model.child_length-1;
        if(start_model_start_off>start_model.child_length-1)
        {
            start_model_start_off=start_model.child_length-1;
        }

        //read start kvs
        GlobalAddress target_start_kvs;
        target_start_kvs.val = start_model.pos_start; 
        target_start_kvs.offset += start_model_start_off * sizeof(kv_pair);
        int kvs_size =  (start_model_end_off - start_model_start_off + 1);
        RdmaOpRegion r_start_kv = fill_RdmaOpRegion(source_start, target_start_kvs,  kvs_size * sizeof(kv_pair), false);
        batch_info info_start_kv = fill_BatchInfo(target_start_kvs, kvs_flag, target_start_kvs.nodeID, cur_off, target_start_kvs.nodeID, false, kvs_size);
        cur_off += sizeof(kv_pair) * kvs_size;
        read_batch.push_back(r_start_kv);
        batch_infos.push_back(info_start_kv);
        

        //read start next
        GlobalAddress target_start_next;
        target_start_next.val = start_model.pos_start;
        target_start_next.offset += (sizeof(kv_pair) * start_model.child_length + sizeof(uint64_t) * start_predict_int);
        int next_size = (start_model_end_off - start_predict_int + 1);
        RdmaOpRegion r_start_next = fill_RdmaOpRegion(source_start + cur_off, target_start_next, sizeof(uint64_t) * next_size, false);
        batch_info info_start_next = fill_BatchInfo(target_start_next, next_flag, target_start_next.nodeID, cur_off, target_start_next.nodeID, false, next_size);
        cur_off += sizeof(uint64_t) * next_size;
        read_batch.push_back(r_start_next);
        batch_infos.push_back(info_start_next);
    

        for(int i = 1; i < target_models.size() - 1; i++)
        {
            //read mid kvs
            GlobalAddress target_mid_kvs;
            target_mid_kvs.val = target_models[i].pos_start;
            int size = target_models[i].child_length;
            RdmaOpRegion r_mid_kv = fill_RdmaOpRegion(source_start + cur_off, target_mid_kvs, size, false);
            batch_info info_mid_kv = fill_BatchInfo(target_mid_kvs, kvs_flag, target_mid_kvs.nodeID, cur_off, target_mid_kvs.nodeID, false, size);
            cur_off += sizeof(kv_pair) * size;
            read_batch.push_back(r_mid_kv);
            batch_infos.push_back(info_mid_kv);

            //read mid next
            GlobalAddress target_mid_next = target_mid_kvs;
            target_mid_next.offset += sizeof(kv_pair) * size;
            RdmaOpRegion r_mid_next = fill_RdmaOpRegion(source_start + cur_off, target_mid_next, size, false);
            batch_info info_mid_next = fill_BatchInfo(target_mid_next, next_flag, target_mid_next.nodeID, cur_off, target_mid_next.nodeID, false, size);
            cur_off += sizeof(uint64_t) * size;
            read_batch.push_back(r_mid_next);
            batch_infos.push_back(info_mid_next);
        }

        //do the predict in end model
        model_mincost end_model=target_models[target_models.size()-1];
        long double end_predict=end_model.slope*key_end+end_model.intercept;
        int end_predict_int;
        if(end_predict < 0) end_predict_int = 0;
        else if(end_predict > end_model.child_length - 1) end_predict_int = end_model.child_length - 1;
        else end_predict_int = (int)end_predict;
        int end_model_start_off=0;
        int end_model_end_off=std::min<int>(end_model.child_length-1,(int)end_predict+Epsilon-1);

        //read end kvs
        GlobalAddress target_end_kvs;
        target_end_kvs.val = end_model.pos_start;
        kvs_size = (end_model_end_off - end_model_start_off + 1);
        RdmaOpRegion r_end_kv = fill_RdmaOpRegion(source_start + cur_off, target_end_kvs, kvs_size * sizeof(kv_pair), false);
        batch_info info_end_kv = fill_BatchInfo(target_end_kvs, kvs_flag, target_end_kvs.nodeID, cur_off, target_end_kvs.nodeID, false, kvs_size);
        cur_off += sizeof(kv_pair) * kvs_size;
        read_batch.push_back(r_end_kv);
        batch_infos.push_back(info_end_kv);
        
        //read end next
        GlobalAddress target_end_next = target_end_kvs;
        target_end_next.offset += (sizeof(kv_pair) * end_model.child_length);
        next_size = (end_predict_int + 1);
        RdmaOpRegion r_end_next = fill_RdmaOpRegion(source_start + cur_off, target_end_next, next_size * sizeof(uint64_t), false);
        batch_info info_end_next = fill_BatchInfo(target_end_next, next_flag, target_end_next.nodeID, cur_off, target_end_next.nodeID, false, next_size);
        cur_off += sizeof(uint64_t) * next_size;
        read_batch.push_back(r_end_next);
        batch_infos.push_back(info_end_next);
    }
    dsm->read_batches_sync(read_batch);
    int loop = 0;
    while(read_batch.size() != 0)
    {
        // std::cout<<"read batch size:"<<read_batch.size()<<std::endl;
        read_batch.clear();
        cur_off = 0;
        std::vector<batch_info> new_batch;
        //std::cout<<"batch size:"<<batch_infos.size()<<std::endl;
        for(int i = 0; i < batch_infos.size(); i++)
        {
            if(batch_infos[i].flag == chain_flag)
            {
                // std::cout<<"in chain"<<std::endl;
                list_node *now = (list_node*)(source_start + batch_infos[i].offset);
                if(now->key >= key_start && now->key <= key_end)
                {
                    kv temp_kv;
                    temp_kv.key = now->key;
                    temp_kv.val = now->val;
                    temp_kv.index = loop + 10;
                    chain_kvs.push_back(temp_kv);
                }
                if((now->next_offset&tail_flag)==0)
                {
                    if((now->next_offset&null_flag)!=0)
                    {
                        RdmaOpRegion r_next = fill_RdmaOpRegion(source_start + cur_off, batch_infos[i].gaddr, sizeof(uint64_t), false);
                        batch_info info_next = fill_BatchInfo(batch_infos[i].gaddr, next_flag, 0, cur_off, batch_infos[i].node_id, false, 1);
                        cur_off += sizeof(uint64_t);
                        read_batch.push_back(r_next);
                        new_batch.push_back(info_next);
                    }
                    else
                    {
                        GlobalAddress next_list_node_gaddr;
                        next_list_node_gaddr.nodeID = batch_infos[i].node_id;
                        uint64_t write_buffer_start_off = write_buffers[batch_infos[i].write_buffer_index]->get_off_write_buffer();
                        next_list_node_gaddr.offset = write_buffer_start_off + (now->next_offset&mask_)*sizeof(list_node);
                        RdmaOpRegion r_listnode = fill_RdmaOpRegion(source_start + cur_off, next_list_node_gaddr, sizeof(list_node), false);
                        batch_info info_listnode = fill_BatchInfo(batch_infos[i].gaddr, chain_flag, batch_infos[i].write_buffer_index, cur_off, batch_infos[i].node_id, false, 1);
                        cur_off += sizeof(list_node);
                        new_batch.push_back(info_listnode);
                        read_batch.push_back(r_listnode);
                    }
                }
            }
            else if(batch_infos[i].flag == model_flag)
            {
                // std::cout<<"in model"<<std::endl;
                model_global *now = (model_global*)(source_start + batch_infos[i].offset);
                if(!batch_infos[i].is_leaf&&!now->is_leaf)
                {
                    int model_num=now->full_model_num;
                    GlobalAddress models_start_gaddr=batch_infos[i].gaddr;
                    models_start_gaddr.offset-=(model_num-1)*sizeof(model_global);
                    RdmaOpRegion r_model = fill_RdmaOpRegion(source_start + cur_off, models_start_gaddr, sizeof(model_global) * model_num, false);
                    batch_info info_model = fill_BatchInfo(models_start_gaddr, model_flag, 0, cur_off, models_start_gaddr.nodeID, false, model_num);
                    cur_off += sizeof(model_global)*model_num;
                    new_batch.push_back(info_model);
                    read_batch.push_back(r_model);
                }
                else
                {
                    model_global *model_buf=(model_global*)(source_start+batch_infos[i].offset);
                    int j=0;
                    while(j<batch_infos[i].length && model_buf[j].is_leaf)
                    {
                        GlobalAddress target_kv = model_buf[j].child_start;
                        RdmaOpRegion r_kvs = fill_RdmaOpRegion(source_start + cur_off, target_kv, model_buf[j].child_length * sizeof(kv_pair), false);
                        batch_info info_kvs = fill_BatchInfo(target_kv, kvs_flag, target_kv.nodeID, cur_off, target_kv.nodeID, false, model_buf[j].child_length);
                        cur_off += sizeof(kv_pair) * model_buf[j].child_length;
                        read_batch.push_back(r_kvs);
                        new_batch.push_back(info_kvs);

                        GlobalAddress target_next = target_kv;
                        target_next.offset += sizeof(kv_pair) * model_buf[j].child_length;
                        RdmaOpRegion r_next = fill_RdmaOpRegion(source_start + cur_off, target_next, sizeof(uint64_t) * model_buf[j].child_length, false);
                        batch_info info_next = fill_BatchInfo(target_next, next_flag, target_next.nodeID, cur_off, target_next.nodeID, false, model_buf[j].child_length);
                        cur_off += sizeof(uint64_t) * model_buf[j].child_length;
                        read_batch.push_back(r_next);
                        new_batch.push_back(info_next);
                        j++;
                    }
                }
            }
            else if(batch_infos[i].flag == next_flag)
            {
                // std::cout<<"in next"<<std::endl;
                uint64_t *next_buf = (uint64_t*)(source_start + batch_infos[i].offset);
                for(int j = 0; j < batch_infos[i].length; j++)
                {
                    uint64_t next = next_buf[j];
                    // std::cout<<"next: "<<next<<std::endl;
                    if((next&chain_flag) != 0)
                    {
                        GlobalAddress list_node_gaddr;
                        list_node_gaddr.val = next&mask_;
                        GlobalAddress next_gaddr = batch_infos[i].gaddr;
                        next_gaddr.offset += (j * sizeof(uint64_t));
                        RdmaOpRegion r_listnode = fill_RdmaOpRegion(source_start + cur_off, list_node_gaddr, sizeof(list_node), false);
                        batch_info info_listnode = fill_BatchInfo(next_gaddr, chain_flag, next_gaddr.nodeID, cur_off, next_gaddr.nodeID, false, 1);
                        cur_off += sizeof(list_node);
                        read_batch.push_back(r_listnode);
                        new_batch.push_back(info_listnode);
                        continue;
                    }
                    else if((next&model_flag) != 0)
                    {
                        // std::cout<<"in model"<<std::endl;
                        GlobalAddress model_root_gaddr;
                        model_root_gaddr.val = (next&mask_);
                        RdmaOpRegion r_model = fill_RdmaOpRegion(source_start + cur_off, model_root_gaddr, sizeof(model_global), false);
                        batch_info info_model = fill_BatchInfo(model_root_gaddr, model_flag, model_root_gaddr.nodeID, cur_off, model_root_gaddr.nodeID, false, 1);
                        cur_off += sizeof(model_global);
                        read_batch.push_back(r_model);
                        new_batch.push_back(info_model);
                        continue;
                    }
                }
            }
            else if(batch_infos[i].flag == kvs_flag)
            {
                // std::cout<<"in kvs"<<std::endl;
                kv_pair *kvs_buf = (kv_pair*)(source_start + batch_infos[i].offset);
                for(int j = 0; j < batch_infos[i].length; j++)
                {
                    if(kvs_buf[j].key >= key_start && kvs_buf[j].key <= key_end)
                    {
                        kv temp_kv;
                        temp_kv.key = kvs_buf[j].key;
                        temp_kv.val = kvs_buf[j].val;
                        temp_kv.index = loop + 20;
                        slot_kvs.push_back(temp_kv);
                    }
                } 
            }
            else
            {
                // std::cout<<"in nothing"<<std::endl;
            }
        }
        dsm->read_batches_sync(read_batch);
        batch_infos = new_batch;
        loop++;
    }
    std::unordered_map<uint64_t, kv> hash;
    for(int i = chain_kvs.size() - 1; i >= 0; i--)
    {
        hash[chain_kvs[i].key] = chain_kvs[i];
    }
    for(int i = slot_kvs.size() - 1; i >= 0; i--)
    {
        hash[slot_kvs[i].key] = slot_kvs[i];
    }
    for(auto kv_now : hash)
    {
        kvs.push_back(kv_now.second);
    } 
    return true;
}

void LLDex::run_coroutine(GenFunc gen_func, WorkFunc work_func, int coro_cnt, Request* req, int req_num, int id)
{
    assert(coro_cnt <= MAX_CORO_NUM);
    using namespace std::placeholders;
    // coro_ops_total = total_ops;
    // coro_ops_cnt_start = 0;
    // coro_ops_cnt_finish = 0;

    assert(coro_cnt <= define::kMaxCoro);
    for (int i = 0; i < coro_cnt; ++i) {
        RequstGen* gen = gen_func(dsm, req, req_num, i, coro_cnt);
        worker[i] =
            CoroCall(std::bind(&LLDex::coro_worker, this, _1, gen, work_func, id, i));
    }

    master = CoroCall(std::bind(&LLDex::coro_master, this, _1, coro_cnt));

    master();
}

void LLDex::coro_worker(CoroYield &yield, RequstGen *gen, WorkFunc work_func, int thread_id, int coro_id) 
{
    CoroContext ctx;
    ctx.coro_id = coro_id;
    ctx.master = &master;
    ctx.yield = &yield;
    Timer coro_timer;

    while (!need_stop) 
    {
        auto r = gen->next();
        coro_timer.begin();
        work_func(this, r, thread_id, &ctx);
        auto us_10 = coro_timer.end() / 100;
        if (us_10 >= LATENCY_WINDOWS) {
          us_10 = LATENCY_WINDOWS - 1;
        }
        latency[thread_id][coro_id][us_10]++;
    }

}

void LLDex::coro_master(CoroYield &yield, int coro_cnt) {
    for (int i = 0; i < coro_cnt; ++i) {
      yield(worker[i]);
    }
  
    while (!need_stop) {
      uint64_t next_coro_id;
  
      if (dsm->poll_rdma_cq_once(next_coro_id)) {
        yield(worker[next_coro_id]);
      }

      if((enable_local_lock || enable_read_delegation) && !busy_waiting_queue.empty())
      {
          uint64_t next_coro_id = busy_waiting_queue.front();
          busy_waiting_queue.pop();
          yield(worker[next_coro_id]);
      }
    }

}
