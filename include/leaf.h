#ifndef LEAF_H
#define LEAF_H
#include "plr.h"
#include "write_buffer.h"
#include "Common.h"
#include "retrainer.h"
#include "Timer.h"
#include "LocalLockTable.h"


struct batch_info
{
    GlobalAddress gaddr;
    uint64_t flag;
    int write_buffer_index;
    int offset;
    int node_id;
    bool is_leaf=false;
    int length;
};

struct node_set
{
    uint64_t node_num;
    uint64_t nodes_gaddr[define::min_model_size];
    uint64_t next_offset;
};

struct kv
{
    uint64_t key;
    uint64_t val;
    int64_t index;
    GlobalAddress gaddr;
};



/*struct kv4scan
{
    uint64_t key;
    uint64_t val;
    uint64_t type;
};*/

struct slot
{
    uint64_t next;
    uint64_t key;
    uint64_t val;
};

struct write_buffer_conf
{
    uint64_t buffer_size;
    int thread_num;
    uint64_t buffer_num;
};

struct model_global
{
    long double slope;
    long double intercept;
    uint64_t key_start;
    uint64_t full_model_num;
    GlobalAddress child_start; //can be the node_set addr in cache
    GlobalAddress sibling; //can be the slot_gaddr of a node_set in cache
    int64_t child_length;
    int64_t threshold;
    int64_t pos_start;
    int write_buffer_index;
    bool is_leaf=false; //can be the set/model flag in cache, true for model and false for cache
    //boost::icl::interval_map<uint64_t,model_global> cache;
    //WRLock cache_lock; 
};

struct model_local
{
    long double slope;
    long double intercept;
    uint64_t key_start;
    uint64_t data_num;
    int64_t pos_start;
    model_local *sibling;
    std::vector<model_local> *child;
};

struct model_mincost
{
    long double slope; //double is enough
    long double intercept; //double is enough
    uint64_t child_length; //can be int
    uint64_t pos_start; //gaddr for cached set
    uint64_t key_start; //index for cached set's slot
};


void make_segment(const std::vector<uint64_t> &keys, const std::vector<uint64_t> &vals, uint64_t Epsilon, std::vector<model_local> &models);
void make_segment_for_kvs(const std::vector<kv> &kvs,uint64_t Epsilon, std::vector<model_local> &models);
void make_segment_par(const std::vector<uint64_t> &keys, uint64_t Epsilon, std::vector<model_local>& models);
void get_start_key_from_model(std::vector<uint64_t>&start_keys,std::vector<model_local>&models);
void build_local(std::vector<std::vector<model_local>> &level_model,std::vector<model_local>leafs,uint64_t Epsilon);
void print_model_info(std::vector<model_local>&models);
void print_local_model(model_local model);
void print_global_model(model_global model);
std::string gaddr2str(GlobalAddress gaddr);

enum RequestType : int {
  INSERT = 0,
  UPDATE,
  SEARCH,
  SCAN
};

struct Request {
    RequestType req_type;
    bool is_search;
    uint64_t k;
    uint64_t v;
    int range_size;
};
  
class RequstGen {
public:
RequstGen() = default;
virtual Request next() { return Request{}; }
};

using CoroFunc = std::function<RequstGen *(int, DSM *, int)>;
using GenFunc = std::function<RequstGen *(DSM *, Request*, int, int, int)>;


class learned_index_local
{
private:
    uint64_t Epsilon;
public:
    learned_index_local(uint64_t Epsilon);
    std::vector<std::vector<model_local>> level_models;
    void build_local(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals);
    void build_local_with_empty(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals,
                        std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals);
    void print_level_model();
};

template <class Type>
struct inplace_replace : boost::icl::identity_based_inplace_combine<Type> {
  void operator()(Type &object, const Type &operand) const { object = operand; }
};
 
template<>
inline std::string boost::icl::unary_template_to_string<inplace_replace>::apply() {
	return "=";
}
 

using ival_map =
    boost::icl::interval_map<uint64_t,         // Key
                 uint64_t,         // Value
                 boost::icl::partial_enricher, // Unmapped intervals have unkown value;
                                   // store identity values
                 std::less,        // Comparator
                 inplace_replace,  // Combination operator
                 boost::icl::inplace_erasure,  // Extraction operator
                 boost::icl::closed_interval<unsigned, std::less> // Interval type
                 >;
using ival=ival_map::interval_type;


class learned_index_global
{
private:
    DSM *dsm;
    uint64_t Epsilon;
    std::vector<std::vector<model_global>> level_models;
    std::vector<ival_map> cache;
    std::vector<std::vector<learned_index_global*>> cache_content;
    std::vector<std::shared_mutex> cache_lock;
    std::vector<int> cached_num;
    write_buffer_conf write_buffer_conf_;
    //write_buffer *write_buffer_;
    //int buffer_num;
    write_buffer ** write_buffers;
public:
    learned_index_global();
    learned_index_global(DSM *dsm,uint64_t Epsilon,write_buffer_conf write_buffer_conf_);
    void print_level_model();
    void write_seg_remote(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals,std::vector<model_local> &local_models);
    void build_remote(std::vector<uint64_t> &keys,std::vector<uint64_t> &vals,learned_index_local &local_index,GlobalAddress &root_model_addr);
    void write_seg_remote_with_empty(std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals,std::vector<model_local> &local_models);
    void build_remote_with_empty(std::vector<std::vector<uint64_t>> &seg_keys, std::vector<std::vector<uint64_t>> &seg_vals, learned_index_local &local_index, GlobalAddress &root_model_addr);
    void read_model_from_remote(GlobalAddress root);
    void read_model_from_remote_oneoff(GlobalAddress root,int model_num);
    void cache_model(int model_now,int slot_size,int slot_pos,slot *read_buf,learned_index_global *temp_model,int &cache_index);
    void cache_set(int model_now,int slot_size,int slot_pos,slot *read_buf,GlobalAddress slot_gaddr);
    void cache_model_range(int model_now,int range_start,int range_end,learned_index_global *temp_model,int &cache_index);
    bool search_cache(int &model_now,uint64_t key,learned_index_global **cache_content,uint64_t &range_start,uint64_t &range_end);
    void get_cache(int &model_now,int content_index,learned_index_global **cache_content);
    
    bool model_binary_search(int level,int model,uint64_t key,int &next_model);
    bool model_search(uint64_t &key,model_global &target_model,int &model_now);
    bool model_scan(uint64_t &key_start,uint64_t &key_end,std::vector<model_global>& target_models);
    bool slot_binary_search(slot* kvs,int size,uint64_t key,int &pos);
    bool write_buffer_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t key, uint64_t val, uint64_t next);
    bool write_next_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t next);
    bool search(uint64_t &key,uint64_t &val);
    bool insert(uint64_t &key,uint64_t &val,int thread_id);
    bool sub_insert(GlobalAddress new_slot,uint64_t key,uint64_t val,int thread_id);
    bool scan(uint64_t &key_start,uint64_t &key_end,std::vector<kv> &res);
    bool update(uint64_t &key,uint64_t &val,int thread_id);
    bool sub_update(GlobalAddress new_slot,uint64_t key,uint64_t val,int thread_id);
};


class LLDex
{
private:
    DSM *dsm;
    uint64_t Epsilon;
    GlobalAddress end;
    std::vector<std::vector<model_mincost>> level_models;
    std::vector<ival_map> cache;
    std::vector<std::vector<LLDex*>> cache_content;
    std::vector<std::shared_mutex> cache_lock;
    std::vector<int> cached_num;
    write_buffer_conf write_buffer_conf_;
    write_buffer ** write_buffers;
    static thread_local CoroCall worker[define::kMaxCoro];
    static thread_local CoroCall master;
    static thread_local uint64_t coro_ops_total;
    static thread_local uint64_t coro_ops_cnt_start;
    static thread_local uint64_t coro_ops_cnt_finish;
    static thread_local CoroQueue busy_waiting_queue;
    LocalLockTable *local_lock_table;
    // std::queue<std::pair<std::vector<uint64_t>, std::vector<kv_pair>>> retrain_queue;
    // std::mutex retrain_mutex;
    // std::condition_variable retrain_cv;
    // std::thread retrain_thread;
    // bool retrain_running = true;

    
public:
    LLDex();
    LLDex(DSM *dsm, uint64_t Epsilon, write_buffer_conf write_buffer_conf_);
    ~LLDex();
    // void retrain_worker();
    using WorkFunc = std::function<void (LLDex *, Request&, int, CoroContext *)>;
    void read_model_from_remote(GlobalAddress root, CoroContext *ctx = nullptr);
    void read_model_from_remote_oneoff(GlobalAddress root,int model_num);
    void print_level_model();
    bool is_set(LLDex *model_in_cache);
    void get_range_start_end(model_mincost &seg_model, uint64_t predict_int, int model_now, uint64_t &range_start, uint64_t &range_end);
    //void cache_model(int model_now,uint64_t range_start, uint64_t range_end, LLDex *temp_model,int &cache_index);
    void cache_set(int model_now, uint64_t range_start, uint64_t range_end, GlobalAddress next_gaddr, GlobalAddress set_gaddr);
    void cache_model(int model_now, uint64_t range_start, uint64_t range_end, LLDex *temp_model, int &cache_index);
    bool search_cache(int &model_now,uint64_t key,LLDex **cache_content,uint64_t &range_start,uint64_t &range_end);
    void fill_cacheset_op(uint64_t dest, LLDex *model_in_cache, std::vector<RdmaOpRegion> &cache_set_op);
    RdmaOpRegion fill_RdmaOpRegion(uint64_t source, uint64_t dest, uint64_t size, bool is_on_chip);
    batch_info fill_BatchInfo(GlobalAddress gaddr, uint64_t flag, int write_buffer_index, int offset, int node_id, bool is_leaf, int length);
    void get_cache(int &model_now,int content_index,LLDex **cache_content);
    bool model_binary_search(int level,int model,uint64_t key,int &next_model);
    bool model_search(uint64_t &key,model_mincost &target_model,int &model_now);
    bool model_scan(uint64_t &key_start,uint64_t &key_end,std::vector<model_mincost>& target_models);
    bool kvs_binary_search(kv_pair* kvs,int size,uint64_t key,int &pos);
    bool kvs_search(kv_pair* kvs, int size, uint64_t key, int &pos, bool &exit_empty);
    void find_key_or_empty(kv_pair* kvs, int size, uint64_t key, int &pos, int &empty_pos);
    bool write_buffer_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t key, uint64_t val, uint64_t next, CoroContext *ctx);
    bool write_next_and_cas(GlobalAddress write_gaddr, GlobalAddress cas_gaddr, uint64_t *write_source, uint64_t *cas_source, uint64_t equal, uint64_t swap_val, uint64_t next, CoroContext *ctx);
    bool search(uint64_t &key, uint64_t &val, CoroContext *ctx = nullptr);
    bool insert(uint64_t &key, uint64_t &val, int thread_id, CoroContext *ctx = nullptr);
    bool sub_insert(GlobalAddress new_slot, uint64_t key, uint64_t val, int thread_id, CoroContext *ctx = nullptr);
    bool scan(uint64_t &key_start,uint64_t &key_end,std::vector<kv> &res);
    void run_coroutine(GenFunc gen_func, WorkFunc work_func, int coro_cnt, Request* req, int req_num, int id);
    void coro_worker(CoroYield &yield, RequstGen *gen, WorkFunc work_func, int thread_id, int coro_id); 
    void coro_master(CoroYield &yield, int coro_cnt);
};





#endif

