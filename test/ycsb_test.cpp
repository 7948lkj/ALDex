#include "DSM.h"
//#include "Tree.h"
#include "leaf.h"
#include "../src/retrain_thread.cpp"
//#include "write_buffer.h"
#include "utils.h"
#include "Timer.h"
#include "zipf.h"

#include <city.h>
#include <stdlib.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <random>



#ifdef LONG_TEST_EPOCH
  #define TEST_EPOCH 40
  #define TIME_INTERVAL 1
#else
#ifdef SHORT_TEST_EPOCH
  #define TEST_EPOCH 5
  #define TIME_INTERVAL 0.2
#else
#ifdef MIDDLE_TEST_EPOCH
  #define TEST_EPOCH 10
  #define TIME_INTERVAL 1
#else
  #define TEST_EPOCH 10
  #define TIME_INTERVAL 0.5
#endif
#endif
#endif

#define USE_CORO
#define MAX_THREAD_REQUEST 100000000
#define LOAD_HEARTBEAT 500000
#define EPOCH_LAT_TEST
#define LOADER_NUM 8 // [CONFIG] 8


//monitor parameters


//running parameters
int kThreadCount;
int kNodeCount;
int kCoroCount = 3;
int fix_range_size = -1;
bool kIsScan = false;

#ifdef USE_CORO
bool kUseCoro = true;
#else
bool kUseCoro = false;
#endif

std::string ycsb_load_path;
std::string ycsb_trans_path;
std::thread th[MAX_APP_THREAD];
uint64_t tp[MAX_APP_THREAD][MAX_CORO_NUM] = {0};
extern volatile bool need_stop;
extern uint64_t latency[MAX_APP_THREAD][MAX_CORO_NUM][LATENCY_WINDOWS];
uint64_t latency_th_all[LATENCY_WINDOWS];
extern uint64_t lock_fail[MAX_APP_THREAD];
extern uint64_t try_lock_op[MAX_APP_THREAD];
extern uint64_t write_handover_num[MAX_APP_THREAD];
extern uint64_t read_handover_num[MAX_APP_THREAD];


// double read_frac=0.95;
// double write_frac=0.05;
// double update_frac=0;
// double range_frac=0;
std::default_random_engine e;
std::uniform_int_distribution<Value> randval(0, UINT64_MAX - 1);

DSM *dsm = nullptr;
learned_index_global *learned_index_g = nullptr;
LLDex *lldex = nullptr;



class RequsetGenBench : public RequstGen {
public:
  RequsetGenBench(DSM* dsm, Request* req, int req_num, int coro_id, int coro_cnt) :
                  dsm(dsm), req(req), req_num(req_num), coro_id(coro_id), coro_cnt(coro_cnt) {
    local_thread_id = dsm->getMyThreadID();
    cur = coro_id;
    epoch_id = 0;
    extra_k = MAX_KEY_SPACE_SIZE + kThreadCount * kCoroCount * dsm->getMyNodeID() + local_thread_id * kCoroCount + coro_id;
    flag = false;
  }

  Request next() override {
    cur = (cur + coro_cnt) % req_num;
    if (cur + coro_cnt >= req_num) {
      // need_stop = true;
      ++ epoch_id;
      flag = true;
    }
    tp[local_thread_id][coro_id]++;
    req[cur].v = randval(e);
    // req[cur].v = random() % 1000;
    return req[cur];
  }

private:
  DSM *dsm;
  Request* req;
  int req_num;
  int coro_id;
  int coro_cnt;
  int local_thread_id;
  int cur;
  uint8_t epoch_id;
  uint64_t extra_k;
  bool flag;
};


RequstGen *gen_func(DSM* dsm, Request* req, int req_num, int coro_id, int coro_cnt) {
  return new RequsetGenBench(dsm, req, req_num, coro_id, coro_cnt);
}


void work_func(LLDex *lldex, Request& r, int thread_id, CoroContext *ctx) {
  if (r.req_type == SEARCH) {
    uint64_t value = 0;
    lldex->search(r.k, value, ctx);
    // assert(value == r.v);
  }
  else if (r.req_type == INSERT) {
    lldex->insert(r.k, r.v, thread_id, ctx);
  }
  else if (r.req_type == UPDATE) {
    lldex->insert(r.k, r.v, thread_id, ctx);
  }
  else {
    uint64_t key_end = r.k + r.range_size;
    std::vector<kv> kvs;
    lldex->scan(r.k, key_end, kvs);
  }
}

std::atomic<int64_t> load_cnt{0};
std::atomic_bool load_finish{false};

void thread_run(int id)
{
  bindCore(id * 2 + 1);
  dsm->registerThread();
  uint64_t my_id= kThreadCount * dsm->getMyNodeID() + id;
  std::cout<<"I am thread "<<my_id<<" on compute nodes"<<std::endl;


  Request* req = new Request[MAX_THREAD_REQUEST];
  int req_num = 0;
  std::ifstream trans_in(ycsb_trans_path + std::to_string(my_id));
  if (!trans_in.is_open()) {
    printf("Error opening trans file\n");
    assert(false);
  }
  std::string op;
  uint64_t int_k;
  int range_size = 0;
  int cnt = 0;
  while(trans_in >> op >> int_k) {
    if (op == "SCAN") trans_in >> range_size;
    else range_size = 0;
    Request r;
    r.req_type = (op == "READ"  ? SEARCH : (
                  op == "INSERT"? INSERT : (
                  op == "UPDATE"? UPDATE : SCAN
    )));
    r.range_size = fix_range_size >= 0 ? fix_range_size : range_size;
    r.k = int_k;
    r.v = int_k + 23;
    req[req_num ++] = r;
    if (++ cnt % LOAD_HEARTBEAT == 0) {
      printf("thread %d: %d trans entries loaded.\n", id, cnt);
    }
  }
  std::cout<<"thread "<<id<<" loaded "<<req_num<<" requests"<<std::endl;
  load_cnt.fetch_add(1);

  if(id == 0){
    while(load_cnt.load() != kThreadCount){}
    std::cout<<"node "<<dsm->getMyNodeID()<<" loaded finish"<<std::endl;
    dsm->barrier("load_finish");
    load_finish = true;
    std::cout<<"load finish"<<std::endl;
    load_cnt.store(-1);
  }

  while(load_cnt.load() != -1){}

  if(!kIsScan && kUseCoro){
    // fix
    lldex->run_coroutine(gen_func, work_func, kCoroCount, req, req_num, id);
  }else{
    Timer timer;
    auto gen = new RequsetGenBench(dsm, req, req_num, 0, 1);
    int req_count = 0;
    while (!need_stop) {
      req_count++;
      auto r = gen->next();

      timer.begin();

      work_func(lldex, r, id, nullptr);

      auto us_10 = timer.end() / 100;

      if (us_10 >= LATENCY_WINDOWS) {
        us_10 = LATENCY_WINDOWS - 1;
      }
      latency[id][0][us_10]++;
    }
    std::cout<<"thread "<<id<<" requests: "<<req_count<<std::endl;
  }
}

void parse_args(int argc, char *argv[]) {
  if (argc != 6 && argc != 7) {
    printf("Usage: ./ycsb_test kNodeCount kThreadCount kCoroCnt workload_type[randint] workload_idx[a/b/c/d/e] [fix_range_size/rm_write_conflict]\n");
    exit(-1);
  }

  kNodeCount = atoi(argv[1]);
  kThreadCount = atoi(argv[2]);
  kCoroCount = atoi(argv[3]);
  assert(std::string(argv[4]) == "randint");  // currently only int workloads is tested
  // kIsScan = (std::string(argv[5]) == "e");

  std::string workload_dir;
  std::ifstream workloads_dir_in("../workloads.conf");
  if (!workloads_dir_in.is_open()) {
    printf("Error opening workloads.conf\n");
    assert(false);
  }
  workloads_dir_in >> workload_dir;
  ycsb_load_path = workload_dir + "/load_" + std::string(argv[4]) + "_workload" + std::string(argv[5]);
  ycsb_trans_path = workload_dir + "/txn_" + std::string(argv[4]) + "_workload" + std::string(argv[5]);
  if (argc == 7) {
    if(kIsScan) fix_range_size = atoi(argv[6]);
  }

  printf("kNodeCount %d, kThreadCount %d, kCoroCnt %d\n", kNodeCount, kThreadCount, kCoroCount);
  printf("ycsb_load: %s\n", ycsb_load_path.c_str());
  printf("ycsb_trans: %s\n", ycsb_trans_path.c_str());
  if (argc == 7) {
    if(kIsScan) printf("fix_range_size: %d\n", fix_range_size);
  }
}

void save_latency(int epoch_id) {
  // sum up local latency cnt
  for (int i = 0; i < LATENCY_WINDOWS; ++i) {
    latency_th_all[i] = 0;
    for (int k = 0; k < MAX_APP_THREAD; ++k)
      for (int j = 0; j < MAX_CORO_NUM; ++j) {
        latency_th_all[i] += latency[k][j][i];
    }
  }
  // store in file
  std::ofstream f_out("../us_lat/epoch_" + std::to_string(epoch_id) + ".lat");
  f_out << std::setiosflags(std::ios::fixed) << std::setprecision(1);
  if (f_out.is_open()) {
    for (int i = 0; i < LATENCY_WINDOWS; ++i) {
      f_out << i / 10.0 << "\t" << latency_th_all[i] << std::endl;
    }
    f_out.close();
  }
  else {
    printf("Fail to write file!\n");
    assert(false);
  }
  memset(latency, 0, sizeof(uint64_t) * MAX_APP_THREAD * MAX_CORO_NUM * LATENCY_WINDOWS);
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  DSMConfig config;
  config.machineNR = kNodeCount;
  dsm = DSM::getInstance(config);
  dsm->registerThread();
  write_buffer_conf write_buffer_conf_;
  write_buffer_conf_.buffer_size=((uint64_t)1<<28);
  write_buffer_conf_.thread_num = kThreadCount;
  write_buffer_conf_.buffer_num=1;
  if(dsm->getMyNodeID() == 0){
    // load dataset
    std::cout<<"I am server0, start loading dataset"<<std::endl;
    std::vector<uint64_t> key_init;
    std::vector<uint64_t> val_init;
    std::string op;
    uint64_t int_k;
    std::ifstream load_in(ycsb_load_path);
    uint64_t cnt = 0;
    if (!load_in.is_open()) {
      printf("Error opening load file\n");
      assert(false);
    }
    while(load_in >> op >> int_k) {
      assert(op == "INSERT");
      key_init.push_back(int_k);
      cnt++;
    }
    sort(key_init.begin(), key_init.end());
    for(int i = 0; i < key_init.size(); i++){
      val_init.push_back(key_init[i] + 23);
    }
    

    std::cout<<"load dataset, "<<cnt<<" entries"<<std::endl;
    
    //alloc write buffer
    int single_buffer_size=write_buffer_conf_.buffer_size/write_buffer_conf_.buffer_num;
    for(int i=0;i<write_buffer_conf_.buffer_num;i++)
    {
      write_buffer *temp=new write_buffer(i,dsm,single_buffer_size,write_buffer_conf_.thread_num);
    }

    //build loacal learned index
    learned_index_local *learned_index_l=new learned_index_local(4);
    std::vector<std::vector<uint64_t>> seg_keys;
    std::vector<std::vector<uint64_t>> seg_vals;
    learned_index_l->build_local_with_empty(key_init, val_init, seg_keys, seg_vals);

    //build global learned index in DM
    learned_index_g = new learned_index_global(dsm, define::Epsilon, write_buffer_conf_);
    GlobalAddress root_model_addr;
    learned_index_g->build_remote_with_empty(seg_keys, seg_vals, *learned_index_l, root_model_addr);
    dsm->Put(0, &root_model_addr, sizeof(root_model_addr));
    std::cout << "Root model address stored in memcached, nodeID = " << root_model_addr.nodeID
              << ", offset = " << std::hex << root_model_addr.offset << std::endl;
    std::cout << std::dec;
    learned_index_g->print_level_model();
  }

  dsm->barrier("root_model_sync");

  GlobalAddress root_model_addr;
  size_t size = dsm->Get(0, &root_model_addr);
  std::cout << "Root model address read from memcached on node " << dsm->getMyNodeID() 
            << ": nodeID=" << root_model_addr.nodeID 
            << ", offset=0x" << std::hex << root_model_addr.offset << std::dec << std::endl;
  sleep(10);
  
  lldex = new LLDex(dsm, define::Epsilon, write_buffer_conf_);
  lldex->read_model_from_remote(root_model_addr);
  lldex->print_level_model();

  dsm->barrier("benchmark");

  memset(latency, 0, sizeof(uint64_t) * MAX_APP_THREAD * MAX_CORO_NUM * LATENCY_WINDOWS);
  // dsm->resetThread();

  std::thread retrain_thread(retrainThread, dsm, write_buffer_conf_);


  timespec s, e;
  uint64_t pre_tp = 0;
  int count = 0;

  for(int i = 0; i < kThreadCount; i++){
    th[i] = std::thread(thread_run, i);
  }

  while(!load_finish.load())
    ;

  clock_gettime(CLOCK_REALTIME, &s);
  while(!need_stop){
    sleep(TIME_INTERVAL);
    clock_gettime(CLOCK_REALTIME, &e);
    int microseconds = (e.tv_sec - s.tv_sec) * 1000000 +
                    (double)(e.tv_nsec - s.tv_nsec) / 1000;
    uint64_t all_tp = 0;
    for(int i = 0; i < MAX_APP_THREAD; i++){
      for(int j = 0; j < MAX_CORO_NUM; j++){
        all_tp += tp[i][j];
      }
    }
    clock_gettime(CLOCK_REALTIME, &s);
    uint64_t cap = all_tp - pre_tp;
    pre_tp = all_tp;

    uint64_t lock_fail_cnt = 0;
    uint64_t try_lock_op_cnt = 0;
    uint64_t write_handover_cnt = 0;
    uint64_t read_handover_cnt = 0;
    for(int i = 0; i < MAX_APP_THREAD; i++){
      lock_fail_cnt += lock_fail[i];
      try_lock_op_cnt += try_lock_op[i];
      write_handover_cnt += write_handover_num[i];
      read_handover_cnt += read_handover_num[i];
    }
    //aviod div zero
    if(try_lock_op_cnt == 0) try_lock_op_cnt = 1;

    double per_node_tp = cap * 1.0 / microseconds;
    uint64_t cluster_tp = dsm->sum(uint64_t(per_node_tp * 1000));

    printf("%d, throughput %.4f\n", dsm->getMyNodeID(), per_node_tp);

    if(dsm->getMyNodeID() == 0){
      std::cout<<"epoch "<<count<<" passed"<<std::endl;
      printf("cluster throughput %.3f Mops\n", cluster_tp / 1000.0);
      printf("avg. lock/cas fail cnt: %.4lf\n", lock_fail_cnt * 1.0 / try_lock_op_cnt);
      printf("avg. write handover cnt: %.4lf\n", write_handover_cnt * 1.0 / try_lock_op_cnt);
      printf("avg. read handover cnt: %.4lf\n", read_handover_cnt * 1.0 / try_lock_op_cnt);
      printf("\n");
    }
    
    save_latency(count);
    count++;
    
    if(count > TEST_EPOCH){
      std::cout<<"All test epochs passed, stop benchmark"<<std::endl;
      need_stop = true;
    }
  }

  for(int i = 0; i < kThreadCount; i++){
    th[i].join();
    std::cout<<"Thread "<<i<<" joined"<<std::endl;
  }

  RetrainManager::getInstance().stop();
  retrain_thread.join();
  std::cout<<"Retrain thread joined"<<std::endl;

  std::cout<<"[END]"<<std::endl;
  return 0;

}