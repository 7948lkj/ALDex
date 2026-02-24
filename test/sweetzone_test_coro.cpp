#include "Common.h"
#include "DSM.h"
#include "Timer.h"

#include <stdlib.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <random>
#include <atomic>
#include <iomanip>
#include <fstream>
#include <functional>
#include <memory>

#define TEST_EPOCH 10
#define TIME_INTERVAL 1
#define MAX_APP_THREAD 64
#define LATENCY_WINDOWS 1000000

// Test parameters
int kThreadCount;
int kNodeCount;
const uint64_t ALLOCATED_SIZE = 4ULL * 1024 * 1024 * 1024; // 1GB
const std::vector<size_t> ACCESS_GRANULARITIES = {16, 32, 64, 128, 256, 512, 1024};
int kCoroCnt = define::kCoroCnt;

// Global variables
DSM *dsm = nullptr;
GlobalAddress allocated_space;
std::thread th[MAX_APP_THREAD];
uint64_t tp[MAX_APP_THREAD];
std::atomic<bool> need_stop{false};
std::atomic<bool> load_finish{false};
std::atomic<int64_t> load_cnt{0};
std::atomic<int64_t> finish_cnt{0};
uint64_t latency[MAX_APP_THREAD][LATENCY_WINDOWS];
std::atomic<size_t> current_granularity{0};
std::atomic<bool> test_ready{false};

// Random number generator
thread_local std::mt19937_64 rng(std::random_device{}());


// Request structure for coroutines
struct SweetRequest {
  GlobalAddress addr;
  size_t size;
};

// Request generator for coroutines
class SweetRequestGen {
public:
  explicit SweetRequestGen(size_t granularity)
      : granularity_(granularity),
        num_blocks_(ALLOCATED_SIZE / granularity),
        dist_(0, num_blocks_ - 1) {}

  SweetRequest next() {
    uint64_t block_idx = dist_(rng);
    uint64_t offset = block_idx * granularity_;
    GlobalAddress addr = allocated_space;
    addr.offset += offset;
    return {addr, granularity_};
  }

private:
  size_t granularity_;
  uint64_t num_blocks_;
  std::uniform_int_distribution<uint64_t> dist_;
};

// Coroutine support
thread_local CoroCall sweet_workers[define::kMaxCoro];
thread_local CoroCall sweet_master;

// Coroutine worker function
void sweet_coro_worker(CoroYield &yield, SweetRequestGen *gen, int thread_id, int coro_id) {
  CoroContext ctx;
  ctx.coro_id = coro_id;
  ctx.master = &sweet_master;
  ctx.yield = &yield;

  char *local_buffer = dsm->get_coro_buf(coro_id);
  Timer coro_timer;

  while (!need_stop.load()) {
    auto req = gen->next();
    
    coro_timer.begin();
    // Post async read and yield to master
    dsm->read_sync(local_buffer, req.addr, req.size, &ctx);
    
    // After resumption, read is complete
    auto us_10 = coro_timer.end() / 100;
    if (us_10 >= LATENCY_WINDOWS) {
      us_10 = LATENCY_WINDOWS - 1;
    }
    latency[thread_id][us_10]++;
    tp[thread_id]++;
  }
}

// Coroutine master function
void sweet_coro_master(CoroYield &yield, int coro_cnt) {
  // Initialize all workers
  for (int i = 0; i < coro_cnt; ++i) {
    yield(sweet_workers[i]);
  }

  // Main polling loop - only resume on actual completions
  while (!need_stop.load()) {
    uint64_t next_coro_id;
    if (dsm->poll_rdma_cq_once(next_coro_id)) {
      // Got a completion, resume the corresponding coroutine
      if (next_coro_id < static_cast<uint64_t>(coro_cnt)) {
        yield(sweet_workers[next_coro_id]);
      }
    }
  }
  
  // Drain any remaining completions after stop
  for (int drain = 0; drain < 1000; ++drain) {
    uint64_t next_coro_id;
    if (dsm->poll_rdma_cq_once(next_coro_id)) {
      if (next_coro_id < static_cast<uint64_t>(coro_cnt)) {
        yield(sweet_workers[next_coro_id]);
      }
    }
  }
}

// Run coroutines for a given granularity
void run_sweet_coroutines(size_t granularity, int thread_id, int coro_cnt) {
  assert(coro_cnt <= define::kMaxCoro);
  using namespace std::placeholders;

  // Create request generators for each coroutine
  std::vector<std::unique_ptr<SweetRequestGen>> gens;
  gens.reserve(coro_cnt);
  for (int i = 0; i < coro_cnt; ++i) {
    gens.emplace_back(std::make_unique<SweetRequestGen>(granularity));
    sweet_workers[i] = CoroCall(
        std::bind(&sweet_coro_worker, _1, gens.back().get(), thread_id, i));
  }

  sweet_master = CoroCall(std::bind(&sweet_coro_master, _1, coro_cnt));
  sweet_master();
}

// void bindCore(uint16_t core) {
//   cpu_set_t cpuset;
//   CPU_ZERO(&cpuset);
//   CPU_SET(core, &cpuset);
//   int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
//   if (rc != 0) {
//     printf("Error calling pthread_setaffinity_np: %d\n", rc);
//   }
// }

void save_latency(int epoch_id, size_t granularity) {
  // Sum up local latency cnt
  uint64_t latency_th_all[LATENCY_WINDOWS] = {0};
  for (int i = 0; i < LATENCY_WINDOWS; ++i) {
    for (int k = 0; k < MAX_APP_THREAD; ++k) {
      latency_th_all[i] += latency[k][i];
    }
  }
  
  // Store in file
  std::string filename = "../us_lat/sweetzone_" + std::to_string(granularity) + 
                         "B_epoch_" + std::to_string(epoch_id) + ".lat";
  std::ofstream f_out(filename);
  f_out << std::setiosflags(std::ios::fixed) << std::setprecision(1);
  if (f_out.is_open()) {
    for (int i = 0; i < LATENCY_WINDOWS; ++i) {
      if (latency_th_all[i] > 0) {
        f_out << i / 10.0 << "\t" << latency_th_all[i] << std::endl;
      }
    }
    f_out.close();
  } else {
    printf("Failed to write latency file!\n");
  }
  
  // Clear latency counters
  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    for (int j = 0; j < LATENCY_WINDOWS; ++j) {
      latency[i][j] = 0;
    }
  }
}

void thread_run(int thread_id) {
  bindCore(thread_id * 2 + 1);
  dsm->registerThread();
  
  uint64_t my_id = kThreadCount * dsm->getMyNodeID() + thread_id;
  printf("Thread %lu started on node %d\n", my_id, dsm->getMyNodeID());
  
  // Thread runs for all test rounds
  for (size_t granularity : ACCESS_GRANULARITIES) {
    // All threads wait for test to be ready
    while (current_granularity.load() != granularity) {
      usleep(100);
    }
    
    printf("Thread %lu testing granularity %lu bytes\n", my_id, granularity);
    
    tp[thread_id] = 0;
    
    // Local synchronization - wait for all threads to be ready
    load_cnt.fetch_add(1);
    if (thread_id == 0) {
      while (load_cnt.load() != kThreadCount) {
        usleep(10);
      }
      printf("Node %d thread loading finished for granularity %lu\n", 
             dsm->getMyNodeID(), granularity);
      load_finish.store(true);
      load_cnt.store(-1);
    }
    while (load_cnt.load() != -1) {
      usleep(10);
    }
    
    // Run coroutines for this granularity
    run_sweet_coroutines(granularity, thread_id, kCoroCnt);
    
    printf("Thread %d finished granularity %lu\n", thread_id, granularity);
    
    // Synchronize with main thread - all threads increment finish counter
    finish_cnt.fetch_add(1);
    if (thread_id == 0) {
      // Wait for all threads to finish
      while (finish_cnt.load() != kThreadCount) {
        usleep(10);
      }
      printf("Node %d all threads finished granularity %lu\n", 
             dsm->getMyNodeID(), granularity);
      // Signal ready for next round
      test_ready.store(true);
    }
    
    // Wait for next round to be ready (main thread sets test_ready back to false)
    while (test_ready.load()) {
      usleep(100);
    }
  }
  
  printf("Thread %d exiting\n", thread_id);
}

void run_test_round(size_t granularity) {
  printf("\n========== Testing with granularity: %lu bytes ==========\n", granularity);
  
  // Reset counters
  need_stop.store(false);
  load_finish.store(false);
  load_cnt.store(0);
  finish_cnt.store(0);
  test_ready.store(false);
  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    tp[i] = 0;
    for (int j = 0; j < LATENCY_WINDOWS; ++j) {
      latency[i][j] = 0;
    }
  }
  
  // Set current granularity and signal threads
  current_granularity.store(granularity);
  
  // Wait for local threads to be ready
  while (!load_finish.load()) {
    usleep(1000);
  }
  
  printf("Main thread: local threads ready for granularity %lu\n", granularity);
  
  // Barrier before starting test - sync with other nodes
  dsm->barrier("test_round_start_" + std::to_string(granularity));
  
  timespec s, e;
  uint64_t pre_tp = 0;
  int count = 0;
  
  clock_gettime(CLOCK_REALTIME, &s);
  
  // Run test epochs
  while (!need_stop.load()) {
    sleep(TIME_INTERVAL);
    clock_gettime(CLOCK_REALTIME, &e);
    
    int microseconds = (e.tv_sec - s.tv_sec) * 1000000 +
                       (e.tv_nsec - s.tv_nsec) / 1000;
    
    uint64_t all_tp = 0;
    for (int i = 0; i < MAX_APP_THREAD; i++) {
      all_tp += tp[i];
    }
    
    clock_gettime(CLOCK_REALTIME, &s);
    uint64_t cap = all_tp - pre_tp;
    pre_tp = all_tp;
    
    double per_node_tp = cap * 1.0 / microseconds;
    uint64_t cluster_tp = dsm->sum(uint64_t(per_node_tp * 1000));
    
    printf("Node %d, throughput %.4f Mops\n", dsm->getMyNodeID(), per_node_tp);
    
    if (dsm->getMyNodeID() == 0) {
      printf("Epoch %d, Granularity %lu B, Cluster throughput %.3f Mops\n", 
             count, granularity, cluster_tp / 1000.0);
    }
    
    save_latency(count, granularity);
    count++;
    
    if (count >= TEST_EPOCH) {
      printf("Test epochs completed for granularity %lu bytes\n", granularity);
      need_stop.store(true);
    }
  }
  
  // Wait for threads to finish this round
  while (!test_ready.load()) {
    usleep(1000);
  }
  
  printf("Main thread: all local threads finished granularity %lu\n", granularity);
  
  dsm->barrier("test_round_end_" + std::to_string(granularity));
  
  // Signal threads to proceed to next round
  test_ready.store(false);
  
  printf("Round %lu bytes completed\n", granularity);
}

void parse_args(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: ./sweetzone_test <kNodeCount> <kThreadCount> <kCoroCnt>\n");
    exit(-1);
  }
  
  kNodeCount = atoi(argv[1]);
  kThreadCount = atoi(argv[2]);
  kCoroCnt = atoi(argv[3]);
  
  if (kThreadCount > MAX_APP_THREAD) {
    printf("Error: kThreadCount (%d) exceeds MAX_APP_THREAD (%d)\n", 
           kThreadCount, MAX_APP_THREAD);
    exit(-1);
  }

  if(kCoroCnt > define::kMaxCoro) {
    printf("Error: kCoroCnt (%d) exceeds define::kMaxCoro (%d)\n", 
           kCoroCnt, define::kMaxCoro);
    exit(-1);
  }
  
  printf("Configuration: kNodeCount=%d, kThreadCount=%d, kCoroCnt=%d\n", kNodeCount, kThreadCount, kCoroCnt);
  printf("Allocated memory: %lu GB\n", ALLOCATED_SIZE / (1024ULL * 1024 * 1024));
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  
  DSMConfig config;
  config.machineNR = kNodeCount;
  dsm = DSM::getInstance(config);
  dsm->registerThread();
  
  // Node 0 allocates 1GB space
  if (dsm->getMyNodeID() == 0) {
    printf("Node 0: Allocating %lu GB of memory...\n", 
           ALLOCATED_SIZE / (1024ULL * 1024 * 1024));
    
    allocated_space = dsm->alloc(ALLOCATED_SIZE);
    
    printf("Node 0: Memory allocated at nodeID=%d, offset=0x%lx\n",
           allocated_space.nodeID, allocated_space.offset);
    
    // Initialize allocated memory with pattern
    printf("Node 0: Initializing allocated memory...\n");
    const size_t CHUNK_SIZE = 1024 * 1024; // 1MB chunks
    char* init_buffer = dsm->get_cache();
    memset(init_buffer, 0xCD, CHUNK_SIZE);
    
    for (uint64_t offset = 0; offset < ALLOCATED_SIZE; offset += CHUNK_SIZE) {
      GlobalAddress write_addr = allocated_space;
      write_addr.offset += offset;
      dsm->write_sync(init_buffer, write_addr, CHUNK_SIZE);
      if (offset % (100 * 1024 * 1024) == 0) {
        printf("  Initialized %lu MB / %lu MB\n", 
               offset / (1024 * 1024), ALLOCATED_SIZE / (1024 * 1024));
      }
    }
    // delete[] init_buffer;
    printf("Node 0: Memory initialization completed\n");
    
    // Store address in memcached for other nodes
    dsm->Put(0, &allocated_space, sizeof(GlobalAddress));
    printf("Node 0: Allocated address stored in memcached\n");
  }
  
  dsm->barrier("allocation_done");
  
  // All nodes retrieve the allocated address
  if (dsm->getMyNodeID() != 0) {
    size_t size = dsm->Get(0, &allocated_space);
    printf("Node %d: Retrieved allocated address from memcached: nodeID=%d, offset=0x%lx\n",
           dsm->getMyNodeID(), allocated_space.nodeID, allocated_space.offset);
  }
  
  dsm->barrier("address_sync");
  sleep(2);
  
  // Start worker threads once for all tests
  for (int i = 0; i < kThreadCount; i++) {
    th[i] = std::thread(thread_run, i);
  }
  
  // Give threads time to start and register
  sleep(1);
  
  // Run tests for each granularity
  for (size_t granularity : ACCESS_GRANULARITIES) {
    run_test_round(granularity);
    sleep(2); // Small delay between rounds
  }
  
  // Join threads
  for (int i = 0; i < kThreadCount; i++) {
    th[i].join();
  }
  
  if (dsm->getMyNodeID() == 0) {
    printf("\n========== All Tests Completed ==========\n");
  }
  
  dsm->barrier("final");
  printf("Node %d exiting\n", dsm->getMyNodeID());
  
  return 0;
}
