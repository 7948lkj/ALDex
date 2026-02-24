#ifndef __COMMON_H__
#define __COMMON_H__

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

#include <atomic>
#include <bitset>
#include <limits>
#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <queue>
#include <unordered_map>
#include <shared_mutex>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <boost/icl/interval_map.hpp>
#include <boost/variant.hpp>

#include "Debug.h"
#include "HugePageAlloc.h"
#include "Rdma.h"

#include "WRLock.h"

// CONFIG_ENABLE_EMBEDDING_LOCK and CONFIG_ENABLE_CRC
// **cannot** be ON at the same time

// #define CONFIG_ENABLE_EMBEDDING_LOCK
// #define CONFIG_ENABLE_CRC

#define LATENCY_WINDOWS 1000000

#define STRUCT_OFFSET(type, field)                                             \
  (char *)&((type *)(0))->field - (char *)((type *)(0))

#define MAX_MACHINE 8

#define CPU_PHYSICAL_CORE_NUM 72

#define MAX_CORO_NUM 8

#define MAX_KEY_SPACE_SIZE 60000000

#define ADD_ROUND(x, n) ((x) = ((x) + 1) % (n))

#define MESSAGE_SIZE 96 // byte

#define POST_RECV_PER_RC_QP 128

#define RAW_RECV_CQ_COUNT 128

// { app thread
#define MAX_APP_THREAD 129

#define APP_MESSAGE_NR 96


#define pre_alloc_size 2000


#define null_next 0xffffffffffffffff
//flag in slot
#define model_flag 0x8000000000000000
#define chain_flag 0x2000000000000000


//flag in list_node or node_set
#define tail_flag 0x8000000000000000

//flag in list_node and slot
#define set_flag 0x4000000000000000
#define null_flag 0x1000000000000000

//flag in retrain
#define slots_flag 0x3000000000000000
#define kvs_flag 0x4000000000000000
#define next_flag 0x1000000000000000


#define mask_ 0x0fffffffffffffff

// }

// { dir thread
#define NR_DIRECTORY 1

#define DIR_MESSAGE_NR 128
// }

#include <boost/coroutine2/all.hpp>
#include <boost/crc.hpp>


using CoroQueue = std::queue<uint16_t>;


void bindCore(uint16_t core);
char *getIP();
char *getMac();

inline int bits_in(std::uint64_t u) {
  auto bs = std::bitset<64>(u);
  return bs.count();
}

#include <boost/coroutine/all.hpp>

using CoroYield = boost::coroutines::symmetric_coroutine<void>::yield_type;
using CoroCall = boost::coroutines::symmetric_coroutine<void>::call_type;
using CoroQueue = std::queue<uint16_t>;


struct CoroContext {
  CoroYield *yield;
  CoroCall *master;
  int coro_id;
};

namespace define {

constexpr uint64_t MB = 1024ull * 1024;
constexpr uint64_t GB = 1024ull * MB;
constexpr uint16_t kCacheLineSize = 64;
constexpr uint64_t dsmSize           = 64;        // GB  [CONFIG] 64
constexpr uint64_t cacheSize         = 4;


constexpr int thread_num = 32;
constexpr int kCoroCnt = 3;
constexpr int max_chain_length=4;//cant be 1
constexpr int min_model_size=8;
constexpr double write_buffer_retrain_rate=0.5;

constexpr uint64_t kLocalLockNum    = 4 * MB;

constexpr uint64_t Epsilon = 4;

// for remote allocate
constexpr uint64_t kChunkSize = 32 * MB;

// for store root pointer
constexpr uint64_t kRootPointerStoreOffest = kChunkSize / 2;
static_assert(kRootPointerStoreOffest % sizeof(uint64_t) == 0, "XX");

constexpr uint64_t rootPointerAddr = 0;

// lock on-chip memory
constexpr uint64_t kLockStartAddr = 0;
constexpr uint64_t kLockChipMemSize = 256 * 1024;

// number of locks
// we do not use 16-bit locks, since 64-bit locks can provide enough concurrency.
// if you want to use 16-bit locks, call *cas_dm_mask*
constexpr uint64_t kNumOfLock = kLockChipMemSize / sizeof(uint64_t);

// level of tree
constexpr uint64_t kMaxLevelOfTree = 7;

constexpr uint16_t kMaxCoro = 8;
constexpr int64_t kPerCoroRdmaBuf = 128 * 1024;

constexpr uint8_t kMaxHandOverTime = 8;

constexpr int kIndexCacheSize = 5; // MB
} // namespace define

static inline unsigned long long asm_rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

// For Tree
using Key = uint64_t;
using Value = uint64_t;
constexpr Key kKeyMin = std::numeric_limits<Key>::min();
constexpr Key kKeyMax = std::numeric_limits<Key>::max();
constexpr Value kValueNull = 0;

struct kv_pair
{
    uint64_t key;
    uint64_t val;
};

// Note: our RNICs can read 1KB data in increasing address order (but not for 4KB)
constexpr uint32_t kInternalPageSize = 1024;
constexpr uint32_t kLeafPageSize = 1024;

constexpr bool enable_cache = true;
constexpr bool enable_local_lock = true;
constexpr bool enable_read_delegation = true;
constexpr bool enable_backoff = false;

__inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

inline void mfence() { asm volatile("mfence" ::: "memory"); }

inline void compiler_barrier() { asm volatile("" ::: "memory"); }


#endif /* __COMMON_H__ */
