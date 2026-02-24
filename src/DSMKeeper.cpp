#include "DSMKeeper.h"

#include "Connection.h"

const char *DSMKeeper::OK = "OK";
const char *DSMKeeper::ServerPrefix = "SPre";

void DSMKeeper::initLocalMeta() {
  localMeta.dsmBase = (uint64_t)dirCon[0]->dsmPool;
  localMeta.lockBase = (uint64_t)dirCon[0]->lockPool;
  localMeta.cacheBase = (uint64_t)thCon[0]->cachePool;

  // per thread APP
  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    localMeta.appTh[i].lid = thCon[i]->ctx.lid;
    localMeta.appTh[i].rKey = thCon[i]->cacheMR->rkey;
    memcpy((char *)localMeta.appTh[i].gid, (char *)(&thCon[i]->ctx.gid),
           16 * sizeof(uint8_t));

    localMeta.appUdQpn[i] = thCon[i]->message->getQPN();
  }

  // per thread DIR
  for (int i = 0; i < NR_DIRECTORY; ++i) {
    localMeta.dirTh[i].lid = dirCon[i]->ctx.lid;
    localMeta.dirTh[i].rKey = dirCon[i]->dsmMR->rkey;
    localMeta.dirTh[i].lock_rkey = dirCon[i]->lockMR->rkey;
    memcpy((char *)localMeta.dirTh[i].gid, (char *)(&dirCon[i]->ctx.gid),
           16 * sizeof(uint8_t));

    localMeta.dirUdQpn[i] = dirCon[i]->message->getQPN();
  }

}

bool DSMKeeper::connectNode(uint16_t remoteID) {

  setDataToRemote(remoteID);

  std::string setK = setKey(remoteID);
  memSet(setK.c_str(), setK.size(), (char *)(&localMeta), sizeof(localMeta));

  std::string getK = getKey(remoteID);
  ExchangeMeta *remoteMeta = (ExchangeMeta *)memGet(getK.c_str(), getK.size());

  setDataFromRemote(remoteID, remoteMeta);

  free(remoteMeta);
  return true;
}

void DSMKeeper::setDataToRemote(uint16_t remoteID) {
  for (int i = 0; i < NR_DIRECTORY; ++i) {
    auto &c = dirCon[i];

    for (int k = 0; k < MAX_APP_THREAD; ++k) {
      localMeta.dirRcQpn2app[i][k] = c->data2app[k][remoteID]->qp_num;
    }
  }

  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    auto &c = thCon[i];
    for (int k = 0; k < NR_DIRECTORY; ++k) {
      localMeta.appRcQpn2dir[i][k] = c->data[k][remoteID]->qp_num;
    }
  
  }
}

void DSMKeeper::setDataFromRemote(uint16_t remoteID, ExchangeMeta *remoteMeta) {
  for (int i = 0; i < NR_DIRECTORY; ++i) {
    auto &c = dirCon[i];

    for (int k = 0; k < MAX_APP_THREAD; ++k) {
      auto &qp = c->data2app[k][remoteID];

      assert(qp->qp_type == IBV_QPT_RC);
      modifyQPtoInit(qp, &c->ctx);
      modifyQPtoRTR(qp, remoteMeta->appRcQpn2dir[k][i],
                    remoteMeta->appTh[k].lid, remoteMeta->appTh[k].gid,
                    &c->ctx);
      modifyQPtoRTS(qp);
    }
  }

  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    auto &c = thCon[i];
    for (int k = 0; k < NR_DIRECTORY; ++k) {
      auto &qp = c->data[k][remoteID];

      assert(qp->qp_type == IBV_QPT_RC);
      modifyQPtoInit(qp, &c->ctx);
      modifyQPtoRTR(qp, remoteMeta->dirRcQpn2app[k][i],
                    remoteMeta->dirTh[k].lid, remoteMeta->dirTh[k].gid,
                    &c->ctx);
      modifyQPtoRTS(qp);
    }
  }

  auto &info = remoteCon[remoteID];
  info.dsmBase = remoteMeta->dsmBase;
  info.cacheBase = remoteMeta->cacheBase;
  info.lockBase = remoteMeta->lockBase;

  for (int i = 0; i < NR_DIRECTORY; ++i) {
    info.dsmRKey[i] = remoteMeta->dirTh[i].rKey;
    info.lockRKey[i] = remoteMeta->dirTh[i].lock_rkey;
    info.dirMessageQPN[i] = remoteMeta->dirUdQpn[i];

    for (int k = 0; k < MAX_APP_THREAD; ++k) {
      struct ibv_ah_attr ahAttr;
      fillAhAttr(&ahAttr, remoteMeta->dirTh[i].lid, remoteMeta->dirTh[i].gid,
                 &thCon[k]->ctx);
      info.appToDirAh[k][i] = ibv_create_ah(thCon[k]->ctx.pd, &ahAttr);

      assert(info.appToDirAh[k][i]);
    }
  }


  for (int i = 0; i < MAX_APP_THREAD; ++i) {
    info.appRKey[i] = remoteMeta->appTh[i].rKey;
    info.appMessageQPN[i] = remoteMeta->appUdQpn[i];

    for (int k = 0; k < NR_DIRECTORY; ++k) {
      struct ibv_ah_attr ahAttr;
      fillAhAttr(&ahAttr, remoteMeta->appTh[i].lid, remoteMeta->appTh[i].gid,
                 &dirCon[k]->ctx);
      info.dirToAppAh[k][i] = ibv_create_ah(dirCon[k]->ctx.pd, &ahAttr);

      assert(info.dirToAppAh[k][i]);
    }
  }
}

void DSMKeeper::connectMySelf() {
  setDataToRemote(getMyNodeID());
  setDataFromRemote(getMyNodeID(), &localMeta);
}

void DSMKeeper::initRouteRule() {

  std::string k =
      std::string(ServerPrefix) + std::to_string(this->getMyNodeID());
  memSet(k.c_str(), k.size(), getMyIP().c_str(), getMyIP().size());
}

void DSMKeeper::barrier(const std::string &barrierKey) {
  std::string key = std::string("barrier-") + barrierKey;
  
  // 1. 使用同步机制确保节点0的初始化先完成
  if (this->getMyNodeID() == 0) {
    memSet(key.c_str(), key.size(), "0", 1);
    // 增加一个初始化完成的标记
    std::string initKey = key + "-init";
    memSet(initKey.c_str(), initKey.size(), "1", 1);
  } else {
    // 其他节点等待初始化完成
    std::string initKey = key + "-init";
    while (true) {
      char* initValue = memGet(initKey.c_str(), initKey.size());
      bool initialized = (initValue && std::string(initValue) == "1");
      free(initValue); // 释放内存
      if (initialized) break;
      usleep(100); // 短暂休眠，减少CPU消耗
    }
  }
  
  // 2. 使用memFetchAndAdd的返回值
  uint64_t myCount = memFetchAndAdd(key.c_str(), key.size());
  
  // 3. 添加超时机制和合理休眠的等待循环
  const int MAX_RETRIES = 300000; // 约30秒（假设每次重试1ms）
  int retries = 0;
  
  while (retries < MAX_RETRIES) {
    size_t valSize;
    char* value = memGet(key.c_str(), key.size(), &valSize);
    
    uint64_t v = 0;
    bool valid = false;
    
    if (value && valSize > 0) {
      try {
        v = std::stoull(value);
        valid = true;
      } catch (const std::exception& e) {
        // 处理转换错误
        valid = false;
      }
      free(value); // 释放内存
    }
    
    if (valid && v == this->getServerNR()) {
      return; // 所有节点都已到达屏障
    }
    
    // 4. 指数退避策略减少CPU使用
    usleep(1000 * (1 + retries / 100)); // 逐渐增加等待时间
    retries++;
  }
  
  // 5. 超时处理
  std::cerr << "Barrier timeout for key: " << barrierKey << std::endl;
  // 可以选择抛出异常或强制退出
}

uint64_t DSMKeeper::sum(const std::string &sum_key, uint64_t value) {
  std::string key_prefix = std::string("sum-") + sum_key;

  std::string key = key_prefix + std::to_string(this->getMyNodeID());
  memSet(key.c_str(), key.size(), (char *)&value, sizeof(value));

  uint64_t ret = 0;
  if (this->getMyNodeID() == 0) {  // only node 0 return the sum
    for (int i = 0; i < this->getServerNR(); ++i) {
      key = key_prefix + std::to_string(i);
      ret += *(uint64_t *)memGet(key.c_str(), key.size());
    }
  }

  return ret;
}