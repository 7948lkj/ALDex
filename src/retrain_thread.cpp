#include "retrainer.h"
#include "leaf.h"  // 假设包含 learned_index_global 的定义


void retrainThread(DSM *dsm, write_buffer_conf write_buffer_conf_) {
    bindCore((CPU_PHYSICAL_CORE_NUM - 1) * 2);
    dsm->registerThread();
    RetrainManager& manager = RetrainManager::getInstance();
    RetrainTask task;
    std::cout<<"retrain thread start"<<std::endl;
    while (manager.getTask(task)) {
        GlobalAddress target_next;
        target_next.val = task.next_addr_content[0];
        uint64_t next_content = task.next_addr_content[1];
        std::vector<kv_pair> chain_kvs = task.chain_kvs;

        std::stable_sort(chain_kvs.begin(), chain_kvs.end(), [](const kv_pair &a, const kv_pair &b) {
            return a.key < b.key;});
        chain_kvs.erase(std::unique(chain_kvs.begin(), chain_kvs.end(), [](const kv_pair &a, const kv_pair &b){
            return a.key == b.key;
        }), chain_kvs.end());

        learned_index_local *local_model = new learned_index_local(define::Epsilon);
        std::vector<uint64_t> init_keys;
        std::vector<uint64_t> init_vals;
        std::vector<std::vector<uint64_t>> seg_keys;
        std::vector<std::vector<uint64_t>> seg_vals;
        std::cout<<"retrain keys: ";
        for(auto &kv : chain_kvs)
        {
            init_keys.push_back(kv.key);
            init_vals.push_back(kv.val);
            std::cout<<kv.key<<" ";
        }
        std::cout<<std::endl;
        local_model->build_local_with_empty(init_keys, init_vals, seg_keys, seg_vals);

        learned_index_global *global_model = new learned_index_global(dsm, define::Epsilon, write_buffer_conf_);
        GlobalAddress submodel_root;
        global_model->build_remote_with_empty(seg_keys, seg_vals, *local_model, submodel_root);
        uint64_t old_val = next_content;
        uint64_t new_val = submodel_root.val | model_flag;
        bool ret = dsm->cas_sync(target_next, old_val, new_val, (uint64_t*)dsm->get_rdma_buffer());
        if(ret)
        {
            std::cout<<"retrain success"<<std::endl;
        }
        // else
        // {
        //     std::cout<<"retrain failed"<<std::endl;
        //     std::cout<<"equal: "<<old_val<<" new equal: "<<*((uint64_t*)dsm->get_rdma_buffer())<<std::endl;
        // }
        delete local_model;
        delete global_model;
    }
}