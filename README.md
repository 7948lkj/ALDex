# ALDex: An Adaptively Constructed Hierarchical Lock-Free Learned Index on Disaggregated Memory
This is the implementation repository of paper: **ALDex: An Adaptively Constructed Hierarchical Lock-Free Learned Index on Disaggregated Memory**.

Thanks to the work of [CHIME](https://github.com/dmemsys/CHIME), we have reused its testing framework and scripts.

## Supported Platform
We strongly recommend you to run ALDex using the r650 instances on [CloudLab](https://www.cloudlab.us/) as the code has been thoroughly tested there. If you want to reproduce the results in the paper, 10 r650 machines are needed.

## Environment Setup
You have to install the necessary dependencies in order to build ALDex.
Note that you should run the following steps(1-4) on **all** nodes you have created, run step5 on node0 of cluster

1) Clone ALDex
    ```shell
    sudo su
    git clone https://github.com/7948lkj/ALDex.git
    cd ALDex
    ```
2) Install dependency.
    ```shell
    sh ./script/installMLNX.sh
    sh ./script/installLibs.sh
    ``` 
3) network
    ```shell
    sudo ip link set ens2f0 up
    # x is your node id in cluster, start from 1
    sudo ip addr add 10.10.1.x/24 dev ens2f0
    ``` 
4) disk
    ```shell
    sudo mkfs.ext4 /dev/nvme0n1
    sudo mkdir -p /mnt/nvme
    sudo mount /dev/nvme0n1 /mnt/nvme
    ``` 
5) memcached
    ```shell
    # Custom IP: Also modify memcached.conf
    path/to/memcached  -d -m 1024 -u root -l 10.10.1.1 -p 11211 -c 1024 -P /tmp/memcached.pid
    ``` 


## YCSB Workloads
You should run the following steps on all nodes.
1) Download YCSB source code.
    ```shell
    sudo su
    mv ALDex/ycsb /mnt/nvme
    cd /mnt/nvme/ycsb
    curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.11.0/ycsb-0.11.0.tar.gz
    tar xfvz ycsb-0.11.0.tar.gz
    mv ycsb-0.11.0 YCSB
    ```

2) Generate YCSB workloads
    ```shell
    # small workloads takes about 20 seconds.
    sh generate_small_workloads.sh

    # full workloads takes about 3 hours.
    sh generate_full_workloads.sh
    ```

## Getting Started
1) Alloc Hugepages on all nodes.
    ```shell
    sudo su
    echo 36864 > /proc/sys/vm/nr_hugepages
    ulimit -l unlimited
    ```

2) Compile
Return to the root directory of ALDex and execute the following commands on **all** nodes to compile:
    ```shell
    mkdir build; cd build; cmake ..; make
    ```

3) Split workloads
    ```shell
    python3 /mnt/nvme/ycsb/split_workload.py <workload_name> randint <CN_num> <client_num_per_CN>
    # key_type: the type of key to test (*i.e.*, `randint`).
    # CN_num: the number of CNs.
    # client_num_per_CN: the number of clients on each CN.
    ```
    **Example**:
    ```shell
    python3 ../ycsb/split_workload.py a randint 10 24
    ```
4) start memcached
    ```shell
    /bin/bash ../script/restartMemc.sh
    ```

5) Execute the following command on **all** nodes to conduct a YCSB evaluation:
    ```shell
    ./ycsb_test <CN_num> <client_num_per_CN> <coro_num_per_client> <key_type> <workload_name>
    ```
    * coro_num_per_client: the number of coroutine in each client.

    **Example**:
    ```shell
    ./ycsb_test 10 24 2 randint a
    ```

6) Latency calculation
    ```shell
    python3 ../us_lat/cluster_latency.py <CN_num> <epoch_start> <epoch_num>
    ```

    **Example**:
    ```shell
    python3 ../us_lat/cluster_latency.py 10 1 10
    ```


## Acknowledgments
This repository adopts [Sherman](https://github.com/thustorage/Sherman)'s codebase, [SMART](https://github.com/dmemsys/SMART)'s Write Combining technique and [CHIME](https://github.com/dmemsys/CHIME)'s testing framework and scripts. We really appreciate it.