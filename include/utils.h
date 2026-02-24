#include "Common.h"

static void load_dataset(std::string file_path, std::vector<uint64_t> &dataset){
    // std::cout << "\r\n********* Running func load_dataset *********" << std::endl;
    // std::cout << "the dataset path is "<< file_path << std::endl;
    std::ifstream fin(file_path.c_str()); //打开文件流操作
    std::string line;
    while (getline(fin, line)) {
        int64_t num=atoll(line.c_str());
        if(num>0)
        {
            dataset.push_back(num);
        }
    }
    fin.close();
    //  std::cout << "Total "<< dataset.size() << " records." <<std::endl;
}
