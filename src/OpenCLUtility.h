#ifndef __DG_OPEN_CL_BENCHMARK_H__
#define __DG_OPEN_CL_BENCHMARK_H__

#include "OpenCLWrapper.h"
#include <chrono>
#include <string>

namespace dg::opencl_benchmark{

    using namespace dg::opencl_wrapper;

    static inline std::string script = "kernel void add(global const ulong * n, \
                                                        global const double *a, \
                                                        global const double *b, \
                                                        global double *c){ \
                                            size_t i = get_global_id(0); \
                                            c[i] = a[i] + b[i];\
                                        }"; 

    void run(std::vector<cl::Device> devices){

        size_t N        = size_t{1} << 25;
    	auto a          = std::vector<double>(N);
    	auto b          = std::vector<double>(N);
	    auto c          = std::vector<double>(N);
        auto program 	= dg::opencl_wrapper::make_program(devices, script.c_str(), script.size());
        auto rs 		= dg::opencl_wrapper::run(program, "add", N, std::make_pair(&N, size_t{1}), std::make_pair(a.data(), a.size()), std::make_pair(b.data(), b.size()), std::make_pair(c.data(), c.size()));
    	
        rs->sync(3);
    }

    auto benchmark(std::vector<cl::Device> devices) -> std::chrono::microseconds{

        auto s = std::chrono::high_resolution_clock::now();
        run(devices);
        auto l = std::chrono::duration_cast<std::chrono::microseconds>(s - std::chrono::high_resolution_clock::now());

        return l;
    }

    auto get_most_flops_device_group() -> std::vector<cl::Device>{

        auto platform_sz = platform_size();
        auto rs = std::vector<cl::Device>{}; 
        auto cur_lapsed = std::chrono::microseconds::max();

        for (size_t i = 0; i < platform_sz; ++i){
            auto device_group = get_double_precision_devices(i);
            if (device_group.empty()){
                continue;
            }
            auto lapsed = benchmark(device_group);
            if (cur_lapsed > lapsed){
                rs = device_group;
                cur_lapsed = lapsed;
            }
        }

        return rs;
    } 

}

#endif