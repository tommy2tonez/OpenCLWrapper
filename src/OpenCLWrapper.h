#ifndef __DG_OPEN_CL_WRAPPER_H__
#define __DG_OPEN_CL_WRAPPER_H__

#include "CL/cl.hpp"
#include <memory>
#include <iostream>
#include <functional>
#include <numeric>
#include <utility>
#include <algorithm>
#include "assert.h"
#include <iostream>

namespace dg::opencl_wrapper::program{
    
    struct Program{
        cl::Device supervisor;
        cl::Context working_env;
        cl::Program content;
    };

    class MemorySynchronizable{

        public:
            virtual ~MemorySynchronizable() noexcept{}
            virtual void sync(size_t arg_idx) = 0;
    };

    template <size_t ARG_SIZE>
    class MemorySynchronizer: public virtual MemorySynchronizable{

        private:

            std::array<std::pair<void *, size_t>, ARG_SIZE> host_mem;
            std::array<cl::Buffer, ARG_SIZE> device_mem;
            cl::CommandQueue cmd_queue;

        public: 

            MemorySynchronizer(std::array<std::pair<void *, size_t>, ARG_SIZE> host_mem,
                               std::array<cl::Buffer, ARG_SIZE> device_mem, 
                               cl::CommandQueue cmd_queue): host_mem(std::move(host_mem)),
                                                            device_mem(std::move(device_mem)),
                                                            cmd_queue(std::move(cmd_queue)){}

            void sync(size_t idx){

                auto [hbuf, hsz] = this->host_mem[idx]; 
                auto err = this->cmd_queue.enqueueReadBuffer(this->device_mem[idx], CL_TRUE, 0, hsz, hbuf);

                if (err != CL_SUCCESS){
                    throw std::exception{};
                }
            }
    };
}

namespace dg::opencl_wrapper{

    auto platform_size() -> size_t{
    
        auto platform   = std::vector<cl::Platform>{};
        auto err        = cl::Platform::get(&platform);

        if (err != CL_SUCCESS){
            throw std::exception{};
        }

        return platform.size();
    }

    auto get_double_precision_devices(size_t platform_idx) -> std::vector<cl::Device>{

        auto platform   = std::vector<cl::Platform>{};
        auto devices    = std::vector<cl::Device>{};
        auto filterer   = [](const cl::Device& cur){
            return cur.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp64") != cl::STRING_CLASS::npos || cur.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_amd_fp64") != cl::STRING_CLASS::npos;
        };

        auto err        = cl::Platform::get(&platform);

        if (err != CL_SUCCESS){
            throw std::exception();
        }
        
        err             = platform[platform_idx].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        if (err != CL_SUCCESS){
            throw std::exception();
        }

        auto last = std::copy_if(devices.begin(), devices.end(), devices.begin(), filterer);
        devices.resize(std::distance(devices.begin(), last));
        
        return devices;
    }

    auto promote_supervisor(std::vector<cl::Device>& devices, size_t supervisor_idx){

        assert(supervisor_idx < devices.size());
        std::swap(devices[supervisor_idx], devices.front());
    } 

    auto make_program(const std::vector<cl::Device>& devices, const char * src, size_t src_len) -> program::Program{

        assert(devices.size() != 0);
        auto supervisor = devices.front();
        auto env        = cl::Context(devices);
        auto program    = cl::Program(env, cl::Program::Sources{{src, src_len}});
        auto error      = program.build(devices);

        if (error != CL_SUCCESS){
            throw std::exception{};
        }

        return program::Program{supervisor, std::move(env), std::move(program)};
    }

    template <class ...Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, bool> = true>
    auto run(const program::Program& program, std::string func_name, size_t map_size, std::pair<Args *, size_t>... args) -> std::unique_ptr<program::MemorySynchronizable>{

        auto idx_seq    = std::make_index_sequence<sizeof...(Args)>{};
        auto func       = cl::Kernel(program.content, func_name.c_str());
        auto dev_buffer = std::array<cl::Buffer, sizeof...(Args)>{};
        auto hst_buffer = std::array<std::pair<void *, size_t>, sizeof...(Args)>();
        auto tup        = std::make_tuple(args...); 
        auto cmd_q      = cl::CommandQueue(program.working_env, program.supervisor); 
        auto err        = cl_int{};

        [&]<size_t ...IDX>(const std::index_sequence<IDX...>){
            (
                [&]{
                    (void) IDX;
                    auto [buf, sz]      = std::get<IDX>(tup);
                    sz                  *= sizeof(Args);
                    dev_buffer[IDX]     = cl::Buffer(program.working_env, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz, buf);
                    hst_buffer[IDX]     = {buf, sz};
                    err                 |= func.setArg(IDX, dev_buffer[IDX]);
                }(), ...
            );
        }(idx_seq);

        if (err != CL_SUCCESS){
            throw std::exception();
        }

        err = cmd_q.enqueueNDRangeKernel(func, cl::NullRange, map_size, cl::NullRange);
        
        if (err != CL_SUCCESS){
            throw std::exception{};
        }

        program::MemorySynchronizer op(std::move(hst_buffer), std::move(dev_buffer), std::move(cmd_q));
        return std::make_unique<decltype(op)>(std::move(op));
    }
}

#endif