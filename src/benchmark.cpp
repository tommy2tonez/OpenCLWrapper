#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "CL/cl.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "OpenCLWrapper.h"
#include "OpenCLUtility.h"

auto read(std::string path) -> std::pair<std::unique_ptr<char[]>, size_t>{

	std::ifstream inp{path, std::ios::binary};
	inp.seekg(0u, inp.end);
	auto sz  = inp.tellg();
	auto buf = std::make_unique<char[]>(sz);
	inp.seekg(0u, inp.beg);
	inp.read(buf.get(), sz);

	return {std::move(buf), sz};
} 

int main() {
	using namespace std::chrono;
    size_t N = 1 << 27;

	std::vector<unsigned char> a(N);
	std::vector<size_t> b(N);
	std::vector<size_t> c(N);

	for (size_t i = 0; i < N; ++i){
		a[i] = i % 0xFF;
	}

	std::iota(b.begin(), b.end(), 0u);

	auto [buf, sz] = read("test.cl"); 
	auto devices 	= dg::opencl_benchmark::get_most_flops_device_group();
	auto program 	= dg::opencl_wrapper::make_program(devices, buf.get(), sz);
	auto l = high_resolution_clock::now();
	auto rs 		= dg::opencl_wrapper::run(program, "add", N, std::make_pair(a.data(), a.size()), std::make_pair(b.data(), b.size()), std::make_pair(c.data(), c.size()));
	rs->sync(2); 
	auto e = high_resolution_clock::now();

	std::cout << duration_cast<milliseconds>(e - l).count() << std::endl;

	for (size_t i = 0; i < 5; ++i){
		std::cout << c[i] << std::endl;
	}
}