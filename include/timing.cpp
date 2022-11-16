#include "sfmm.hpp"
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace sfmm {
namespace detail {

func_data_t* operator_initialize(void* ptr) {
	for (int i = 0; i < operator_count(); i++) {
		auto* entry = operator_data(i);
		if (entry->func_ptr == ptr) {
			return entry;
		}
	}
	printf( "Error file %s line %i\n", __FILE__, __LINE__);
	return nullptr;
}

void operator_update_timing(func_data_t* ptr, double time) {
	auto& entry = *ptr;
	sfmm_detail_atomic_inc_int(&(entry.ncalls), 1);
	sfmm_detail_atomic_inc_dbl(&(entry.time), time);
}

}

}
