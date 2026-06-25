
#include <stdatomic.h>


void sfmm_detail_atomic_inc_dbl(double* num, double val) {
	double old, next;
	do {
		old = *num;
		next = old + val;
	} while (!atomic_compare_exchange_weak(num, &old, next));
}

void sfmm_detail_atomic_inc_int(unsigned long long* num, int val) {
	unsigned long long old, next;
	do {
		old = *num;
		next = old + val;
	} while (!atomic_compare_exchange_weak(num, &old, next));
}

