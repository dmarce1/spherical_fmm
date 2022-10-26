#pragma once

#include "sfmm.hpp"

namespace fmm {
namespace detail {
template<class T, int P>
__device__ void regular_harmonic(expansion_type<T, P>* Yptr, T* xptr, T* yptr, T* zptr) {
	const int& tid = threadIdx.x;
	const int& bsz = blockDim.x;
	const T& x = *xptr;
	const T& y = *yptr;
	const T& z = *zptr;
	const T r2 = x * x + y * y + z * z;
	T* Y = Yptr->data();
	if (tid == 0) {
		Y[0] = T(1);
		Yptr->trace2() = r2;
		for (int m = 1; m <= P; m++) {
			const int mm1 = m - 1;
			const int mm12 = mm1 * mm1;
			const int m2 = m * m;
			const T ax = Y[mm12 + mm1];
			const T ay = Y[mm12 - mm1];
			const T inv = T(1) / T(2 * m);
			const T invx = inv * x;
			const T invy = inv * y;
			Y[m2 + m] = (ax * invx - ay * invy);
			Y[m2 - m] = (ax * invy + ay * invx);
		}
	}
	__syncthreads();
	for (int m = tid; m < P; m += bsz) {
		const int mp1 = m + 1;
		const int m2 = m * m;
		Y[mp1 + m] = z * Y[m2 + m];
		Y[mp1 - m] = z * Y[m2 - m];
		for (int n = m + 2; n <= P; n++) {
			const T n2 = n * n;
			const T nm1 = n - 1;
			const T nm2 = n - 2;
			const T nm12 = nm1 * nm1;
			const T nm22 = nm2 * nm2;
			const T inv = T(1) / (T(n2) - T(m * m));
			const T tmp1 = inv * T(2 * n - 1) * z;
			const T tmp2 = inv * r2;
			Y[n2 + m] = tmp1 * Y[nm12 + m] - tmp2 * Y[nm22 + m];
			Y[n2 - m] = tmp1 * Y[nm12 - m] - tmp2 * Y[nm22 - m];
		}
	}
	__syncthreads();
}
}

template<class T, int P>
__device__ void greens(expansion_type<T, P>* Yptr, T* xptr, T* yptr, T* zptr) {
	const int& tid = threadIdx.x;
	const int& bsz = blockDim.x;
	T x = *xptr;
	T y = *yptr;
	T z = *zptr;
	const T r2 = x * x + y * y + z * z;
	const T r2inv = T(1) / r2;
	x *= r2inv;
	y *= r2inv;
	z *= r2inv;
	T* Y = Yptr->data();
	if (tid == 0) {
		Y[0] = T(1);
		Yptr->trace2() = T(0);
		for (int m = 1; m <= P; m++) {
			const int mm1 = m - 1;
			const int mm12 = mm1 * mm1;
			const int m2 = m * m;
			const T& ax = Y[mm12 + mm1];
			const T& ay = Y[mm12 - mm1];
			const T c0 = T(2 * m - 1);
			const T c0x = c0 * x;
			const T c0y = c0 * y;
			Y[m2 + m] = (ax * c0x - ay * c0y);
			Y[m2 - m] = (ax * c0y + ay * c0x);
		}
	}
	__syncthreads();
	for (int m = tid; m < P; m += bsz) {
		const int mp1 = m + 1;
		const int m2 = m * m;
		const T c0 = T(2 * m + 1) * z;
		Y[mp1 + m] = c0 * Y[m2 + m];
		Y[mp1 - m] = c0 * Y[m2 - m];
		for (int n = m + 2; n <= P; n++) {
			const T n2 = n * n;
			const T nm1 = n - 1;
			const T nm2 = n - 2;
			const T nm12 = nm1 * nm1;
			const T nm22 = nm2 * nm2;
			const T tmp1 = T(2 * n - 1) * z;
			const T tmp2 = T(nm12 - m * m) * r2;
			Y[n2 + m] = tmp1 * Y[nm12 + m] - tmp2 * Y[nm22 + m];
			Y[n2 - m] = tmp1 * Y[nm12 - m] - tmp2 * Y[nm22 - m];
		}
	}
	__syncthreads();
}

namespace detail {
template<int P>
struct m2l_indices {
	int L[(P + 2) * (P + 1) / 2];
	int M[(P + 2) * (P + 1) / 2];
	__device__ constexpr m2l_indices() :
			L(), M() {
		for (int l = 0; l <= P; l++) {
			for (int m = 0; m <= l; m++) {
				const int i = l * (l + 1) / 2 + m;
				L[i] = l;
				M[i] = m;
			}
		}
	}
	__device__ int l_index(int i) const {
		return L[i];
	}
	__device__ int m_index(int i) const {
		return M[i];
	}
};
}

template<class T, int P>
__device__ void M2L(expansion_type<T, P>* Lptr, multipole_type<T, P>* Mptr, T* xptr, T* yptr, T * zptr) {
	const int& tid = threadIdx.x;
	const int& bsz = blockDim.x;
	detail::m2l_indices<P> indices;
	const T& x = *xptr;
	const T& y = *yptr;
	const T& z = *zptr;
	__shared__ expansion_type<T, P> O;
	greens(&O, xptr, yptr, zptr);
	T* L = Lptr->data();
	T* M = Mptr->data();
	for (int i = tid; i < (P + 2) * (P + 1) / 2; i += bsz) {
		const int n = indices.l_index(i);
		const int m = indices.m_index(i);
		T& lx = L[n * n + m];
		T& ly = L[n * n - m];
		lx = T(0);
		ly = T(0);
		const int kmax = min(P - n, P - 1);
		for (int k = 0; k <= kmax; k++) {
			for (int l = -k; l <= k; l++) {
				const int k2 = k * k;
				const int nk = n + k;
				const int nk2 = nk * nk;
				const int ml = m + l;
				const T& mx = M[k2 + l];
				const T my = T(l != 0) * (-M[k2 - l]);
				const T& ox = O[nk2 + ml];
				const T oy = T(ml != 0) * O[nk2 - ml];
				lx += mx * ox - my * oy;
				if (m != 0) {
					ly += mx * oy + my * ox;
				}
			}
		}
	}
	__syncthreads();
}

}
