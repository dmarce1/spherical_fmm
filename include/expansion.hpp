inline double reduce_sum(double A) {
	return A;
}

inline float reduce_sum(float A) {
	return A;
}

template<class T, int P>
struct is_compound_type<expansion<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, class V, typename std::enable_if<!is_compound_type<V>::value && !type_traits<V>::is_simd>::type* = nullptr>
inline void load(V& dest, const T& src, int index = -1) {
	dest = src;
}

template<class T, class V, typename std::enable_if<!is_compound_type<V>::value && type_traits<V>::is_simd>::type* = nullptr>
inline void load(V& dest, const T& src, int index = -1) {
	if (index == -1) {
		dest = src;
	} else {
		dest[index] = src;
	}
}

template<class T, class V, typename std::enable_if<is_compound_type<V>::value>::type* = nullptr>
inline void load(V& dest, const T& src, int index = -1) {
	dest.load(src, index);
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline void accumulate(force_type<typename type_traits<T>::type>& dest, const force_type<T>& src, int index) {
	dest.potential += src.potential[index];
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] += src.force[dim][index];
	}
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline void accumulate(force_type<T>& dest, const force_type<T>& src, int index) {
	dest.potential += src.potential;
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] += src.force[dim];
	}
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline void store(force_type<typename type_traits<T>::type>& dest, const force_type<T>& src, int index) {
	dest.potential += src.potential[index];
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] += src.force[dim][index];
	}
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline void store(force_type<T>& dest, const force_type<T>& src, int index) {
	dest.potential = src.potential;
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] = src.force[dim];
	}
}

template<class V, typename std::enable_if<!is_compound_type<V>::value && type_traits<V>::is_simd>::type* = nullptr>
inline void apply_padding(V& A, int n) {
	A.pad(n);
}

template<class V, typename std::enable_if<!is_compound_type<V>::value && !type_traits<V>::is_simd>::type* = nullptr>
inline void apply_padding(V& A, int n) {
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline T create_mask(int end) {
	return T(1);
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline T create_mask(int end) {
	return T::mask(end);
}

template<class V, typename std::enable_if<!is_compound_type<V>::value>::type* = nullptr>
inline void apply_mask(V& A, int n) {
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline force_type<typename type_traits<T>::type> reduce_sum(const force_type<T>& f) {
	force_type<typename type_traits<T>::type> result;
	result.potential = reduce_sum(f.potential);
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		result.force[dim] = reduce_sum(f.force[dim]);
	}
	return result;
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline force_type<T> reduce_sum(const force_type<T>& f) {
	return f;
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline T reduce_max(const T& A) {
	return A;
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline T reduce_min(const T& A) {
	return A;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline typename type_traits<T>::type reduce_max(const T& A) {
	typename type_traits<T>::type result = A[0];
	for (int i = 1; i < T::size(); i++) {
		result = std::max(result, A[i]);
	}
	return result;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline typename type_traits<T>::type reduce_min(const T& A) {
	typename type_traits<T>::type result = A[0];
	for (int i = 1; i < T::size(); i++) {
		result = std::min(result, A[i]);
	}
	return result;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline constexpr size_t simd_size() {
	return T::size();
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline constexpr size_t simd_size() {
	return 1;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline typename type_traits<T>::type& access(T& A, int index) {
	return A[index];
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline T& access(T& A, int index) {
	return A;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
inline typename type_traits<T>::type access(const T& A, int index) {
	return A[index];
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
inline T& access(const T& A, int index) {
	return A;
}

template<class T>
inline vec3<T>& vec3<T>::load(const vec3<typename type_traits<T>::type>& other, int index) {
	if (index == -1) {
		for (int dim = 0; dim < SFMM_NDIM; dim++) {
			(*this)[dim] = other[dim];
		}
	} else {
		for (int dim = 0; dim < SFMM_NDIM; dim++) {
			access((*this)[dim], index) = other[dim];
		}
	}
	return *this;
}

namespace detail {

template<class ...Args>
std::string print2str(const char* fstr, Args&&...args) {
	std::string result;
	char* str;
	if (!asprintf(&str, fstr, std::forward<Args>(args)...)) {
		printf("Error in %s on line %i\n", __FILE__, __LINE__);
		abort();
	}
	result = str;
	free(str);
	return result;
}

struct func_data_t {
	void* func_ptr;
	std::string type;
	std::string name;
	int P;
	int periodic;
	int scaled;
	int nodip;
	int nopot;
	int nrot;
	int flops;
	double time;
	unsigned long long ncalls;
};

func_data_t* operator_initialize(void*);
void operator_update_timing(func_data_t*, double);
int operator_count();
func_data_t* operator_data(int index);
void operator_flops_initialize();
int operator_best_rotation(int, int, int, int, int, const char*, const char*);
void operator_write_new_bestops_source();
}

std::string operator_profiling_results();

