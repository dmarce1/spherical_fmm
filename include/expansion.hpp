inline double reduce_sum(double A) {
	return A;
}

inline float reduce_sum(float A) {
	return A;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
expansion<typename type_traits<T>::type, P> reduce_sum(const expansion<T, P>& A) {
	constexpr int end = expansion<T, P>::size();
	expansion<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
expansion_periodic<typename type_traits<T>::type, P> reduce_sum(const expansion_periodic<T, P>& A) {
	constexpr int end = expansion_periodic<T, P>::size();
	expansion_periodic<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	B.trace2() = reduce_sum(A.trace2());
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
expansion_scaled<typename type_traits<T>::type, P> reduce_sum(const expansion_scaled<T, P>& A) {
	constexpr int end = expansion_scaled<T, P>::size();
	expansion_scaled<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
expansion_periodic_scaled<typename type_traits<T>::type, P> reduce_sum(const expansion_periodic_scaled<T, P>& A) {
	constexpr int end = expansion_periodic_scaled<T, P>::size();
	expansion_periodic_scaled<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	B.trace2() = reduce_sum(A.trace2());
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole<typename type_traits<T>::type, P> reduce_sum(const multipole<T, P>& A) {
	constexpr int end = multipole<T, P>::size();
	multipole<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole_periodic<typename type_traits<T>::type, P> reduce_sum(const multipole_periodic<T, P>& A) {
	constexpr int end = multipole_periodic<T, P>::size();
	multipole_periodic<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	B.trace2() = reduce_sum(A.trace2());
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole_scaled<typename type_traits<T>::type, P> reduce_sum(const multipole_scaled<T, P>& A) {
	constexpr int end = multipole_scaled<T, P>::size();
	multipole_scaled<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole_periodic_scaled<typename type_traits<T>::type, P> reduce_sum(const multipole_periodic_scaled<T, P>& A) {
	constexpr int end = multipole_periodic_scaled<T, P>::size();
	multipole_periodic_scaled<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	B.trace2() = reduce_sum(A.trace2());
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole_wo_dipole<typename type_traits<T>::type, P> reduce_sum(const multipole_wo_dipole<T, P>& A) {
	constexpr int end = multipole_wo_dipole<T, P>::size();
	multipole_wo_dipole<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole_periodic_wo_dipole<typename type_traits<T>::type, P> reduce_sum(const multipole_periodic_wo_dipole<T, P>& A) {
	constexpr int end = multipole_periodic_wo_dipole<T, P>::size();
	multipole_periodic_wo_dipole<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	B.trace2() = reduce_sum(A.trace2());
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole_scaled_wo_dipole<typename type_traits<T>::type, P> reduce_sum(const multipole_scaled_wo_dipole<T, P>& A) {
	constexpr int end = multipole_scaled_wo_dipole<T, P>::size();
	multipole_scaled_wo_dipole<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	return B;
}

template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>
multipole_periodic_scaled_wo_dipole<typename type_traits<T>::type, P> reduce_sum(const multipole_periodic_scaled_wo_dipole<T, P>& A) {
	constexpr int end = multipole_periodic_scaled_wo_dipole<T, P>::size();
	multipole_periodic_scaled_wo_dipole<typename type_traits<T>::type, P> B;
	for (int i = 0; i < end; i++) {
		B[i] = reduce_sum(A[i]);
	}
	B.trace2() = reduce_sum(A.trace2());
	return B;
}

template<class T, int P>
struct is_compound_type<expansion<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<expansion_periodic<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<expansion_scaled<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<expansion_periodic_scaled<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole_periodic<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole_scaled<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole_periodic_scaled<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole_wo_dipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole_periodic_wo_dipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole_scaled_wo_dipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct is_compound_type<multipole_periodic_scaled_wo_dipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct has_trace2<expansion_periodic<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct has_trace2<expansion_periodic_scaled<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct has_trace2<multipole_periodic<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct has_trace2<multipole_periodic_scaled<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct has_trace2<multipole_periodic_wo_dipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, int P>
struct has_trace2<multipole_periodic_scaled_wo_dipole<T, P>> {
	static constexpr bool value = true;
};

template<class T, class V, typename std::enable_if<!is_compound_type<V>::value && !type_traits<V>::is_simd>::type* = nullptr>
void load(V& dest, const T& src, int index = -1) {
	dest = src;
}

template<class T, class V, typename std::enable_if<!is_compound_type<V>::value && type_traits<V>::is_simd>::type* = nullptr>
void load(V& dest, const T& src, int index = -1) {
	if (index == -1) {
		dest = src;
	} else {
		dest[index] = src;
	}
}

template<class T, class V, typename std::enable_if<is_compound_type<V>::value>::type* = nullptr>
void load(V& dest, const T& src, int index = -1) {
	dest.load(src, index);
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
void accumulate(force_type<typename type_traits<T>::type>& dest, const force_type<T>& src, int index) {
	dest.potential += src.potential[index];
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] += src.force[dim][index];
	}
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
void accumulate(force_type<T>& dest, const force_type<T>& src, int index) {
	dest.potential += src.potential;
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] += src.force[dim];
	}
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
void store(force_type<typename type_traits<T>::type>& dest, const force_type<T>& src, int index) {
	dest.potential += src.potential[index];
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] += src.force[dim][index];
	}
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
void store(force_type<T>& dest, const force_type<T>& src, int index) {
	dest.potential = src.potential;
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		dest.force[dim] = src.force[dim];
	}
}

template<class V, typename std::enable_if<!is_compound_type<V>::value && type_traits<V>::is_simd>::type* = nullptr>
void apply_padding(V& A, int n) {
	A.pad(n);
}

template<class V, typename std::enable_if<!is_compound_type<V>::value && !type_traits<V>::is_simd>::type* = nullptr>
void apply_padding(V& A, int n) {
}

template<class V, typename std::enable_if<is_compound_type<V>::value && !has_trace2<V>::value>::type* = nullptr>
void apply_padding(V& A, int n) {
	for (int i = 0; i < V::size(); i++) {
		apply_padding(A[i], n);
	}
}

template<class V, typename std::enable_if<is_compound_type<V>::value && has_trace2<V>::value>::type* = nullptr>
void apply_padding(V& A, int n) {
	for (int i = 0; i < V::size(); i++) {
		apply_padding(A[i], n);
	}
	apply_padding(A.trace2(), n);
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
T create_mask(int end) {
	return T(1);
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
T create_mask(int end) {
	return T::mask(end);
}

template<class V, typename std::enable_if<!is_compound_type<V>::value>::type* = nullptr>
void apply_mask(V& A, int n) {
}

template<class V, typename std::enable_if<is_compound_type<V>::value && !has_trace2<V>::value>::type* = nullptr>
void apply_mask(V& A, int n) {
	const auto mask = create_mask<typename V::type>(n);
	for (int i = 0; i < V::size(); i++) {
		A[i] *= mask;
	}
}

template<class V, typename std::enable_if<is_compound_type<V>::value && has_trace2<V>::value>::type* = nullptr>
void apply_mask(V& A, int n) {
	const auto mask = create_mask<typename V::type>(n);
	for (int i = 0; i < V::size(); i++) {
		A[i] *= mask;
	}
	A.trace2() *= mask;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
force_type<typename type_traits<T>::type> reduce_sum(const force_type<T>& f) {
	force_type<typename type_traits<T>::type> result;
	result.potential = reduce_sum(f.potential);
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		result.force[dim] = reduce_sum(f.force[dim]);
	}
	return result;
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
force_type<T> reduce_sum(const force_type<T>& f) {
	return f;
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
T reduce_max(const T& A) {
	return A;
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
T reduce_min(const T& A) {
	return A;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
typename type_traits<T>::type reduce_max(const T& A) {
	typename type_traits<T>::type result = A[0];
	for (int i = 1; i < T::size(); i++) {
		result = std::max(result, A[i]);
	}
	return result;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
typename type_traits<T>::type reduce_min(const T& A) {
	typename type_traits<T>::type result = A[0];
	for (int i = 1; i < T::size(); i++) {
		result = std::min(result, A[i]);
	}
	return result;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
constexpr size_t simd_size() {
	return T::size();
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
constexpr size_t simd_size() {
	return 1;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
typename type_traits<T>::type& access(T& A, int index) {
	return A[index];
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
T& access(T& A, int index) {
	return A;
}

template<class T, typename std::enable_if<type_traits<T>::is_simd>::type* = nullptr>
typename type_traits<T>::type access(const T& A, int index) {
	return A[index];
}

template<class T, typename std::enable_if<!type_traits<T>::is_simd>::type* = nullptr>
T& access(const T& A, int index) {
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
