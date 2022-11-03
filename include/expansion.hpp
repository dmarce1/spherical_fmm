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
	multipole_scaled<typename type_traits<T>::type, P>  B;
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

