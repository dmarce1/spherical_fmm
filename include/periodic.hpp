
template<class T, typename std::enable_if<!is_vec3<T>::value>* = nullptr>
inline T distance(const T& a, const T& b) {
	const T c = a - b;
	const T absc = abs(c);
	return copysign(min(absc, T(1) - absc), c * (T(0.5) - absc));
}

template<class T, typename std::enable_if<is_vec3<vec3<T>>::value>* = nullptr>
inline vec3<T> distance(const vec3<T>& a, const vec3<T>& b) {
	vec3<T> d;
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		d[dim] = distance(a[dim], b[dim]);
	}
	return d;
}

