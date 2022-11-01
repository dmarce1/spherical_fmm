namespace detail {

template<class T>
struct vsize_helper {
	static constexpr size_t size = 0;
};

}

template<class T>
size_t vsize() {
	detail::vsize_helper<T> help;
	return help.size;
}

template<class T, class V>
vec3<T> vaccess(const vec3<V>& v, int i) {
	vec3<T> result;
	for (int dim = 0; dim < SFMM_NDIM; dim++) {
		result[dim] = v[dim][i];
	}
	return result;
}

namespace detail {
template<class T, class V>
struct vec3_vaccess_helper {
	vec3<V>& ref;
	int index;
	vec3_vaccess_helper(vec3<V>& ref_, int i) :
			ref(ref_), index(i) {
	}
	operator vec3<T>() const {
		const T& cref(ref);
		return vaccess(cref, index);
	}
	vec3_vaccess_helper operator=(const vec3<T>& v) const {
		for (int dim = 0; dim < SFMM_NDIM; dim++) {
			ref[dim][index] = v[dim];
		}
	}
};
}

template<class T, class V>
detail::vec3_vaccess_helper<T, V> vaccess(vec3<V>& v, int i) {
	return detail::vec3_vaccess_helper < T, V > (v, i);
}

