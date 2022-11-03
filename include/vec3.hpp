#define SFMM_NDIM 3

#define SFMM_VEC3_OP1( op ) \
		SFMM_PREFIX inline vec3 operator op (const vec3& other) const { \
			vec3 result; \
			for( int dim = 0; dim < SFMM_NDIM; dim++ ) { \
				result[dim] = (*this)[dim] op other[dim]; \
			} \
			return result; \
		} \
		SFMM_PREFIX inline vec3& operator op##= (const vec3& other) { \
			for( int dim = 0; dim < SFMM_NDIM; dim++ ) { \
				(*this)[dim] op##= other[dim]; \
			} \
			return *this; \
		}

#define SFMM_VEC3_OP2( op ) \
		SFMM_PREFIX inline vec3 operator op (const T other) const { \
			vec3 result; \
			for( int dim = 0; dim < SFMM_NDIM; dim++ ) { \
				result[dim] = (*this)[dim] op other; \
			} \
			return result; \
		} \
		SFMM_PREFIX inline vec3& operator op##= (const T other) { \
			for( int dim = 0; dim < SFMM_NDIM; dim++ ) { \
				(*this)[dim] op##= other; \
			} \
			return *this; \
		}

template<class T>
struct vec3: public std::array<T, SFMM_NDIM> {
	SFMM_VEC3_OP1( + )SFMM_VEC3_OP1( - )SFMM_VEC3_OP1( * )SFMM_VEC3_OP1( / )SFMM_VEC3_OP2( * )SFMM_VEC3_OP2( / )SFMM_PREFIX
	static size_t constexpr size() {
		return SFMM_NDIM;
	}
	vec3 inline operator-() const {
		vec3 result;
		for (int dim = 0; dim < SFMM_NDIM; dim++) {
			result[dim] = -(*this)[dim];
		}
		return result;
	}
	SFMM_PREFIX
	inline vec3& operator=(T a) {
		for (int dim = 0; dim < SFMM_NDIM; dim++) {
			(*this)[dim] = a;
		}
		return *this;
	}
	SFMM_PREFIX inline vec3<T>(T x, T y, T z) {
		(*this)[0] = x;
		(*this)[1] = y;
		(*this)[2] = z;
	}
	inline vec3<T>& load(const vec3<typename type_traits<T>::type>& other, int index = -1) {
		if (index == -1) {
			for (int dim = 0; dim < SFMM_NDIM; dim++) {
				(*this)[dim] = other[dim];
			}
		} else {
			for (int dim = 0; dim < SFMM_NDIM; dim++) {
				(*this)[dim][index] = other[dim];
			}
		}
		return *this;
	}
	vec3() = default;

	vec3(const vec3&) = default;

	vec3& operator=(const vec3&) = default;
};

template<class T>
SFMM_PREFIX T sqr(const T& a) {
	return a * a;
}

template<class T>
SFMM_PREFIX inline T abs(vec3<T> vec) {
	return sqrt(sqr(vec[0])+sqr(vec[1])+sqr(vec[2]));
}

template<class T>
struct is_compound_type {
	static constexpr bool value = false;
};

template<class T>
struct is_compound_type<vec3<T>> {
	static constexpr bool value = true;
};

template<class T>
struct has_trace2 {
	static constexpr bool value = false;
};

template<class T>
struct force_type {
	T potential;
	vec3<T> force;SFMM_PREFIX
	inline void init() {
		potential = force[0] = force[1] = force[2] = T(0);
	}
};

