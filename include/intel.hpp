
class simd_i32;
class simd_f32;

class simd_i32 {
	__m256i v;
public:
	simd_i32() = default;
	simd_i32(const simd_i32&) = default;
	simd_i32(simd_i32&&) = default;
	simd_i32& operator=(const simd_i32&) = default;
	simd_i32& operator=(simd_i32&&) = default;
	simd_i32(const simd_f32& other);
	int operator[](int i) const {
		return v[i];
	}
	int& operator[](int i) {
		return *((int*) (&v[0]));
	}
	inline simd_i32(int a) {
		v = _mm256_set1_epi32(a);
	}
	inline simd_i32& operator+=(const simd_i32& other) {
		v = _mm256_add_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator-=(const simd_i32& other) {
		v = _mm256_sub_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator*=(const simd_i32& other) {
		v = _mm256_mul_epi32(v, other.v);
		return *this;
	}
	inline simd_i32& operator&=(const simd_i32& other) {
		v = _mm256_and_si256(v, other.v);
		return *this;
	}
	inline simd_i32& operator^=(const simd_i32& other) {
		v = _mm256_xor_si256(v, other.v);
		return *this;
	}
	inline simd_i32& operator|=(const simd_i32& other) {
		v = _mm256_or_si256(v, other.v);
		return *this;
	}
	inline simd_i32 operator+(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_add_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator-(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_sub_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator!() const {
		simd_i32 result;
		result.v = _mm256_andnot_si256(v, simd_i32(0xFFFFFFFF).v);
		return result;
	}
	inline simd_i32 operator*(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_mul_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator&(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_and_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator^(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_xor_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator|(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_or_si256(v, other.v);
		return result;
	}
	inline simd_i32 operator>>(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_srlv_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator<<(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_sllv_epi32(v, other.v);
		return result;
	}
	inline simd_i32 operator-() const {
		return simd_i32(0) - *this;
	}
	inline simd_i32 operator+() const {
		return *this;
	}
	inline simd_i32 operator==(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_cmpeq_epi32(v, other.v);
		return -result;
	}
	inline simd_i32 operator!=(const simd_i32& other) const {
		return simd_i32(1) - (*this == other);
	}
	inline simd_i32 operator>(const simd_i32& other) const {
		simd_i32 result;
		result.v = _mm256_cmpgt_epi32(v, other.v);
		return -result;
	}
	inline simd_i32 operator>=(const simd_i32& other) const {
		return ((*this == other) + (*this > other)) > simd_i32(0);
	}
	inline simd_i32 operator<(const simd_i32& other) const {
		return simd_i32(1) - (*this >= other);
	}
	inline simd_i32 operator<=(const simd_i32& other) const {
		return simd_i32(1) - (*this > other);
	}
	inline static constexpr int size() {
		return 8;
	}
	inline simd_i32& pad(int n) {
		const int& e = size();
		for (int i = n; i < e; i++) {
			v[i] = v[0];
		}
		return *this;
	}
	static inline simd_i32 mask(int n) {
		simd_i32 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0;
		}
		return mk;
	}
	inline void set_NaN() {
		for (int i = 0; i < size(); i++) {
			v[i] = std::numeric_limits<int>::signaling_NaN();
		}
	}
	friend simd_f32;
};

class simd_f32 {
	__m256 v;
public:
	simd_f32() = default;
	simd_f32(const simd_f32&) = default;
	simd_f32(simd_f32&&) = default;
	simd_f32& operator=(const simd_f32&) = default;
	simd_f32& operator=(simd_f32&&) = default;
	float operator[](int i) const {
		return v[i];
	}
	float& operator[](int i) {
		return v[i];
	}
	simd_f32(float a) {
		v = _mm256_broadcast_ss(&a);
	}
	simd_f32(const simd_i32& other) {
		v = _mm256_cvtepi32_ps(other.v);
	}
	inline simd_f32& operator+=(const simd_f32& other) {
		v = _mm256_add_ps(v, other.v);
		return *this;
	}
	inline simd_f32& operator-=(const simd_f32& other) {
		v = _mm256_sub_ps(v, other.v);
		return *this;
	}
	inline simd_f32& operator*=(const simd_f32& other) {
		v = _mm256_mul_ps(v, other.v);
		return *this;
	}
	inline simd_f32& operator/=(const simd_f32& other) {
		v = _mm256_div_ps(v, other.v);
		return *this;
	}
	inline simd_f32 operator+(const simd_f32& other) const {
		simd_f32 result;
		result.v = _mm256_add_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator-(const simd_f32& other) const {
		simd_f32 result;
		result.v = _mm256_sub_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator*(const simd_f32& other) const {
		simd_f32 result;
		result.v = _mm256_mul_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator/(const simd_f32& other) const {
		simd_f32 result;
		result.v = _mm256_div_ps(v, other.v);
		return result;
	}
	inline simd_f32 operator-() const {
		return simd_f32(0) - *this;
	}
	inline simd_f32 operator+() const {
		return *this;
	}
	inline simd_i32 operator==(const simd_f32& other) const {
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_EQ_OS));
		return -result;
	}
	inline simd_i32 operator!=(const simd_f32& other) const {
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_NEQ_OS));
		return -result;
	}
	inline simd_i32 operator>(const simd_f32& other) const {
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_GT_OS));
		return -result;
	}
	inline simd_i32 operator>=(const simd_f32& other) const {
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_GE_OS));
		return -result;
	}
	inline simd_i32 operator<(const simd_f32& other) const {
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_LT_OS));
		return -result;
	}
	inline simd_i32 operator<=(const simd_f32& other) const {
		simd_i32 result;
		result.v = _mm256_castps_si256(_mm256_cmp_ps(v, other.v, _CMP_LE_OS));
		return -result;
	}
	inline static constexpr int size() {
		return 8;
	}
	inline simd_f32& pad(int n) {
		const int& e = size();
		for (int i = n; i < e; i++) {
			v[i] = v[0];
		}
		return *this;
	}
	static inline simd_f32 mask(int n) {
		simd_f32 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1.f;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0.f;
		}
		return mk;
	}
	inline void set_NaN() {
		for (int i = 0; i < size(); i++) {
			v[i] = std::numeric_limits<float>::signaling_NaN();
		}
	}
	friend simd_f32 sqrt(simd_f32);
	friend simd_f32 rsqrt(simd_f32);
	friend simd_f32 fma(simd_f32, simd_f32, simd_f32);
	friend float reduce_sum(simd_f32);
	friend class simd_i32;
};

inline float reduce_sum(simd_f32 x) {
	__m128 a = *((__m128*) &x.v);
	__m128 b = *(((__m128*) &x.v) + 1);
	a = _mm_add_ps(a,b);
	return a[0] + a[1] + a[2] + a[3];
}

inline simd_f32 fma(simd_f32 a, simd_f32 b, simd_f32 c) {
	simd_f32 result;
	result.v = _mm256_fmadd_ps(a.v, b.v, c.v);
	return result;
}

inline simd_i32::simd_i32(const simd_f32& other) {
	v = _mm256_cvtps_epi32(other.v);
}

inline simd_f32 sqrt(simd_f32 x) {
	x.v = _mm256_sqrt_ps(x.v);
	return x;
}

inline simd_f32 rsqrt(simd_f32 x) {
	x.v = _mm256_rsqrt_ps(x.v);
	return x;
}

using simd_ui32 = simd_i32;




class simd_i64;
class simd_f64;

class simd_i64 {
	__m256i v;
public:
	simd_i64() = default;
	simd_i64(const simd_i64&) = default;
	simd_i64(simd_i64&&) = default;
	simd_i64& operator=(const simd_i64&) = default;
	simd_i64& operator=(simd_i64&&) = default;
	simd_i64(const simd_f64& other);
	long long operator[](int i) const {
		return v[i];
	}
	long long& operator[](int i) {
		return *((long long*) (&v[0]));
	}
	inline simd_i64(long long a) {
		v = _mm256_set_epi64x(a, a, a, a);
	}
	inline simd_i64& operator+=(const simd_i64& other) {
		v = _mm256_add_epi64(v, other.v);
		return *this;
	}
	inline simd_i64& operator-=(const simd_i64& other) {
		v = _mm256_sub_epi64(v, other.v);
		return *this;
	}
	inline simd_i64& operator&=(const simd_i64& other) {
		v = _mm256_and_si256(v, other.v);
		return *this;
	}
	inline simd_i64& operator^=(const simd_i64& other) {
		v = _mm256_xor_si256(v, other.v);
		return *this;
	}
	inline simd_i64& operator|=(const simd_i64& other) {
		v = _mm256_or_si256(v, other.v);
		return *this;
	}
	inline simd_i64 operator+(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_add_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator-(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_sub_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator!() const {
		simd_i64 result;
		result.v = _mm256_andnot_si256(v, simd_i64(0xFFFFFFFFFFFFFFFFLL).v);
		return result;
	}
	inline simd_i64 operator&(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_and_si256(v, other.v);
		return result;
	}
	inline simd_i64 operator^(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_xor_si256(v, other.v);
		return result;
	}
	inline simd_i64 operator|(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_or_si256(v, other.v);
		return result;
	}
	inline simd_i64 operator>>(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_srlv_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator<<(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_sllv_epi64(v, other.v);
		return result;
	}
	inline simd_i64 operator-() const {
		return simd_i64(0) - *this;
	}
	inline simd_i64 operator+() const {
		return *this;
	}
	inline simd_i64 operator==(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_cmpeq_epi64(v, other.v);
		return -result;
	}
	inline simd_i64 operator!=(const simd_i64& other) const {
		return simd_i64(1) - (*this == other);
	}
	inline simd_i64 operator>(const simd_i64& other) const {
		simd_i64 result;
		result.v = _mm256_cmpgt_epi64(v, other.v);
		return -result;
	}
	inline simd_i64 operator>=(const simd_i64& other) const {
		return ((*this == other) + (*this > other)) > simd_i64(0);
	}
	inline simd_i64 operator<(const simd_i64& other) const {
		return simd_i64(1) - (*this >= other);
	}
	inline simd_i64 operator<=(const simd_i64& other) const {
		return simd_i64(1) - (*this > other);
	}
	inline static constexpr int size() {
		return 4;
	}
	inline simd_i64& pad(int n) {
		const int& e = size();
		for (int i = n; i < e; i++) {
			v[i] = v[0];
		}
		return *this;
	}
	static inline simd_i64 mask(int n) {
		simd_i64 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0;
		}
		return mk;
	}
	inline void set_NaN() {
		for (int i = 0; i < size(); i++) {
			v[i] = std::numeric_limits<int>::signaling_NaN();
		}
	}
	friend simd_f64;
};

class simd_f64 {
	__m256d v;
public:
	simd_f64() = default;
	simd_f64(const simd_f64&) = default;
	simd_f64(simd_f64&&) = default;
	simd_f64& operator=(const simd_f64&) = default;
	simd_f64& operator=(simd_f64&&) = default;
	double operator[](int i) const {
		return v[i];
	}
	double& operator[](int i) {
		return v[i];
	}
	simd_f64(double a) {
		v = _mm256_broadcast_sd(&a);
	}
	inline simd_f64(const simd_i64& other) {
		v[0] = (double) other[0];
		v[1] = (double) other[1];
		v[2] = (double) other[2];
		v[3] = (double) other[3];
	}
	inline simd_f64& operator+=(const simd_f64& other) {
		v = _mm256_add_pd(v, other.v);
		return *this;
	}
	inline simd_f64& operator-=(const simd_f64& other) {
		v = _mm256_sub_pd(v, other.v);
		return *this;
	}
	inline simd_f64& operator*=(const simd_f64& other) {
		v = _mm256_mul_pd(v, other.v);
		return *this;
	}
	inline simd_f64& operator/=(const simd_f64& other) {
		v = _mm256_div_pd(v, other.v);
		return *this;
	}
	inline simd_f64 operator+(const simd_f64& other) const {
		simd_f64 result;
		result.v = _mm256_add_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator-(const simd_f64& other) const {
		simd_f64 result;
		result.v = _mm256_sub_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator*(const simd_f64& other) const {
		simd_f64 result;
		result.v = _mm256_mul_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator/(const simd_f64& other) const {
		simd_f64 result;
		result.v = _mm256_div_pd(v, other.v);
		return result;
	}
	inline simd_f64 operator-() const {
		return simd_f64(0) - *this;
	}
	inline simd_f64 operator+() const {
		return *this;
	}
	inline simd_i64 operator==(const simd_f64& other) const {
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_EQ_OS));
		return -result;
	}
	inline simd_i64 operator!=(const simd_f64& other) const {
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_NEQ_OS));
		return -result;
	}
	inline simd_i64 operator>(const simd_f64& other) const {
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_GT_OS));
		return -result;
	}
	inline simd_i64 operator>=(const simd_f64& other) const {
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_GE_OS));
		return -result;
	}
	inline simd_i64 operator<(const simd_f64& other) const {
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_LT_OS));
		return -result;
	}
	inline simd_i64 operator<=(const simd_f64& other) const {
		simd_i64 result;
		result.v = _mm256_castpd_si256(_mm256_cmp_pd(v, other.v, _CMP_LE_OS));
		return -result;
	}
	inline static constexpr int size() {
		return 4;
	}
	inline simd_f64& pad(int n) {
		const int& e = size();
		for (int i = n; i < e; i++) {
			v[i] = v[0];
		}
		return *this;
	}
	static inline simd_f64 mask(int n) {
		simd_f64 mk;
		for (int i = 0; i < n; i++) {
			mk[i] = 1.f;
		}
		for (int i = n; i < size(); i++) {
			mk[i] = 0.f;
		}
		return mk;
	}
	inline void set_NaN() {
		for (int i = 0; i < size(); i++) {
			v[i] = std::numeric_limits<double>::signaling_NaN();
		}
	}
	friend simd_f64 sqrt(simd_f64);
	friend simd_f64 rsqrt(simd_f64);
	friend simd_f64 fma(simd_f64, simd_f64, simd_f64);
	friend double reduce_sum(simd_f64);
	friend class simd_i64;
};

inline double reduce_sum(simd_f64 x) {
	__m128d a = *((__m128d*) &x.v);
	__m128d b = *(((__m128d*) &x.v) + 1);
	a = _mm_add_pd(a,b);
	return a[0] + a[1];
}

inline simd_f64 fma(simd_f64 a, simd_f64 b, simd_f64 c) {
	simd_f64 result;
	result.v = _mm256_fmadd_pd(a.v, b.v, c.v);
	return result;
}

inline simd_i64::simd_i64(const simd_f64& other) {
	v[0] = (long long)(other.v[0]);
	v[1] = (long long)(other.v[1]);
	v[2] = (long long)(other.v[2]);
	v[3] = (long long)(other.v[3]);
}

inline simd_f64 sqrt(simd_f64 x) {
	x.v = _mm256_sqrt_pd(x.v);
	return x;
}

inline simd_f64 rsqrt(simd_f64 x) {
	return simd_f64(1) / sqrt(x);
}

using simd_ui64 = simd_i64;


