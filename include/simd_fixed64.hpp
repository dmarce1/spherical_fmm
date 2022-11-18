
class simd_fixed64 {
	static const simd_f64 c0d;
	static const simd_f64 c0di;
	simd_ui64 i;
public:
	SFMM_PREFIX constexpr simd_fixed64() :
			i() {
	}
	SFMM_PREFIX simd_fixed64( fixed64 other ) : i(other.i) {
	}
	SFMM_PREFIX simd_fixed64(const simd_fixed64&) = default;
	SFMM_PREFIX simd_fixed64(simd_fixed64&&) = default;
	SFMM_PREFIX
	simd_fixed64& operator=(const simd_fixed64&) = default;
	SFMM_PREFIX
	simd_fixed64& operator=(simd_fixed64&&) = default;
	SFMM_PREFIX
	inline fixed64& operator[](int j) {
		return *((fixed64*) &(i[j]));
	}
	SFMM_PREFIX simd_f64 inline operator-(const simd_fixed64& other) const {
		return simd_f64(simd_i64(i - other.i)) * c0di;
	}
	SFMM_PREFIX void pad(int n) {
		i.pad(n);
	}
};
