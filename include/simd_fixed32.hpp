
class simd_fixed32 {
	static const simd_f32 c0s;
	static const simd_f32 c0si;
	simd_ui32 i;
public:
	SFMM_PREFIX constexpr simd_fixed32() :
			i() {
	}
	SFMM_PREFIX simd_fixed32( fixed32 other ) : i(other.i) {
	}
	SFMM_PREFIX simd_fixed32(const simd_fixed32&) = default;
	SFMM_PREFIX simd_fixed32(simd_fixed32&&) = default;
	SFMM_PREFIX
	simd_fixed32& operator=(const simd_fixed32&) = default;
	SFMM_PREFIX
	simd_fixed32& operator=(simd_fixed32&&) = default;
	SFMM_PREFIX
	inline fixed32& operator[](int j) {
		return *((fixed32*) &(i[j]));
	}
	simd_f32 inline operator-(const simd_fixed32& other) const {
		return simd_f32(simd_i32(i - other.i)) * c0si;
	}
	SFMM_PREFIX void pad(int n) {
		i.pad(n);
	}
};
