
class fixed64 {
	static constexpr double c0d = double(std::numeric_limits<std::uint64_t>::max()) + double(1);
	static constexpr double c0di = double(1) / c0d;
	std::uint64_t i;
public:
	SFMM_PREFIX constexpr fixed64() :
			i() {
	}
	SFMM_PREFIX fixed64(const fixed64&) = default;
	SFMM_PREFIX fixed64(fixed64&&) = default;
	SFMM_PREFIX
	fixed64& operator=(const fixed64&) = default;
	SFMM_PREFIX
	fixed64& operator=(fixed64&&) = default;
	SFMM_PREFIX
	inline fixed64& operator=(double other) {
		i = other * c0d;
		return *this;
	}
	SFMM_PREFIX inline fixed64(double other) {
		*this = other;
	}
	SFMM_PREFIX inline double to_double() const {
		return (i + double(0.5)) * c0di;
	}
	SFMM_PREFIX
	inline double operator-(const fixed64& other) const {
		return double((std::int64_t)(i - other.i)) * c0di;
	}
	friend class simd_fixed64;
};
