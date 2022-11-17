
class fixed32 {
	static constexpr double c0d = double(std::numeric_limits<unsigned>::max()) + double(1);
	static constexpr float c0s = float(std::numeric_limits<unsigned>::max()) + float(1);
	static constexpr double c0di = double(1) / c0d;
	static constexpr float c0si = float(1) / c0s;
	unsigned i;
	SFMM_PREFIX constexpr fixed32() :
			i() {
	}
	SFMM_PREFIX fixed32(const fixed32&) = default;
	SFMM_PREFIX fixed32(fixed32&&) = default;
	SFMM_PREFIX
	fixed32& operator=(const fixed32&) = default;
	SFMM_PREFIX
	fixed32& operator=(fixed32&&) = default;
	SFMM_PREFIX
	inline fixed32& operator=(double other) {
		i = other * c0d;
		return *this;
	}
	SFMM_PREFIX inline fixed32(double other) {
		*this = other;
	}
	SFMM_PREFIX inline operator double() const {
		return (i + double(0.5)) * c0di;
	}
	SFMM_PREFIX
	inline float operator-(const fixed32& other) const {
		return float(i - other.i) * c0si;
	}
};
