template<class T>
class complex {
	T x, y;
public:
	SFMM_PREFIX complex() = default;
	SFMM_PREFIX complex(T a);
	SFMM_PREFIX complex(T a, T b);
	SFMM_PREFIX complex& operator+=(complex other);
	SFMM_PREFIX complex& operator-=(complex other);
	SFMM_PREFIX complex operator*(complex other) const;
	SFMM_PREFIX complex operator/(complex other) const;
	SFMM_PREFIX complex operator/=(complex other);
	SFMM_PREFIX complex operator/(T other) const;
	SFMM_PREFIX complex operator*(T other) const;
	SFMM_PREFIX complex& operator*=(T other);
	SFMM_PREFIX complex& operator*=(complex other);
	SFMM_PREFIX complex operator+(complex other) const;
	SFMM_PREFIX complex operator-(complex other) const;
	SFMM_PREFIX complex conj() const;
	SFMM_PREFIX T real() const;
	SFMM_PREFIX T imag() const;
	SFMM_PREFIX T& real();
	SFMM_PREFIX T& imag();
	SFMM_PREFIX T norm() const;
	SFMM_PREFIX T abs() const;
	SFMM_PREFIX complex operator-() const;
};
