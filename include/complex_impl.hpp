template<class T>
SFMM_PREFIX complex<T>::complex(T a) :
		x(a), y(0.0) {
}

template<class T>
SFMM_PREFIX complex<T>::complex(T a, T b) :
		x(a), y(b) {
}

template<class T>
SFMM_PREFIX complex<T>& complex<T>::operator+=(complex<T> other) {
	x += other.x;
	y += other.y;
	return *this;
}

template<class T>
SFMM_PREFIX complex<T>& complex<T>::operator-=(complex<T> other) {
	x -= other.x;
	y -= other.y;
	return *this;
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator*(complex<T> other) const {
	complex<T> a;
	a.x = x * other.x - y * other.y;
	a.y = fma(x, other.y, y * other.x);
	return a;
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator/(complex<T> other) const {
	return *this * other.conj() / other.norm();
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator/=(complex<T> other) {
	*this = *this * other.conj() / other.norm();
	return *this;
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator/(T other) const {
	complex<T> b;
	b.x = x / other;
	b.y = y / other;
	return b;
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator*(T other) const {
	complex<T> b;
	b.x = x * other;
	b.y = y * other;
	return b;
}

template<class T>
SFMM_PREFIX complex<T>& complex<T>::operator*=(T other) {
	x *= other;
	y *= other;
	return *this;
}

template<class T>
SFMM_PREFIX complex<T>& complex<T>::operator*=(complex<T> other) {
	*this = *this * other;
	return *this;
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator+(complex<T> other) const {
	complex<T> a;
	a.x = x + other.x;
	a.y = y + other.y;
	return a;
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator-(complex<T> other) const {
	complex<T> a;
	a.x = x - other.x;
	a.y = y - other.y;
	return a;
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::conj() const {
	complex<T> a;
	a.x = x;
	a.y = -y;
	return a;
}

template<class T>
SFMM_PREFIX T complex<T>::real() const {
	return x;
}

template<class T>
SFMM_PREFIX T complex<T>::imag() const {
	return y;
}

template<class T>
SFMM_PREFIX T& complex<T>::real() {
	return x;
}

template<class T>
SFMM_PREFIX T& complex<T>::imag() {
	return y;
}

template<class T>
SFMM_PREFIX T complex<T>::norm() const {
	return fma(x,x,y*y);
}

template<class T>
SFMM_PREFIX T complex<T>::abs() const {
	return sqrt(norm());
}

template<class T>
SFMM_PREFIX complex<T> complex<T>::operator-() const {
	complex<T> a;
	a.x = -x;
	a.y = -y;
	return a;
}

template<class T>
SFMM_PREFIX inline complex<T> operator*(T a, complex<T> b) {
	return complex<T>(a*b.real(),a*b.imag());
}

template<class T>
inline void swap(complex<T>& a, complex<T>& b) {
	std::swap(a.real(), b.real());
	std::swap(a.imag(), b.imag());
}

