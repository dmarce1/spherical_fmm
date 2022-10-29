#include <stdio.h>
#include <utility>
#include <cmath>
#include <complex>
#include <array>
#include <limits>
#include <math.h>
#include <climits>
#include <functional>
#ifdef TEST_TYPE_FLOAT
#include "sfmmf.hpp"
bool scaled = true;
#define SCALED
#endif
#ifdef TEST_TYPE_DOUBLE
#include "sfmmd.hpp"
bool scaled = false;
#endif
#ifdef TEST_TYPE_VEC_FLOAT
#include "sfmmvf.hpp"
bool scaled = true;
#define SCALED
#endif
#ifdef TEST_TYPE_VEC_DOUBLE
#include "sfmmvd.hpp"
bool scaled = false;
#endif

#include "timer.hpp"

template<class T>
using complex = std::complex<T>;

#ifdef TEST_TYPE_VEC_DOUBLE
#define VECTOR
using vec_real = sfmm::v2df;
using real = double;
#define FLAG
#endif
#ifdef TEST_TYPE_VEC_FLOAT
#define VECTOR
using vec_real = sfmm::v8sf;
using real = float;
#define FLAG
#endif
#ifdef TEST_TYPE_FLOAT
using real = float;
#define FLAG
#endif
#ifdef TEST_TYPE_DOUBLE
using real = double;
#define FLAG
#endif

#ifndef FLAG
#error
#endif

constexpr double dfactorial(int n) {
	if (n == 1 || n == 0) {
		return 1;
	} else {
		return n * dfactorial(n - 2);
	}
}

double rand1() {
	return (double) (rand() + 0.5) / (RAND_MAX);
}

template<class T>
T sqr(T a) {
	return a * a;
}

template<class T>
T sqr(T a, T b, T c) {
	return a * a + b * b + c * c;
}

double Pm(int l, int m, double x) {
	if (m > l) {
		return 0.0;
	} else if (l == 0) {
		return 1.0;
	} else if (l == m) {
		return -(2 * l - 1) * Pm(l - 1, l - 1, x) * sqrt(1.0 - x * x);
	} else {
		return ((2 * l - 1) * x * Pm(l - 1, m, x) - (l - 1 + m) * Pm(l - 2, m, x)) / (l - m);
	}
}

constexpr int index(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template<class T>
constexpr T nonepow(int m) {
	return m % 2 == 0 ? T(1) : T(-1);
}

double nonepow(int m) {
	return m % 2 == 0 ? double(1) : double(-1);
}

template<class T, int P>
struct spherical_expansion: public std::array<complex<T>, (P + 1) * (P + 2) / 2> {

	inline complex<T> operator()(int n, int m) const {
		if (m >= 0) {
			return (*this)[index(n, m)];
		} else {
			return std::conj((*this)[index(n, -m)]) * nonepow<T>(m);
		}
	}
	void print() const {
		for (int l = 0; l <= P; l++) {
			for (int m = 0; m <= l; m++) {
				printf("%e + i %e  ", (*this)(l, m).real(), (*this)(l, m).imag());
			}
			printf("\n");
		}
	}
};

#define MBITS 23

template<int P>
constexpr int multi_bits(int n = 1) {
	if (n == P + 1) {
		return 0;
	} else {
		return (2 * n + 1) * (MBITS - n + 1) + multi_bits<P>(n + 1);
	}
}

template<class T, int P>
spherical_expansion<T, P> spherical_regular_harmonic(T x, T y, T z) {
	const T r2 = x * x + y * y + z * z;
	const complex<T> R = complex<T>(x, y);
	spherical_expansion<T, P> Y;
	Y[index(0, 0)] = complex<T>(T(1), T(0));
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / T(2 * m);
		}
		if (m + 1 <= P) {
			Y[index(m + 1, m)] = z * Y[index(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			const T inv = T(1) / (T(n * n) - T(m * m));
			Y[index(n, m)] = inv * (T(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
		}
	}
	return Y;
}

#include <vector>

int cindex(int l, int m) {
	return l * (l + 1) / 2 + m;
}
std::vector<complex<float>> spherical_singular_harmonic2(int P, float x, float y, float z) {
	const float r2 = x * x + y * y + z * z;
	const float r2inv = float(1) / r2;
	complex<float> R = complex<float>(x, y);
	std::vector<complex<float>> O((P + 1) * (P + 1));
	O[cindex(0, 0)] = complex<float>(sqrt(r2inv), float(0));
	R *= r2inv;
	z *= r2inv;
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			O[cindex(m, m)] = O[cindex(m - 1, m - 1)] * R * float(2 * m - 1);
		}
		if (m + 1 <= P) {
			O[cindex(m + 1, m)] = float(2 * m + 1) * z * O[cindex(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			O[cindex(n, m)] = (float(2 * n - 1) * z * O[cindex(n - 1, m)] - float((n - 1) * (n - 1) - m * m) * r2inv * O[cindex(n - 2, m)]);
		}
	}
	return O;
}

template<class T, int P>
spherical_expansion<T, P> spherical_singular_harmonic(T x, T y, T z) {
	const T r2 = x * x + y * y + z * z;
	const T r2inv = T(1) / r2;
	complex<T> R = complex<T>(x, y);
	spherical_expansion<T, P> O;
	O[index(0, 0)] = complex<T>(sqrt(r2inv), T(0));
	R *= r2inv;
	z *= r2inv;
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			O[index(m, m)] = O[index(m - 1, m - 1)] * R * T(2 * m - 1);
		}
		if (m + 1 <= P) {
			O[index(m + 1, m)] = T(2 * m + 1) * z * O[index(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			O[index(n, m)] = (T(2 * n - 1) * z * O[index(n - 1, m)] - T((n - 1) * (n - 1) - m * m) * r2inv * O[index(n - 2, m)]);
		}
	}
	return O;
}

template<class T, int P>
void greens(T* G, T x, T y, T z) {
	auto G0 = spherical_singular_harmonic<float, P>(x, y, z);
	for (int l = 0; l <= P; l++) {
		for (int m = 0; m <= l; m++) {
			if (m == 0) {
				G[l * (l + 1)] = G0[cindex(l, 0)].real();
			} else {
				G[l * (l + 1) + m] = G0[cindex(l, m)].real();
				G[l * (l + 1) - m] = G0[cindex(l, m)].imag();
			}
		}
	}
}

template<class T, int P>
void spherical_expansion_M2M(spherical_expansion<T, P>& M, T x, T y, T z) {
	const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	const auto M0 = M;
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			M[index(n, m)] = complex<T>(T(0), T(0));
			for (int k = 0; k <= n; k++) {
				for (int l = -k; l <= k; l++) {
					if (abs(m - l) > n - k) {
						continue;
					}
					if (-abs(m - l) < k - n) {
						continue;
					}
					M[index(n, m)] += Y(k, l) * M0(n - k, m - l);
				}
			}
		}
	}
}

template<class T, int P>
void spherical_expansion_L2L(spherical_expansion<T, P>& L, T x, T y, T z) {
	const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	const auto L0 = L;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = complex<T>(T(0), T(0));
			for (int k = 0; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					L[index(n, m)] += Y(k, l).conj() * L0(n + k, m + l);
				}
			}
		}
	}
}

void random_unit(real& x, real& y, real& z) {
	const real theta = acos(2 * rand1() - 1.0);
	const real phi = rand1() * 2.0 * M_PI;
	x = cos(phi) * sin(theta);
	y = sin(phi) * sin(theta);
	z = cos(theta);
}

void random_vector(real& x, real& y, real& z) {
	do {
		x = 2 * rand1() - 1;
		y = 2 * rand1() - 1;
		z = 2 * rand1() - 1;
	} while (sqr(x, y, z) > 1);
}

template<class T, int P>
spherical_expansion<T, P> spherical_expansion_ref_M2L(spherical_expansion<T, P - 1> M, T x, T y, T z) {
	const auto O = spherical_singular_harmonic<T, P>(x, y, z);
	spherical_expansion<T, P> L;
	int count = 0;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = complex<T>(T(0), T(0));
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					L[index(n, m)] += M(k, l).conj() * O(n + k, m + l);
					count += 9;
				}
			}
		}
	}
	return L;
}

#include <fenv.h>

void ewald_compute(long double& pot, long double& fx, long double& fy, long double& fz, long double dx0, long double dx1, long double dx2) {
	const long double cons1 = (long double) (4.0L / sqrtl(4.0L * atanl(1)));
	fx = 0.0;
	fy = 0.0;
	fz = 0.0;
	pot = 0.0;
	const auto r2 = sqr(dx0, dx1, dx2);  // 5

	if (r2 > 0.) {
		const long double dx = dx0;
		const long double dy = dx1;
		const long double dz = dx2;
		const long double r2 = sqr(dx, dy, dz);
		const long double r = sqrtl(r2);
		const long double rinv = 1.L / r;
		const long double r2inv = rinv * rinv;
		const long double r3inv = r2inv * rinv;
		long double exp0 = expl(-4.0L * r2);
		long double erf0 = erfl(2.0L * r);
		const long double expfactor = cons1 * r * exp0;
		const long double d0 = erf0 * rinv;
		const long double d1 = (expfactor - erf0) * r3inv;
		pot += d0;
		fx -= dx * d1;
		fy -= dy * d1;
		fz -= dz * d1;
		for (int xi = -4; xi <= +4; xi++) {
			for (int yi = -4; yi <= +4; yi++) {
				for (int zi = -4; zi <= +4; zi++) {
					const bool center = sqr(xi, yi, zi) == 0;
					if (center || sqr(xi, yi, zi) > 20) {
						continue;
					}
					const long double dx = dx0 - xi;
					const long double dy = dx1 - yi;
					const long double dz = dx2 - zi;
					const long double r2 = sqr(dx, dy, dz);
					const long double r = sqrtl(r2);
					const long double rinv = 1.L / r;
					const long double r2inv = rinv * rinv;
					const long double r3inv = r2inv * rinv;
					long double exp0 = expl(-4.0L * r2);
					long double erfc0 = erfcl(2.0L * r);
					const long double expfactor = cons1 * r * exp0;
					const long double d0 = -erfc0 * rinv;
					const long double d1 = (expfactor + erfc0) * r3inv;
					pot += d0;
					fx -= dx * d1;
					fy -= dy * d1;
					fz -= dz * d1;
				}
			}
		}
		pot += (long double) (atanl(1));
		for (int xi = -4; xi <= +4; xi++) {
			for (int yi = -4; yi <= +4; yi++) {
				for (int zi = -4; zi <= +4; zi++) {
					const long double hx = xi;
					const long double hy = yi;
					const long double hz = zi;
					const long double h2 = sqr(hx, hy, hz);
					if (h2 > 0.0L && h2 <= 20) {
						const long double hdotx = dx0 * hx + dx1 * hy + dx2 * hz;
						const long double omega = (long double) (8.0L * atanl(1)) * hdotx;
						long double c, s;
						sincosl(omega, &s, &c);
						const long double c0 = -1.L / h2 * exp((long double) (-sqr(4.0L * atanl(1)) * 0.25L) * h2) * (long double) (1.L / (4.0L * atanl(1)));
						const long double c1 = -s * 8.0L * atanl(1) * c0;
						pot += c0 * c;
						fx -= c1 * hx;
						fy -= c1 * hy;
						fz -= c1 * hz;
					}
				}
			}
		}
	} else {
		pot += 2.837291L;
	}
}

template<class T>
void ewald_limits(double& r2, double& h2, double alpha) {
	r2 = 0;
	h2 = 0;
	const auto threesqr = [](int i) {
		for( int x = 0; x <= i; x++ ) {
			for( int y = 0; y <= i; y++) {
				for( int z = 0; z <= i; z++) {
					if( x*x + y*y + z*z == i ) {
						return true;
					}
				}
			}
		}
		return false;
	};

	double dx = 0.01;

	int R2 = 1;
	while (1) {
		double m = double(0);
		const int e = sqrt(R2);
		for (double x0 = 0; x0 < 0.5; x0 += dx) {
			for (double y0 = x0; y0 < 0.5; y0 += dx) {
				for (double z0 = y0; z0 < 0.5; z0 += dx) {
					for (int xi = 0; xi <= e; xi++) {
						for (int yi = xi; yi <= e; yi++) {
							for (int zi = yi; zi <= e; zi++) {
								if (xi * xi + yi * yi + zi * zi != R2) {
									continue;
								}
								const double x = x0 - xi;
								const double y = y0 - yi;
								const double z = z0 - zi;
								const double r0 = sqrt(x0 * x0 + y0 * y0 + z0 * z0);
								if (r0 != double(0)) {
									const double r = sqrt(x * x + y * y + z * z);
									double a = r0 * erfc(alpha * r) / r;
									m = std::max(m, fabs(a));
								}
							}
						}
					}
				}
			}
		}
		if (m < std::numeric_limits<T>::epsilon()) {
			break;
		}
		while (!threesqr(++R2))
			;
	}
	while (!threesqr(--R2))
		;
	r2 = R2;
	R2 = 1;
	while (1) {
		double m = double(0);
		const int e = sqrt(R2);
		int n = 0;
		for (double x0 = 0; x0 < 0.5; x0 += dx) {
			for (double y0 = x0; y0 < 0.5; y0 += dx) {
				for (double z0 = y0; z0 < 0.5; z0 += dx) {
					for (int xi = 0; xi <= e; xi++) {
						for (int yi = xi; yi <= e; yi++) {
							for (int zi = yi; zi <= e; zi++) {
								if (xi * xi + yi * yi + zi * zi != R2) {
									continue;
								}
								const double x = x0 - xi;
								const double y = y0 - yi;
								const double z = z0 - zi;
								const double h2 = xi * xi + yi * yi + zi * zi;
								const double hdotx = x * xi + y * yi + z * zi;
								const double r = sqrt(x * x + y * y + z * z);
								double a = r * fabs(cos(2.0 * M_PI * hdotx) * exp(-M_PI * M_PI * h2 / (alpha * alpha)) / (h2 * sqrt(M_PI)));
								m = std::max(m, fabs(a));
							}
						}
					}
				}
			}
		}
		if (m < std::numeric_limits<T>::epsilon()) {
			break;
		}
		while (!threesqr(++R2))
			;
	}
	while (!threesqr(--R2))
		;
	h2 = R2;
}

enum test_type {
	CC, PC, CP, EWALD
};

template<class T, int P>
void M2L_ewald3(T* L, const T* M, T x0, T y0, T z0) {
	constexpr T alpha = 2.f;
	const auto index = [](int l, int m) {
		return l * (l + 1) + m;
	};
	T L2[(P + 1) * (P + 1) + 1];
	T G[(P + 1) * (P + 1) + 1];
	T G0[(P + 1) * (P + 1) + 1];
	for (int i = 0; i < (P + 1) * (P + 1) + 1; i++) {
		L2[i] = L[i];
	}
	for (int l = 0; l <= P; l++) {
		for (int m = -l; m <= l; m++) {
			G[index(l, m)] = T(0);
		}
	}
	G[(P + 1) * (P + 1)] = T(0);
	for (int ix = -3; ix <= 3; ix++) {
		for (int iy = -3; iy <= 3; iy++) {
			for (int iz = -3; iz <= 3; iz++) {
				const T x = x0 - ix;
				const T y = y0 - iy;
				const T z = z0 - iz;
				const T r2 = sqr(x, y, z);
				if (r2 <= sqr(2.6)) {
					const T r = sqrt(x * x + y * y + z * z);
					greens<real, P>(G0, x, y, z);
					T gamma1 = sqrt(M_PI) * erfc(alpha * r);
					T gamma0inv = 1.0f / sqrt(M_PI);
					for (int l = 0; l <= P; l++) {
						const T gamma = gamma1 * gamma0inv;
						if (ix * ix + iy * iy + iz * iz == 0) {
							if ((x0 * x0 + y0 * y0 + z0 * z0) == 0.0) {
								if (l == 0) {
									G[index(0, 0)] += T(2) * alpha / sqrt(M_PI);
								}
							} else {
								for (int m = -l; m <= l; m++) {
									G[index(l, m)] -= (gamma - nonepow(l)) * G0[index(l, m)];
								}
							}
						} else {
							for (int m = -l; m <= l; m++) {
								G[index(l, m)] -= gamma * G0[index(l, m)];
							}
						}
						const T x = alpha * alpha * r * r;
						const T s = l + 0.5f;
						gamma0inv /= -s;
						gamma1 = s * gamma1 + pow(x, s) * exp(-x);
					}
				}
			}
		}
	}
	for (int hx = -2; hx <= 2; hx++) {

		for (int hy = -2; hy <= 2; hy++) {
			for (int hz = -2; hz <= 2; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= 8 && h2 > 0) {
					const T h = sqrt(h2);
					greens<real, P>(G0, (T) hx, (T) hy, (T) hz);
					const T hdotx = hx * x0 + hy * y0 + hz * z0;
					T gamma0inv = 1.0f / sqrt(M_PI);
					T hpow = 1.f / h;
					T pipow = 1.f / sqrt(M_PI);
					for (int l = 0; l <= P; l++) {
						for (int m = 0; m <= l; m++) {
							const float phi = T(2.0 * M_PI) * hdotx;
							float Rx, Ry, ax, ay, bx, by;
							sincosf(phi, &Ry, &Rx);
							if (m == 0) {
								ax = G0[index(l, m)] * Rx;
								ay = G0[index(l, m)] * Ry;
							} else {
								ax = G0[index(l, m)] * Rx - G0[index(l, -m)] * Ry;
								ay = G0[index(l, m)] * Ry + G0[index(l, -m)] * Rx;
							}
							T c0 = gamma0inv * hpow * pipow * exp(-h * h * T(M_PI * M_PI) / (alpha * alpha));
							ax *= c0;
							ay *= c0;
							if (l % 4 == 1) {
								T tmp = ax;
								ax = -ay;
								ay = tmp;
							} else if (l % 4 == 2) {
								ax = -ax;
								ay = -ay;
							} else if (l % 4 == 3) {
								T tmp = ax;
								ax = ay;
								ay = -tmp;
							}
							G[index(l, m)] -= ax;
							if (m != 0) {
								G[index(l, -m)] -= ay;
							}
						}
						const T s = l + 0.5f;
						gamma0inv /= s;
						hpow *= h * h;
						pipow *= M_PI;
					}
				}
			}
		}
	}
	G[(P + 1) * (P + 1)] = T(4.0 * M_PI / 3.0);
	G[0] += T(M_PI / (alpha * alpha));
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			L[index(n, m)] = L[index(n, -m)] = 0;
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					auto mx = M[index(k, abs(l))];
					auto my = M[index(k, -abs(l))];
					auto gx = G[index(n + k, abs(l + m))];
					auto gy = G[index(n + k, -abs(l + m))];
					if (l == 0) {
						if ((l + m) == 0) {
							L[index(n, m)] += mx * gx;
						} else if ((l + m) < 0) {
							if (abs((l + m)) % 2 == 0) {
								L[index(n, m)] += mx * gx;
								if (m != 0) {
									L[index(n, -m)] -= mx * gy;
								}
							} else {
								L[index(n, m)] -= mx * gx;
								if (m != 0) {
									L[index(n, -m)] += mx * gy;
								}
							}
						} else {
							L[index(n, m)] += mx * gx;
							if (m != 0) {
								L[index(n, -m)] += mx * gy;
							}
						}
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							if ((l + m) == 0) {
								L[index(n, m)] += mx * gx;
								if (m != 0) {
									L[index(n, -m)] += gx * my;
								}
							} else if ((l + m) < 0) {
								if (abs((l + m)) % 2 == 0) {
									L[index(n, m)] += mx * gx + my * gy;
									if (m != 0) {
										L[index(n, -m)] -= mx * gy - gx * my;
									}
								} else {
									L[index(n, m)] -= mx * gx + my * gy;
									if (m != 0) {
										L[index(n, -m)] += mx * gy - gx * my;
									}
								}
							} else {
								L[index(n, m)] += mx * gx - my * gy;
								if (m != 0) {
									L[index(n, -m)] += mx * gy + gx * my;
								}
							}
						} else {
							if ((l + m) == 0) {
								L[index(n, m)] -= mx * gx;
								if (m != 0) {
									L[index(n, -m)] -= gx * my;
								}
							} else if ((l + m) < 0) {
								if (abs((l + m)) % 2 == 0) {
									L[index(n, m)] -= mx * gx + my * gy;
									if (m != 0) {
										L[index(n, -m)] += mx * gy - gx * my;
									}
								} else {
									L[index(n, m)] += mx * gx + my * gy;
									if (m != 0) {
										L[index(n, -m)] -= mx * gy - gx * my;
									}
								}
							} else {
								L[index(n, m)] -= mx * gx - my * gy;
								if (m != 0) {
									L[index(n, -m)] -= mx * gy + gx * my;
								}
							}
						}
					} else {
						if ((l + m) == 0) {
							L[index(n, m)] += mx * gx;
							if (m != 0) {
								L[index(n, -m)] -= gx * my;
							}
						} else if ((l + m) < 0) {
							if (abs((l + m)) % 2 == 0) {
								L[index(n, m)] += mx * gx - my * gy;
								if (m != 0) {
									L[index(n, -m)] -= mx * gy + gx * my;
								}
							} else {
								L[index(n, m)] -= mx * gx - my * gy;
								if (m != 0) {
									L[index(n, -m)] += mx * gy + gx * my;
								}
							}
						} else {
							L[index(n, m)] += mx * gx + my * gy;
							if (m != 0) {
								L[index(n, -m)] += mx * gy - gx * my;
							}
						}

					}
				}
			}
		}
	}
	//L[index(0, 0)] += M[index(0, 0)] * T(M_PI / (alpha * alpha));
	L[index(0, 0)] -= T(0.5) * G[(P + 1) * (P + 1)] * M[P * P];
	L[index(1, -1)] -= 2.0 * G[(P + 1) * (P + 1)] * M[index(1, -1)];
	L[index(1, +0)] -= G[(P + 1) * (P + 1)] * M[index(1, +0)];
	L[index(1, +1)] -= 2.0 * G[(P + 1) * (P + 1)] * M[index(1, +1)];
	L[(P + 1) * (P + 1)] -= T(0.5) * G[(P + 1) * (P + 1)] * M[index(0, 0)];

//	M2L_ewald<real>(L2, M, x0, y0, z0);
//	L = L2;
}

template<int P>

real test_M2L(test_type type, real theta = 0.5) {

	real err = 0.0;
	using namespace sfmm;
	int N = 4000;
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);
	real err2 = 0.0;
	real norm = 0.0;
	long double phi, fx, fy, fz;
	int flags = 0;
//	sfmm_set_scale_factor_float(1000);
//	sfmm_set_scale_factor_double(1000);
	for (int i = 0; i < N; i++) {
		if (type == EWALD) {
			real x0, x1, x2, y0, y1, y2, z0, z1, z2;
			random_unit(x0, y0, z0);
			random_unit(x1, y1, z1);
			random_unit(x2, y2, z2);
			const auto alpha = 0.45 * rand1() + 0.05;
			x1 *= alpha;
			y1 *= alpha;
			z1 *= alpha;
			x0 *= 0.5 * theta * alpha;
			y0 *= 0.5 * theta * alpha;
			z0 *= 0.5 * theta * alpha;
			x2 *= 0.5 * theta * alpha;
			y2 *= 0.5 * theta * alpha;
			z2 *= 0.5 * theta * alpha;
			//					x0 = y0 = z0 = 0.0;
//						x2 = y2 = z2 = 0.0;
			double f0 = rand1();
			double f1 = rand1();
			double f2 = rand1();
			double g0 = rand1();
			double g1 = rand1();
			double g2 = rand1();
#ifdef VECTOR
#ifdef SCALED
			multipole_periodic_scaled<vec_real, P> M;
			expansion_periodic_scaled<vec_real, P> L;
			M.init(0.1);
			L.init(0.01);
#else
			multipole_periodic<vec_real, P> M;
			expansion_periodic<vec_real, P> L;
			M.init();
			L.init();
#endif
			force_type<vec_real> f;
			f.init();
			P2M(M, vec_real(real(0.5)), vec_real(real(-x0 * f0)), vec_real(real(-y0 * f1)), vec_real(real(-z0 * f2)));
			M2M(M, vec_real(real(-real(x0) * (1 - f0))), vec_real(real(-real(y0) * (1 - f1))), vec_real(real(-real(z0) * (1 - f2))));
			M2L_ewald(L, M, vec_real(x1), vec_real(y1), vec_real(z1));
			L2L(L, vec_real(x2 * g0), vec_real(y2 * g1), vec_real(z2 * g2));
			L2P(f, L, vec_real(x2 * (real(1) - g0)), vec_real(y2 * (real(1) - g1)), vec_real(z2 * (real(1) - g2)));
			ewald_compute(phi, fx, fy, fz, (-x2 + x1) + x0, (-y2 + y1) + y0, (-z2 + z1) + z0);
			fx *= 0.5;
			fy *= 0.5;
			fz *= 0.5;
			phi *= 0.5;
			//		printf( "%e %e %e\n", fx, fy, fz);
			//	abort();
			const double fa = std::sqrt(fx * fx + fy * fy + fz * fz);
			const double fn = std::sqrt(sqr(f.force[0][0], f.force[1][0], f.force[2][0]));
//			printf( "%e %e %e\n",fa, fn, fn/fa-1.0);

			//	printf("%Le %e %e %e\n", fx, -f.force[0][0], f.force[1][0], f.force[2][1]);
			err += fabs(phi - f.potential[0]);
			norm += fabs(phi);
#else
#ifdef SCALED
			multipole_periodic_scaled<real, P> M;
			expansion_periodic_scaled<real, P> L;
			M.init(0.1);
			L.init(0.01);
#else
			multipole_periodic < vec_real, P > M;
			expansion_periodic < vec_real, P > L;
			M.init();
			L.init();
#endif
			force_type < real > f;
			f.init();
			P2M(M, 0.5, -x0 * f0, -y0 * f1, -z0 * f2);
			M2M(M, -real(x0) * (1 - f0), -real(y0) * (1 - f1), -real(z0) * (1 - f2));
			M2L_ewald(L, M, x1, y1, z1);
			L2L(L, x2 * g0, y2 * g1, z2 * g2);
			L2P(f, L, x2 * (1 - g0), y2 * (1 - g1), z2 * (1 - g2));
			ewald_compute(phi, fx, fy, fz, (-x2 + x1) + x0, (-y2 + y1) + y0, (-z2 + z1) + z0);
			fx *= 0.5;
			fy *= 0.5;
			fz *= 0.5;
			phi *= 0.5;
			//		printf( "%e %e %e\n", fx, fy, fz);
			//	abort();
			//		printf( "%e %e %e\n", phi, L0[0], phi/ L0[0]);
			const double fa = std::sqrt(fx * fx + fy * fy + fz * fz);
			const double fn = std::sqrt(sqr(f.force[0], f.force[1], f.force[2]));
			//	printf( "%e %e\n", fx, -L2[2], fy, -L2[1],  fz, -L2[2]);
			err += fabs(phi - f.potential);
			norm += fabs(phi);
#endif
		} else {
			real xa, x1, x2, ya, y1, y2, za, z1, z2, x3, y3, z3, xb, yb, zb;
			random_unit(xa, ya, za);
			random_unit(x1, y1, z1);
			random_unit(x2, y2, z2);
			if (type == CP) {
				xa = ya = za = 0;
			} else if (type == PC) {
				x2 = y2 = z2 = 0;
			}
			if (type == CC) {
				x1 /= 0.5 * theta;
				y1 /= 0.5 * theta;
				z1 /= 0.5 * theta;
			} else {
				x1 /= theta;
				y1 /= theta;
				z1 /= theta;
			}
			real eps = 1;
			xa *= eps;
			ya *= eps;
			za *= eps;
			x1 *= eps;
			y1 *= eps;
			z1 *= eps;
			x2 *= eps;
			y2 *= eps;
			z2 *= eps;
			xb = -xa;
			yb = -ya;
			zb = -za;
			eps = 0.0;
			double f0 = rand1();
			double f1 = rand1();
			double f2 = rand1();
			double g0 = rand1();
			double g1 = rand1();
			double g2 = rand1();

#ifdef VECTOR
			using T = vec_real;
#else
			using T = real;
#endif
			force_type<T> f;
#ifdef SCALED
			multipole_periodic_scaled_wo_dipole<T, P> M1(0.1);
			multipole_periodic_scaled_wo_dipole<T, P> M;
			expansion_periodic_scaled<T, P> L;
			M.init(0.1);
			L.init(0.01);
#else
			multipole_periodic_wo_dipole<T, P> M1;
			multipole_periodic_wo_dipole<T, P> M;
			expansion_periodic<T, P> L;
			M.init();
			L.init();
#endif
			f.init();
/*			P2M(M1, T(1.0), -T(xa), -T(ya), -T(za));
			M += M1;
			P2M(M1, T(1.0), -T(xb), -T(yb), -T(zb));
			M += M1;
			M2M(M, -T(0), -T(0), -T(0));*/
			P2M(M1, T(1.0), -T(0), -T(0), -T(0));
			M2M(M1, -T(xa), -T(ya), -T(za));
			M += M1;
			P2M(M1, T(1.0), -T(0), -T(0), -T(0));
			M2M(M1, -T(xb), -T(yb), -T(zb));
			M += M1;
			if (type == CC) {
				M2L(L, M, T(x1), T(y1), T(z1));
				L2L(L, T(x2 * g0), T(y2 * g1), T(z2 * g2));
				L2P(f, L, T(x2 * (1 - g0)), T(y2 * (1 - g1)), T(z2 * (1 - g2)));
			} else if (type == PC) {
				M2P(f, M, T(x1), T(y1), T(z1));
			} else if (type == CP) {
				P2L(L, T(2.0), T(x1), T(y1), T(z1));
				L2L(L, T(x2 * g0), T(y2 * g1), T(z2 * g2));
				L2P(f, L, T(x2 * (1 - g0)), T(y2 * (1 - g1)), T(z2 * (1 - g2)));
			}
			real dx = (x2 + x1) - xa;
			real dy = (y2 + y1) - ya;
			real dz = (z2 + z1) - za;
			real r = std::sqrt(sqr(dx, dy, dz));
			real phi = 1.0 / r;
			dx = (x2 + x1) - xb;
			dy = (y2 + y1) - yb;
			dz = (z2 + z1) - zb;
			r = std::sqrt(sqr(dx, dy, dz));
			phi += 1.0 / r;
#ifdef VECTOR
			err += fabs(phi - f.potential[0]);
#else
			err += fabs(phi - f.potential);
#endif
			norm += fabs(phi);
		}
	}
	err /= norm;
	return err;
}

template<int NMAX, int N = 3>
struct run_tests {
	void operator()(test_type type, real theta) {
		auto a = test_M2L<N>(type, theta);
		printf("%i %e\n", N, a);
		run_tests<NMAX, N + 1> run;
		run(type, theta);
	}
};

template<int NMAX>
struct run_tests<NMAX, NMAX> {
	void operator()(test_type type, real theta) {

	}
};

float erfc2_float(float x) {
	constexpr float x0 = 2.75;
	if (x < x0) {
		constexpr int N = 25;
		const float q = 2.0 * x * x;
		float y = 1.0 / dfactorial(2 * N + 1);
		for (int n = N - 1; n >= 0; n--) {
			y = fma(y, q, 1.0 / dfactorial(2 * n + 1));
		}
		y *= (float) (2.0 / sqrt(M_PI)) * x * expf(-x * x);
		y = 1.0f - y;
		return y;
	} else {
		constexpr int N = x0 * x0 + 0.5;
		float q = 1.0 / (2.0 * x * x);
		float y = dfactorial(2 * N - 1) * nonepow(N);
		for (int i = N - 1; i >= 1; i--) {
			y = fma(y, q, dfactorial(2 * i - 1) * nonepow(i));
		}
		y *= q;
		y += 1.0;
		y *= exp(-x * x) / sqrt(M_PI) / x;
		return y;
	}
}

struct test_res {
	double aerr;
	double rerr;
	double tm1;
	double tm2;
};

template<class T>
test_res test_unary(std::function<T(T)> f1, std::function<T(T)> f2, T a, T b) {
	std::pair<T, T> err;
	err.first = 0.0;
	err.second = 0.0;
	constexpr int N = 1000000;
	std::vector<T> nums(N);
	for (int i = 0; i < N; i++) {
		nums[i] = (b - a) * rand1() + a;
	}
	timer tm1, tm2;
	for (int i = 0; i < N; i++) {
		T num = nums[i];
		T res2 = f2(nums[i]);
		T res1 = f1(nums[i]);
		T a = fabs(res2 - res1);
		T r = a / fabs(res1);
		err.first = std::max(err.first, a);
		err.second = std::max(err.second, r);
	}
	tm1.start();
	for (int i = 0; i < N; i++) {
		T res1 = f1(nums[i]);
	}
	tm1.stop();
	tm2.start();
	for (int i = 0; i < N; i++) {
		T res2 = f2(nums[i]);
	}
	tm2.stop();
	test_res res;
	res.aerr = err.first;
	res.rerr = err.second;
	res.tm1 = tm1.read();
	res.tm2 = tm2.read();
	return res;
}

template<class T, class F>
test_res test_binary(F* f1, F* f2, T a, T b) {
	std::pair<T, T> err;
	err.first = 0.0;
	err.second = 0.0;
	constexpr int N = 1000000;
	std::vector<T> nums(N);
	for (int i = 0; i < N; i++) {
		nums[i] = (b - a) * rand1() + a;
	}
	timer tm1, tm2;
	for (int i = 0; i < N; i++) {
		T num = nums[i];
		T res2a, res2b;
		f2(nums[i], &res2a, &res2b);
		T res1a, res1b;
		f1(nums[i], &res1a, &res1b);
		T a = fabs(res2a - res1a);
		T r = a / fabs(res1a);
		err.first = std::max(err.first, a);
		err.second = std::max(err.second, r);
		a = fabs(res2b - res1b);
		r = a / fabs(res1b);
		err.first = std::max(err.first, a);
		err.second = std::max(err.second, r);
	}
	tm1.start();
	for (int i = 0; i < N; i++) {
		T res1a, res1b;
		f1(nums[i], &res1a, &res1b);
	}
	tm1.stop();
	tm2.start();
	for (int i = 0; i < N; i++) {
		T res2a, res2b;
		f2(nums[i], &res2a, &res2b);
	}
	tm2.stop();
	test_res res;
	res.aerr = err.first;
	res.rerr = err.second;
	res.tm1 = tm1.read();
	res.tm2 = tm2.read();
	return res;
}

int main() {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_OVERFLOW);
	feenableexcept(FE_INVALID);
	/*
	 typedef sfmm::vec_float T;
	 typedef sfmm::vec_int32_t V ;
	 typedef sfmm::vec_uint32_t U ;

	 T a = float(1);
	 T b = float(0);
	 */

	/*	auto res = test_unary<double>(static_cast<double (*)(double)>(&std::sqrt), static_cast<double(*)(double)>(&sfmm::detail::sqrt), 0.0, 10.0);
	 printf("sqrt(double)     %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 res = test_unary<double>([](double x) {return 1.0/sqrt(x);}, static_cast<double (*)(double)>(&sfmm::detail::rsqrt), 0.0, 10.0);
	 printf("rsqrt(double)    %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 res = test_binary<double>(static_cast<void (*)(double, double*, double*)>(&sincos), static_cast<void(*)(double,double*,double*)>(&sfmm::detail::sincos), 0.001, 10.0);
	 printf("sincos(double)   %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 res = test_binary<double>(static_cast<void (*)(double, double*, double*)>([](double x, double* e1, double* e2) {
	 *e1 = erfc(x);
	 *e2 = exp(-x*x);
	 }), static_cast<void(*)(double,double*,double*)>(&sfmm::detail::erfcexp), 0.01, 5.0);
	 printf("erfcexp(double)  %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 printf("\n");
	 res = test_unary<float>(static_cast<float (*)(float)>(&std::sqrt), static_cast<float(*)(float)>(&sfmm::detail::sqrt), 0.0, 10.0);
	 printf("sqrt(float)       %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 res = test_unary<float>([](float x) {return 1.0f/sqrtf(x);}, static_cast<float (*)(float)>(&sfmm::detail::rsqrt), 0.0, 10.0);
	 printf("rsqrt(float)      %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 res = test_binary<float>(static_cast<void (*)(float, float*, float*)>(&sincosf), static_cast<void(*)(float,float*,float*)>(&sfmm::detail::sincos), 0.001, 10.0);
	 printf("sincos(float)     %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 res = test_binary<float>(static_cast<void (*)(float, float*, float*)>([](float x, float* e1, float* e2) {
	 *e1 = erfcf(x);
	 *e2 = expf(-x*x);
	 }), static_cast<void(*)(float,float*,float*)>(&sfmm::detail::erfcexp), 0.01, 5.0);
	 printf("erfcexp(float)    %e %e %e %e %e\n", res.aerr, res.rerr, res.tm1, res.tm2, 1 - res.tm2 / res.tm1);

	 printf("\n");
	 */
	run_tests<PMAX + 1, PMIN> run;
	real theta = 0.5;
	printf("M2L\n");
	run(CC, theta);
	printf("M2P\n");
	run(PC, theta);
	printf("P2L\n");
	run(CP, theta);
	printf("EWALD\n");
	run(EWALD, theta);
}
