#include <stdio.h>
#include <utility>
#include <cmath>
#include <complex>
#include <array>
#include <limits>
#include <climits>
#include "spherical_fmm.hpp"

template<class T>
using complex = std::complex<T>;

using real = double;

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
std::vector<complex<double>> spherical_singular_harmonic2(int P, double x, double y, double z) {
	const double r2 = x * x + y * y + z * z;
	const double r2inv = double(1) / r2;
	complex<double> R = complex<double>(x, y);
	std::vector<complex<double>> O((P + 1) * (P + 1));
	O[cindex(0, 0)] = complex<double>(sqrt(r2inv), double(0));
	R *= r2inv;
	z *= r2inv;
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			O[cindex(m, m)] = O[cindex(m - 1, m - 1)] * R * double(2 * m - 1);
		}
		if (m + 1 <= P) {
			O[cindex(m + 1, m)] = double(2 * m + 1) * z * O[cindex(m, m)];
		}
		for (int n = m + 2; n <= P; n++) {
			O[cindex(n, m)] = (double(2 * n - 1) * z * O[cindex(n - 1, m)] - double((n - 1) * (n - 1) - m * m) * r2inv * O[cindex(n - 2, m)]);
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
	auto G0 = spherical_singular_harmonic<double, P>(x, y, z);
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

void ewald_compute(double& pot, double& fx, double& fy, double& fz, double dx0, double dx1, double dx2) {
	const double cons1 = double(4.0f / sqrtf(M_PI));
	fx = 0.0;
	fy = 0.0;
	fz = 0.0;
	pot = 0.0;
	const auto r2 = sqr(dx0, dx1, dx2);  // 5

	if (r2 > 0.) {
		const double dx = dx0;
		const double dy = dx1;
		const double dz = dx2;
		const double r2 = sqr(dx, dy, dz);
		const double r = sqrt(r2);
		const double rinv = 1. / r;
		const double r2inv = rinv * rinv;
		const double r3inv = r2inv * rinv;
		double exp0 = exp(-4.0 * r2);
		double erf0 = erf(2.0 * r);
		const double expfactor = cons1 * r * exp0;
		const double d0 = erf0 * rinv;
		const double d1 = (expfactor - erf0) * r3inv;
		pot += d0;
		fx -= dx * d1;
		fy -= dy * d1;
		fz -= dz * d1;
		for (int xi = -4; xi <= +4; xi++) {
			for (int yi = -4; yi <= +4; yi++) {
				for (int zi = -4; zi <= +4; zi++) {
					const bool center = sqr(xi, yi, zi) == 0;
					if (center) {
						continue;
					}
					const double dx = dx0 - xi;
					const double dy = dx1 - yi;
					const double dz = dx2 - zi;
					const double r2 = sqr(dx, dy, dz);
					if (r2 < 3.6 * 3.6) {
						const double r = sqrt(r2);
						const double rinv = 1. / r;
						const double r2inv = rinv * rinv;
						const double r3inv = r2inv * rinv;
						double exp0 = exp(-4.0 * r2);
						double erfc0 = erfc(2.0 * r);
						const double expfactor = cons1 * r * exp0;
						const double d0 = -erfc0 * rinv;
						const double d1 = (expfactor + erfc0) * r3inv;
						pot += d0;
						fx -= dx * d1;
						fy -= dy * d1;
						fz -= dz * d1;
					}
				}
			}
		}
		pot += double(M_PI / 4.);
		for (int xi = -4; xi <= +4; xi++) {
			for (int yi = -4; yi <= +4; yi++) {
				for (int zi = -4; zi <= +4; zi++) {
					const double hx = xi;
					const double hy = yi;
					const double hz = zi;
					const double h2 = sqr(hx, hy, hz);
					if (h2 > 0.0 && h2 <= 10) {
						const double hdotx = dx0 * hx + dx1 * hy + dx2 * hz;
						const double omega = double(2.0 * M_PI) * hdotx;
						double c, s;
						sincos(omega, &s, &c);
						const double c0 = -1. / h2 * exp(double(-M_PI * M_PI * 0.25f) * h2) * double(1. / M_PI);
						const double c1 = -s * 2.0 * M_PI * c0;
						pot += c0 * c;
						fx -= c1 * hx;
						fy -= c1 * hy;
						fz -= c1 * hz;
					}
				}
			}
		}
	} else {
		pot += 2.837291f;
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

/*template<class T, int P>
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
							const double phi = T(2.0 * M_PI) * hdotx;
							double Rx, Ry, ax, ay, bx, by;
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
}*/

template<int P>

real test_M2L(test_type type, real theta = 0.5) {
	real err = 0.0;
	int N = 10000;
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_INVALID);
	feenableexcept(FE_OVERFLOW);

	real err2 = 0.0;
	real norm = 0.0;
	double phi, fx, fy, fz;
	for (int i = 0; i < N; i++) {
		if (type == EWALD) {
			real x0, x1, x2, y0, y1, y2, z0, z1, z2;
			random_vector(x0, y0, z0);
			random_unit(x1, y1, z1);
			random_vector(x2, y2, z2);
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
			real M[P * P + 1];
			real L[(P + 1) * (P + 1) + 1];
			real L0[(P + 1) * (P + 1) + 1];
			real L2[4] = { 0, 0, 0, 0 };
			for (int n = 0; n <= (P > 2 ? P * P : (P * P - 1)); n++) {
				M[n] = (0);
			}
			P2M_double(P, M, -x0 * f0, -y0 * f1, -z0 * f2, FMM_CALC_POT);
			for (int n = 0; n <= (P > 2 ? P * P : (P * P - 1)); n++) {
				M[n] *= (0.5);
			}
			M2M_double(P, M, -real(x0) * (1 - f0), -real(y0) * (1 - f1), -real(z0) * (1 - f2), FMM_CALC_POT);
			for (int n = 0; n <= (P > 1 ? (P + 1) * (P + 1) : (P + 1) * (P + 1) - 1); n++) {
				L[n] = (0);
				L0[n] = (0);
			}
			//	g0 = g1 = g2 = 0.0;
			//		M2L_ewald3<real,P>( L0, M, x1, y1, z1);
			M2L_ewald_double(P, L, M, x1, y1, z1, FMM_CALC_POT);
			L2L_double(P, L, x2 * g0, y2 * g1, z2 * g2, FMM_CALC_POT);
			L2P_double(P, L2, L, x2 * (1 - g0), y2 * (1 - g1), z2 * (1 - g2), FMM_CALC_POT);
			ewald_compute(phi, fx, fy, fz, (-x2 + x1) + x0, (-y2 + y1) + y0, (-z2 + z1) + z0);
			for (int l = 0; l <= P; l++) {
				for (int m = -l; m <= l; m++) {
					int i = l * (l + 1) + m;
					//				printf( "%e %e %e %i %i %e %e %e\n", x1, y1, z1, l, m, L[i], L0[i], L[i] - L0[i]);
				}
			}
			fx *= 0.5;
			fy *= 0.5;
			fz *= 0.5;
			phi *= 0.5;
			//		printf( "%e %e %e\n", fx, fy, fz);
			//	abort();
			//		printf( "%e %e %e\n", phi, L0[0], phi/ L0[0]);
			err += fabs((phi - L2[0]));
			norm += fabs(phi);

		} else {
			real x0, x1, x2, y0, y1, y2, z0, z1, z2;
			random_vector(x0, y0, z0);
			random_unit(x1, y1, z1);
			random_vector(x2, y2, z2);
			if (type == CP) {
				x0 = y0 = z0 = 0;
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
			double f0 = rand1();
			double f1 = rand1();
			double f2 = rand1();
			double g0 = rand1();
			double g1 = rand1();
			double g2 = rand1();
			real M[P * P + 1];
			real L[(P + 1) * (P + 1) + 1];
			for (int n = 0; n <= (P > 2 ? P * P : (P * P - 1)); n++) {
				M[n] = (0);
			}
			P2M_double(P, M, -x0 * f0, -y0 * f1, -z0 * f2, FMM_CALC_POT);

			M2M_double(P, M, -real(x0) * (1 - f0), -real(y0) * (1 - f1), -real(z0) * (1 - f2), FMM_CALC_POT);
			for (int n = 0; n <= (P > 1 ? (P + 1) * (P + 1) : (P + 1) * (P + 1) - 1); n++) {
				L[n] = (0);
			}
			real L2[4];
			for (int l = 0; l < 4; l++) {
				L2[l] = 0.0;
			}
			if (type == CC) {
				M2L_double(P, L, M, x1, y1, z1, FMM_CALC_POT);
				L2L_double(P, L, x2 * g0, y2 * g1, z2 * g2, FMM_CALC_POT);
				L2P_double(P, L2, L, x2 * (1 - g0), y2 * (1 - g1), z2 * (1 - g2), FMM_CALC_POT);
			} else if (type == PC) {
				M2P_double(P, L2, M, x1, y1, z1, FMM_CALC_POT);
			} else if (type == CP) {
				P2L_double(P, L, 1.0, x1, y1, z1, FMM_CALC_POT);
				L2L_double(P, L, x2 * g0, y2 * g1, z2 * g2, FMM_CALC_POT);
				L2P_double(P, L2, L, x2 * (1 - g0), y2 * (1 - g1), z2 * (1 - g2), FMM_CALC_POT);
			}
			const real dx = (x2 + x1) - x0;
			const real dy = (y2 + y1) - y0;
			const real dz = (z2 + z1) - z0;
			const real r = sqrt(sqr(dx, dy, dz));
			const real phi = 1.0 / r;
			err += fabs((L2[0] - phi));
			norm += fabs(phi);
		}
	}
	err /= norm;
	return err;
}

template<int NMAX, int N = 3>
struct run_tests {
	void operator()(test_type type) {
		auto a = test_M2L<N>(type);
		printf("%i %e\n", N, a);
		run_tests<NMAX, N + 1> run;
		run(type);
	}
};

template<int NMAX>
struct run_tests<NMAX, NMAX> {
	void operator()(test_type type) {

	}
};

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}

template<int Nmax>
struct const_factorial {
	double y[Nmax + 1];
	constexpr const_factorial() :
			y() {
		y[0] = 1.0;
		for (int i = 1; i <= Nmax; i++) {
			y[i] = i * y[i - 1];
		}
	}
	constexpr double operator()(int n) const {
		return y[n];
	}
};

constexpr double const_exp(double x0) {
	constexpr int N = 12;
	if (x0 < 0.0) {
		return 1.0 / const_exp(-x0);
	} else {
		constexpr const_factorial<N> factorial;
		int k = x0 / 0.6931471805599453094172 + 0.5;
		double x = x0 - k * 0.6931471805599453094172;
		double y = 1.0 / factorial(N);
		for (int i = N - 1; i >= 0; i--) {
			y = y * x + 1.0 / factorial(i);
		}
		return y * (1 << k);
	}
}

constexpr int ewald_real_size() {
	int i = 0;
	for (int xi = -4; xi <= +4; xi++) {
		for (int yi = -4; yi <= +4; yi++) {
			for (int zi = -4; zi <= +4; zi++) {
				const int r2 = xi * xi + yi * yi + zi * zi;
				if (r2 < 3.1 * 3.1 && r2 > 0) {
					i++;
				}
			}
		}
	}
	return i;
}

constexpr int ewald_four_size() {
	int i = 0;
	for (int xi = -2; xi <= +2; xi++) {
		for (int yi = -2; yi <= +2; yi++) {
			for (int zi = -2; zi <= +2; zi++) {
				const int r2 = xi * xi + yi * yi + zi * zi;
				if (r2 <= 8 && r2 > 0) {
					i++;
				}
			}
		}
	}
	return i;
}

template<class T, int P>
constexpr T const_S(int n, int m0, T x, T y, T z) {
	const T r2 = x * x + y * y + z * z;
	const T r2inv = T(1) / r2;
	const T m = m0 >= 0 ? m0 : -m0;
	T Ox = T(1), Oy = T(0), Oxm1 = T(0), Oym1 = T(0), Oxm2 = T(0);
	x *= r2inv;
	y *= r2inv;
	Oxm1 = Ox;
	Oym1 = Oy;
	for (int m1 = 1; m1 <= m; m1++) {
		const T tmp = Ox;
		Ox = (tmp * x - Oy * y) * T(2 * m1 - 1);
		Oy = (tmp * y + Oy * x) * T(2 * m1 - 1);
		Oxm1 = Ox;
		Oym1 = Oy;
	}
	if (m0 < 0) {
		Oxm1 = Oym1;
	}
	for (int n1 = m + 1; n1 <= n; n1++) {
		Ox = T(2 * n - 1) * z * Oxm1 - T((n - 1) * (n - 1) - m * m) * r2inv * Oxm2;
		Oxm2 = Oxm1;
		Oxm1 = Ox;
	}
	return Ox;
}


int main() {
//	return 0;
	run_tests<13, 3> run;
	printf("M2L\n");
	run(CC);
	printf("M2P\n");
	run(PC);
	printf("P2L\n");
	run(CP);
	printf("EWALD\n");
	run(EWALD);
}
