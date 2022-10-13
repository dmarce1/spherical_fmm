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
		for (int xi = -3; xi <= +3; xi++) {
			for (int yi = -3; yi <= +3; yi++) {
				for (int zi = -3; zi <= +3; zi++) {
					const bool center = sqr(xi, yi, zi) == 0;
					if (center || sqr(xi, yi, zi) > 12) {
						continue;
					}
					const double dx = dx0 - xi;
					const double dy = dx1 - yi;
					const double dz = dx2 - zi;
					const double r2 = sqr(dx, dy, dz);
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
		pot += double(M_PI / 4.);
		for (int xi = -3; xi <= +3; xi++) {
			for (int yi = -3; yi <= +3; yi++) {
				for (int zi = -3; zi <= +3; zi++) {
					const double hx = xi;
					const double hy = yi;
					const double hz = zi;
					const double h2 = sqr(hx, hy, hz);
					if (h2 > 0.0 && h2 <= 13) {
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
	int N = 4000;
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

float sqrt_float(float x) {
	int e = (int&) x;
	e += (~0x3F100000 + 1);
	e >>= 1;
	e += 0x3F100000;
	float y = (float&) e;
	y = 0.5f * (y + x / y);
	y = 0.5f * (y + x / y);
	y = 0.5f * (y + x / y);
	y = 0.5f * (y + x / y);
	return y;
}

double sqrt_double(double x) {
	long long e = (long long&) x;
	e += (~(long long) 0x3FF0000000000000 + 1);
	e >>= 1;
	e += (long long) 0x3FF0000000000000;
	double y = (double&) e;
	y = 0.5 * (y + x / y);
	y = 0.5 * (y + x / y);
	y = 0.5 * (y + x / y);
	y = 0.5 * (y + x / y);
	return y;
}

void sincos_float(float x, float* s, float* c) {
	constexpr const_factorial<11> factorial;
	int ssgn = (((int&) x & 0x80000000) >> 30) - 1;
	int j = ((int&) x & 0x7FFFFFFF);
	x = (float&) j;
	int i = x * float(1.0 / M_PI);
	x -= i * float(M_PI);
	x -= float(M_PI * 0.5);
	float x2 = x * x;
	*c = -1.0 / factorial(11);
	*s = -1.0 / factorial(10);
	*c = fmaf(*c, x2, 1.0 / factorial(9));
	*s = fmaf(*s, x2, 1.0 / factorial(8));
	*c = fmaf(*c, x2, -1.0 / factorial(7));
	*s = fmaf(*s, x2, -1.0 / factorial(6));
	*c = fmaf(*c, x2, 1.0 / factorial(5));
	*s = fmaf(*s, x2, 1.0 / factorial(4));
	*c = fmaf(*c, x2, -1.0 / factorial(3));
	*s = fmaf(*s, x2, -1.0 / factorial(2));
	*c = fmaf(*c, x2, 1.0 / factorial(1));
	*s = fmaf(*s, x2, 1.0 / factorial(0));
	*c *= x;
	int k = (((i & 1) << 1) - 1);
	*s *= ssgn * k;
	*c *= k;
}

void sincos_double(double x, double* s, double* c) {
	constexpr const_factorial<21> factorial;
	long long int ssgn = (((long long int&) x & 0x8000000000000000LL) >> 62LL) - 1LL;
	long long int j = ((long long int&) x & 0x7FFFFFFFFFFFFFFFLL);
	x = (double&) j;
	long long int i = x * double(1.0 / M_PI);
	x -= i * M_PI;
	x -= double(M_PI * 0.5);
	double x2 = x * x;
	*c = 1.0 / factorial(21);
	*s = 1.0 / factorial(20);
	*c = fma(*c, x2, -1.0 / factorial(19));
	*s = fma(*s, x2, -1.0 / factorial(18));
	*c = fma(*c, x2, 1.0 / factorial(17));
	*s = fma(*s, x2, 1.0 / factorial(16));
	*c = fma(*c, x2, -1.0 / factorial(15));
	*s = fma(*s, x2, -1.0 / factorial(14));
	*c = fma(*c, x2, 1.0 / factorial(13));
	*s = fma(*s, x2, 1.0 / factorial(12));
	*c = fma(*c, x2, -1.0 / factorial(11));
	*s = fma(*s, x2, -1.0 / factorial(10));
	*c = fma(*c, x2, 1.0 / factorial(9));
	*s = fma(*s, x2, 1.0 / factorial(8));
	*c = fma(*c, x2, -1.0 / factorial(7));
	*s = fma(*s, x2, -1.0 / factorial(6));
	*c = fma(*c, x2, 1.0 / factorial(5));
	*s = fma(*s, x2, 1.0 / factorial(4));
	*c = fma(*c, x2, -1.0 / factorial(3));
	*s = fma(*s, x2, -1.0 / factorial(2));
	*c = fma(*c, x2, 1.0 / factorial(1));
	*s = fma(*s, x2, 1.0 / factorial(0));
	*c *= x;
	long long int k = (((i & 1) << 1) - 1);
	*s *= ssgn * k;
	*c *= k;
}

constexpr double dfactorial(int n) {
	if (n == 1 || n == 0) {
		return 1;
	} else {
		return n * dfactorial(n - 2);
	}
}

constexpr double factorial(int n) {
	if (n == 0) {
		return 1;
	} else {
		return n * factorial(n - 1);
	}
}

constexpr double cnk(int n, int k) {
	return factorial(n) * factorial(2 * n + 1 - k) * 0.5 / (factorial(n - k) * factorial(k + 1) * factorial(2 * n + 1));
}

constexpr double pki(int k, int i) {
	return nonepow<double>(i + k) * factorial(k) / (factorial(i) * factorial(k - 2 * i)) * pow(2, (k - 2 * i));
}

using polynomial = std::vector<double>;

polynomial poly_mult(polynomial a, polynomial b) {
	polynomial c(a.size() + b.size(), 0.0);
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < b.size(); j++) {
			c[i + j] += a[i] * b[j];
		}
	}
	return c;
}

polynomial poly_add(polynomial a, polynomial b) {
	polynomial c(std::max(a.size(), b.size()), 0.0);
	for (int i = 0; i < c.size(); i++) {
		if (i >= a.size()) {
			c[i] = b[i];
		} else if (i >= b.size()) {
			c[i] = a[i];
		} else {
			c[i] = a[i] + b[i];
		}
	}
	return c;
}

double poly_eval(polynomial a, double x) {
	double y = 0.0;
	y = a.back();
	for (int i = a.size() - 2; i >= 0; i--) {
		y = fma(x, y, double(a[i]));
	}
	return y;
}

polynomial pkx(int k) {
	polynomial y;
	for (int i = 0; i <= k / 2; i++) {
		if (y.size() <= k - 2 * i) {
			y.resize(k - 2 * i + 1, 0.0);
		}
		y[k - 2 * i] += pki(k, i);
	}
	return y;
}

double pochhammer(double a, int n) {
	if (n == 0) {
		return 1.0;
	} else {
		return (a + n - 1) * pochhammer(a, n - 1);
	}
}

double gamma(double x) {
	if (x > 1.0) {
		return tgamma(x);
	} else {
		return gamma(1.0 + x) / x;
	}
}

double hypergeo(double a, int i) {
	double y = 0.0;
	int done = 0;
	int n = i / 2;
	const auto g1 = gamma(0.5 + n);
	const auto g2 = gamma(1.0 + n);
	const auto g3 = gamma(1.0 - 0.5 * i + n);
	const auto g4 = gamma(1.5 - 0.5 * i + n);
	auto z = (g1 / g3) * (g2 / g4) / (gamma(0.5) * gamma(1.0)) * pow(-a * a, n) / factorial(n);
	y += z;
	for (++n;; n++) {
		z *= (n - 0.5);
		z /= (n - 0.5 * i);
		z /= (n - 0.5 * i + 0.5);
		z *= -a * a;
		y += z;
		if (y) {
			if (fabs(z / y) < (10 - done) * std::numeric_limits<double>::epsilon()) {
				done++;
				if (done == 10) {
					break;
				}
			} else {
				done = 0;
			}
		}
	}
//	printf( "\n");
	return y;
}

void erfc_double_create(int i) {
	std::vector<double> c0(1, erfc(i));
	double err;
	double x0 = i;
	double xmax = sqrt(i + 0.75);
	double xmin = sqrt(std::max(0.0, i - 0.75));
	xmax -= x0;
	xmin -= x0;
	int n = 1;
	do {
		double c1, c2;
		if (i == 0) {
			c1 = 0.0;
			c2 = nonepow((n + 1) / 2) / (2 * ((n + 1) / 2) + 1) / factorial((n + 1) / 2);
		} else {
			c2 = -(1 << n) * pow(x0, 1 - n) * hypergeo(x0, n) / factorial(n);
			c1 = -(1 << (n + 1)) * pow(x0, 1 - (n + 1)) * hypergeo(x0, (n + 1)) / factorial(n + 1);
		}
		err = fabs((c2 + c1 * xmax) * pow(xmax, n));
		err = std::max(err, fabs((c2 + c1 * xmin) * pow(xmin, n)));

		if (err > std::numeric_limits<double>::epsilon()) {
			c0.push_back(c2);
			c0.push_back(c1);
		}
		n += 2;
	} while (err > std::numeric_limits<double>::epsilon());
}



double erf_double(double x);

static int N1 = 58;
static int N2 = 15;

double erfc1_double(double x) {
	constexpr int N = 51;
	constexpr double c0[N] = { 1.5417257900280018852e-8, -1.2698234671866558268e-7, 5.0792938687466233073e-7, -1.3121509160928776877e-6,
			2.4549920365608679319e-6, -3.5343419836695253847e-6, 4.0577914351431357311e-6, -3.7959659297660776487e-6, 2.9264391936639771349e-6,
			-1.8631747969134645771e-6, 9.7028398087939794881e-7, -4.0077792841735885016e-7, 1.2017256123590621089e-7, -1.7432381111955779115e-8,
			-5.8855705165843298536e-9, 5.2972657465156914314e-9, -1.9619829796563405661e-9, 3.3902885661963547618e-10, 5.449519885206690639e-11,
			-5.6649970057793101492e-11, 1.7497284973974059992e-11, -1.5401588896183612903e-12, -9.5485866992008160798e-13, 4.5996403017767892727e-13,
			-7.720942042791545628e-14, -1.0556894443355771748e-14, 8.9498939526324557708e-15, -1.8999048717660430384e-15, -7.277120798336296798e-17,
			1.4642316953650200603e-16, -3.4362054810804757693e-17, -2.6414051454190806842e-19, 2.1443852179986680066e-18, -5.043427120322544597e-19,
			-3.6489999780744875762e-21, 2.8806005872407609211e-20, -6.2044044807659249396e-21, -1.7233627673269238901e-22, 3.5400331474254875021e-22,
			-6.4010879354598909094e-23, -4.4441394626915675505e-24, 3.9115690351951320278e-24, -5.3859668207240296229e-25, -7.7397660968811423558e-26,
			3.798462825970030578e-26, -3.3911062141902652569e-27, -1.025047850118420974e-27, 3.1564161337682830322e-28, -1.0805338838117337057e-29,
			-1.085079147261258634e-29, 2.1595195043768972257e-30 };
	x -= 4.0;
	double y = c0[40 - 1];
	for (int n = 40 - 2; n >= 0; n--) {
		y = fma(y, x, c0[n]);
	}
	return y;
}

double erfc02_double(double x) {
	constexpr int N = 60;
	constexpr double c0[N + 1] = { 0.15729920705028513066, -0.41510749742059470334, 0.41510749742059470334, -0.13836916580686490111, -0.069184582903432450557,
			0.069184582903432450557, -0.0046123055268954967038, -0.01515471815979948917, 0.0047770307242846215861, 0.0018851883701199847638,
			-0.0012262875805634852347, -0.000085523991371727464132, 0.00020005514713217961292, -0.00001871663923714299038, -0.000023707092917618642639,
			5.4782439136144749702e-6, 2.0810470178536989366e-6, -8.4904713963144343778e-7, -1.2328726086225257871e-7, 9.7385801574591139527e-8,
			1.9412656084384987673e-9, -8.9959787718381029827e-9, 6.4974130753173241251e-10, 6.9020255115771561091e-10, -1.0930785305190424683e-10,
			-4.4170900677939190423e-11, 1.1469726123674405183e-11, 2.2964662043673653157e-12, -9.5295626119961216479e-13, -8.6999537449087987462e-14,
			6.7139682527845269542e-14, 1.0941851832004162369e-15, -4.1292544687793769952e-15, 1.8601591348811962216e-16, 2.2459468423499485974e-16,
			-2.315083093966012229e-17, -1.0834825684288445954e-17, 1.8023015127965290308e-18, 4.5998373920471192063e-19, -1.1358237255499409868e-19,
			-1.673034558991574761e-20, 6.2182028698065850645e-21, 4.8114692907614392188e-22, -3.0471150463263347992e-22, -7.5111800443919715459e-24,
			1.3568774364804531701e-23, -2.7063084489174606623e-25, -5.5332638996409214387e-25, 3.4091630490543129263e-26, 2.0722739850991718161e-26,
			-2.1647449112201341996e-27, -7.1151137814712743521e-28, 1.0899270425830332517e-28, 2.2220217463041891355e-29, -4.7835724297505193226e-30,
			-6.1909670998969171609e-31, 1.8984625341296174533e-31, 1.4673514322247610681e-32, -6.9375562658765708089e-33, -2.5365987898730140379e-34,
			2.3578768151474118064e-34 };
	x -= 1.0;
	double y = c0[N];
	for (int n = N - 1; n >= 0; n--) {
		y = fma(y, x, c0[n]);
	}
	return y;
}

double erfc24_double(double x) {
	constexpr int N = 60;
	constexpr double c0[N + 1] = { 0.000022090496998585441373, -0.00013925305194674785389, 0.00041775915584024356167, -0.00078910062769823783871,
			0.0010443978896006089042, -0.0010165472792112593334, 0.00073804117531776362562, -0.00039057165522206898067, 0.00013477706099131667287,
			-0.000013906885478808813451, -0.000015616235111171009329, 0.000010793618593534720017, -3.0307130678020555646e-6, -1.2338633446163999819e-7,
			4.5253432810580908467e-7, -1.6573732792802534838e-7, 9.3558263606651124303e-9, 1.4977795982415109037e-8, -5.9709857312013879724e-9,
			3.9655378820946255485e-10, 4.4670619596676640987e-10, -1.6350901778088463397e-10, 5.917507363032050835e-12, 1.2028197183551233037e-11,
			-3.4787346654048557896e-12, -8.7265464375095810005e-14, 2.77029359377996076e-13, -5.5346590946171791105e-14, -7.1949457068465155992e-15,
			5.1692940761619351482e-15, -5.7073587318249637038e-16, -2.1192107445649773325e-16, 7.425551637082496029e-17, -1.0586671656207698049e-18,
			-4.0487851594829634913e-18, 7.5279344997100450365e-19, 9.3040290754706802889e-20, -5.4648832057798036546e-20, 3.8642530458693273864e-21,
			2.1342536287576572607e-21, -5.0839652603548761562e-22, -2.7107693167426974747e-23, 2.7491413822314776444e-23, -2.6052078867631847922e-24,
			-8.6529340712977777216e-25, 2.2852794835550263837e-25, 6.9774269427220925551e-27, -1.0403924693888314061e-26, 1.0159501263058830009e-27,
			2.9140105648359150401e-28, -7.4776784788383947047e-29, -2.4016737642654052075e-30, 3.0967536390285100672e-30, -2.6168955421691720239e-31,
			-8.3453913423474974363e-32, 1.8443825150235392305e-32, 9.5018193186935641895e-34, -7.356146289580190822e-34, 4.3907993567536178823e-35,
			2.0040936861400229439e-35, -3.4428866956977056398e-36 };
	x -= 3.0;
	double y = c0[N];
	for (int n = N - 1; n >= 0; n--) {
		y = fma(y, x, c0[n]);
	}
	return y;
}

double erfc46_double(double x) {
	constexpr int N = 60;
	constexpr double c0[N + 1] = { 1.5374597944280348502e-12, -1.5670866531017335308e-11, 7.8354332655086676541e-11, -2.5595748667328314337e-10,
			6.1377560579817896624e-10, -1.1507639655943729895e-9, 1.7542664477777739248e-9, -2.232103505017207276e-9, 2.4142151424619861111e-9,
			-2.2484411434266387087e-9, 1.8192473403222856223e-9, -1.2859344859140824134e-9, 7.9596853518260115928e-10, -4.3093375340898926936e-10,
			2.0284694010322075381e-10, -8.1877590599129450144e-11, 2.7508017779080151729e-11, -7.1505703186725763687e-12, 1.0958836119077553182e-12,
			1.3409456342535567037e-13, -1.7086783441972833901e-13, 6.9233270175671791258e-14, -1.6675916363857018623e-14, 1.5037633451982783876e-15,
			7.0267164966323331565e-16, -3.9635718299716133598e-16, 1.0055547163916174591e-16, -9.012198684222203901e-18, -3.6978715350481689606e-18,
			1.8744615255683329059e-18, -3.8679659362312538038e-19, 7.8711931225530167407e-21, 2.093520740866541092e-20, -6.8061366897454720951e-21,
			8.0763978026864482396e-22, 1.4672898843013606566e-22, -8.4344961118107516674e-23, 1.508495256186875481e-23, 3.4950551394900563364e-25,
			-8.428465485012887967e-25, 1.9368444542011423241e-25, -7.1535045030153456223e-27, -7.2949005733872513864e-27, 2.021287713217001065e-27,
			-1.3550849287462015934e-28, -5.768030629890364721e-29, 1.8299944544673050854e-29, -1.4924823883664547381e-30, -4.3534050335890379062e-31,
			1.4849353151306146466e-31, -1.2640466170998103585e-32, -3.2282874520709693114e-33, 1.0974635059034708549e-33, -8.7589152016896217273e-35,
			-2.3659662489519906974e-35, 7.4278342956672554871e-36, -4.9677445251585732622e-37, -1.6881832052459837121e-37, 4.5936231512642798586e-38,
			-2.161813482620867581e-39, -1.1449527968136307241e-39 };
	x -= 5.0;
	double y = c0[N];
	for (int n = N - 1; n >= 0; n--) {
		y = fma(y, x, c0[n]);
	}
	return y;
}

#define List(...) { __VA_ARGS__}

double erfc_double(double x) {
	constexpr int N = 39;
	constexpr double c0[5][N + 1] = List(List(0.15729920705028513066,-0.41510749742059470334,0.41510749742059470334,
					-0.13836916580686490111,-0.069184582903432450557,0.069184582903432450557,
					-0.0046123055268954967038,-0.01515471815979948917,0.0047770307242846215861,
					0.0018851883701199847638,-0.0012262875805634852347,-0.000085523991371727464132,
					0.00020005514713217961292,-0.00001871663923714299038,-0.000023707092917618642639,
					5.4782439136144749702e-6,2.0810470178536989366e-6,-8.4904713963144343778e-7,
					-1.2328726086225257871e-7,9.7385801574591139527e-8,1.9412656084384987673e-9,
					-8.9959787718381029827e-9,6.4974130753173241251e-10,6.9020255115771561091e-10,
					-1.0930785305190424683e-10,-4.4170900677939190423e-11,1.1469726123674405183e-11,
					2.2964662043673653157e-12,-9.5295626119961216479e-13,-8.6999537449087987462e-14,
					6.7139682527845269542e-14,1.0941851832004162369e-15,-4.1292544687793769952e-15,
					1.8601591348811962216e-16,2.2459468423499485974e-16,-2.315083093966012229e-17,
					-1.0834825684288445954e-17,1.8023015127965290308e-18,4.5998373920471192063e-19,
					-1.1358237255499409868e-19),List(0.000022090496998585441373,
					-0.00013925305194674785389,0.00041775915584024356167,-0.00078910062769823783871,
					0.0010443978896006089042,-0.0010165472792112593334,0.00073804117531776362562,
					-0.00039057165522206898067,0.00013477706099131667287,-0.000013906885478808813451,
					-0.000015616235111171009329,0.000010793618593534720017,-3.0307130678020555646e-6,
					-1.2338633446163999819e-7,4.5253432810580908467e-7,-1.6573732792802534838e-7,
					9.3558263606651124303e-9,1.4977795982415109037e-8,-5.9709857312013879724e-9,
					3.9655378820946255485e-10,4.4670619596676640987e-10,-1.6350901778088463397e-10,
					5.917507363032050835e-12,1.2028197183551233037e-11,-3.4787346654048557896e-12,
					-8.7265464375095810005e-14,2.77029359377996076e-13,-5.5346590946171791105e-14,
					-7.1949457068465155992e-15,5.1692940761619351482e-15,-5.7073587318249637038e-16,
					-2.1192107445649773325e-16,7.425551637082496029e-17,-1.0586671656207698049e-18,
					-4.0487851594829634913e-18,7.5279344997100450365e-19,9.3040290754706802889e-20,
					-5.4648832057798036546e-20,3.8642530458693273864e-21,2.1342536287576572607e-21),
			List(1.5374597944280348502e-12,-1.5670866531017335308e-11,7.8354332655086676541e-11,
					-2.5595748667328314337e-10,6.1377560579817896624e-10,-1.1507639655943729895e-9,
					1.7542664477777739248e-9,-2.232103505017207276e-9,2.4142151424619861111e-9,
					-2.2484411434266387087e-9,1.8192473403222856223e-9,-1.2859344859140824134e-9,
					7.9596853518260115928e-10,-4.3093375340898926936e-10,2.0284694010322075381e-10,
					-8.1877590599129450144e-11,2.7508017779080151729e-11,-7.1505703186725763687e-12,
					1.0958836119077553182e-12,1.3409456342535567037e-13,-1.7086783441972833901e-13,
					6.9233270175671791258e-14,-1.6675916363857018623e-14,1.5037633451982783876e-15,
					7.0267164966323331565e-16,-3.9635718299716133598e-16,1.0055547163916174591e-16,
					-9.012198684222203901e-18,-3.6978715350481689606e-18,1.8744615255683329059e-18,
					-3.8679659362312538038e-19,7.8711931225530167407e-21,2.093520740866541092e-20,
					-6.8061366897454720951e-21,8.0763978026864482396e-22,1.4672898843013606566e-22,
					-8.4344961118107516674e-23,1.508495256186875481e-23,3.4950551394900563364e-25,
					-8.428465485012887967e-25),List(4.1838256077794143986e-23,-5.91596295800306936e-22,
					4.141174070602148552e-21,-1.9128280230876590931e-20,6.5568589451200685407e-20,
					-1.7785356639409894186e-19,3.9750669773257734823e-19,-7.5266730822846447221e-19,
					1.231987782742831966e-18,-1.7700734632222038553e-18,2.259082798245693048e-18,
					-2.5855479037854305211e-18,2.6741872819548669643e-18,-2.5152654454175524265e-18,
					2.1626253642806468928e-18,-1.7070365229435734662e-18,1.2413506650762179787e-18,
					-8.3401269591457949725e-19,5.1886209263147367655e-19,-2.9940589380712182335e-19,
					1.6042876952095092804e-19,-7.9863408336180072771e-20,3.6932232186144814444e-20,
					-1.5851510599235256507e-20,6.3028409361655448543e-21,-2.3143084449780021196e-21,
					7.8072552431746859822e-22,-2.3998385840834538999e-22,6.6291231764346812579e-23,
					-1.6043145932972043684e-23,3.2197796206703772773e-24,-4.535537382464300245e-25,
					3.685025361620961693e-27,2.5065757963326203296e-26,-1.0531391980570808798e-26,
					2.8223550900606517397e-27,-5.2922169797690821761e-28,5.1924083690777867792e-29,
					7.971043316155374219e-30,-5.4541006999407701087e-30),
			List(4.1370317465138102381e-37,-7.4920734282459744915e-36,6.7428660854213770423e-35,
					-4.0207460731586729771e-34,1.7868595126366649162e-33,-6.3120718632972335091e-33,
					1.845971971985525655e-32,-4.5964976455033223149e-32,9.9465542798141482825e-32,
					-1.8999345128558206115e-31,3.2430544914993366868e-31,-4.995918065804325751e-31,
					7.002505206055073977e-31,-8.9912238914115436699e-31,1.0636737723324062875e-30,
					-1.1650886119528398615e-30,1.1866294150081641107e-30,-1.1279287248667870148e-30,
					1.0038367598986129901e-30,-8.3886998117450248627e-31,6.5988265843507837542e-31,
					-4.8971594702856457303e-31,3.4354398623245513764e-31,-2.2821215172500498505e-31,
					1.4377517286218122782e-31,-8.6021859495186768509e-32,4.8936351500691302914e-32,
					-2.6497321264336454898e-32,1.3667999810094668424e-32,-6.7214489606078132036e-33,
					3.1530900782436517937e-33,-1.4116393575659341966e-33,6.0333604518868162707e-34,
					-2.4621204721552338732e-34,9.5932664130075996376e-35,-3.5681340614606694302e-35,
					1.266335192568019814e-35,-4.2854040576713962415e-36,1.3814493341110779735e-36,
					-4.2361096104203374629e-37));
	double del = 0.25;
	const int j = std::max(int(x + del) / 2, 0);
	double x0 = 2 * j + 1;
	x -= x0;
	double y = c0[j][N];
	for (int n = N - 1; n >= 0; n--) {
		y = fma(y, x, c0[j][n]);
	}
	return y;

	/*	double y;
	 double q = 1.0 / (2.0 * x * x);
	 const int N = N2;
	 y = nonepow<double>(N) * dfactorial(2 * N - 1);
	 for (int n = N - 1; n >= 1; n--) {
	 y = fma(y, q, nonepow<double>(n) * dfactorial(2 * n - 1));
	 }
	 y *= q;
	 //	y /= pow((2.0 * x * x), N);
	 return exp(-x * x) / x / sqrt(M_PI) * (1.0 + y);*/
}

double erf_double(double x) {
	const int N = N1;
	double y;
	double q = (2.0 * x * x);
	y = 1.0 / dfactorial(2 * N + 1);
	for (int n = N - 1; n >= 0; n--) {
		y = fma(y, q, 1.0 / dfactorial(2 * n + 1));
	}
	return exp(-x * x) * x * 2.0 / sqrt(M_PI) * y;
}

double erf_double2(double x) {
	const int N = N1;
	double y;
	double q = x * x;
	y = nonepow(N) / factorial(N) / (2 * N + 1);
	for (int n = N - 1; n >= 0; n--) {
		y = fma(y, q, nonepow(n) / factorial(n) / (2 * n + 1));
	}
	y *= x;
	return 2.0 / sqrt(M_PI) * y;
}

float erfc_float(float x) {
	float erfc0;
	float exp0 = expf(-x * x);
	float t = float(1) / (float(1) + 0.3275911 * x);
	erfc0 = float(1.061405429);
	erfc0 = fmaf(erfc0, t, float(-1.453152027));
	erfc0 = fmaf(erfc0, t, float(1.421413741));
	erfc0 = fmaf(erfc0, t, float(-0.284496736));
	erfc0 = fmaf(erfc0, t, float(0.254829592));
	erfc0 *= t;
	erfc0 *= exp0;
	return erfc0;
}

double exp_double(double x0) {
	constexpr int N = 18;
	constexpr const_factorial<N> factorial;
	long long int k = x0 / 0.6931471805599453094172 + 0.5;
	k -= x0 < 0.0;
	double x = x0 - k * 0.6931471805599453094172;
	double y = 1.0 / factorial(N);
	for (int i = N - 1; i >= 0; i--) {
		y = y * x + 1.0 / factorial(i);
	}
	k = (k + (long long) 1023) << (long long) 52;
	return y * (double&) (k);
}

double exp_float(double x0) {
	constexpr int N = 7;
	constexpr const_factorial<N> factorial;
	int k = x0 / 0.6931471805599453094172 + 0.5;
	k -= x0 < 0.0;
	double x = x0 - k * 0.6931471805599453094172;
	double y = 1.0 / factorial(N);
	for (int i = N - 1; i >= 0; i--) {
		y = y * x + 1.0 / factorial(i);
	}
	k = (k + 127) << 23;
	return y * (float&) (k);
}

float rsqrt_float(float x) {
	int i = *((int*) &x);
	i >>= 1;
	i = 0x5F3759DF - i;
	float y = *((float*) &i);
	y *= fmaf(-0.5, x * y * y, 1.5);
	y *= fmaf(-0.5, x * y * y, 1.5);
	y *= fmaf(-0.5, x * y * y, 1.5);
	return y;
}

double rsqrt_double(double x) {
	long long i = *((long long*) &x);
	i >>= 1;
	i = 0x5FE6EB50C7B537A9 - i;
	double y = *((double*) &i);
	y *= fma(-0.5, x * y * y, 1.5);
	y *= fma(-0.5, x * y * y, 1.5);
	y *= fma(-0.5, x * y * y, 1.5);
	y *= fma(-0.5, x * y * y, 1.5);
	return y;
}
double erfc2_double(double x) {
	constexpr int N = 45;
	constexpr double a = (2);
	static double c0[N + 1];
	static bool init = false;
	double b = 1;
	if (!init) {
		double q[N + 1];
		init = true;
		c0[0] = (a+b) * exp(a * a) * erfc(a);
		q[0] = exp(a * a) * erfc(a);
		q[1] = -2.0 / sqrt(M_PI) + 2 * a * exp(a * a) * erfc(a);
		for (int n = 2; n <= N; n++) {
			q[n] = 2 * (a * q[n - 1] + q[n - 2]) / n;
		}
		for (int n = 1; n <= N; n++) {
			c0[n] = q[n - 1] + (a+b) * q[n];
			printf("%i %e\n", n, c0[n]);
		}
	}
	x -= a;
	double y = c0[N];
	for (int n = N - 1; n >= 0; n--) {
		y = fma(y, x, c0[n]);
	}
	return y * exp(-(x+a) * (x+a)) / (a+x+b);
}
int main() {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_OVERFLOW);
	feenableexcept(FE_INVALID);

	/*	erfc_double_create(0);
	 return 0;*/
	double err_min = 1e99, shft_min, shft;
	//for (N1 = 3; N1 <= 100; N1++) {
	//	for (N2 = 3; N2 <= 100; N2++) {
	double emax = 0.0;
	for (double r = 0.0; r < 4.0; r += 0.05) {
		double e1;
		e1 = erfc2_double(r);
		double e0 = erfc(r);
		double err1 = fabs((e1 - e0));
		emax = std::max(emax, err1);
		printf("%e %e %e %e\n", r, e0, e1, fabs((e1 - e0)));
	}
	printf("%e\n", emax);

//	}
//}
//	printf("%i %i %e\n", N1_min, N2_min, err_min);
	//return 0;
	run_tests<21, 3> run;
	printf("M2L\n");
	run(CC);
	printf("M2P\n");
	run(PC);
	printf("P2L\n");
	run(CP);
	printf("EWALD\n");
	run(EWALD);
}
