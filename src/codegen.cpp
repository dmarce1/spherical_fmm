#include <stdio.h>
#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>
#include <climits>
#include <unordered_map>
#include <complex>
#include <set>
#include <algorithm>

using complex = std::complex<double>;

static int ntab = 0;
static int tprint_on = true;

#define FLOAT
#define DOUBLE
#define VEC_DOUBLE
#define VEC_FLOAT
#define VEC_DOUBLE_SIZE 4
#define VEC_FLOAT_SIZE 8

static bool nophi = false;
static bool fmaops = true;
static bool periodic = true;
static int pmin = 3;
static int pmax = 25;
static std::string type = "float";
static std::string sitype = "int";
static std::string uitype = "unsigned";
static int rsqrt_flops;
static int sqrt_flops;
static int sincos_flops;
static int erfcexp_flops;
static const int divops = 4;
static const char* prefix = "";
static std::string inter_src = "#include \"spherical_fmm.h\"\n\n";
static std::string inter_header = "\n\nenum fmm_calcpot_type {FMM_CALC_POT, FMM_NOCALC_POT};\n\n";

static bool is_float(std::string str) {
	return str == "float" || str == "vec_float";
}

static bool is_double(std::string str) {
	return str == "double" || str == "vec_double";
}

static bool is_vec(std::string str) {
	return str == "vec_double" || str == "vec_float";
}

const double ewald_r2 = (2.6 + 0.5 * sqrt(3));
const int ewald_h2 = 8;

static FILE* fp = nullptr;
int exp_sz(int P) {
	if (periodic && P > 1) {
		return (P + 1) * (P + 1) + 1;
	} else {
		return (P + 1) * (P + 1);
	}
}

int half_exp_sz(int P) {
	if (periodic && P >= 2) {
		return (P + 2) * (P + 1) / 2 + 1;
	} else {
		return (P + 2) * (P + 1) / 2;
	}
}

int mul_sz(int P) {
	if (periodic && P > 2) {
		return P * P + 1;
	} else {
		return P * P;
	}
}

void set_file(std::string file) {
	if (fp != nullptr) {
		fclose(fp);
	}
	fp = fopen(file.c_str(), "at");
	if (fp == nullptr) {
		printf("unable to open %s for writing\n", file.c_str());
		exit(0xFF);
	}
}

struct closer {
	~closer() {
		if (fp != nullptr) {
			fclose(fp);
		}
	}
};

static closer closer_instance;

void indent() {
	ntab++;
}

void deindent() {
	ntab--;
}
constexpr double dfactorial(int n) {
	if (n == 1 || n == 0) {
		return 1;
	} else {
		return n * dfactorial(n - 2);
	}
}

template<class ...Args>
void tprint(const char* str, Args&&...args) {
	if (fp == nullptr) {
		return;
	}
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			fprintf(fp, "\t");
		}
		fprintf(fp, str, std::forward<Args>(args)...);
	}
}

void tprint(const char* str) {
	if (fp == nullptr) {
		return;
	}
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			fprintf(fp, "\t");
		}
		fprintf(fp, "%s", str);
	}
}

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

double nonepow(int i) {
	return i % 2 == 0 ? 1.0 : -1.0;
}

template<class T>
void ewald_limits(int& r2, int& h2, double alpha) {
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

enum arg_type {
	LIT, PTR, CPTR
};

template<class ...Args>
std::string func_args(const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == CPTR) {
		str += std::string("const ");
	}
	str += std::string(type) + " ";
	if (atype == PTR || atype == CPTR) {
		str += "*";
	}
	str += std::string(arg);
	return str;
}

template<class ...Args>
std::string func_args(const char* arg, arg_type atype, Args&& ...args) {
	std::string str;
	if (atype == CPTR) {
		str += std::string("const ");
	}
	str += std::string(type) + " ";
	if (atype == PTR || atype == CPTR) {
		str += "*";
	}
	str += std::string(arg) + ", ";
	str += func_args(std::forward<Args>(args)...);
	return str;
}
template<class ...Args>
std::string func_args2(const char* arg, arg_type atype, int term) {
	std::string str;
	str += std::string(arg);
	return str;
}

template<class ...Args>
std::string func_args2(const char* arg, arg_type atype, Args&& ...args) {
	std::string str;
	str += std::string(arg) + ", ";
	str += func_args2(std::forward<Args>(args)...);
	return str;
}

template<class ... Args>
void func_header(const char* func, int P, bool nopot, bool pub, Args&& ...args) {
	if (tprint_on) {
		static std::set<std::string> igen;
		std::string func_name = std::string(func) + "_" + type + "_P" + std::to_string(P);
		if (nopot) {
			func_name += "_nopot";
		}
		std::string file_name = func_name + ".c";
		func_name = "void " + func_name;
		if (prefix[0]) {
			func_name = std::string(prefix) + " " + func_name;
		}
		func_name += "(" + func_args(std::forward<Args>(args)..., 0);
		func_name += ")";
		if (!pub) {
			set_file("./generated_code/include/detail/spherical_fmm.h");
		} else {
			set_file("./generated_code/include/spherical_fmm.h");
		}
		tprint("%s;\n", func_name.c_str());
		std::string func1 = std::string(func) + std::string("_") + type;
		if (pub && igen.find(func1) == igen.end() && !nopot) {
			igen.insert(func1);
			if (prefix[0]) {
				inter_header += std::string(prefix) + " ";
			}
			inter_header += std::string("void ") + func1 + "( int P, " + func_args(std::forward<Args>(args)..., 0);
			inter_header += std::string(", enum fmm_calcpot_type calcpot");
			inter_header += std::string(");\n");
			if (prefix[0]) {
				inter_src += std::string(prefix) + " ";
			}
			inter_src += std::string("void ") + func1 + "( int P, " + func_args(std::forward<Args>(args)..., 0);
			inter_src += std::string(", enum fmm_calcpot_type calcpot");
			inter_src += std::string(") {\n");
			inter_src += std::string("\ttypedef void(*func_type)(") + func_args(std::forward<Args>(args)..., 0) + ");\n";
			const int np = pmax - pmin + 1;
			inter_src += std::string("\tstatic const func_type funcs[") + std::to_string(np) + "] = {";
			for (int p = pmin; p <= pmax; p++) {
				inter_src += std::string("&") + func + "_" + type + "_P" + std::to_string(p);
				if (p != pmax) {
					inter_src += std::string(", ");
				} else {
					inter_src += std::string("};\n");
				}
			}
			inter_src += std::string("\tstatic const func_type funcs_nopot[") + std::to_string(np) + "] = {";
			for (int p = pmin; p <= pmax; p++) {
				inter_src += std::string("&") + func + "_" + type + "_P" + std::to_string(p);
				if (p != pmax) {
					inter_src += std::string(", ");
				} else {
					inter_src += std::string("};\n");
				}
			}
			inter_src += std::string("\t") + "if( calcpot == FMM_CALC_POT ) {\n";
			inter_src += std::string("\t\t") + "(*funcs[P - " + std::to_string(pmin) + "])(" + func_args2(std::forward<Args>(args)..., 0) + ");\n";
			inter_src += std::string("\t") + "} else {\n";
			inter_src += std::string("\t\t") + "(*funcs_nopot[P - " + std::to_string(pmin) + "])(" + func_args2(std::forward<Args>(args)..., 0) + ");\n";
			inter_src += std::string("\t") + "}\n";
			inter_src += std::string("}") + "\n\n";
		}
		file_name = std::string("./generated_code/src/") + file_name;
//		printf("%s ", file_name.c_str());
		set_file(file_name);
		tprint("#include <stdio.h>\n");
		tprint("#include \"spherical_fmm.h\"\n");
		tprint("#include \"math_%s.h\"\n", type.c_str());
		tprint("#include \"typecast_%s.h\"\n", type.c_str());
		tprint("#include \"detail/spherical_fmm.h\"\n");
		tprint("\n");
		tprint("%s {\n", func_name.c_str());
		indent();
		tprint("typedef %s T;\n", type.c_str());
		tprint("typedef %s U;\n", uitype.c_str());
		tprint("typedef %s V;\n", sitype.c_str());
	} else {
		indent();
	}
}

void set_tprint(bool c) {
	tprint_on = c;
}

int index(int l, int m) {
	return l * (l + 1) + m;
}

constexpr int cindex(int l, int m) {
	return l * (l + 1) / 2 + m;
}

template<class T>
T nonepow(int m) {
	return m % 2 == 0 ? double(1) : double(-1);
}

template<int P>
struct spherical_expansion: public std::array<complex, (P + 1) * (P + 2) / 2> {
	inline complex operator()(int n, int m) const {
		if (m >= 0) {
			return (*this)[cindex(n, m)];
		} else {
			return (*this)[cindex(n, -m)].conj() * nonepow<double>(m);
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

struct hash {
	size_t operator()(std::array<int, 3> i) const {
		return i[0] * 12345 + i[1] * 42 + i[2];
	}
};

double Brot(int n, int m, int l) {
	static std::unordered_map<std::array<int, 3>, double, hash> values;
	std::array<int, 3> key;
	key[0] = n;
	key[1] = m;
	key[2] = l;
	if (values.find(key) != values.end()) {
		return values[key];
	} else {
		double v;
		if (n == 0 && m == 0 && l == 0) {
			v = 1.0;
		} else if (abs(l) > n) {
			v = 0.0;
		} else if (m == 0) {
			v = 0.5 * (Brot(n - 1, m, l - 1) - Brot(n - 1, m, l + 1));
		} else if (m > 0) {
			v = 0.5 * (Brot(n - 1, m - 1, l - 1) + Brot(n - 1, m - 1, l + 1) + 2.0 * Brot(n - 1, m - 1, l));
		} else {
			v = 0.5 * (Brot(n - 1, m + 1, l - 1) + Brot(n - 1, m + 1, l + 1) - 2.0 * Brot(n - 1, m + 1, l));
		}
		values[key] = v;
		return v;
	}
}

bool close21(double a) {
	return std::abs(1.0 - a) < 1.0e-20;
}

int z_rot(int P, const char* name, bool noevenhi, bool exclude, bool noimaghi) {
	//noevenhi = false;
	int flops = 0;
	tprint("Rx = cosphi;\n");
	tprint("Ry = sinphi;\n");
	int mmin = 1;
	bool initR = true;
	for (int m = 1; m <= P; m++) {
		for (int l = m; l <= P; l++) {
			if (noevenhi && l == P) {
				if ((((P + l) / 2) % 2 == 1) ? m % 2 == 0 : m % 2 == 1) {
					continue;
				}
			}
			if (!initR) {
				tprint("tmp = Rx;\n");
				tprint("Rx = Rx * cosphi - Ry * sinphi;\n");
				tprint("Ry = FMA(tmp, sinphi, Ry * cosphi);\n");
				flops += 6 - fmaops;
				initR = true;
			}
			if (exclude && l == P && m % 2 == 1) {
				tprint("%s[%i] = -%s[%i] * Ry;\n", name, index(l, m), name, index(l, -m));
				tprint("%s[%i] *= Rx;\n", name, index(l, -m));
				flops += 3;
			} else if ((exclude && l == P && m % 2 == 0) || (noimaghi && l == P)) {
				tprint("%s[%i] = %s[%i] * Ry;\n", name, index(l, -m), name, index(l, m));
				tprint("%s[%i] *= Rx;\n", name, index(l, m));
				flops += 2;
			} else {
				if (noevenhi && ((l >= P - 1 && m % 2 == P % 2))) {
					tprint("%s[%i] = %s[%i] * Ry;\n", name, index(l, -m), name, index(l, m));
					tprint("%s[%i] *= Rx;\n", name, index(l, m));
					flops += 2;
				} else {
					tprint("tmp = %s[%i];\n", name, index(l, m));
					tprint("%s[%i] = %s[%i] * Rx - %s[%i] * Ry;\n", name, index(l, m), name, index(l, m), name, index(l, -m));
					tprint("%s[%i] = FMA(tmp, Ry, %s[%i] * Rx);\n", name, index(l, -m), name, index(l, -m));
					flops += 6 - fmaops;
				}
			}

		}
		initR = false;
	}
	return flops;
}

int m2l(int P, int Q, const char* mname, const char* lname) {
	int flops = 0;
	tprint("A[0] = rinv;\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = rinv * A[%i];\n", n, n - 1);
		flops++;
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, factorial(n));
		flops++;
	}
	for (int n = nophi; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			bool pfirst = true;
			bool nfirst = true;
			const int maxk = std::min(P - n, P - 1);
			bool looped = true;
			for (int k = m; k <= maxk; k++) {
				looped = true;
				if (pfirst) {
					pfirst = false;
					tprint("%s[%i] = %s[%i] * A[%i];\n", lname, index(n, m), mname, index(k, m), n + k);
					flops += 1;
				} else {
					tprint("%s[%i] = FMA(%s[%i], A[%i], %s[%i]);\n", lname, index(n, m), mname, index(k, m), n + k, lname, index(n, m));
					flops += 2 - fmaops;
				}
				if (m != 0) {
					if (nfirst) {
						nfirst = false;
						tprint("%s[%i] = %s[%i] * A[%i];\n", lname, index(n, -m), mname, index(k, -m), n + k);
						flops += 1;
					} else {
						tprint("%s[%i] = FMA(%s[%i], A[%i], %s[%i]);\n", lname, index(n, -m), mname, index(k, -m), n + k, lname, index(n, -m));
						flops += 2 - fmaops;
					}
				}
			}
			if (m % 2 != 0) {
				if (!pfirst) {
					tprint("%s[%i] = -%s[%i];\n", lname, index(n, m), lname, index(n, m));
					flops++;
				}
				if (!nfirst) {
					tprint("%s[%i] = -%s[%i];\n", lname, index(n, -m), lname, index(n, -m));
					flops++;
				}
			}
		}
	}
	return flops;

}

int xz_swap(int P, const char* name, bool inv, bool m_restrict, bool l_restrict, bool noevenhi) {
	//noevenhi = false;
//	tprint("template<class T>\n");
//	tprint(" inline void %s( array<T,%i>& %s ) {\n", fname, (P + 1) * (P + 1), name);
	int flops = 0;
	auto brot = [inv](int n, int m, int l) {
		if( inv ) {
			return Brot(n,m,l);
		} else {
			return Brot(n,l,m);
		}
	};
	for (int n = 1; n <= P; n++) {
		int lmax = n;
		if (l_restrict && lmax > (P) - n) {
			lmax = (P) - n;
		}
		for (int m = -lmax; m <= lmax; m++) {
			if (noevenhi && P == n && P % 2 != abs(m) % 2) {
			} else {
				tprint("A[%i] = %s[%i];\n", m + P, name, index(n, m));
			}
		}
		std::vector<std::vector<std::pair<float, int>>>ops(2 * n + 1);
		int mmax = n;
		if (m_restrict && mmax > (P) - n) {
			mmax = (P + 1) - n;
		}
		int mmin = 0;
		int stride = 1;
		for (int m = 0; m <= mmax; m += stride) {
			for (int l = 0; l <= lmax; l++) {
				if (noevenhi && P == n && P % 2 != abs(l) % 2) {
					continue;
				}
				double r = l == 0 ? brot(n, m, 0) : brot(n, m, l) + nonepow<double>(l) * brot(n, m, -l);
				double i = l == 0 ? 0.0 : brot(n, m, l) - nonepow<double>(l) * brot(n, m, -l);
				if (r != 0.0) {
					ops[n + m].push_back(std::make_pair(r, P + l));
				}
				if (i != 0.0 && m != 0) {
					ops[n - m].push_back(std::make_pair(i, P - l));
				}
			}
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			std::sort(ops[m].begin(), ops[m].end(), [](std::pair<float,int> a, std::pair<float,int> b) {
				return a.first < b.first;
			});
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			if (ops[m].size() == 0 && !m_restrict && !l_restrict) {
				//			fprintf(stderr, " %i %i %i %i %i\n", m_restrict, l_restrict, noevenhi, n, m - n);
				//			tprint("%s[%i] = TCAST(0);\n", name, index(n, m - n));
			}
			for (int l = 0; l < ops[m].size(); l++) {
				int len = 1;
				while (len + l < ops[m].size()) {
					if (ops[m][len + l].first == ops[m][l].first && !close21(ops[m][l].first)) {
						len++;
					} else {
						break;
					}
				}
				if (len == 1) {
					if (close21(ops[m][l].first)) {
						tprint("%s[%i] %s= A[%i];\n", name, index(n, m - n), l == 0 ? "" : "+", ops[m][l].second);
						flops += 1 - (l == 0);
					} else {
						if (l == 0) {
							tprint("%s[%i] = TCAST(%.20e) * A[%i];\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second);
							flops += 1;
						} else {
							tprint("%s[%i] = FMA(TCAST(%.20e), A[%i], %s[%i]);\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second, name, index(n, m - n));
							flops += 2 - fmaops;
						}
					}
				} else {
					tprint("tmp = A[%i];\n", ops[m][l].second);
					for (int p = 1; p < len; p++) {
						tprint("tmp += A[%i];\n", ops[m][l + p].second);
						flops++;
					}
					if (l == 0) {
						tprint("%s[%i] = TCAST(%.20e) * tmp;\n", name, index(n, m - n), ops[m][l].first);
						flops += 1;
					} else {
						tprint("%s[%i] = FMA(TCAST(%.20e), tmp, %s[%i]);\n", name, index(n, m - n), ops[m][l].first, name, index(n, m - n));
						flops += 2 - fmaops;
					}
				}
				l += len - 1;
			}

		}
	}
	return flops;
}

int greens_body(int P, const char* M = nullptr) {
	int flops = 0;
	tprint("T r2, r2inv, ax, ay;\n");
	tprint("r2 = FMA(x, x, FMA(y, y, z * z));\n");
	flops += 5 - 2 * fmaops;
	if (M) {
		tprint("r2inv = %s / r2;\n", M);
	} else {
		tprint("r2inv = TCAST(1) / r2;\n");
	}
	tprint("O[0] = rsqrt_%s(r2);\n", type.c_str());
	flops += rsqrt_flops;
	if (M) {
		tprint("O[0] *= %s;\n", M);
		flops++;
	}
	tprint("O[%i] = TCAST(0);\n", (P + 1) * (P + 1));
	tprint("x *= r2inv;\n");
	tprint("y *= r2inv;\n");
	tprint("z *= r2inv;\n");
	flops += 3;
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			tprint("O[%i] = y * O[0];\n", index(m, -m));
			flops += 2;
		} else if (m > 0) {
			tprint("ax = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("ay = O[%i] * TCAST(%i);\n", index(m - 1, -(m - 1)), 2 * m - 1);
			tprint("O[%i] = x * ax - y * ay;\n", index(m, m));
			tprint("O[%i] = FMA(y, ax, x * ay);\n", index(m, -m));
			flops += 8 - fmaops;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			flops += 2;
			if (m != 0) {
				tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, -m), 2 * m + 1, index(m, -m));
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = TCAST(%i) * z;\n", 2 * n - 1);
				tprint("ay = TCAST(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = (ax * O[%i] + ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				tprint("O[%i] = FMA(ax, O[%i], ay * O[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
				flops += 8 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					flops += 4;
				} else {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - TCAST(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
					flops += 5;
				}

			}
		}
	}
	return flops;
}

int m2lg_body(int P, int Q) {
	int flops = 0;
	for (int n = nophi; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			const int kmax = std::min(P - n, P - 1);
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			for (int k = 0; k <= kmax; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					bool greal = false;
					bool mreal = false;
					int gxsgn = 1;
					int gysgn = 1;
					int mxsgn = 1;
					int mysgn = 1;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					char* mxstr = nullptr;
					char* mystr = nullptr;
					if (m + l > 0) {
						asprintf(&gxstr, "O[%i]", index(n + k, m + l));
						asprintf(&gystr, "O[%i]", index(n + k, -m - l));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&gxstr, "O[%i]", index(n + k, -m - l));
							asprintf(&gystr, "O[%i]", index(n + k, m + l));
							gysgn = -1;
						} else {
							asprintf(&gxstr, "O[%i]", index(n + k, -m - l));
							asprintf(&gystr, "O[%i]", index(n + k, m + l));
							gxsgn = -1;
						}
					} else {
						greal = true;
						asprintf(&gxstr, "O[%i]", index(n + k, 0));
					}
					if (l > 0) {
						asprintf(&mxstr, "M[%i]", index(k, l));
						asprintf(&mystr, "M[%i]", index(k, -l));
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
						} else {
							asprintf(&mxstr, "M[%i]", index(k, -l));
							asprintf(&mystr, "M[%i]", index(k, l));
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						asprintf(&mxstr, "M[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
						if( sgn == 1) {
							if( m >= 0 ) {
								pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							} else {
								pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							}
						} else {
							if( m >= 0 ) {
								neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							} else {
								neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
							}
						}
					};
					if (!mreal) {
						if (!greal) {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
							add_work(-mysgn * gysgn, 1, mystr, gystr);
							flops += 4;
							if (m > 0) {
								add_work(mysgn * gxsgn, -1, mystr, gxstr);
								add_work(mxsgn * gysgn, -1, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
							if (m > 0) {
								add_work(mysgn * gxsgn, -1, mystr, gxstr);
							}
						}
					} else {
						if (!greal) {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
							if (m > 0) {
								add_work(mxsgn * gysgn, -1, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
				//		if(!fmaops) {
				//	}

			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}

		}
	}
	return flops;
}
std::vector<complex> spherical_singular_harmonic(int P, double x, double y, double z) {
	const double r2 = x * x + y * y + z * z;
	const double r2inv = double(1) / r2;
	complex R = complex(x, y);
	std::vector<complex> O((P + 1) * (P + 1));
	O[cindex(0, 0)] = complex(sqrt(r2inv), double(0));
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

bool close2zero(double a) {
	return abs(a) < 1e-10;
}

int p2l(int P) {
	int flops = 0;
	func_header("P2L", P, nophi, true, "L", PTR, "M", LIT, "x", LIT, "y", LIT, "z", LIT);
	tprint("int n;\n");
	tprint("T O[%i];\n", exp_sz(P));
	flops += greens_body(P, "M");
	tprint("for (n = %i; n < %i; n++) {\n;", nophi, (P + 1) * (P + 1));
	indent();
	tprint("L[n] += O[n];\n");
	flops += (P + 1) * (P + 1) - nophi;
	deindent();
	tprint("}\n");
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int greens(int P) {
	int flops = 0;
	func_header("greens", P, false, true, "O", PTR, "x", LIT, "y", LIT, "z", LIT);
	flops += greens_body(P);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int greens_xz(int P) {
	int flops = 0;
	func_header("greens_xz", P, false, false, "O", PTR, "x", LIT, "z", LIT, "r2inv", LIT);
	tprint("T ax, ay;\n");
	tprint("O[0] = sqrt_%s(r2inv);\n", type.c_str());
	flops += sqrt_flops;
	tprint("O[%i] = TCAST(0);\n", (P + 2) * (P + 1) / 2);
	tprint("x *= r2inv;\n");
	flops += 1;
	tprint("z *= r2inv;\n");
	flops += 1;
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			flops += 1;
		} else if (m > 0) {
			tprint("ax = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("O[%i] = x * ax;\n", index(m, m));
			flops += 2;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			flops += 2;
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = TCAST(%i) * z;\n", 2 * n - 1);
				tprint("ay = TCAST(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = FMA(ax, O[%i], ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				flops += 5 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					flops += 4;
				} else {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - TCAST(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
					flops += 5;
				}
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}
int m2l_rot1(int P, int Q) {
	int flops = 0;
	if (Q > 1) {
		func_header("M2L", P, nophi, true, "L0", PTR, "M0", CPTR, "x", LIT, "y", LIT, "z", LIT);
	} else {
		func_header("M2P", P, nophi, true, "L0", PTR, "M0", CPTR, "x", LIT, "y", LIT, "z", LIT);
	}
	tprint("int n;\n");
	tprint("T R2, Rzero, r2, rzero, tmp1, Rinv, r2inv, R, cosphi, sinphi;\n");
	tprint("T L[%i];\n", exp_sz(Q));
	tprint("T M[%i];\n", mul_sz(P));
	tprint("T O[%i];\n", half_exp_sz(P));
	tprint("T Rx, Ry, tmp;\n");
	tprint("for(n = 0; n < %i; n++) {\n", P * P + 1);
	indent();
	tprint("M[n] = M0[n];\n");
	deindent();
	tprint("}\n");
	const auto tpo = tprint_on;

	set_tprint(false);
	flops += greens_xz(P);
	set_tprint(tpo);
	tprint("R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = TCAST(R2<TCAST(1e-37));\n");
	flops++;
	tprint("r2 = FMA(z, z, R2);\n");
	tprint("rzero = TCAST(r2<TCAST(1e-37));\n");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops + rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("r2inv = TCAST(1) / (r2+rzero);\n");
	flops += 7 - fmaops;
	tprint("cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P - 1, "M", false, false, false);
	tprint("greens_xz_%s_P%i(O, R, z, r2inv);\n", type.c_str(), P);
	const auto oindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int n = nophi; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			bool nfirst = true;
			bool pfirst = true;
			const int kmax = std::min(P - n, P - 1);
			for (int sgn = -1; sgn <= 1; sgn += 2) {
				for (int k = 0; k <= kmax; k++) {
					const int lmin = std::max(-k, -n - k - m);
					const int lmax = std::min(k, n + k - m);
					for (int l = lmin; l <= lmax; l++) {
						bool mreal = false;
						int gxsgn = 1;
						int mxsgn = 1;
						int mysgn = 1;
						char* gxstr = nullptr;
						char* mxstr = nullptr;
						char* mystr = nullptr;
						if (m + l > 0) {
							asprintf(&gxstr, "O[%i]", oindex(n + k, m + l));
						} else if (m + l < 0) {
							if (abs(m + l) % 2 == 0) {
								asprintf(&gxstr, "O[%i]", oindex(n + k, -m - l));
							} else {
								asprintf(&gxstr, "O[%i]", oindex(n + k, -m - l));
								gxsgn = -1;
							}
						} else {
							asprintf(&gxstr, "O[%i]", oindex(n + k, 0));
						}
						if (l > 0) {
							asprintf(&mxstr, "M[%i]", index(k, l));
							asprintf(&mystr, "M[%i]", index(k, -l));
							mysgn = -1;
						} else if (l < 0) {
							if (l % 2 == 0) {
								asprintf(&mxstr, "M[%i]", index(k, -l));
								asprintf(&mystr, "M[%i]", index(k, l));
							} else {
								asprintf(&mxstr, "M[%i]", index(k, -l));
								asprintf(&mystr, "M[%i]", index(k, l));
								mxsgn = -1;
								mysgn = -1;
							}
						} else {
							mreal = true;
							asprintf(&mxstr, "M[%i]", index(k, 0));
						}
						if (!mreal) {
							if (mxsgn * gxsgn == sgn) {
								if (pfirst) {
									tprint("L[%i] = %s * %s;\n", index(n, m), mxstr, gxstr);
									pfirst = false;
									flops += 1;
								} else {
									tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), mxstr, gxstr, index(n, m));
									flops += 2 - fmaops;
								}
							}
							if (mysgn * gxsgn == sgn) {
								if (m > 0) {
									if (nfirst) {
										tprint("L[%i] = %s * %s;\n", index(n, -m), mystr, gxstr);
										nfirst = false;
										flops += 1;
									} else {
										tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, -m), mystr, gxstr, index(n, -m));
										flops += 2 - fmaops;
									}
								}
							}
						} else {
							if (mxsgn * gxsgn == sgn) {
								if (pfirst) {
									tprint("L[%i] = %s * %s;\n", index(n, m), mxstr, gxstr);
									pfirst = false;
									flops += 1;
								} else {
									tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), mxstr, gxstr, index(n, m));
									flops += 2 - fmaops;
								}
							}
						}
						if (gxstr) {
							free(gxstr);
						}
						if (mxstr) {
							free(mxstr);
						}
						if (mystr) {
							free(mystr);
						}
					}
				}
				if (!pfirst && sgn == -1) {
					tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
					flops++;
				}
				if (!nfirst && sgn == -1) {
					tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
					flops++;
				}

			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	flops += z_rot(Q, "L", false, false, Q == P);
	flops++;
	tprint("for(n = %i; n < %i; n++) {\n", nophi, (Q + 1) * (Q + 1));
	indent();
	tprint("L0[n] += L[n];\n");
	deindent();
	tprint("}\n");
	flops += (Q + 1) * (Q + 1);

	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int m2l_ewald(int P) {
	int flops = 0;
	if (periodic) {
		func_header("M2L_ewald", P, nophi, true, "L", PTR, "M", CPTR, "x", LIT, "y", LIT, "z", LIT);
		tprint("T G[%i];\n", exp_sz(P));
		tprint("greens_ewald_%s_P%i(G, x, y, z);\n", type.c_str(), P);
		tprint("M2LG_%s_P%i%s(L,M,G);\n", type.c_str(), P, nophi ? "_nopot" : "");
		deindent();
		tprint("}");
		tprint("\n");
	}
	return flops;
}

int m2lg(int P, int Q) {
	int flops = 0;
	func_header("M2LG", P, nophi, true, "L", PTR, "M", CPTR, "O", CPTR);
	flops += m2lg_body(P, Q);
	if (!nophi && P > 2 && periodic) {
		tprint("L[%i] = FMA(TCAST(-0.5) * O[%i], M[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), P * P, index(0, 0));
		flops += 3 - fmaops;
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = FMA(TCAST(-2) * O[%i], M[%i], L[%i]);\n", index(1, -1), (P + 1) * (P + 1), index(1, -1), index(1, -1));
		tprint("L[%i] -= O[%i] * M[%i];\n", index(1, +0), (P + 1) * (P + 1), index(1, +0), index(1, +0));
		tprint("L[%i] = FMA(TCAST(-2) * O[%i], M[%i], L[%i]);\n", index(1, +1), (P + 1) * (P + 1), index(1, +1), index(1, +1));
		tprint("L[%i] = FMA(TCAST(-0.5) * O[%i], M[%i], L[%i]);\n", (P + 1) * (P + 1), (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		flops += 10 - 3 * fmaops;
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int m2l_norot(int P, int Q) {
	int flops = 0;
	if (Q > 1) {
		func_header("M2L", P, nophi, true, "L", PTR, "M", CPTR, "x", LIT, "y", LIT, "z", LIT);
	} else {
		func_header("M2P", P, nophi, true, "L", PTR, "M", CPTR, "x", LIT, "y", LIT, "z", LIT);
	}
	const auto c = tprint_on;
	set_tprint(false);
	flops += greens(P);
	set_tprint(c);
	tprint("T O[%i];\n", exp_sz(P));
	tprint("greens_%s_P%i(O, x, y, z);\n", type.c_str(), P);
	flops += m2lg_body(P, Q);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int ewald_greens(int P, double alpha) {
	int R2, H2;
	if (is_float(type)) {
		ewald_limits<float>(R2, H2, alpha);
	} else {
		ewald_limits<double>(R2, H2, alpha);
	}
	int R = sqrt(R2);
	int H = sqrt(H2);
	if (!periodic) {
		return 0;
	}
	int flops = 0;
	func_header("greens_ewald", P, false, false, "G", PTR, "x0", LIT, "y0", LIT, "z0", LIT);
	const auto c = tprint_on;

	//set_tprint(false);
	//flops += greens(P);
	//set_tprint(c);
	tprint("T Gr[%i];\n", exp_sz(P));
	set_tprint(false);
	flops += greens(P);
	set_tprint(c);
	int cnt = 0;
	for (int ix = -R; ix <= R; ix++) {
		for (int iy = -R; iy <= R; iy++) {
			for (int iz = -R; iz <= R; iz++) {
				const int i2 = ix * ix + iy * iy + iz * iz;
				if (i2 > R2 && i2 != 0) {
					continue;
				}
				cnt++;
			}
		}
	}
	flops *= cnt + 1;
	bool first = true;
	int eflops;
	tprint("T sw, r, r2, xxx, gam1, exp0, xfac, xpow, gam0inv, gam, x, y, z, x2, x2y2, hdotx, phi;\n");
	const auto name = [](const char* base, int hx, int hy, int hz) {
		std::string s = base;
		const auto add_symbol = [&s](int h) {
			if( h < 0 ) {
				s += "n";
				s += std::to_string(-h);
			} else if( h > 0 ) {
				s += "p";
				s += std::to_string(h);
			} else {
				s += "x0";
			}
		};
		add_symbol(hx);
		add_symbol(hy);
		add_symbol(hz);
		return s;
	};
	const auto cosname = [name](int hx, int hy, int hz) {
		return name("cos", hx, hy, hz);
	};
	const auto sinname = [name](int hx, int hy, int hz) {
		return name("sin", hx, hy, hz);
	};
	bool close = false;
	std::string str = "T ";
	int icnt = 0;
	for (int hx = -H; hx <= H; hx++) {
		for (int hy = -H; hy <= H; hy++) {
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= H2 && h2 > 0) {
					if (close) {
						str += std::string(", ");
					}
					if (icnt == 8) {
						str += "\n\t  ";
						icnt = 0;
					}
					str += cosname(hx, hy, hz).c_str() + std::string(", ");
					str += sinname(hx, hy, hz).c_str();
					close = true;
					icnt++;
				}
			}
		}
	}
	str += ";";
	tprint("%s\n", str.c_str());
	tprint("int ix, iy, iz, ii;\n");
	tprint("r2 = FMA(x0, x0, FMA(y0, y0, z0 * z0));\n");
	flops += 5 - 2 * fmaops;
	tprint("r = sqrt_%s(r2);\n", type.c_str());
	tprint("greens_%s_P%i(Gr, x0, y0, z0);\n", type.c_str(), P);
	tprint("xxx = TCAST(%.20e) * r;\n", alpha);
	flops++;
	tprint("erfcexp_%s(xxx, &gam1, &exp0);\n", type.c_str());
	flops += erfcexp_flops;
	tprint("gam1 *= TCAST(%.20e);\n", sqrt(M_PI));
	flops += 1;
	tprint("xfac = TCAST(%.20e) * r2;\n", alpha * alpha);
	flops += 1;
	tprint("xpow = TCAST(%.20e) * r;\n", alpha);
	flops++;
	double gam0inv = 1.0 / sqrt(M_PI);
	tprint("sw = r2 > TCAST(0);\n");
	flops += 1;
	for (int l = 0; l <= P; l++) {
		tprint("gam = gam1 * TCAST(%.20e);\n", gam0inv);
		flops++;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = sw*(TCAST(%.1e) - gam) * Gr[%i];\n", index(l, m), nonepow<double>(l), index(l, m));
			flops += 4;
		}
		if (l == 0) {
			tprint("G[%i] += (TCAST(1) - sw)*TCAST(%.20e);\n", index(0, 0), (2) * alpha / sqrt(M_PI));
			flops += 3;
		}
		gam0inv *= 1.0 / -(l + 0.5);
		if (l != P) {
			tprint("gam1 = TCAST(%.20e) * gam1 + xpow * exp0;\n", l + 0.5);
			flops += 3;
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
				flops++;
			}
		}
	}
	tprint("for (ix = -%i; ix <= %i; ix++) {\n", R, R);
	indent();
	tprint("for (iy = -%i; iy <= %i; iy++) {\n", R, R);
	indent();
	tprint("for (iz = -%i; iz <= %i; iz++) {\n", R, R);
	indent();
	tprint("ii = ix * ix + iy * iy + iz * iz;\n");
	tprint("if (ii > %i || ii == 0) {\n", R2);
	indent();
	tprint("continue;\n");
	deindent();
	tprint("}\n");
	tprint("x = x0 - TCAST(ix);\n");
	flops++;
	tprint("y = y0 - TCAST(iy);\n");
	flops++;
	tprint("z = z0 - TCAST(iz);\n");
	tprint("r2 = FMA(x, x, FMA(y, y, z * z));\n");
	flops += 3 * cnt;
	tprint("r = sqrt_%s(r2);\n", type.c_str());
	flops += sqrt_flops * cnt;
	tprint("greens_%s_P%i(Gr, x, y, z);\n", type.c_str(), P);
	tprint("xxx = TCAST(%.20e) * r;\n", alpha);
	flops += cnt;
	tprint("erfcexp_%s(xxx, &gam1, &exp0);\n", type.c_str());
	flops += erfcexp_flops * cnt;
	tprint("gam1 *= TCAST(%.20e);\n", -sqrt(M_PI));
	flops += cnt;
	tprint("xfac = TCAST(%.20e) * r2;\n", alpha * alpha);
	flops += cnt;
	tprint("xpow = TCAST(%.20e) * r;\n", alpha);
	flops++;
	tprint("gam0inv = TCAST(%.20e);\n", 1.0 / sqrt(M_PI));
	for (int l = 0; l <= P; l++) {
		tprint("gam = gam1 * gam0inv;\n");
		flops += cnt;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = FMA(gam, Gr[%i], G[%i]);\n", index(l, m), index(l, m), index(l, m));
			flops += (2 - fmaops) * cnt;
		}
		if (l != P) {
			tprint("gam0inv *= TCAST(%.20e);\n", 1.0 / -(l + 0.5));
			tprint("gam1 = FMA(TCAST(%.20e), gam1, -xpow * exp0);\n", l + 0.5);
			flops += (4 - fmaops) * cnt;
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
				flops += cnt;
			}
		}
	}
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");

	int rflops = flops;
	for (int hx = -H; hx <= H; hx++) {
		if (hx) {
			if (abs(hx) == 1) {
				tprint("x2 = %cx0;\n", hx < 0 ? '-' : ' ');
				flops += hx < 0;
			} else {
				tprint("x2 = TCAST(%i) * x0;\n", hx);
				flops++;
			}
		} else {
			tprint("x2 = TCAST(0);\n");
		}
		for (int hy = -H; hy <= H; hy++) {
			if (hx * hx + hy * hy > H2) {
				continue;
			}
			if (hy) {
				if (abs(hy) == 1) {
					tprint("x2y2 = x2 %c y0;\n", hy > 0 ? '+' : '-');
					flops++;
				} else {
					tprint("x2y2 = FMA(TCAST(%i), y0, x2);\n", hy);
					flops += 2 - fmaops;
				}
			} else {
				tprint("x2y2 = x2;\n", hy);
			}
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 > H2 || h2 == 0) {
					continue;
				}
				if (hz) {
					if (abs(hz) == 1) {
						tprint("hdotx = x2y2 %c z0;\n", hz > 0 ? '+' : '-');
						flops++;
					} else {
						tprint("hdotx = FMA(TCAST(%i), z0, x2y2);\n", hz);
						flops += 2 - fmaops;
					}
				} else {
					tprint("hdotx = x2y2;\n", hz);
				}
				tprint("phi = TCAST(%.20e) * hdotx;\n", 2.0 * M_PI);
				flops++;
				tprint("sincos_%s(phi, &%s, &%s);\n", type.c_str(), sinname(hx, hy, hz).c_str(), cosname(hx, hy, hz).c_str());
				flops += sincos_flops;
			}
		}
	}
	using table_type = std::unordered_map<double, std::vector<std::pair<int, std::string>>>;
	table_type ops[(P + 1) * (P + 1)];
	for (int hx = -H; hx <= H; hx++) {
		for (int hy = -H; hy <= H; hy++) {
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= H2 && h2 > 0) {
					const double h = sqrt(h2);
					bool init = false;
					const auto G0 = spherical_singular_harmonic(P, (double) hx, (double) hy, (double) hz);
					flops++;
					flops += 8;
					double gam0inv = 1.0 / sqrt(M_PI);
					double hpow = 1. / h;
					double pipow = 1. / sqrt(M_PI);
					for (int l = 0; l <= P; l++) {
						for (int m = 0; m <= l; m++) {
							double c0 = gam0inv * hpow * pipow * exp(-h * h * double(M_PI * M_PI) / (alpha * alpha));
							std::string ax = cosname(hx, hy, hz);
							std::string ay = sinname(hx, hy, hz);
							int xsgn = 1;
							int ysgn = 1;
							if (l % 4 == 3) {
								std::swap(ax, ay);
								ysgn = -1;
							} else if (l % 4 == 2) {
								ysgn = xsgn = -1;
							} else if (l % 4 == 1) {
								std::swap(ax, ay);
								xsgn = -1;
							}
							if (G0[cindex(l, m)].real() != (0)) {
								const double a = c0 * G0[cindex(l, m)].real();
								//						tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, m), -xsgn * c0 * G0[cindex(l, m)].real(), ax.c_str());
								ops[index(l, m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -xsgn * a), ax));
								if (m != 0) {
									//								tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, -m), -ysgn * c0 * G0[cindex(l, m)].real(), ay.c_str());
									ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -ysgn * a), ay));
								}
							}
							if (G0[cindex(l, m)].imag() != (0)) {
								const double a = c0 * G0[cindex(l, m)].imag();
//								tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, m), ysgn * c0 * G0[cindex(l, m)].imag(), ay.c_str());
								ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, ysgn * a), ay));
								if (m != 0) {
									//	tprint("G[%i] += TCAST(%.20e) * %s;\n", index(l, -m), -xsgn * c0 * G0[cindex(l, m)].imag(), ax.c_str());
									ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -xsgn * a), ax));
								}
							}
						}
						gam0inv /= l + 0.5;
						hpow *= h * h;
						pipow *= M_PI;

					}
				}
			}
		}
	}
	for (int ii = 0; ii < (P + 1) * (P + 1); ii++) {
		table_type next_ops;
		for (auto j = ops[ii].begin(); j != ops[ii].end(); j++) {
			std::vector<int> remove;
			const std::vector<std::pair<int, std::string>>& these_ops = j->second;
			for (int dim = 0; dim < 3; dim++) {
				for (int hx = 1; hx <= H; hx++) {
					std::string sym1, sym2;
					if (dim == 0) {
						sym1 = sinname(hx, 0, 0);
						sym2 = sinname(-hx, 0, 0);
					} else if (dim == 1) {
						sym1 = sinname(0, hx, 0);
						sym2 = sinname(0, -hx, 0);
					} else {
						sym1 = sinname(0, 0, hx);
						sym2 = sinname(0, 0, -hx);
					}
					for (int k = 0; k < these_ops.size(); k++) {
						if (these_ops[k].second == sym1) {
							for (int l = 0; l < these_ops.size(); l++) {
								if (these_ops[l].second == sym2) {
									remove.push_back(l);
									remove.push_back(k);
									if (these_ops[l].first * these_ops[k].first < 0) {
										double a = 2 * j->first;
										next_ops[a].push_back(std::make_pair(these_ops[k].first > 0 ? 1 : -1, these_ops[k].second));
									}
								}
							}
						}
					}
					if (dim == 0) {
						sym1 = cosname(hx, 0, 0);
						sym2 = cosname(-hx, 0, 0);
					} else if (dim == 1) {
						sym1 = cosname(0, hx, 0);
						sym2 = cosname(0, -hx, 0);
					} else {
						sym1 = cosname(0, 0, hx);
						sym2 = cosname(0, 0, -hx);
					}
					for (int k = 0; k < these_ops.size(); k++) {
						if (these_ops[k].second == sym1) {
							for (int l = 0; l < these_ops.size(); l++) {
								if (these_ops[l].second == sym2) {
									remove.push_back(l);
									remove.push_back(k);
									if (these_ops[l].first * these_ops[k].first > 0) {
										double a = 2 * j->first;
										next_ops[a].push_back(std::make_pair(these_ops[k].first > 0 ? 1 : -1, these_ops[k].second));
									}
								}
							}
						}
					}
				}
			}
			std::vector<std::pair<int, std::string>> these_next_ops;
			for (int k = 0; k < these_ops.size(); k++) {
				bool use = true;
				for (int l = 0; l < remove.size(); l++) {
					if (k == remove[l]) {
						use = false;
						break;
					}
				}
				if (use) {
					these_next_ops.push_back(these_ops[k]);
				}
			}
			next_ops[j->first].insert(next_ops[j->first].end(), these_next_ops.begin(), these_next_ops.end());
		}
		ops[ii] = std::move(next_ops);
	}
	for (int ii = 0; ii < (P + 1) * (P + 1); ii++) {
		std::vector<std::pair<double, std::vector<std::pair<int, std::string> > > > sorted_ops(ops[ii].begin(), ops[ii].end());
		std::sort(sorted_ops.begin(), sorted_ops.end(),
				[](const std::pair<double,std::vector<std::pair<int, std::string> > >& a, const std::pair<double,std::vector<std::pair<int, std::string> > > & b ) {
					return fabs(a.first) * sqrt(a.second.size()) < fabs(b.first) * sqrt(b.second.size());
				});
		for (auto j = sorted_ops.begin(); j != sorted_ops.end(); j++) {
			auto op = j->second;
			if (op.size()) {
				int sgn = op[0].first > 0 ? 1 : -1;
				if (sgn > 0) {
					tprint("G[%i] = FMA(TCAST(+%.20e), ", ii, sgn * j->first);
				} else {
					tprint("G[%i] = FMA(TCAST(%.20e), ", ii, sgn * j->first);
				}
				flops += 2 - fmaops;
				for (int k = 0; k < op.size(); k++) {
					if (tprint_on) {
						if (k != 0) {
							fprintf(fp, " %c ", sgn * op[k].first > 0 ? '+' : '-');
						}
						fprintf(fp, "%s", op[k].second.c_str());
					}
					if (!(k == 0 && op[k].first > 0)) {
						flops++;
					}
				}
				if (tprint_on) {
					fprintf(fp, ", G[%i]);\n", ii);
				}
			}
		}
	}
	tprint("G[%i] = TCAST(%.20e);\n", (P + 1) * (P + 1), (4.0 * M_PI / 3.0));
	if (!nophi) {
		tprint("G[%i] += TCAST(%.20e);\n", index(0, 0), M_PI / (alpha * alpha));
		flops++;
	}
	int fflops = flops - rflops;
	fprintf(stderr, "%e %i %i %i %i %i\n", alpha, R2, H2, rflops, fflops, rflops + fflops);
	deindent();
	tprint("/* flops = %i */\n", flops);
	tprint("}");
	tprint("\n");
	return flops;

}

int m2l_rot2(int P, int Q) {
	int flops = 0;
	if (Q > 1) {
		func_header("M2L", P, nophi, true, "L0", PTR, "M0", CPTR, "x", LIT, "y", LIT, "z", LIT);
	} else {
		func_header("M2P", P, nophi, true, "L0", PTR, "M0", CPTR, "x", LIT, "y", LIT, "z", LIT);
	}
	tprint("T L[%i];\n", exp_sz(Q));
	tprint("T M[%i];\n", mul_sz(P));
	tprint("T A[%i];\n", 2 * P + 1);
	tprint("T Rx, Ry, tmp, R, Rinv, R2, Rzero, tmp1, r2, rzero, r2inv, r2przero, rinv, cosphi, sinphi, cosphi0, sinphi0;\n");
	tprint( "int n;\n");
	tprint("bool sw;\n");
	tprint("for(n = 0; n < %i; n++) {\n", P * P + 1);
	indent();
	tprint("M[n] = M0[n];\n");
	deindent();
	tprint("}\n");

	/*	tprint("for( int n = 0; n < %i; n++) {\n", Q == 1 ? 4 : (Q + 1) * (Q + 1) + 1);
	 indent();
	 tprint("L[n] = TCAST(0);\n");
	 deindent();
	 tprint("}\n");*/

	tprint("R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = TCAST(R2<TCAST(1e-37));\n");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops += rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("r2 = (FMA(z, z, R2));\n");
	tprint("rzero = TCAST(r2<TCAST(1e-37));\n");
	tprint("r2inv = TCAST(1) / r2;\n");
	flops += 3 + divops - fmaops;
	tprint("r2przero = (r2 + rzero);");
	flops += 1;
	tprint("rinv = rsqrt_%s(r2przero);\n", type.c_str());
	flops += rsqrt_flops;

	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sw = false;\n");
	if (tprint_on) {
		fprintf(fp, "MULTIPOLE_ROTATION:\n");
	}
	flops += 2 * z_rot(P - 1, "M", false, false, false);
	tprint("if( sw ) {\n");
	indent();
	tprint("goto CONTINUE_AFTER_ROTATION;\n");
	deindent();
	tprint("}\n");
	flops += xz_swap(P - 1, "M", false, false, false, false);

	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = FMA(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += 2;
	tprint("sw = true;\n");
	tprint("goto MULTIPOLE_ROTATION;\n");
	if (tprint_on) {
		fprintf(fp, "CONTINUE_AFTER_ROTATION:\n");
	}
	flops += xz_swap(P - 1, "M", false, true, false, false);
	flops += m2l(P, Q, "M", "L");
	flops += xz_swap(Q, "L", true, false, true, false);

	tprint("sinphi = -sinphi;\n");
	flops += 1;
	flops += z_rot(Q, "L", true, false, false);
	flops += xz_swap(Q, "L", true, false, false, true);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(Q, "L", false, true, false);
	tprint("for(n = 0; n < %i; n++) {\n", (Q + 1) * (Q + 1));
	indent();
	tprint("L0[n] += L[n];\n");
	deindent();
	tprint("}\n");
	flops += (Q + 1) * (Q + 1);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int regular_harmonic(int P) {
	int flops = 0;
	func_header("regular_harmonic", P, false, false, "Y", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("T ax, ay, r2;\n");
	tprint("r2 = FMA(x, x, FMA(y, y, z * z));\n");
	flops += 5 - 2 * fmaops;
	tprint("Y[0] = TCAST(1);\n");
	tprint("Y[%i] = r2;\n", (P + 1) * (P + 1));
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / TCAST(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay = Y[%i] * TCAST(%.20e);\n", index(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax - y * ay;\n", index(m, m));
				tprint("Y[%i] = FMA(y, ax, x * ay);\n", index(m, -m));
				flops += 6 - fmaops;
			} else {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				tprint("Y[%i] = y * ax;\n", index(m, -m));
				flops += 3;
			}
		}
		if (m + 1 <= P) {
//			Y[index(m + 1, m)] = z * Y[index(m, m)];
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, -m), index(m, -m));
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
//			Y[index(n, m)] = inv * (TCAST(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax =  TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("Y[%i] = FMA(ax, Y[%i], ay * Y[%i]);\n", index(n, m), index(n - 1, m), -(double) inv, index(n - 2, m));
			flops += 5 - fmaops;
			if (m != 0) {
				tprint("Y[%i] = FMA(ax, Y[%i], ay * Y[%i]);\n", index(n, -m), index(n - 1, -m), -(double) inv, index(n - 2, -m));
				flops += 3 - fmaops;
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int regular_harmonic_xz(int P) {
	int flops = 0;
	func_header("regular_harmonic_xz", P, false, false, "Y", PTR, "x", LIT, "z", LIT);
	tprint("T ax, ay, r2;\n");
	tprint("r2 = FMA(x, x, z * z);\n");
	flops += 3 - fmaops;
	tprint("Y[0] = TCAST(1);\n");
	tprint("Y[%i] = r2;\n", (P + 2) * (P + 1) / 2);
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / TCAST(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				flops += 2;
			} else {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				flops += 2;
			}
		}
		if (m + 1 <= P) {
//			Y[index(m + 1, m)] = z * Y[index(m, m)];
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				flops += 1;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
//			Y[index(n, m)] = inv * (TCAST(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax =  TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("Y[%i] = FMA(ax, Y[%i], ay * Y[%i]);\n", index(n, m), index(n - 1, m), -(double) inv, index(n - 2, m));
			flops += 5 - fmaops;
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int M2M_norot(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	func_header("M2M", P + 1, nophi, true, "M", PTR, "x", LIT, "y", LIT, "z", LIT);
//const auto Y = spherical_regular_harmonic<T, P>(-x, -y, -z);
	tprint("T Y[%i];\n", exp_sz(P));
	tprint("regular_harmonic_%s_P%i(Y, -x, -y, -z);\n", type.c_str(), P);
	flops += 3;
	if (P > 1 && !nophi && periodic) {
		tprint("M[%i] = FMA(TCAST(-4) * x, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 1), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(TCAST(-4) * y, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, -1), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(TCAST(-2) * z, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(x * x, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(y * y, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(z * z, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		flops += 18 - 6 * fmaops;
	}
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= n; k++) {
				const int lmin = std::max(-k, m - n + k);
				const int lmax = std::min(k, m + n - k);
				for (int l = -k; l <= k; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (abs(m - l) > n - k) {
						continue;
					}
					if (-abs(m - l) < k - n) {
						continue;
					}
					if (m - l > 0) {
						asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "M[%i]", index(n - k, 0));
					}
					if (l > 0) {
						asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
						asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gysgn = -1;
						} else {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
						}
					} else {
						asprintf(&gxstr, "Y[%i]", index(k, 0));
					}
					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (gystr && mystr) {
						add_work(-mysgn * gysgn, 1, mystr, gystr);
					}
					if (m > 0) {
						if (gystr) {
							add_work(mxsgn * gysgn, -1, mxstr, gystr);
						}
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int M2M_rot1(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic_xz(P);
	set_tprint(c);

	func_header("M2M", P + 1, nophi, true, "M", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("T Y[%i];\n", exp_sz(P));
	tprint("T Rx, Ry, tmp, R, Rinv, tmp1, R2, Rzero, cosphi, sinphi;\n");
	tprint("R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = (R2<TCAST(1e-37));\n");
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops += rsqrt_flops;
	flops++;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P, "M", false, false, false);
	if (P > 1 && !nophi && periodic) {
		tprint("M[%i] = FMA(TCAST(-4)*R, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 1), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(TCAST(-2)*z, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(R * R, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(z * z, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		flops += 12;
	}
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("regular_harmonic_xz_%s_P%i(Y, -R, -z);\n", type.c_str(), P);
	flops += 2;
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= n; k++) {
				const int lmin = std::max(-k, m - n + k);
				const int lmax = std::min(k, m + n - k);
				for (int l = -k; l <= k; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					if (abs(m - l) > n - k) {
						continue;
					}
					if (-abs(m - l) < k - n) {
						continue;
					}
					if (m - l > 0) {
						asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							asprintf(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "M[%i]", index(n - k, 0));
					}
					asprintf(&gxstr, "Y[%i]", yindex(k, abs(l)));
					if (l < 0 && abs(l) % 2 != 0) {
						gxsgn = -1;
					}
					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (m > 0) {
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("M[%i] = FMA(%s, %s, M[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	flops += z_rot(P, "M", false, false, false);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int M2M_rot2(int P) {
	int flops = 0;
	func_header("M2M", P + 1, nophi, true, "M", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("T A[%i];\n", 2 * P + 1);
	tprint("T Rx, Ry, tmp, R, Rinv, R2, Rzero, tmp1, r2, rzero, r2inv, r, rinv, cosphi, sinphi, cosphi0, sinphi0;\n");
	tprint("R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = TCAST(R2<TCAST(1e-37));");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops += rsqrt_flops;
	tprint("r2 = FMA(z, z, R2);\n");
	flops += 2 - fmaops;
	tprint("rzero = TCAST(r2<TCAST(1e-37));");
	flops += 1;
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops += rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("r = TCAST(1) / rinv;\n");
	flops += divops;
	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	flops += z_rot(P, "M", false, false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = FMA(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "M", false, false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	if (P > 1 && !nophi && periodic) {
		tprint("M[%i] = FMA(TCAST(-2) * r, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(r * r, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		flops += 6 - 2 * fmaops;
	}
	tprint("T c0[%i];\n", P + 1);
	tprint("c0[0] = TCAST(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("c0[%i] = -r * c0[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("c0[%i] *= TCAST(%.20e);\n", n, 1.0 / factorial(n));
		flops += 1;
	}
	for (int n = P; n >= 0; n--) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= n; k++) {
				if (abs(m) > n - k) {
					continue;
				}
				if (-abs(m) < k - n) {
					continue;
				}
				tprint("M[%i] = FMA(M[%i], c0[%i], M[%i]);\n", index(n, m), index(n - k, m), k, index(n, m));
				flops += 2 - fmaops;
				if (m > 0) {
					tprint("M[%i] = FMA(M[%i], c0[%i], M[%i]);\n", index(n, -m), index(n - k, -m), k, index(n, -m));
					flops += 2 - fmaops;
				}
			}

		}
	}
	flops += xz_swap(P, "M", false, false, false, false);
	tprint("sinphi = -sinphi;\n");
	flops += z_rot(P, "M", false, false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(P, "M", false, false, false);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2L_norot(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	func_header("L2L", P, nophi, true, "L", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("T Y[%i];\n", exp_sz(P));
	tprint("regular_harmonic_%s_P%i(Y, -x, -y, -z);\n", type.c_str(), P);
	flops += 3;
	for (int n = nophi; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (m + l > 0) {
						asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "L[%i]", index(n + k, 0));
					}
					if (l > 0) {
						asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
						asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						asprintf(&gxstr, "Y[%i]", index(k, 0));
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};

					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (gystr && mystr) {
						add_work(-mysgn * gysgn, 1, mystr, gystr);
					}
					if (m > 0) {
						if (gystr) {
							add_work(mxsgn * gysgn, -1, mxstr, gystr);
						}
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = FMA(TCAST(-2) * x, L[%i], L[%i]);\n", index(1, 1), (P + 1) * (P + 1), index(1, 1));
		tprint("L[%i] = FMA(TCAST(-2) * y, L[%i], L[%i]);\n", index(1, -1), (P + 1) * (P + 1), index(1, -1));
		tprint("L[%i] = FMA(TCAST(-2) * z, L[%i], L[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
		flops += 9 - 3 * fmaops;
		if (!nophi) {
			tprint("L[%i] = FMA(x * x, L[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			tprint("L[%i] = FMA(y * y, L[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			tprint("L[%i] = FMA(z * z, L[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			flops += 9 - 3 * fmaops;
		}
	}

	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2L_rot1(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic_xz(P);
	set_tprint(c);

	func_header("L2L", P, nophi, true, "L", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("T Y[%i];\n", half_exp_sz(P));
	tprint("T Rx, Ry, tmp, R, Rinv, tmp1, R2, Rzero, cosphi, sinphi;\n");
	tprint("R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = (R2<TCAST(1e-37));\n");
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops += rsqrt_flops;
	flops++;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P, "L", false, false, false);
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("regular_harmonic_xz_%s_P%i(Y, -R, -z);\n", type.c_str(), P);
	flops += 2;
	for (int n = 0; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			for (int k = 1; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					if (m + l > 0) {
						asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "L[%i]", index(n + k, 0));
					}
					asprintf(&gxstr, "Y[%i]", yindex(k, abs(l)));
					if (l < 0 && abs(l) % 2 != 0) {
						gxsgn = -1;
					}
					//	L[index(n, m)] += Y(k, l) * M(n + k, m + l);
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					add_work(mxsgn * gxsgn, +1, mxstr, gxstr);
					if (m > 0) {
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("L[%i] = FMA(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	flops++;
	if (P > 1 && periodic) {
		tprint("L[%i] = FMA(TCAST(-2) * R, L[%i], L[%i]);\n", index(1, 1), (P + 1) * (P + 1), index(1, 1));
		tprint("L[%i] = FMA(TCAST(-2) * z, L[%i], L[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
		flops += 6;
		if (!nophi) {
			tprint("L[%i] = FMA(R2, L[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			tprint("L[%i] = FMA(z * z, L[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			flops += 5;
		}
	}
	flops += z_rot(P, "L", false, false, false);
	flops++;
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2L_rot2(int P) {
	int flops = 0;
	func_header("L2L", P, nophi, true, "L", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("T A[%i];\n", 2 * P + 1);
	tprint("T Rx, Ry, tmp, R, Rinv, tmp1, R2, Rzero, r2, rzero, r, rinv, cosphi, sinphi, cosphi0, sinphi0;\n");
	tprint("R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("Rzero = TCAST(R2<TCAST(1e-37));");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("Rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops += rsqrt_flops;
	tprint("r2 = FMA(z, z, R2);\n");
	flops += 2 - fmaops;
	tprint("rzero = TCAST(r2<TCAST(1e-37));");
	flops += 1;
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = rsqrt_%s(tmp1);\n", type.c_str());
	flops += rsqrt_flops;
	tprint("R = TCAST(1) / Rinv;\n");
	flops += divops;
	tprint("r = TCAST(1) / rinv;\n");
	flops += divops;
	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	flops += z_rot(P, "L", false, false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = FMA(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "L", false, false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("A[0] = TCAST(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = -r * A[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, 1.0 / factorial(n));
		flops += 1;
	}
	for (int n = nophi; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			for (int k = 1; k <= P - n; k++) {
				if (abs(m) > n + k) {
					continue;
				}
				if (-abs(m) < -(k + n)) {
					continue;
				}
				tprint("L[%i] = FMA(L[%i], A[%i], L[%i]);\n", index(n, m), index(n + k, m), k, index(n, m));
				flops += 2 - fmaops;
				if (m > 0) {
					tprint("L[%i] = FMA(L[%i], A[%i], L[%i]);\n", index(n, -m), index(n + k, -m), k, index(n, -m));
					flops += 2 - fmaops;
				}
			}

		}
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = FMA(TCAST(-2) * r, L[%i], L[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
		flops += 3 - fmaops;
		if (!nophi) {
			tprint("L[%i] = FMA(r * r, L[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			flops += 3 - fmaops;
		}
	}
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("sinphi = -sinphi;\n");
	flops += z_rot(P, "L", false, false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(P, "L", false, false, false);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int L2P(int P) {
	int flops = 0;
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	func_header("L2P", P, nophi, true, "L1", PTR, "L", CPTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("T Y[%i];\n", exp_sz(P));
//tprint("expansion_type<T,1> L1;\n");
	tprint("regular_harmonic_%s_P%i(Y, -x, -y, -z);\n", type.c_str(), P);
	flops += 3;
	for (int n = nophi; n <= 1; n++) {
		for (int m = 0; m <= n; m++) {
			std::vector<std::pair<std::string, std::string>> pos_real;
			std::vector<std::pair<std::string, std::string>> neg_real;
			std::vector<std::pair<std::string, std::string>> pos_imag;
			std::vector<std::pair<std::string, std::string>> neg_imag;
			const auto add_work = [&pos_real,&pos_imag,&neg_real,&neg_imag](int sgn, int m, char* mstr, char* gstr) {
				if( sgn == 1) {
					if( m >= 0 ) {
						pos_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						pos_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				} else {
					if( m >= 0 ) {
						neg_real.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					} else {
						neg_imag.push_back(std::make_pair(std::string(mstr),std::string(gstr)));
					}
				}
			};
			bool pfirst = true;
			bool nfirst = true;
			for (int k = 0; k <= P - n; k++) {
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					char* mxstr = nullptr;
					char* mystr = nullptr;
					char* gxstr = nullptr;
					char* gystr = nullptr;
					int mxsgn = 1;
					int mysgn = 1;
					int gxsgn = 1;
					int gysgn = 1;
					if (m + l > 0) {
						asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
						asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mysgn = -1;
						} else {
							asprintf(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							asprintf(&mystr, "L[%i]", index(n + k, -abs(m + l)));
							mxsgn = -1;
						}
					} else {
						asprintf(&mxstr, "L[%i]", index(n + k, 0));
					}
					if (l > 0) {
						asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
						asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							asprintf(&gxstr, "Y[%i]", index(k, abs(l)));
							asprintf(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						asprintf(&gxstr, "Y[%i]", index(k, 0));
					}
					add_work(mxsgn * gxsgn, 1, mxstr, gxstr);
					if (gystr && mystr) {
						add_work(-mysgn * gysgn, 1, mystr, gystr);
					}
					if (m > 0) {
						if (gystr) {
							add_work(mxsgn * gysgn, -1, mxstr, gystr);
						}
						if (mystr) {
							add_work(mysgn * gxsgn, -1, mystr, gxstr);
						}
					}
					if (gxstr) {
						free(gxstr);
					}
					if (mxstr) {
						free(mxstr);
					}
					if (gystr) {
						free(gystr);
					}
					if (mystr) {
						free(mystr);
					}
				}
			}
			if (fmaops && neg_real.size() >= 2) {
				tprint("L1[%i] = -L1[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L1[%i] = FMA(%s, %s, L1[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
					flops++;
				}
				tprint("L1[%i] = -L1[%i];\n", index(n, m), index(n, m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("L1[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("L1[%i] = FMA(%s, %s, L1[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
				flops += 2 - fmaops;
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("L1[%i] = -L1[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L1[%i] = FMA(%s, %s, L1[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
					flops++;
				}
				tprint("L1[%i] = -L1[%i];\n", index(n, -m), index(n, -m));
				flops += 2;
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("L1[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					flops += 2;
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("L1[%i] = FMA(%s, %s, L1[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
				flops += 2 - fmaops;
			}
		}
	}
	if (P >= 1 && periodic) {
		tprint("L1[%i] = FMA(TCAST(-2) * x, L[%i], L1[%i]);\n", index(1, 1), (P + 1) * (P + 1), index(1, 1));
		tprint("L1[%i] = FMA(TCAST(-2) * y, L[%i], L1[%i]);\n", index(1, -1), (P + 1) * (P + 1), index(1, -1));
		tprint("L1[%i] = FMA(TCAST(-2) * z, L[%i], L1[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
		flops += 9;
		if (!nophi) {
			tprint("L1[%i] = FMA(x * x, L[%i], L1[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			tprint("L1[%i] = FMA(y * y, L[%i], L1[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			tprint("L1[%i] = FMA(z * z, L[%i], L1[%i]);\n", index(0, 0), (P + 1) * (P + 1), index(0, 0));
			flops += 9;
		}
	}
	deindent();
	tprint("}\n");
	tprint("\n");

	return flops;
}

int P2M(int P) {
	int flops = 0;
	tprint("\n");
	const auto c = tprint_on;
	set_tprint(false);
	flops += regular_harmonic(P);
	set_tprint(c);
	func_header("P2M", P + 1, nophi, true, "M", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("regular_harmonic_%s_P%i(M, -x, -y, -z);\n", type.c_str(), P);
	if (!nophi) {
		tprint("M[%i] = FMA(x, x, FMA(y, y, z * z));\n", (P + 1) * (P + 1));
		flops += 5 - 2 * fmaops;
	}
	deindent();
	tprint("}\n");
	tprint("\n");
	return flops + 3;
}

void math_functions() {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
#ifdef FLOAT
	fp = fopen("./generated_code/include/math_float.h", "wt");
	tprint("#ifndef SPHERICAL_FMM_MATH_FLOAT\n");
	tprint("#define SPHERICAL_FMM_MATH_FLOAT\n");
	tprint("\n");
	tprint("#define FMA(a, b, c) fmaf((a),(b),(c))\n");
	tprint("\n");
	tprint("#ifdef __cplusplus\n");
	tprint("extern \"C\" {\n");
	tprint("#endif\n");
	tprint("float rsqrt_float( float );\n");
	tprint("float sqrt_float( float );\n");
	tprint("void sincos_float( float, float*, float* );\n");
	tprint("void erfcexp_float( float, float*, float* );\n");
	tprint("#ifdef __cplusplus\n");
	tprint("}\n");
	tprint("#endif\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
	fp = fopen("./generated_code/src/math_float.c", "wt");
	tprint("\n");
	tprint("#include \"typecast_float.h\"\n");
	tprint("#include <math.h>\n");
	tprint("\n");
	tprint("#define TCAST(a) ((float)(a))\n");
	tprint("#define UCAST(a) ((unsigned)(a))\n");
	tprint("#define VCAST(a) ((int)(a))\n");
	tprint("\n");
	tprint("typedef float T;\n");
	tprint("typedef unsigned U;\n");
	tprint("typedef int V;\n");
	tprint("\n");
	tprint("T rsqrt_float( T x ) {\n");
	indent();
	tprint("V i;\n");
	tprint("T y;\n");
	tprint("x += TCAST(%.20e);\n", std::numeric_limits<float>::min());
	tprint("i = *((V*) &x);\n");
	tprint("i >>= VCAST(1);\n");
	tprint("i = VCAST(0x5F3759DF) - i;\n");
	tprint("y = *((T*) &i);\n");
	tprint("y *= fmaf(TCAST(-0.5), x * y * y, TCAST(1.5));\n");
	tprint("y *= fmaf(TCAST(-0.5), x * y * y, TCAST(1.5));\n");
	tprint("y *= fmaf(TCAST(-0.5), x * y * y, TCAST(1.5));\n");
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T sqrt_float(float x) {\n");
	indent();
	tprint("return TCAST(1) / rsqrt_float(x);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("void sincos_float( T x, T* s, T* c ) {\n");
	indent();
	tprint("V ssgn, j, i, k;\n");
	tprint("T x2;\n");
	tprint("ssgn = VCAST(((*((U*) &x) & UCAST(0x80000000)) >> UCAST(30)) - UCAST(1));\n");
	tprint("j = VCAST((*((U*) &x) & UCAST(0x7FFFFFFF)));\n");
	tprint("x = *((T*) &j);\n");
	tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
	tprint("x -= i * TCAST(%.20e);\n", M_PI);
	tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
	tprint("x2 = x * x;\n");
	{
		constexpr int N = 11;
		tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
		tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
		for (int n = N - 2; n >= 0; n -= 2) {
			tprint("%s = fmaf(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = fmaf(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
		}
	}
	tprint("%s *= x;\n", cout);
	tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
	tprint("%s *= TCAST(ssgn * k);\n", sout);
	tprint("%s *= TCAST(k);\n", cout);
	deindent();
	tprint("}\n");
	tprint("\n");

	tprint("T exp_float( T x ) {\n");
	{
		indent();
		constexpr int N = 7;
		tprint("V k;\n");
		tprint("T y, xxx;\n");
		tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
		tprint("k -= x < TCAST(0);\n");
		tprint("xxx = x - k * TCAST(0.6931471805599453094172);\n");
		tprint("y = TCAST(%.20e);\n", 1.0 / factorial(N));
		for (int i = N - 1; i >= 0; i--) {
			tprint("y = fmaf(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-fmaops);
		}
		tprint("k = (k + VCAST(127)) << VCAST(23);\n");
		tprint("y *= *((T*) (&k));\n"); //1
		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	{

		tprint("void erfcexp_float( T x, T* erfc0, T* exp0 ) {\n");
		indent();
		tprint("T x2, nx2, q, x0;\n");
		tprint("x2 = x * x;\n");
		tprint("nx2 = -x2;\n");
		tprint("*exp0 = exp_float(nx2);\n");

		constexpr double x0 = 2.75;
		tprint("if (x < TCAST(%.20e) ) {\n", x0);
		{
			indent();
			constexpr int N = 25;
			tprint("q = TCAST(2) * x * x;\n");
			tprint("*erfc0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N + 1));
			for (int n = N - 1; n >= 0; n--) {
				tprint("*erfc0 = fmaf(*erfc0, q, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n + 1));
			}
			tprint("*erfc0 *= TCAST(%.20e) * x * *exp0;\n", 2.0 / sqrt(M_PI));
			tprint("*erfc0 = TCAST(1) - *erfc0;\n");
			deindent();
		}
		tprint("} else  {\n");
		{
			indent();
			constexpr int N = x0 * x0 + 0.5;
			tprint("q = TCAST(1) / (TCAST(2) * x * x);\n");
			tprint("*erfc0 = TCAST(%.20e);\n", dfactorial(2 * N - 1) * nonepow(N));
			for (int i = N - 1; i >= 1; i--) {
				tprint("*erfc0 = fmaf(*erfc0, q, TCAST(%.20e));\n", dfactorial(2 * i - 1) * nonepow(i));
			}
			tprint("*erfc0 = fmaf(*erfc0, q, TCAST(1));\n");
			tprint("*erfc0 *= *exp0 * TCAST(%.20e) / x;\n", 1.0 / sqrt(M_PI));
			deindent();
		}
		tprint("}\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	fclose(fp);
#endif
#ifdef DOUBLE
	fp = fopen("./generated_code/include/math_double.h", "wt");
	tprint("#ifndef SPHERICAL_FMM_MATH_DOUBLE\n");
	tprint("#define SPHERICAL_FMM_MATH_DOUBLE\n");
	tprint("\n");
	tprint("#define FMA(a, b, c) fma((a),(b),(c))\n");
	tprint("\n");
	tprint("#ifdef __cplusplus\n");
	tprint("extern \"C\" {\n");
	tprint("#endif\n");
	tprint("double rsqrt_double( double );\n");
	tprint("double sqrt_double( double );\n");
	tprint("void sincos_double( double, double*, double* );\n");
	tprint("void erfcexp_double( double, double*, double* );\n");
	tprint("#ifdef __cplusplus\n");
	tprint("}\n");
	tprint("#endif\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
	fp = fopen("./generated_code/src/math_double.c", "wt");
	tprint("\n");
	tprint("#include <math.h>\n");
	tprint("#include \"typecast_double.h\"\n");
	tprint("\n");
	tprint("#define TCAST(a) ((double)(a))\n");
	tprint("#define UCAST(a) ((unsigned long long)(a))\n");
	tprint("#define VCAST(a) ((long long)(a))\n");
	tprint("\n");
	tprint("typedef double T;\n");
	tprint("typedef unsigned long long U;\n");
	tprint("typedef long long V;\n");
	tprint("\n");
	tprint("double rsqrt_double( T x ) {\n");
	indent();
	tprint("V i;\n");
	tprint("T y;\n");
	tprint("x += TCAST(%.20e);\n", std::numeric_limits<double>::min());
	tprint("i = *((V*) &x);\n");
	tprint("i >>= VCAST(1);\n");
	tprint("i = VCAST(0x5FE6EB50C7B537A9) - i;\n");
	tprint("y = *((T*) &i);\n");
	tprint("y *= fma(TCAST(-0.5), x * y * y, TCAST(1.5));\n");
	tprint("y *= fma(TCAST(-0.5), x * y * y, TCAST(1.5));\n");
	tprint("y *= fma(TCAST(-0.5), x * y * y, TCAST(1.5));\n");
	tprint("y *= fma(TCAST(-0.5), x * y * y, TCAST(1.5));\n");
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T sqrt_double(T x) {\n");
	indent();
	tprint("return TCAST(1) / rsqrt_double(x);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("void sincos_double( T x, T* s, T* c ) {\n");
	indent();
	tprint("V ssgn, j, i, k;\n");
	tprint("T x2;\n");
	tprint("ssgn = VCAST(((*((U*) &x) & UCAST(0x8000000000000000LL)) >> UCAST(62LL)) - UCAST(1LL));\n");
	tprint("j = VCAST((*((U*) &x) & UCAST(0x7FFFFFFFFFFFFFFFLL)));\n");
	tprint("x = *((T*) &j);\n");
	tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
	tprint("x -= i * TCAST(%.20e);\n", M_PI);
	tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
	tprint("x2 = x * x;\n");
	{
		constexpr int N = 21;
		tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
		tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
		for (int n = N - 2; n >= 0; n -= 2) {
			tprint("%s = fmaf(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = fmaf(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
		}
	}
	tprint("%s *= x;\n", cout);
	tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
	tprint("%s *= TCAST(ssgn * k);\n", sout);
	tprint("%s *= TCAST(k);\n", cout);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T exp_double( T x ) {\n");
	{
		indent();
		constexpr int N = 18;
		tprint("V k;\n");
		tprint("T xxx, y;\n");
		tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
		tprint("k -= x < TCAST(0);\n");
		tprint("xxx = x - k * TCAST(0.6931471805599453094172);\n");
		tprint("y = TCAST(%.20e);\n", 1.0 / factorial(N));
		for (int i = N - 1; i >= 0; i--) {
			tprint("y = fma(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-fmaops);
		}
		tprint("k = (k + VCAST(1023)) << VCAST(52);\n");
		tprint("y *= *((T*) (&k));\n"); //1
		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	tprint("void erfcexp_double( T x, T* erfc0, T* exp0 ) {\n");
	indent();
	tprint("T x2, nx2, q, x0;\n");
	tprint("x2 = x * x;\n");
	tprint("nx2 = -x2;\n");
	tprint("*exp0 = exp_double(nx2);\n");

	constexpr double x0 = 1.0;
	constexpr double x1 = 4.7;
	tprint("if (x < TCAST(%.20e) ) {\n", x0);
	{
		indent();
		constexpr int N = 17;
		tprint("q = TCAST(2) * x * x;\n");
		tprint("*erfc0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N + 1));
		for (int n = N - 1; n >= 0; n--) {
			tprint("*erfc0 = fma(*erfc0, q, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n + 1));
		}
		tprint("*erfc0 *= TCAST(%.20e) * x * *exp0;\n", 2.0 / sqrt(M_PI));
		tprint("*erfc0 = TCAST(1) - *erfc0;\n");
		deindent();
	}
	tprint("} else if (x < TCAST(%.20e) ) {\n", x1);
	indent();
	{
		constexpr int N = 35;
		constexpr double a = (x1 - x0) * 0.5 + x0;
		static double c0[N + 1];
		static bool init = false;
		if (!init) {
			double q[N + 1];
			init = true;
			c0[0] = (a) * exp(a * a) * erfc(a);
			q[0] = exp(a * a) * erfc(a);
			q[1] = -2.0 / sqrt(M_PI) + 2 * a * exp(a * a) * erfc(a);
			for (int n = 2; n <= N; n++) {
				q[n] = 2 * (a * q[n - 1] + q[n - 2]) / n;
			}
			for (int n = 1; n <= N; n++) {
				c0[n] = q[n - 1] + (a) * q[n];
			}
		}
		tprint("x0 = x;\n");
		tprint("x -= TCAST(%.20e);\n", a);
		tprint("*erfc0 = TCAST(%.20e);\n", c0[N]);
		for (int n = N - 1; n >= 0; n--) {
			tprint("*erfc0 = fma(*erfc0, x, TCAST(%.20e));\n", c0[n]);
		}
		tprint("*erfc0 *= *exp0 / x0;\n");
		deindent();
	}
	tprint("} else  {\n");
	{
		indent();
		constexpr int N = x1 * x1 + 0.5;
		tprint("q = TCAST(1) / (TCAST(2) * x * x);\n");
		tprint("*erfc0 = TCAST(%.20e);\n", dfactorial(2 * N - 1) * nonepow(N));
		for (int i = N - 1; i >= 1; i--) {
			tprint("*erfc0 = fma(*erfc0, q, TCAST(%.20e));\n", dfactorial(2 * i - 1) * nonepow(i));
		}
		tprint("*erfc0 = fma(*erfc0, q, TCAST(1));\n");
		tprint("*erfc0 *= *exp0 * TCAST(%.20e) / x;\n", 1.0 / sqrt(M_PI));
		deindent();
	}
	tprint("}\n");

	deindent();
	tprint("}\n");
	tprint("\n");

	fclose(fp);
#endif
	fp = nullptr;
}


void typecast_functions() {
	if (fp) {
		fclose(fp);
	}
#ifdef FLOAT
	fp = fopen("./generated_code/include/typecast_float.h", "wt");
	tprint("#ifndef SPHERICAL_FMM_TYPECAST_FLOAT\n");
	tprint("#define SPHERICAL_FMM_TYPECAST_FLOAT\n");
	tprint("\n");
	tprint("#define TCAST(a) ((float)(a))\n");
	tprint("#define UCAST(a) ((unsigned)(a))\n");
	tprint("#define VCAST(a) ((int)(a))\n");
	tprint("\n");
	tprint("typedef float T;\n");
	tprint("typedef unsigned U;\n");
	tprint("typedef int V;\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
#endif
#ifdef DOUBLE
	fp = fopen("./generated_code/include/typecast_double.h", "wt");
	tprint("#ifndef SPHERICAL_FMM_TYPECAST_DOUBLE\n");
	tprint("#define SPHERICAL_FMM_TYPECAST_DOUBLE\n");
	tprint("\n");
	tprint("#define TCAST(a) ((double)(a))\n");
	tprint("#define UCAST(a) ((unsigned long long)(a))\n");
	tprint("#define VCAST(a) ((long long)(a))\n");
	tprint("\n");
	tprint("typedef double T;\n");
	tprint("typedef unsigned long long U;\n");
	tprint("typedef long long V;\n");
	tprint("\n");
	tprint("#endif\n");
	fclose(fp);
#endif
	fp = nullptr;
}

int main() {
	system("[ -e ./generated_code ] && rm -r  ./generated_code\n");
	system("mkdir generated_code\n");
	system("mkdir ./generated_code/include\n");
	system("mkdir ./generated_code/include/detail\n");
	system("mkdir ./generated_code/src\n");
	math_functions();
	typecast_functions();
	set_file("./generated_code/include/spherical_fmm.h");
	tprint("#ifndef SPHERICAL_FMM_HEADER\n");
	tprint("#define SPHERICAL_FMM_HEADER\n");
	tprint("\n");
	tprint("#include <math.h>\n");
	tprint("\n");

	tprint("#ifdef __cplusplus\n");
	tprint("extern \"C\" {\n");
	tprint("#else\n");
	tprint("#define bool int\n");
	tprint("#define true 1\n");
	tprint("#define false 0\n");
	tprint("#endif\n");

	static int rsqrt_float_flops = 15 - 3 * fmaops;
	static int rsqrt_double_flops = 15 - 3 * fmaops;
	static int sqrt_float_flops = 15 - 3 * fmaops + divops;
	static int sqrt_double_flops = 15 - 3 * fmaops + divops;
	static int sincos_float_flops = 31 - fmaops * 10;
	static int sincos_double_flops = 41 - fmaops * 20;
	static int exp_double_flops = 4 + divops + 17 * (2 - fmaops);
	static int exp_float_flops = 4 + divops + 7 * (2 - fmaops);
	static int erfcexp_float_flops = exp_float_flops + 56 - 24 * fmaops;
	static int erfcexp_double_flops = exp_double_flops + 72 - 35 * fmaops + divops;

	const int rsqrt_flops_array[] = { rsqrt_float_flops, rsqrt_double_flops };
	const int sqrt_flops_array[] = { sqrt_float_flops, sqrt_double_flops };
	const int sincos_flops_array[] = { sincos_float_flops, sincos_double_flops };
	const int erfcexp_flops_array[] = { erfcexp_float_flops, erfcexp_double_flops };
	const char* rtypenames[] = { "float", "double" };
	const char* sitypenames[] = { "int", "long long" };
	const char* uitypenames[] = { "unsigned", "unsigned long long" };
	const int ntypenames = 2;

	for (int ti = 0; ti < ntypenames; ti++) {
		type = rtypenames[ti];
		sitype = sitypenames[ti];
		uitype = uitypenames[ti];
		rsqrt_flops = rsqrt_flops_array[ti];
		sqrt_flops = sqrt_flops_array[ti];
		sincos_flops = sincos_flops_array[ti];
		erfcexp_flops = erfcexp_flops_array[ti];
		std::vector<double> alphas(pmax + 1);
		for (int b = 0; b < 2; b++) {
			nophi = b != 0;
			std::vector<int> pc_flops(pmax + 1);
			std::vector<int> cp_flops(pmax + 1);
			std::vector<int> cc_flops(pmax + 1);
			std::vector<int> m2m_flops(pmax + 1);
			std::vector<int> l2l_flops(pmax + 1);
			std::vector<int> l2p_flops(pmax + 1);
			std::vector<int> p2m_flops(pmax + 1);
			std::vector<int> pc_rot(pmax + 1);
			std::vector<int> cc_rot(pmax + 1);
			std::vector<int> m2m_rot(pmax + 1);
			std::vector<int> l2l_rot(pmax + 1);

			set_tprint(false);
			fprintf(stderr, "%2s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s ", "p", "M2L", "eff", "-r", "M2P", "eff", "-r", "P2L",
					"eff", "M2M", "eff", "-r", "L2L", "eff", "-r", "P2M", "eff", "L2P", "eff");
			if (b == 0) {
				fprintf(stderr, " %8s %8s %8s %8s\n", "CC_ewald", "green", "m2l", "alpha");
			} else {
				fprintf(stderr, " \n");

			}
			int eflopsg;
			for (int P = pmin; P <= pmax; P++) {
				auto r0 = m2l_norot(P, P);
				auto r1 = m2l_rot1(P, P);
				auto r2 = m2l_rot2(P, P);
				if (r0 <= r1 && r0 <= r2) {
					cc_flops[P] = r0;
					cc_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					cc_flops[P] = r1;
					cc_rot[P] = 1;
				} else {
					cc_flops[P] = r2;
					cc_rot[P] = 2;
				}
				r0 = m2l_norot(P, 1);
				r1 = m2l_rot1(P, 1);
				r2 = m2l_rot2(P, 1);
				if (r0 <= r1 && r0 <= r2) {
					pc_flops[P] = r0;
					pc_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					pc_flops[P] = r1;
					pc_rot[P] = 1;
				} else {
					pc_flops[P] = r2;
					pc_rot[P] = 2;
				}
				cp_flops[P] = p2l(P);
				r0 = M2M_norot(P - 1);
				r1 = M2M_rot1(P - 1);
				r2 = M2M_rot2(P - 1);
				if (r0 <= r1 && r0 <= r2) {
					m2m_flops[P] = r0;
					m2m_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					m2m_flops[P] = r1;
					m2m_rot[P] = 1;
				} else {
					m2m_flops[P] = r2;
					m2m_rot[P] = 2;
				}
				r0 = L2L_norot(P);
				r1 = L2L_rot1(P);
				r2 = L2L_rot2(P);
				if (r0 <= r1 && r0 <= r2) {
					l2l_flops[P] = r0;
					l2l_rot[P] = 0;
				} else if (r1 <= r0 && r1 <= r2) {
					l2l_flops[P] = r1;
					l2l_rot[P] = 1;
				} else {
					l2l_flops[P] = r2;
					l2l_rot[P] = 2;
				}
				l2p_flops[P] = L2P(P);
				p2m_flops[P] = P2M(P - 1);
				int eflopsm = m2lg(P, P);
				if (b == 0) {
					double best_alpha;
					int best_ops = 1000000000;
					for (double alpha = 1.9; alpha <= 2.6; alpha += 0.1) {
						int ops = ewald_greens(P, alpha);

//					printf( "%i %e %i\n", P, alpha, ops);
						if (ops < best_ops) {
							best_ops = ops;
							best_alpha = alpha;
						}
					}
					alphas[P] = best_alpha;
					eflopsg = ewald_greens(P, best_alpha);

				}
				fprintf(stderr, "%2i %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f ", P, cc_flops[P],
						cc_flops[P] / pow(P + 1, 3), cc_rot[P], pc_flops[P], pc_flops[P] / pow(P + 1, 2), pc_rot[P], cp_flops[P], cp_flops[P] / pow(P + 1, 2),
						m2m_flops[P], m2m_flops[P] / pow(P + 1, 3), m2m_rot[P], l2l_flops[P], l2l_flops[P] / pow(P + 1, 3), l2l_rot[P], p2m_flops[P],
						p2m_flops[P] / pow(P + 1, 2), l2p_flops[P], l2p_flops[P] / pow(P + 1, 2));
				if (b == 0) {
					fprintf(stderr, " %8i %8i %8i %f \n", eflopsg + eflopsm, eflopsg, eflopsm, alphas[P]);
				} else {
					fprintf(stderr, "\n");
				}
			}
			set_tprint(true);
			if (b == 0)
				regular_harmonic(pmin - 1);
			if (b == 0)
				regular_harmonic_xz(pmin - 1);
			for (int P = 3; P <= pmax; P++) {
				if (b == 0)
					greens(P);
				if (b == 0)
					greens_xz(P);
				switch (cc_rot[P]) {
				case 0:
					m2l_norot(P, P);
					break;
				case 1:
					m2l_rot1(P, P);
					break;
				case 2:
					m2l_rot2(P, P);
					break;
				};
				if (P > 1) {
					switch (pc_rot[P]) {
					case 0:
						m2l_norot(P, 1);
						break;
					case 1:
						m2l_rot1(P, 1);
						break;
					case 2:
						m2l_rot2(P, 1);
						break;
					};
				}
				p2l(P);
				if (b == 0)
					regular_harmonic(P);
				if (b == 0)
					regular_harmonic_xz(P);
				switch (m2m_rot[P]) {
				case 0:
					M2M_norot(P - 1);
					break;
				case 1:
					M2M_rot1(P - 1);
					break;
				case 2:
					M2M_rot2(P - 1);
					break;
				};
				switch (l2l_rot[P]) {
				case 0:
					L2L_norot(P);
					break;
				case 1:
					L2L_rot1(P);
					break;
				case 2:
					L2L_rot2(P);
					break;
				};
				L2P(P);
				P2M(P - 1);
				if (b == 0)
					ewald_greens(P, alphas[P]);
				m2lg(P, P);
				m2l_ewald(P);
			}
		}
	}
//	printf("./generated_code/include/spherical_fmm.h");
	fflush(stdout);
	set_file("./generated_code/include/spherical_fmm.h");
	tprint(inter_header.c_str());
	tprint("#ifdef __cplusplus\n");
	tprint("}\n");
	tprint("#endif\n");
	tprint("\n#endif\n");
	set_file("./generated_code/src/interface.c");
	tprint(inter_src.c_str());
	return 0;
}
