#include <stdio.h>
#include <cmath>
#include <utility>
#include <algorithm>
#include <vector>
#include <climits>
#include <unordered_map>
#include <complex>
#include <set>

using complex = std::complex<double>;

static int ntab = 0;
static int tprint_on = true;

static bool nophi = false;
static bool fmaops = true;
static bool periodic = true;
static int pmin = 3;
static int pmax = 12;
static const char* type = "float";
static const char* prefix = "";
static std::string inter_src = "#include \"spherical_fmm.hpp\"\n\n";
static std::string inter_header = "\n\nenum fmm_calcpot_type {FMM_CALC_POT, FMM_NOCALC_POT};\n\n";

const double ewald_r2 = (2.6 + 0.5 * sqrt(3));
const int ewald_h2 = 8;

static FILE* fp = nullptr;

int exp_sz(int P) {
	if (periodic) {
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
		std::string file_name = func_name + ".cpp";
		func_name = "void " + func_name;
		if (prefix[0]) {
			func_name = std::string(prefix) + " " + func_name;
		}
		func_name += "(" + func_args(std::forward<Args>(args)..., 0);
		func_name += ")";
		if (!pub) {
			set_file("./generated_code/include/detail/spherical_fmm.hpp");
		} else {
			set_file("./generated_code/include/spherical_fmm.hpp");
		}
		tprint("%s;\n", func_name.c_str());
		std::string func1 = std::string(func) + std::string("_") + std::string(type);
		if (pub && igen.find(func1) == igen.end() && !nopot) {
			igen.insert(func1);
			if (prefix[0]) {
				inter_header += std::string(prefix) + " ";
			}
			inter_header += std::string("void ") + func1 + "( int P, " + func_args(std::forward<Args>(args)..., 0);
			inter_header += std::string(", fmm_calcpot_type calcpot");
			inter_header += std::string(");\n");
			if (prefix[0]) {
				inter_src += std::string(prefix) + " ";
			}
			inter_src += std::string("void ") + func1 + "( int P, " + func_args(std::forward<Args>(args)..., 0);
			inter_src += std::string(", fmm_calcpot_type calcpot");
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
		tprint("#include \"spherical_fmm.hpp\"\n");
		tprint("#include \"detail/spherical_fmm.hpp\"\n");
		tprint("\n");
		tprint("%s {\n", func_name.c_str());
		indent();
		tprint("typedef %s T;\n", type);
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

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

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
	tprint("\n");
//	tprint("template<class T>\n");
	tprint("{\n");
//	tprint(" inline void %s( array<T,%i>& %s, T cosphi, T sinphi ) {\n", fname, (P + 1) * (P + 1), name);
	indent();
	tprint("T tmp;\n");
	tprint("T Rx = cosphi;\n");
	tprint("T Ry = sinphi;\n");
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
	deindent();
	tprint("}\n");
	return flops;
}

int m2l(int P, int Q, const char* mname, const char* lname) {
	int flops = 0;
	tprint("{\n");
	indent();
	tprint("T c0[%i];\n", P + 1);
	tprint("c0[0] = rinv;\n");
	for (int n = 1; n <= P; n++) {
		tprint("c0[%i] = rinv * c0[%i];\n", n, n - 1);
		flops++;
	}
	for (int n = 2; n <= P; n++) {
		tprint("c0[%i] *= T(%.16e);\n", n, factorial(n));
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
					tprint("%s[%i] = %s[%i] * c0[%i];\n", lname, index(n, m), mname, index(k, m), n + k);
					flops += 1;
				} else {
					tprint("%s[%i] = FMA(%s[%i], c0[%i], %s[%i]);\n", lname, index(n, m), mname, index(k, m), n + k, lname, index(n, m));
					flops += 2 - fmaops;
				}
				if (m != 0) {
					if (nfirst) {
						nfirst = false;
						tprint("%s[%i] = %s[%i] * c0[%i];\n", lname, index(n, -m), mname, index(k, -m), n + k);
						flops += 1;
					} else {
						tprint("%s[%i] = FMA(%s[%i], c0[%i], %s[%i]);\n", lname, index(n, -m), mname, index(k, -m), n + k, lname, index(n, -m));
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
	deindent();
	tprint("}\n");
	return flops;

}

int xz_swap(int P, const char* name, bool inv, bool m_restrict, bool l_restrict, bool noevenhi) {
	//noevenhi = false;
	tprint("\n");
//	tprint("template<class T>\n");
//	tprint(" inline void %s( array<T,%i>& %s ) {\n", fname, (P + 1) * (P + 1), name);
	tprint("{\n");
	indent();
	tprint("T A[%i];\n", 2 * P + 1);
	tprint("T tmp;\n");
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
		std::vector < std::vector<std::pair<float, int>>>ops(2 * n + 1);
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
				//			tprint("%s[%i] = T(0);\n", name, index(n, m - n));
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
							tprint("%s[%i] = T(%.16e) * A[%i];\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second);
							flops += 1;
						} else {
							tprint("%s[%i] = FMA(T(%.16e), A[%i], %s[%i]);\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second, name, index(n, m - n));
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
						tprint("%s[%i] = T(%.16e) * tmp;\n", name, index(n, m - n), ops[m][l].first);
						flops += 1;
					} else {
						tprint("%s[%i] = FMA(T(%.16e), tmp, %s[%i]);\n", name, index(n, m - n), ops[m][l].first, name, index(n, m - n));
						flops += 2 - fmaops;
					}
				}
				l += len - 1;
			}

		}
	}
	deindent();
	tprint("}\n");
	return flops;
}

int greens_body(int P, const char* M = nullptr) {
	int flops = 0;
	tprint("const T r2 = FMA(x, x, FMA(y, y, z * z));\n");
	flops += 5 - 2 * fmaops;
	if (M) {
		tprint("const T r2inv = %s / r2;\n", M);
	} else {
		tprint("const T r2inv = T(1) / r2;\n");
	}
	flops += 4;
	tprint("O[0] = SQRT(r2inv);\n");
	tprint("O[%i] = T(0);\n", (P + 1) * (P + 1));
	flops += 4;
	tprint("x *= r2inv;\n");
	tprint("y *= r2inv;\n");
	tprint("z *= r2inv;\n");
	flops += 3;
	tprint("T ax;\n");
	tprint("T ay;\n");
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			tprint("O[%i] = y * O[0];\n", index(m, -m));
			flops += 2;
		} else if (m > 0) {
			tprint("ax = O[%i] * T(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("ay = O[%i] * T(%i);\n", index(m - 1, -(m - 1)), 2 * m - 1);
			tprint("O[%i] = x * ax - y * ay;\n", index(m, m));
			tprint("O[%i] = FMA(y, ax, x * ay);\n", index(m, -m));
			flops += 8 - fmaops;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = T(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			flops += 2;
			if (m != 0) {
				tprint("O[%i] = T(%i) * z * O[%i];\n", index(m + 1, -m), 2 * m + 1, index(m, -m));
				flops += 2;
			}
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = T(%i) * z;\n", 2 * n - 1);
				tprint("ay = T(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = (ax * O[%i] + ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				tprint("O[%i] = FMA(ax, O[%i], ay * O[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
				flops += 8 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (T(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					flops += 4;
				} else {
					tprint("O[%i] = (T(%i) * z * O[%i] - T(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
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
	tprint("T O[%i];\n", exp_sz(P));
	flops += greens_body(P, "M");
	for (int n = nophi; n < (P + 1) * (P + 1); n++) {
		tprint("L[%i] += O[%i];\n", n, n);
		flops++;
	}
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
	tprint("O[0] = SQRT(r2inv);\n");
	tprint("O[%i] = T(0);\n", (P + 1) * (P + 1));
	flops += 4;
	tprint("x *= r2inv;\n");
	flops += 1;
	tprint("z *= r2inv;\n");
	flops += 1;
	tprint("T ax;\n");
	tprint("T ay;\n");
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			flops += 1;
		} else if (m > 0) {
			tprint("ax = O[%i] * T(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("O[%i] = x * ax;\n", index(m, m));
			flops += 2;
		}
		if (m + 1 <= P) {
			tprint("O[%i] = T(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			flops += 2;
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = T(%i) * z;\n", 2 * n - 1);
				tprint("ay = T(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = FMA(ax, O[%i], ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				flops += 5 - fmaops;
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (T(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
					flops += 4;
				} else {
					tprint("O[%i] = (T(%i) * z * O[%i] - T(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
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
	tprint("T L[%i];\n", exp_sz(Q));
	tprint("T M[%i];\n", mul_sz(P));
	tprint("T O[%i];\n", half_exp_sz(P));
	tprint("for( int n = 0; n < %i; n++) {\n", P * P + 1);
	indent();
	tprint("M[n] = M0[n];\n");
	deindent();
	tprint("}\n");
	const auto tpo = tprint_on;

	set_tprint(false);
	flops += greens_xz(P);
	set_tprint(tpo);
	/*	tprint("for( int n = 0; n < %i; n++) {\n", (Q + 1) * (Q + 1));
	 indent();
	 tprint("L[n] = T(0);\n");
	 deindent();
	 tprint("}\n");*/
	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = T(R<T(1e-37));\n");
	flops++;
	tprint("const T r2 = FMA(z, z, R2);\n");
	tprint("const T rzero = T(r2<T(1e-37));\n");
	flops++;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T r2inv = T(1) / (r2+rzero);\n");
	flops += 7 - fmaops;
	tprint("T cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("T sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P - 1, "M", false, false, false);
	tprint("greens_xz_%s_P%i(O, R, z, r2inv);\n", type, P);
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
	flops += z_rot(Q, "L", false, false, true);
	flops++;
	tprint("for( int n = 0; n < %i; n++) {\n", (Q + 1) * (Q + 1));
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
		tprint("greens_ewald_%s_P%i(G, x, y, z);\n", type, P);
		tprint("M2LG_%s_P%i%s(L,M,G);\n", type, P, nophi ? "_nopot" : "");
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
		tprint("L[%i] = FMA(T(-0.5) * O[%i], M[%i], L[%i]);\n", index(0, 0), (P + 1) * (P + 1), P * P, index(0, 0));
		flops += 3 - fmaops;
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = FMA(T(-2) * O[%i], M[%i], L[%i]);\n", index(1, -1), (P + 1) * (P + 1), index(1, -1), index(1, -1));
		tprint("L[%i] -= O[%i] * M[%i];\n", index(1, +0), (P + 1) * (P + 1), index(1, +0), index(1, +0));
		tprint("L[%i] = FMA(T(-2) * O[%i], M[%i], L[%i]);\n", index(1, +1), (P + 1) * (P + 1), index(1, +1), index(1, +1));
		tprint("L[%i] = FMA(T(-0.5) * O[%i], M[%i], L[%i]);\n", (P + 1) * (P + 1), (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
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
	tprint("greens_%s_P%i(O, x, y, z);\n", type, P);
	flops += m2lg_body(P, Q);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int ewald_greens(int P) {
	if (!periodic) {
		return 0;
	}
	int flops = 0;
	func_header("greens_ewald", P, false, false, "G", PTR, "x0", LIT, "y0", LIT, "z0", LIT);
	const auto c = tprint_on;

	//set_tprint(false);
	//flops += greens(P);
	//set_tprint(c);

	constexpr double alpha = 2.0;
	tprint("T Gr[%i];\n", exp_sz(P));
	set_tprint(false);
	flops += greens(P);
	set_tprint(c);
	int cnt = 0;
	for (int ix = -3; ix <= 3; ix++) {
		for (int iy = -3; iy <= 3; iy++) {
			for (int iz = -3; iz <= 3; iz++) {
				if (ix * ix + iy * iy + iz * iz > 3.1 * 3.1) {
					continue;
				}
				cnt++;
			}
		}
	}
	flops *= cnt;
	bool first = true;
	tprint("T sw;\n");
	for (int ix = -3; ix <= 3; ix++) {
		for (int iy = -3; iy <= 3; iy++) {
			for (int iz = -3; iz <= 3; iz++) {
				if (ix * ix + iy * iy + iz * iz > 3.46 * 3.46) {
					continue;
				}
				tprint("{\n");
				indent();
				if (ix == 0) {
					tprint("const T& x = x0;\n");
				} else {
					tprint("const T x = x0 - T(%i);\n", ix);
					flops++;
				}
				if (iy == 0) {
					tprint("const T& y = y0;\n");
				} else {
					tprint("const T y = y0 - T(%i);\n", iy);
					flops++;
				}
				if (iz == 0) {
					tprint("const T& z = z0;\n");
				} else {
					tprint("const T z = z0 - T(%i);\n", iz);
					flops++;
				}
				tprint("const T r2 = x * x + y * y + z * z;\n");
				flops += 5;
				tprint("const T r = sqrt(r2);\n");
				flops += 4;
				tprint("greens_%s_P%i(Gr, x, y, z);\n", type, P);
				tprint("T gamma1 = T(%.16e) * erfc(T(%.16e) * r);\n", sqrt(M_PI), alpha);
				flops += 10;
				tprint("const T xfac = T(%.16e) * r2;\n", alpha * alpha);
				flops += 1;
				tprint("const T exp0 = exp(-xfac);\n");
				flops += 9;
				tprint("T xpow = T(%.16e) * r;\n", alpha);
				flops++;
				double gamma0inv = 1.0f / sqrt(M_PI);
				tprint("T gamma;\n");
				if (ix * ix + iy * iy + iz * iz == 0) {
					tprint("sw = (x0 * x0 + y0 * y0 + z0 * z0) > T(0);\n");
					flops += 5;
				}
				for (int l = 0; l <= P; l++) {
					tprint("gamma = gamma1 * T(%.16e);\n", gamma0inv);
					flops++;
					if (ix * ix + iy * iy + iz * iz == 0) {
						for (int m = -l; m <= l; m++) {
							tprint("G[%i] -= sw*(gamma - T(%.1e)) * Gr[%i];\n", index(l, m), nonepow<double>(l), index(l, m));
							flops += 4;
						}
						if (l == 0) {
							tprint("G[%i] += (T(1) - sw)*T(%.16e);\n", index(0, 0), (2) * alpha / sqrt(M_PI));
							flops += 3;
						}

					} else {
						if (first) {
							for (int m = -l; m <= l; m++) {
								tprint("G[%i] = -gamma * Gr[%i];\n", index(l, m), index(l, m));
							}
						} else {
							for (int m = -l; m <= l; m++) {
								tprint("G[%i] -= gamma * Gr[%i];\n", index(l, m), index(l, m));
							}
						}
						flops += 2;
					}
					gamma0inv *= 1.0 / -(l + 0.5);
					if (l != P) {
						tprint("gamma1 = T(%.16e) * gamma1 + xpow * exp0;\n", l + 0.5);
						flops += 3;
						if (l != P - 1) {
							tprint("xpow *= xfac;\n");
							flops++;
						}
					}
				}
				deindent();
				tprint("}\n");
				first = false;
			}
		}
	}
	tprint("T cosphi, sinphi, hdotx, phi;\n");

	for (int hx = -2; hx <= 2; hx++) {
		for (int hy = -2; hy <= 2; hy++) {
			for (int hz = -2; hz <= 2; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= 8 && h2 > 0) {
					const double h = sqrt(h2);
					bool init = false;
					if (hx) {
						if (hx == 1) {
							tprint("hdotx %s= x0;\n", init ? "+" : "");
							flops += init;
							init = true;
						} else if (hx == -1) {
							if (init) {
								tprint("hdotx -= x0;\n");
								flops++;
							} else {
								tprint("hdotx = -x0;\n");
								flops++;
							}
							init = true;
						} else {
							tprint("hdotx %s= T(%i) * x0;\n", init ? "+" : "", hx);
							flops += 1 + init;
							init = true;
						}
					}
					if (hy) {
						if (hy == 1) {
							tprint("hdotx %s= y0;\n", init ? "+" : "");
							flops += init;
							init = true;
						} else if (hy == -1) {
							if (init) {
								tprint("hdotx -= y0;\n");
								flops++;
							} else {
								flops++;
								tprint("hdotx = -y0;\n");
							}
							init = true;
						} else {
							flops++;
							tprint("hdotx %s= T(%i) * y0;\n", init ? "+" : "", hy);
							flops += 1 + init;
							init = true;
						}
					}
					if (hz) {
						if (hz == 1) {
							tprint("hdotx %s= z0;\n", init ? "+" : "");
							flops += init;
							init = true;
						} else if (hz == -1) {
							if (init) {
								flops++;
								tprint("hdotx -= z0;\n");
							} else {
								flops++;
								tprint("hdotx = -z0;\n");
							}
							init = true;
						} else {
							flops++;
							tprint("hdotx %s= T(%i) * z0;\n", init ? "+" : "", hz);
							flops += 1 + init;
							init = true;
						}
					}

					const auto G0 = spherical_singular_harmonic(P, (double) hx, (double) hy, (double) hz);
					tprint("phi = T(%.16e) * hdotx;\n", 2.0 * M_PI);
					flops++;
					tprint("SINCOS(phi, &sinphi, &cosphi);\n");
					flops += 8;
					double gamma0inv = 1.0f / sqrt(M_PI);
					double hpow = 1.f / h;
					double pipow = 1.f / sqrt(M_PI);
					for (int l = 0; l <= P; l++) {
						for (int m = 0; m <= l; m++) {
							double c0 = gamma0inv * hpow * pipow * exp(-h * h * double(M_PI * M_PI) / (alpha * alpha));
							std::string ax = "cosphi";
							std::string ay = "sinphi";
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
								tprint("G[%i] += T(%.16e) * %s;\n", index(l, m), -xsgn * c0 * G0[cindex(l, m)].real(), ax.c_str());
								flops += 2;
								if (m != 0) {
									tprint("G[%i] += T(%.16e) * %s;\n", index(l, -m), -ysgn * c0 * G0[cindex(l, m)].real(), ay.c_str());
									flops += 2;
								}
							}
							if (G0[cindex(l, m)].imag() != (0)) {
								flops += 2;
								tprint("G[%i] += T(%.16e) * %s;\n", index(l, m), ysgn * c0 * G0[cindex(l, m)].imag(), ay.c_str());
								if (m != 0) {
									flops += 2;
									tprint("G[%i] += T(%.16e) * %s;\n", index(l, -m), -xsgn * c0 * G0[cindex(l, m)].imag(), ax.c_str());
								}
							}
						}
						gamma0inv /= l + 0.5f;
						hpow *= h * h;
						pipow *= M_PI;

					}
				}
			}
		}
	}

	tprint("G[%i] = T(%.16e);\n", (P + 1) * (P + 1), (4.0 * M_PI / 3.0));
	if (!nophi) {
		tprint("G[%i] += T(%.16e);\n", index(0, 0), M_PI / (alpha * alpha));
		flops++;
	}

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
	tprint("for( int n = 0; n < %i; n++) {\n", P * P + 1);
	indent();
	tprint("M[n] = M0[n];\n");
	deindent();
	tprint("}\n");

	/*	tprint("for( int n = 0; n < %i; n++) {\n", Q == 1 ? 4 : (Q + 1) * (Q + 1) + 1);
	 indent();
	 tprint("L[n] = T(0);\n");
	 deindent();
	 tprint("}\n");*/

	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = T(R<T(1e-37));\n");
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T r2 = (FMA(z, z, R2));\n");
	tprint("const T rzero = T(r2<T(1e-37));\n");
	tprint("const T r2inv = T(1) / r2;\n");
	flops += 6 - fmaops;
	tprint("const T rinv = T(1) / sqrt(r2 + rzero);\n");
	flops += 5;
	tprint("T cosphi0;\n");
	tprint("T cosphi;\n");
	tprint("T sinphi0;\n");
	tprint("T sinphi;\n");

	tprint("cosphi = y * Rinv;\n");
	flops++;
	tprint("sinphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("const auto multi_rot = [&M,&cosphi,&sinphi]()\n");
	flops += 2 * z_rot(P - 1, "M", false, false, false);
	tprint(";\n");
	tprint("multi_rot();\n");

	flops += xz_swap(P - 1, "M", false, false, false, false);

	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = FMA(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += 2;
	tprint("multi_rot();\n");
	flops += xz_swap(P - 1, "M", false, true, false, false);
	flops += m2l(P, Q, "M", "L");
	flops += xz_swap(Q, "L", true, false, true, false);

	tprint("sinphi = -sinphi;\n");
	flops += 1;
	flops += z_rot(Q, "L", true, false, false);
	//	flops += z_rot(P, "L", true);
	//	flops += xz_swap(P, "L", true, false, false, true);
	flops += xz_swap(Q, "L", true, false, false, true);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	flops += 1;
	flops += z_rot(Q, "L", false, true, false);
	tprint("for( int n = 0; n < %i; n++) {\n", (Q + 1) * (Q + 1));
	indent();
	tprint("L0[n] += L[n];\n");
	deindent();
	tprint("}\n");
	flops += (Q + 1) * (Q + 1);
	tprint("\n");
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int regular_harmonic(int P) {
	int flops = 0;
	func_header("regular_harmonic", P, false, false, "Y", PTR, "x", LIT, "y", LIT, "z", LIT);
	tprint("const T r2 = FMA(x, x, FMA(y, y, z * z));\n");
	flops += 5 - 2 * fmaops;
	tprint("T ax;\n");
	tprint("T ay;\n");
	tprint("Y[0] = T(1);\n");
	tprint("Y[%i] = r2;\n", (P + 1) * (P + 1));
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / T(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay = Y[%i] * T(%.16e);\n", index(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax - y * ay;\n", index(m, m));
				tprint("Y[%i] = FMA(y, ax, x * ay);\n", index(m, -m));
				flops += 6 - fmaops;
			} else {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
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
//			Y[index(n, m)] = inv * (T(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax =  T(%.16e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  T(%.16e) * r2;\n", -(double) inv);
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
	tprint("const T r2 = FMA(x, x, z * z);\n");
	flops += 3 - fmaops;
	tprint("T ax;\n");
	tprint("T ay;\n");
	tprint("Y[0] = T(1);\n");
	tprint("Y[%i] = r2;\n", (P + 2) * (P + 1) / 2);
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	Y[index(m, m)] = Y[index(m - 1, m - 1)] * R / T(2 * m);
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				flops += 2;
			} else {
				tprint("ax = Y[%i] * T(%.16e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
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
//			Y[index(n, m)] = inv * (T(2 * n - 1) * z * Y[index(n - 1, m)] - r2 * Y[index(n - 2, m)]);
			tprint("ax =  T(%.16e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  T(%.16e) * r2;\n", -(double) inv);
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
	tprint("regular_harmonic_%s_P%i(Y, -x, -y, -z);\n", type, P);
	flops += 3;
	if (P > 1 && !nophi && periodic) {
		tprint("M[%i] = FMA(T(-4) * x, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 1), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(T(-4) * y, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, -1), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(T(-2) * z, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 0), (P + 1) * (P + 1));
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
	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = (R<T(1e-37));\n");
	flops++;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T r2 = FMA(z, z, R2);\n");
	tprint("const T rzero = (r2<T(1e-37));\n");
	tprint("const T r2inv = T(1) / (r2 + rzero);\n");
	flops += 6 - fmaops;
	tprint("T cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("T sinphi = -y * Rinv;\n");
	flops += 2;
	flops += z_rot(P, "M", false, false, false);
	tprint("T Y[%i];\n", exp_sz(P));
	if (P > 1 && !nophi && periodic) {
		tprint("M[%i] = FMA(T(-4)*R, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 1), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(T(-2)*z, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(R * R, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(z * z, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		flops += 12;
	}
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("regular_harmonic_xz_%s_P%i(Y, -R, -z);\n", type, P);
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
	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T r = SQRT(FMA(z, z, R2));\n");
	flops += 6 - fmaops;
	tprint("const T Rzero = T(R<T(1e-37));");
	tprint("const T rzero = T(r<T(1e-37));");
	flops += 2;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T rinv = T(1) / (r + rzero);\n");
	flops += 5;
	tprint("T cosphi = y * Rinv;\n");
	flops++;
	tprint("T sinphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	flops += z_rot(P, "M", false, false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	tprint("T cosphi0 = cosphi;\n");
	tprint("T sinphi0 = sinphi;\n");
	tprint("cosphi = FMA(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "M", false, false, false);
	flops += xz_swap(P, "M", false, false, false, false);
	if (P > 1 && !nophi && periodic) {
		tprint("M[%i] = FMA(T(-2) * r, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(1, 0), (P + 1) * (P + 1));
		tprint("M[%i] = FMA(r * r, M[%i], M[%i]);\n", (P + 1) * (P + 1), index(0, 0), (P + 1) * (P + 1));
		flops += 6 - 2 * fmaops;
	}
	tprint("T c0[%i];\n", P + 1);
	tprint("c0[0] = T(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("c0[%i] = -r * c0[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("c0[%i] *= T(%.16e);\n", n, 1.0 / factorial(n));
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
	tprint("regular_harmonic_%s_P%i(Y, -x, -y, -z);\n", type, P);
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
		tprint("L[%i] = FMA(T(-2) * x, L[%i], L[%i]);\n", index(1, 1), (P + 1) * (P + 1), index(1, 1));
		tprint("L[%i] = FMA(T(-2) * y, L[%i], L[%i]);\n", index(1, -1), (P + 1) * (P + 1), index(1, -1));
		tprint("L[%i] = FMA(T(-2) * z, L[%i], L[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
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
	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T Rzero = (R<T(1e-37));\n");
	flops++;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("T cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("T sinphi = -y * Rinv;\n");
	flops += z_rot(P, "L", false, false, false);
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("regular_harmonic_xz_%s_P%i(Y, -R, -z);\n", type, P);
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
		tprint("L[%i] = FMA(T(-2) * R, L[%i], L[%i]);\n", index(1, 1), (P + 1) * (P + 1), index(1, 1));
		tprint("L[%i] = FMA(T(-2) * z, L[%i], L[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
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
	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("const T R = SQRT(R2);\n");
	flops += 4;
	tprint("const T r = SQRT(R2 + z * z);\n");
	flops += 6;
	tprint("const T Rzero = T(R<T(1e-37));");
	tprint("const T rzero = T(r<T(1e-37));");
	flops += 2;
	tprint("const T Rinv = T(1) / (R + Rzero);\n");
	flops += 5;
	tprint("const T rinv = T(1) / (r + rzero);\n");
	flops += 5;
	tprint("T cosphi = y * Rinv;\n");
	flops++;
	tprint("T sinphi = FMA(x, Rinv, Rzero);\n");
	flops += 1 - fmaops;
	flops += 1;
	flops += z_rot(P, "L", false, false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("T cosphi0 = cosphi;\n");
	tprint("T sinphi0 = sinphi;\n");
	tprint("cosphi = FMA(z, rinv, rzero);\n");
	flops += 2 - fmaops;
	tprint("sinphi = -R * rinv;\n");
	flops += z_rot(P, "L", false, false, false);
	flops += xz_swap(P, "L", true, false, false, false);
	tprint("T c0[%i];\n", P + 1);
	tprint("c0[0] = T(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("c0[%i] = -r * c0[%i];\n", n, n - 1);
		flops += 2;
	}
	for (int n = 2; n <= P; n++) {
		tprint("c0[%i] *= T(%.16e);\n", n, 1.0 / factorial(n));
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
				tprint("L[%i] = FMA(L[%i], c0[%i], L[%i]);\n", index(n, m), index(n + k, m), k, index(n, m));
				flops += 2 - fmaops;
				if (m > 0) {
					tprint("L[%i] = FMA(L[%i], c0[%i], L[%i]);\n", index(n, -m), index(n + k, -m), k, index(n, -m));
					flops += 2 - fmaops;
				}
			}

		}
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = FMA(T(-2) * r, L[%i], L[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
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
	tprint("regular_harmonic_%s_P%i(Y, -x, -y, -z);\n", type, P);
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
		tprint("L1[%i] = FMA(T(-2) * x, L[%i], L1[%i]);\n", index(1, 1), (P + 1) * (P + 1), index(1, 1));
		tprint("L1[%i] = FMA(T(-2) * y, L[%i], L1[%i]);\n", index(1, -1), (P + 1) * (P + 1), index(1, -1));
		tprint("L1[%i] = FMA(T(-2) * z, L[%i], L1[%i]);\n", index(1, 0), (P + 1) * (P + 1), index(1, 0));
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
	tprint("regular_harmonic_%s_P%i(M, -x, -y, -z);\n", type, P);
	if (!nophi) {
		tprint("M[%i] = FMA(x, x, FMA(y, y, z * z));\n", (P + 1) * (P + 1));
		flops += 5 - 2 * fmaops;
	}
	deindent();
	tprint("}\n");
	tprint("\n");
	return flops + 3;
}

int main() {

	system("[ -e ./generated_code ] && rm -r  ./generated_code\n");
	system("mkdir generated_code\n");
	system("mkdir ./generated_code/include\n");
	system("mkdir ./generated_code/include/detail\n");
	system("mkdir ./generated_code/src\n");
	set_file("./generated_code/include/spherical_fmm.hpp");
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#include <cmath>\n");
	tprint("\n");
	tprint("inline float SQRT(float x) {\n");
	indent();
	tprint("return sqrtf(x);\n");
	deindent();
	tprint("}\n");
	tprint("inline double SQRT(double x) {\n");
	indent();
	tprint("return sqrt(x);\n");
	deindent();
	tprint("}\n");
	tprint("inline float EXP(float x) {\n");
	indent();
	tprint("return expf(x);\n");
	deindent();
	tprint("}\n");
	tprint("inline double EXP(double x) {\n");
	indent();
	tprint("return exp(x);\n");
	deindent();
	tprint("}\n");
	tprint("inline float ERFC(float x) {\n");
	indent();
	tprint("return erfcf(x);\n");
	deindent();
	tprint("}\n");
	tprint("inline double ERFC(double x) {\n");
	indent();
	tprint("return erfc(x);\n");
	deindent();
	tprint("}\n");
	tprint("inline void SINCOS(float x, float * s, float* c) {\n");
	indent();
	tprint("sincosf(x, s, c);\n");
	deindent();
	tprint("}\n");
	tprint("inline void SINCOS(double x, double* s, double* c) {\n");
	indent();
	tprint("sincos(x, s, c);\n");
	deindent();
	tprint("}\n");
	tprint("inline float FMA(float a, float b, float c) {\n");
	indent();
	tprint("return fmaf(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("inline double FMA(double a, double b, double c) {\n");
	indent();
	tprint("return fma(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
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
		fprintf(stderr, "%2s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s %2s %5s %5s %2s %5s %5s %5s %5s %8s %8s %8s\n", "p", "M2L", "eff", "-r", "M2P", "eff", "-r",
				"P2L", "eff", "M2M", "eff", "-r", "L2L", "eff", "-r", "P2M", "eff", "L2P", "eff", "CC_ewald", "green", "m2l");
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
			int eflopsg = ewald_greens(P);
			int eflopsm = m2lg(P, P);

			fprintf(stderr, "%2i %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f %2i %5i %5.2f %2i %5i %5.2f %5i %5.2f %8i %8i %8i\n", P, cc_flops[P],
					cc_flops[P] / pow(P + 1, 3), cc_rot[P], pc_flops[P], pc_flops[P] / pow(P + 1, 2), pc_rot[P], cp_flops[P], cp_flops[P] / pow(P + 1, 2),
					m2m_flops[P], m2m_flops[P] / pow(P + 1, 3), m2m_rot[P], l2l_flops[P], l2l_flops[P] / pow(P + 1, 3), l2l_rot[P], p2m_flops[P],
					p2m_flops[P] / pow(P + 1, 2), l2p_flops[P], l2p_flops[P] / pow(P + 1, 2), eflopsg + eflopsm, eflopsg, eflopsm);
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
				ewald_greens(P);
			m2lg(P, P);
			m2l_ewald(P);
		}
	}
//	printf("./generated_code/include/spherical_fmm.hpp");
	fflush (stdout);
	set_file("./generated_code/include/spherical_fmm.hpp");
	tprint(inter_header.c_str());
	set_file("./generated_code/src/interface.cpp");
	tprint(inter_src.c_str());
	return 0;
}
