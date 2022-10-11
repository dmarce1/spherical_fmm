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

static bool nophi = false;
static bool fmaops = true;
static bool periodic = true;
static int pmin = 3;
static int pmax = 12;
static std::string type = "float";
static std::string sitype = "int";
static std::string uitype = "unsigned";
static const int divops = 4;
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

int do_exp(const char* in, const char* out) {
	if (type == "double") {
		tprint("{\n");
		indent();
		constexpr int N = 18;
		tprint("V k = %s / T(0.6931471805599453094172) + T(0.5);\n", in); // 1 + divops
		tprint("k -= %s < T(0.0);\n", in);
		tprint("T xxx = %s - k * T(0.6931471805599453094172);\n", in);
		// 2;
		tprint("%s = T(%.16e);\n", out, 1.0 / factorial(N));
		for (int i = N - 1; i >= 0; i--) {
			tprint("%s = FMA(%s, xxx, T(%.16e));\n", out, out, 1.0 / factorial(i)); //17*(2-fmaops);
		}
		tprint("k = (k + V(1023)) << V(52);\n");
		tprint("%s *= (T&) (k);\n", out); //1
//		tprint("printf( \"%%e %%e\\n\", %s, exp(%s));\n", out, in);
		deindent();
		tprint("}\n");
		return 4 + divops + 17 * (2 - fmaops);
	} else if (type == "float") {
		tprint("{\n");
		indent();
		constexpr int N = 7;
		tprint("V k =  %s / T(0.6931471805599453094172) + T(0.5);\n", in); // 1 + divops
		tprint("k -= %s < T(0);\n", in);
		tprint("T xxx = %s - k * T(0.6931471805599453094172);\n", in);
		// 2;
		tprint("%s = T(%.16e);\n", out, 1.0 / factorial(N));
		for (int i = N - 1; i >= 0; i--) {
			tprint("%s = FMA(%s, xxx, T(%.16e));\n", out, out, 1.0 / factorial(i)); //17*(2-fmaops);
		}
		tprint("k = (k + V(127)) << V(23);\n");
		tprint("%s *= (T&) (k);\n", out); //1
		deindent();
		tprint("}\n");
		return 4 + divops + 17 * (2 - fmaops);
	}
}

double nonepow(int i) {
	return i % 2 == 0 ? 1.0 : -1.0;
}
int do_erfcexp(const char* in, const char* erfc0, const char* exp0) {
	int flops = 0;
	tprint("{\n");
	indent();
	tprint("const T x2 = %s * %s;\n", in, in);
	tprint("const T nx2 = -x2;\n");
	flops += 2;
	flops += do_exp("nx2", exp0);
	if (type == "double") {
		tprint("if( %s < T(4.05) ) {\n", in);
		indent();
		{
			const int N = 59;
			tprint("T q = T(2) * x2;\n");
			flops++;
			tprint("%s = T(%.16e);\n", erfc0, 1.0 / dfactorial(2 * N + 1));
			for (int n = N - 1; n >= 0; n--) {
				tprint("%s = FMA(%s, q, T(%.16e));\n", erfc0, erfc0, 1.0 / dfactorial(2 * n + 1));
				flops += (2 - fmaops);
			}
			tprint("%s *= %s * %s * T(%.16e);\n", erfc0, exp0, in, 2.0 / sqrt(M_PI));
			flops += 3;
			tprint("%s = T(1) - %s;\n", erfc0, erfc0);
			flops++;
		}
		deindent();
		tprint("} else {\n");
		indent();
		const int N = 11;
		tprint("T q = T(1) / (T(2) * x2);\n");
		flops += 1 + divops;
		tprint("%s = T(%.16e);\n", erfc0, nonepow(N) * dfactorial(2 * N + 1));
		for (int n = N - 1; n >= 1; n--) {
			tprint("%s = FMA(%s, q, T(%.16e));\n", erfc0, erfc0, nonepow(n) * dfactorial(2 * n + 1));
			flops += (2 - fmaops);
		}
		tprint("%s = %s / %s * T(%.16e) * (T(1) + %s);\n", erfc0, exp0, in, 1.0 / sqrt(M_PI), erfc0);
		flops += 4 + divops;

		deindent();
		tprint("}\n");
		deindent();
		tprint("}\n");
	} else {
		tprint("T t = T(1) / (T(1) + 0.3275911 * %s);\n", in);
		flops += divops + 2;
		tprint("%s = T(1.061405429);\n", erfc0);
		tprint("%s = FMA(%s, t, T(-1.453152027));\n", erfc0, erfc0);
		flops += 2 - fmaops;
		tprint("%s = FMA(%s, t, T(1.421413741));\n", erfc0, erfc0);
		flops += 2 - fmaops;
		tprint("%s = FMA(%s, t, T(-0.284496736));\n", erfc0, erfc0);
		flops += 2 - fmaops;
		tprint("%s = FMA(%s, t, T(0.254829592));\n", erfc0, erfc0);
		flops += 2 - fmaops;
		tprint("%s *= t;\n", erfc0);
		flops += 1;
		tprint("%s *= %s;\n", erfc0, exp0);
		flops += 1;
		deindent();
		tprint("}\n");
	}
	return flops;
}

int do_sincos(const char* in, const char* sout, const char* cout) {
	tprint("{\n");
	indent();
	if (type == "float") {
		tprint("V ssgn = V((((U&) %s & U(0x80000000)) >> U(30)) - U(1));\n", in);
		tprint("V j = V(((U&) %s & U(0x7FFFFFFF)));\n", in);
		tprint("%s = (T&) j;\n", in);
		tprint("V i = %s * T(%.16e);\n", in, 1.0 / M_PI);
		tprint("%s -= i * T(%.16e);\n", in, M_PI);
		tprint("%s -= T(%.16e);\n", in, 0.5 * M_PI);
		tprint("T x2 = %s * %s;\n", in, in);
		tprint("%s = T(%.16e);\n", cout, -1.0 / factorial(11));
		tprint("%s = T(%.16e);\n", sout, -1.0 / factorial(10));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(9));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(8));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, -1.0 / factorial(7));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, -1.0 / factorial(6));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(5));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(4));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, -1.0 / factorial(3));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, -1.0 / factorial(2));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(1));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(0));
		tprint("%s *= %s;\n", cout, in);
		tprint("V k = (((i & V(1)) << V(1)) - V(1));\n");
		tprint("%s *= T(ssgn * k);\n", sout);
		tprint("%s *= T(k);\n", cout);
		deindent();
		tprint("}\n");
		return 31 - fmaops * 10;
	} else {
		tprint("V ssgn = V((((U&) %s & U(0x8000000000000000LL)) >> U(62LL)) - U(1LL));\n", in);
		tprint("V j = V(((U&) %s & U(0x7FFFFFFFFFFFFFFFLL)));\n", in);
		tprint("%s = (T&) j;\n", in);
		tprint("V i = %s * T(%.16e);\n", in, 1.0 / M_PI);
		tprint("%s -= i * T(%.16e);\n", in, M_PI);
		tprint("%s -= T(%.16e);\n", in, 0.5 * M_PI);
		tprint("T x2 = %s * %s;\n", in, in);
		tprint("%s = T(%.16e);\n", cout, 1.0 / factorial(21));
		tprint("%s = T(%.16e);\n", sout, 1.0 / factorial(20));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, -1.0 / factorial(19));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, -1.0 / factorial(18));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(17));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(16));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, -1.0 / factorial(15));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, -1.0 / factorial(14));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(13));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(12));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, -1.0 / factorial(11));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, -1.0 / factorial(10));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(9));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(8));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, -1.0 / factorial(7));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, -1.0 / factorial(6));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(5));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(4));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, -1.0 / factorial(3));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, -1.0 / factorial(2));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", cout, cout, 1.0 / factorial(1));
		tprint("%s = FMA(%s, x2, T(%.16e));\n", sout, sout, 1.0 / factorial(0));
		tprint("%s *= %s;\n", cout, in);
		tprint("V k = (((i & V(1)) << V(1)) - V(1));\n");
		tprint("%s *= T(ssgn * k);\n", sout);
		tprint("%s *= T(k);\n", cout);
		deindent();
		tprint("}\n");
		return 41 - fmaops * 20;
	}

}

int do_rsqrt(const char* xin, const char* yout) {
	if (type == "float") {
		tprint("{\n");
		indent();
		tprint("T xxx = %s + T(%.16e);\n", xin, std::numeric_limits<float>::min());
		tprint("V i = *((V*) &xxx);\n");
		tprint("i >>= V(1);\n");
		tprint("i = V(0x5F3759DF) - i;\n");
		tprint("%s = *((T*) &i);\n", yout);
		tprint("%s *= FMA(T(-0.5), xxx * %s * %s, T(1.5));\n", yout, yout, yout);
		tprint("%s *= FMA(T(-0.5), xxx * %s * %s, T(1.5));\n", yout, yout, yout);
		tprint("%s *= FMA(T(-0.5), xxx * %s * %s, T(1.5));\n", yout, yout, yout);
		deindent();
		tprint("}\n");
		return 15 - 3 * fmaops;
	} else if (type == "double") {
		tprint("{\n");
		indent();
		tprint("T xxx = %s + T(%.16e);\n", xin, std::numeric_limits<double>::min());
		tprint("V i = *((V*) &xxx);\n");
		tprint("i >>= V(1);\n");
		tprint("i = V(0x5FE6EB50C7B537A9) - i;\n");
		tprint("%s = *((T*) &i);\n", yout);
		tprint("%s *= FMA(T(-0.5), xxx * %s * %s, T(1.5));\n", yout, yout, yout);
		tprint("%s *= FMA(T(-0.5), xxx * %s * %s, T(1.5));\n", yout, yout, yout);
		tprint("%s *= FMA(T(-0.5), xxx * %s * %s, T(1.5));\n", yout, yout, yout);
		tprint("%s *= FMA(T(-0.5), xxx * %s * %s, T(1.5));\n", yout, yout, yout);
		deindent();
		tprint("}\n");
		return 20 - 4 * fmaops;
	}
}

int do_sqrt(const char* xin, const char* yout) {
	int flops = 0;
	if (type == "float") {
		flops += do_rsqrt(xin, yout);
	} else if (type == "double") {
		flops += do_rsqrt(xin, yout);
	}
	tprint("%s = T(1) / %s;\n", yout, yout);
	flops += divops;
	return flops;
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
		std::string func1 = std::string(func) + std::string("_") + type;
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
		tprint("#include <stdio.h>\n");
		tprint("#include \"detail/spherical_fmm.hpp\"\n");
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
	flops += do_rsqrt("r2", "O[0]");
	if (M) {
		tprint("O[0] *= %s;\n", M);
		flops++;
	}
	tprint("O[%i] = T(0);\n", (P + 1) * (P + 1));
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
	flops += do_sqrt("r2inv", "O[0]");
	tprint("O[%i] = T(0);\n", (P + 1) * (P + 1));
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
	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("const T Rzero = T(R2<T(1e-37));\n");
	flops++;
	tprint("const T r2 = FMA(z, z, R2);\n");
	tprint("const T rzero = T(r2<T(1e-37));\n");
	flops++;
	tprint("T tmp1 = R2 + Rzero;\n");
	flops++;
	tprint("T Rinv;\n");
	flops += do_rsqrt("R2", "Rinv");
	tprint("T R = T(1) / Rinv;\n");
	flops += divops;
	tprint("const T r2inv = T(1) / (r2+rzero);\n");
	flops += 7 - fmaops;
	tprint("T cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("T sinphi = -y * Rinv;\n");
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
	tprint("greens_%s_P%i(O, x, y, z);\n", type.c_str(), P);
	flops += m2lg_body(P, Q);
	deindent();
	tprint("}");
	tprint("\n");
	return flops;
}

int ewald_greens(int P, double alpha) {
	int R2, H2;
	if (type == "float") {
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
				if (ix * ix + iy * iy + iz * iz > R2 && ix * ix + iy * iy + iz * iz != 0) {
					continue;
				}
				cnt++;
			}
		}
	}
	flops *= cnt + 1;
	bool first = true;
	int eflops;
	tprint("T sw;\n");
	tprint("const auto erfcexp = [](T x, T& erfc0, T& exp0) {\n");
	indent();
	eflops = do_erfcexp("x", "erfc0", "exp0");
	deindent();
	tprint("};\n");
	tprint("{\n");
	indent();
	tprint("const T r2 = x0 * x0 + y0 * y0 + z0 * z0;\n");
	flops += 5;
	tprint("T r;\n");
	flops += do_sqrt("r2", "r");
	tprint("greens_%s_P%i(Gr, x0, y0, z0);\n", type.c_str(), P);
	tprint("T xxx = T(%.16e) * r;\n", alpha);
	flops++;
	tprint("T gamma1, exp0;\n", sqrt(M_PI), alpha);
	tprint("erfcexp(xxx, gamma1, exp0);\n");
	flops += eflops;
	tprint("gamma1 *= T(%.16e);\n", sqrt(M_PI));
	flops += 1;
	tprint("const T xfac = T(%.16e) * r2;\n", alpha * alpha);
	flops += 1;
	tprint("T xpow = T(%.16e) * r;\n", alpha);
	flops++;
	double gamma0inv = 1.0f / sqrt(M_PI);
	tprint("T gamma;\n");
	tprint("sw = r2 > T(0);\n");
	flops += 1;
	for (int l = 0; l <= P; l++) {
		tprint("gamma = gamma1 * T(%.16e);\n", gamma0inv);
		flops++;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = sw*(T(%.1e) - gamma) * Gr[%i];\n", index(l, m), nonepow<double>(l), index(l, m));
			flops += 4;
		}
		if (l == 0) {
			tprint("G[%i] += (T(1) - sw)*T(%.16e);\n", index(0, 0), (2) * alpha / sqrt(M_PI));
			flops += 3;
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

	tprint("for (int ix = -%i; ix <= %i; ix++) {\n", R, R);
	indent();
	tprint("for (int iy = -%i; iy <= %i; iy++) {\n", R, R);
	indent();
	tprint("for (int iz = -%i; iz <= %i; iz++) {\n", R, R);
	indent();
	tprint("int ii = ix * ix + iy * iy + iz * iz;\n");
	tprint("if (ii > %i || ii == 0) {\n", R2);
	indent();
	tprint("continue;\n");
	deindent();
	tprint("}\n");
	tprint("const T x = x0 - T(ix);\n");
	flops += cnt;
	tprint("const T y = y0 - T(iy);\n");
	flops += cnt;
	tprint("const T z = z0 - T(iz);\n");
	flops += cnt;
	tprint("const T r2 = FMA(x, x, FMA(y, y, z * z));\n");
	flops += 3 * cnt;
	tprint("T r;\n");
	flops += do_sqrt("r2", "r");
	tprint("greens_%s_P%i(Gr, x, y, z);\n", type.c_str(), P);

	tprint("T xxx = T(%.16e) * r;\n", alpha);
	flops += cnt;
	tprint("T gamma1, exp0;\n", sqrt(M_PI), alpha);
	tprint("erfcexp(xxx, gamma1, exp0);\n");
	flops += cnt * eflops;
	tprint("gamma1 *= T(%.16e);\n", -sqrt(M_PI));
	flops += cnt;
	tprint("const T xfac = T(%.16e) * r2;\n", alpha * alpha);
	flops += cnt;
	tprint("T xpow = T(%.16e) * r;\n", alpha);
	flops++;
	tprint("T gamma0inv = T(%.16e);\n", 1.0f / sqrt(M_PI));
	tprint("T gamma;\n");
	for (int l = 0; l <= P; l++) {
		tprint("gamma = gamma1 * gamma0inv;\n");
		flops += cnt;
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = FMA(gamma, Gr[%i], G[%i]);\n", index(l, m), index(l, m), index(l, m));
			flops += (2 - fmaops) * cnt;
		}
		if (l != P) {
			tprint("gamma0inv *= T(%.16e);\n", 1.0 / -(l + 0.5));
			tprint("gamma1 = FMA(T(%.16e), gamma1, -xpow * exp0);\n", l + 0.5);
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
	for (int hx = -H; hx <= H; hx++) {
		for (int hy = -H; hy <= H; hy++) {
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= H2 && h2 > 0) {
					tprint("T %s;\n", cosname(hx, hy, hz).c_str());
					tprint("T %s;\n", sinname(hx, hy, hz).c_str());
				}
			}
		}
	}
	tprint("T x2;\n");
	tprint("T x2y2;\n");
	tprint("T hdotx;\n");
	tprint("T phi;\n");
	tprint("const auto sincos = []( T phi, T& s, T& c ) {\n");
	indent();
	do_sincos("phi", "s", "c");
	deindent();
	tprint("};\n");
	for (int hx = -H; hx <= H; hx++) {
		if (hx) {
			if (abs(hx) == 1) {
				tprint("x2 = %cx0;\n", hx < 0 ? '-' : ' ');
				flops += hx < 0;
			} else {
				tprint("x2 = T(%i) * x0;\n", hx);
				flops++;
			}
		} else {
			tprint("x2 = T(0);\n");
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
					tprint("x2y2 = FMA(T(%i), y0, x2);\n", hy);
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
						tprint("hdotx = FMA(T(%i), z0, x2y2);\n", hz);
						flops += 2 - fmaops;
					}
				} else {
					tprint("hdotx = x2y2;\n", hz);
				}
				tprint("phi = T(%.16e) * hdotx;\n", 2.0 * M_PI);
				flops++;
				tprint("sincos(phi, %s, %s);\n", sinname(hx, hy, hz).c_str(), cosname(hx, hy, hz).c_str());
				const auto pb = tprint_on;
				tprint_on = false;
				flops += do_sincos("", "", "");
				tprint_on = pb;
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
					double gamma0inv = 1.0f / sqrt(M_PI);
					double hpow = 1.f / h;
					double pipow = 1.f / sqrt(M_PI);
					for (int l = 0; l <= P; l++) {
						for (int m = 0; m <= l; m++) {
							double c0 = gamma0inv * hpow * pipow * exp(-h * h * double(M_PI * M_PI) / (alpha * alpha));
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
								//						tprint("G[%i] += T(%.16e) * %s;\n", index(l, m), -xsgn * c0 * G0[cindex(l, m)].real(), ax.c_str());
								ops[index(l, m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -xsgn * a), ax));
								if (m != 0) {
									//								tprint("G[%i] += T(%.16e) * %s;\n", index(l, -m), -ysgn * c0 * G0[cindex(l, m)].real(), ay.c_str());
									ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -ysgn * a), ay));
								}
							}
							if (G0[cindex(l, m)].imag() != (0)) {
								const double a = c0 * G0[cindex(l, m)].imag();
//								tprint("G[%i] += T(%.16e) * %s;\n", index(l, m), ysgn * c0 * G0[cindex(l, m)].imag(), ay.c_str());
								ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, ysgn * a), ay));
								if (m != 0) {
									//	tprint("G[%i] += T(%.16e) * %s;\n", index(l, -m), -xsgn * c0 * G0[cindex(l, m)].imag(), ax.c_str());
									ops[index(l, -m)][fabs(a)].push_back(std::make_pair(copysign(1.0, -xsgn * a), ax));
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
		for (auto j = ops[ii].begin(); j != ops[ii].end(); j++) {
			auto op = j->second;
			if (op.size()) {
				int sgn = op[0].first > 0 ? 1 : -1;
				if (sgn > 0) {
					tprint("G[%i] = FMA(T(+%.16e), ", ii, sgn * j->first);
				} else {
					tprint("G[%i] = FMA(T(%.16e), ", ii, sgn * j->first);
				}
				flops += 2 - fmaops;
				for (int k = 0; k < op.size(); k++) {
					if (tprint_on) {
						if (k != 0) {
							fprintf(fp, " %c ", sgn * op[k].first > 0 ? '+' : '-');
						}
						fprintf(fp, "%s", j->second[k].second.c_str());
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
	tprint("G[%i] = T(%.16e);\n", (P + 1) * (P + 1), (4.0 * M_PI / 3.0));
	if (!nophi) {
		tprint("G[%i] += T(%.16e);\n", index(0, 0), M_PI / (alpha * alpha));
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
	tprint("T R, Rinv;\n");
	tprint("const T Rzero = T(R2<T(1e-37));\n");
	flops++;
	tprint("const T tmp1 = R2 + Rzero;\n");
	flops++;
	flops += do_rsqrt("tmp1", "Rinv");
	tprint("R = T(1) / Rinv;\n");
	flops += divops;
	tprint("const T r2 = (FMA(z, z, R2));\n");
	tprint("const T rzero = T(r2<T(1e-37));\n");
	tprint("const T r2inv = T(1) / r2;\n");
	flops += 3 + divops - fmaops;
	tprint("const T r2przero = (r2 + rzero);");
	flops += 1;
	tprint("T rinv;\n");
	flops += do_rsqrt("r2przero", "rinv");
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
	tprint("regular_harmonic_%s_P%i(Y, -x, -y, -z);\n", type.c_str(), P);
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
	tprint("T R, Rinv, tmp1;");
	tprint("const T Rzero = (R2<T(1e-37));\n");
	tprint("tmp1 = R2 + Rzero;\n");
	flops += do_rsqrt("tmp1", "Rinv");
	flops++;
	tprint("R = T(1) / Rinv;\n");
	flops += divops;
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
	tprint("const T R2 = FMA(x, x, y * y);\n");
	flops += 3 - fmaops;
	tprint("T R, Rinv, tmp1;");
	tprint("const T Rzero = T(R2<T(1e-37));");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	flops += do_rsqrt("tmp1", "Rinv");
	tprint("T r2 = FMA(z, z, R2);\n");
	flops += 2 - fmaops;
	tprint("const T rzero = T(r2<T(1e-37));");
	flops += 1;
	tprint("T r, rinv;\n");
	tprint("tmp1 = r2 + rzero;\n");
	flops += do_rsqrt("tmp1", "rinv");
	tprint("R = T(1) / Rinv;\n");
	flops += divops;
	tprint("r = T(1) / rinv;\n");
	flops += divops;
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
	tprint("T R, Rinv, tmp1;");
	tprint("const T Rzero = (R2<T(1e-37));\n");
	tprint("tmp1 = R2 + Rzero;\n");
	flops += do_rsqrt("tmp1", "Rinv");
	flops++;
	tprint("R = T(1) / Rinv;\n");
	flops += divops;
	tprint("T cosphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
	tprint("T sinphi = -y * Rinv;\n");
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
	tprint("T R, Rinv, tmp1;");
	tprint("const T Rzero = T(R2<T(1e-37));");
	flops++;
	tprint("tmp1 = R2 + Rzero;\n");
	flops++;
	flops += do_rsqrt("tmp1", "Rinv");
	tprint("T r2 = FMA(z, z, R2);\n");
	flops += 2 - fmaops;
	tprint("const T rzero = T(r2<T(1e-37));");
	flops += 1;
	tprint("T r, rinv;\n");
	tprint("tmp1 = r2 + rzero;\n");
	flops += do_rsqrt("tmp1", "rinv");
	tprint("R = T(1) / Rinv;\n");
	flops += divops;
	tprint("r = T(1) / rinv;\n");
	flops += divops;
	tprint("T cosphi = y * Rinv;\n");
	flops++;
	tprint("T sinphi = FMA(x, Rinv, Rzero);\n");
	flops += 2 - fmaops;
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
	const char* rtypenames[] = { "float", "double" };
	const char* sitypenames[] = { "int", "long long" };
	const char* uitypenames[] = { "unsigned", "unsigned long long" };
	const int ntypenames = 2;

	for (int ti = 0; ti < ntypenames; ti++) {
		type = rtypenames[ti];
		sitype = sitypenames[ti];
		uitype = uitypenames[ti];
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
//	printf("./generated_code/include/spherical_fmm.hpp");
	fflush(stdout);
	set_file("./generated_code/include/spherical_fmm.hpp");
	tprint(inter_header.c_str());
	set_file("./generated_code/src/interface.cpp");
	tprint(inter_src.c_str());
	return 0;
}
