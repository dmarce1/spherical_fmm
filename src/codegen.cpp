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
#include <cstring>
#include <functional>
#include <array>
#include <stack>

#if USE_DOUBLE_FLAG == 1
#define USE_DOUBLE
#endif
#if USE_FLOAT_FLAG == 1
#define USE_FLOAT
#endif
#if USE_SIMD_FLAG == 1
#define USE_SIMD
#endif

enum stage_t {
	PRE1, PRE2, POST1, POST2, XZ1, XZ2, FULL
};

#define NO_DIPOLE

struct flops_t {
	int r;
	int i;
	int fma;
	int rdiv;
	int idiv;
	int con;
	int icmp;
	int rcmp;
	int asgn;
	flops_t() {
		reset();
	}
	flops_t(const flops_t&) = default;
	flops_t(flops_t&&) = default;
	flops_t& operator=(const flops_t&) = default;
	flops_t& operator=(flops_t&&) = default;
	void reset() {
		r = 0;
		i = 0;
		icmp = 0;
		rcmp = 0;
		fma = 0;
		rdiv = 0;
		idiv = 0;
		con = 0;
		asgn = 0;
	}
	flops_t& operator+=(const flops_t& other) {
		r += other.r;
		i += other.i;
		fma += other.fma;
		rdiv += other.rdiv;
		idiv += other.idiv;
		con += other.con;
		asgn += other.asgn;
		icmp += other.icmp;
		rcmp += other.rcmp;
		return *this;
	}
	flops_t& operator*=(int a) {
		r *= a;
		i *= a;
		fma *= a;
		rdiv *= a;
		idiv *= a;
		con *= a;
		asgn *= a;
		icmp *= a;
		rcmp *= a;
		return *this;
	}
	int load() const;
};

using complex = std::complex<double>;

static int ntab = 0;
static int tprint_on = true;

#define TAB0() if( ntab != 0 ) { printf( "Tab mismatch (%i) %s %i\n", ntab,  __FILE__, __LINE__); ntab = 0; }

#ifdef NO_DIPOLE
constexpr static int nodip = true;
#else
constexpr static int nodip = false;
#endif
#ifdef USE_PERIODIC
constexpr static int periodic = 1;
#else
constexpr static int periodic = 0;
#endif
#ifdef USE_SCALED
constexpr static int scaled = 1;
#else
constexpr static int scaled = 0;
#endif

std::array<std::unordered_map<std::string, flops_t>, PMAX + 1> flops_map;

//#define FLOAT
//#define DOUBLE
//#define CUDA_FLOAT
//#define CUDA_DOUBLE
//#define VEC_DOUBLE
//#define VEC_FLOAT
//#define VEC_DOUBLE_SIZE 2
//#define VEC_FLOAT_SIZE 8
std::vector<int> precision;
std::vector<int> m2monly;
std::vector<int> simd;
std::vector<int> simd_size;
std::vector<std::string> base_rtype;
std::vector<std::string> rtype;
std::vector<std::string> base_itype;
std::vector<std::string> itype;
std::vector<std::string> base_uitype;
std::vector<std::string> uitype;

static std::string root_dir = std::string(ROOT_DIR) + "/include/";

#define ASPRINTF(...) if( asprintf(__VA_ARGS__) == 0 ) {printf( "ASPRINTF error %s %i\n", __FILE__, __LINE__); abort(); }
#define SYSTEM(...) if( system(__VA_ARGS__) != 0 ) {printf( "SYSTEM error %s %i\n", __FILE__, __LINE__); abort(); }

#define DEBUGNAN

static int nopot = false;
static bool fmaops = true;
static int pmin = PMIN;
static int pmax = PMAX;
static std::string vtype = "simd_f32";
static std::string type;
static int typenum = -1;

static const int divops = 4;
static const char* prefix = "";
static std::string detail_header;
static std::string detail_header_vec;
static std::vector<std::string> lines[2];

static std::string header = "sfmm.hpp";
bool enable_scaled = true;
static std::string full_header = std::string("./generated_code/include/") + header;
//static std::string full_detail_header = std::string("./generated_code/include/detail/") + header;

static const char* pot_name() {
	return nopot ? "_wo_potential" : "";
}

static const char* period_name() {
	return periodic ? "_periodic" : "";
}

static const char* scaled_name() {
	return scaled ? "_scaled" : "";
}

std::string random_macro() {
	static std::set<int> used;
	char* macro;
	int num;
	do {
		num = rand();
	} while (used.find(num) != used.end());
	ASPRINTF(&macro, "SFMM_MACRO_%0i_%s", (unsigned ) num, type.c_str());
	used.insert(num);
	std::string result = macro;
	free(macro);
	return result;
}

std::string type_macro() {
	return "SFMM_MACRO_" + type;
}

static const char* dip_name() {
	return nodip ? "_wo_dipole" : "";
}

template<class ...Args>
std::string print2str(const char* fstr, Args&&...args) {
	std::string result;
	char* str;
	ASPRINTF(&str, fstr, std::forward<Args>(args)...);
	result = str;
	free(str);
	return result;
}

template<class ...Args>
std::string print2str(const char* fstr) {
	std::string result = fstr;
	return result;
}

static std::string vec_header() {
	std::string str = "\n";
	std::string macro;
#ifdef USE_SIMD
	str += print2str("#ifndef __CUDACC__\n");
	str += "\n";
#ifdef USE_FLOAT
	str += print2str("SFMM_SIMD_FACTORY(simd_f32, float, simd_i32, int32_t, simd_ui32, uint32_t, %i);\n", FLOAT_SIMD_WIDTH);
	if ( M2M_SIMD_WIDTH == FLOAT_SIMD_WIDTH) {
		str += print2str("using m2m_simd_f32 = simd_f32;\n");
		str += print2str("using m2m_simd_i32 = simd_i32;\n");
		str += print2str("using m2m_simd_ui32 = simd_ui32;\n");
	} else {
		str += print2str("SFMM_SIMD_FACTORY(m2m_simd_f32, float, m2m_simd_i32, int32_t, m2m_simd_ui32, uint32_t, %i);\n", M2M_SIMD_WIDTH);
	}
	for (int width = std::min(M2M_SIMD_WIDTH,FLOAT_SIMD_WIDTH); width <= std::max(M2M_SIMD_WIDTH, FLOAT_SIMD_WIDTH); width += std::abs(FLOAT_SIMD_WIDTH - M2M_SIMD_WIDTH)) {
		str += print2str("\ninline float reduce_sum(const %s& v) {\n", width == FLOAT_SIMD_WIDTH ? "simd_f32" : "m2m_simd_f32" );
		for (int sz = width; sz > 1; sz /= 2) {
			str += print2str("\ttypedef float type%i __attribute__ ((vector_size(%i*sizeof(float))));\n", sz, sz);
		}
		str += print2str("\ttype%i a%i = *((type%i*)(&v));\n", width, width, width);
		for (int sz = width / 2; sz > 1; sz /= 2) {
			str += print2str("\ttype%i a%i = *((type%i*)(&a%i));\n", sz, sz, sz, 2 * sz);
			str += print2str("\tconst type%i& b%i = *(((type%i*)(&a%i))+1);\n", sz, sz, sz, 2 * sz);
			str += print2str("\ta%i += b%i;\n", sz, sz);
		}
		str += print2str("\treturn a2[0] + a2[1];\n");
		str += "}\n";
		str += "\n";
	}
#endif
#ifdef USE_DOUBLE
	str += print2str("SFMM_SIMD_FACTORY(simd_f64, double, simd_i64, int64_t, simd_ui64, uint64_t, %i);\n", DOUBLE_SIMD_WIDTH);
	if ( M2M_SIMD_WIDTH == FLOAT_SIMD_WIDTH) {
		str += print2str("using m2m_simd_f64 = simd_f64;\n");
		str += print2str("using m2m_simd_i64 = simd_i64;\n");
		str += print2str("using m2m_simd_ui64 = simd_ui64;\n");
	} else {
		str += print2str("SFMM_SIMD_FACTORY(m2m_simd_f64, double, m2m_simd_i64, int64_t, m2m_simd_ui64, uint64_t, %i);\n", M2M_SIMD_WIDTH);
	}
	for (int width = std::min(M2M_SIMD_WIDTH,DOUBLE_SIMD_WIDTH); width <= std::max(M2M_SIMD_WIDTH, DOUBLE_SIMD_WIDTH); width += std::abs(DOUBLE_SIMD_WIDTH - M2M_SIMD_WIDTH)) {
		str += print2str("\ninline double reduce_sum(const %s& v) {\n", width == DOUBLE_SIMD_WIDTH ? "simd_f64" : "m2m_simd_f64");
		for (int sz = width; sz > 1; sz /= 2) {
			str += print2str("\ttypedef double type%i __attribute__ ((vector_size(%i*sizeof(double))));\n", sz, sz);
		}
		str += print2str("\ttype%i a%i = *((type%i*)(&v));\n", width, width, width);
		for (int sz = width / 2; sz > 1; sz /= 2) {
			str += print2str("\ttype%i a%i = *((type%i*)(&a%i));\n", sz, sz, sz, 2 * sz);
			str += print2str("\tconst type%i& b%i = *(((type%i*)(&a%i))+1);\n", sz, sz, sz, 2 * sz);
			str += print2str("\ta%i += b%i;\n", sz, sz);
		}
		str += print2str("\treturn a2[0] + a2[1];\n");
		str += "}\n";
		str += "\n";
	}
#endif

	str += "#endif /* __CUDACC__ */\n";
#endif
	return str;
}

static int cuda = 0;

static bool is_float(std::string str) {
	return precision[typenum] == 1;
}

static bool is_double(std::string str) {
	return precision[typenum] == 2;
}

static bool is_vec(std::string str) {
	return simd[typenum];
}

const double ewald_r2 = (2.6 + 0.5 * sqrt(3));
const int ewald_h2 = 8;

static FILE* fp = nullptr;

int xyexp_sz(int P) {
	int index = 0;
	for (int l = 0; l <= P; l++) {
		for (int m = -l; m <= l; m++) {
			if (l % 2 != abs(m) % 2) {
				continue;
			}
			index++;
		}
	}
	return index;
}

int exp_sz(int P) {
	return (P + 1) * (P + 1);
}

int half_exp_sz(int P) {
	return (P + 2) * (P + 1) / 2;
}

int mul_sz(int P) {
	if (nodip) {
		if (P <= 1) {
			return 1;
		} else {
			return P * P - 3;
		}
	} else {
		return P * P;
	}
}

static flops_t greens_ewald_real_flops;

double tiny() {
	if (is_float(type)) {
		return 10.0 * std::numeric_limits<float>::min();
	} else {
		return 10.0 * std::numeric_limits<double>::min();
	}
}
double huge() {
	if (is_float(type)) {
		return 0.1 * std::numeric_limits<float>::max();
	} else {
		return 0.1 * std::numeric_limits<double>::max();
	}
}

double factorial(int n) {
	return n == 0 ? 1.0 : n * factorial(n - 1);
}

flops_t rescale_flops(int P) {
	flops_t fps;
	fps.asgn++;
	fps.r += (P + 1) * (P + 2) + 1;
	fps.rdiv++;
	return fps;
}

flops_t init_flops(int P) {
	flops_t fps;
	fps.asgn += (P + 1) * (P + 1) + 2;
	return fps;
}

flops_t copy_flops(int P) {
	flops_t fps;
	fps.asgn += (P + 1) * (P + 1) + 2;
	return fps;
}

flops_t accumulate_flops(int P) {
	flops_t fps;
	fps = rescale_flops(P);
	fps.r += (P + 1) * (P + 1) + 1;
	//fps.a++;
	return fps;
}

flops_t sqrt_flops() {
	flops_t fps;
	fps.r += 4;
	/*	fps.r += 10;
	 fps.i += 2;
	 fps.asgn += 2;
	 fps.fma += 2;
	 fps.rdiv++;
	 if (!is_float(type)) {
	 fps.fma += 2;
	 }*/
	return fps;
}

flops_t rsqrt_flops() {
	flops_t fps;
	fps.r += 4;
	/*	fps.r += 10;
	 fps.i += 2;
	 fps.asgn += 2;
	 fps.fma += 2;
	 if (!is_float(type)) {
	 fps.fma += 2;
	 }*/
	return fps;
}

flops_t sincos_flops() {
	flops_t fps;
	fps.r += 8;
	fps.fma += 10;
	fps.asgn += 3;
	fps.i += 8;
	fps.con += 2;
	if (!is_float(type)) {
		fps.fma += 10;
	}
	return fps;
}

flops_t erfcexp_flops() {
	flops_t fps;
	fps.r += 4;
	fps.rdiv += 1;
	fps.con += 2;
	fps.rcmp++;
	fps.i += 2;
	fps.fma += 7;
	fps.asgn++;
	if (!is_float(type)) {
		fps.fma += 11;
	}
	if (is_float(type)) {
		if (is_vec(type)) {
			fps.r += 14;
			fps.rcmp++;
			fps.con++;
			fps.rdiv += 2;
			fps.asgn += 2;
			fps.fma += 34;
		} else {
			fps.r += 7;
			fps.rcmp++;
			fps.asgn++;
			fps.fma += 25;
		}
	} else {
		if (is_vec(type)) {
			fps.r += 16;
			fps.rcmp += 4;
			fps.asgn += 5;
			fps.rdiv += 2;
			fps.fma += 76;
		} else {
			fps.r += 4;
			fps.rcmp += 2;
			fps.asgn += 2;
			fps.rdiv++;
			fps.fma += 36;
		}
	}
	return fps;

}

//#define COUNT_POT_FLOPS
flops_t parse_flops(const char* line);

flops_t count_flops(std::string fname) {
	FILE* fp = fopen(fname.c_str(), "rt");
	if (!fp) {
		printf("Unable to open %s\n", fname.c_str());
		abort();
	}
	constexpr int N = 1000;
	char line[N];
	flops_t fps;
	int ln = 0;
	bool inpot = false;
	while (fgets(line, N - 2, fp)) {
#ifdef COUNT_POT_FLOPS
		inpot = false;
#endif
		if (line[0] != '#') {
			int j = 0;
			fps += parse_flops(line);
		}
//		printf( "%i, %i : %s", fps.r, cnt, line);
	}

//	printf( "%i\n", ln);
	fclose(fp);
	return fps;
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

flops_t running_flops;

void reset_running_flops() {
	running_flops.reset();
}

flops_t get_running_flops(bool simd = true) {
	flops_t r = running_flops;
	//running_flops.reset();
	return r;
}

template<class ...Args>
void tprint(const char* str, Args&&...args) {
	if (fp == nullptr) {
		return;
	}
	auto line = print2str(str, std::forward<Args>(args)...);
	running_flops += parse_flops(line.c_str());
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			fprintf(fp, "\t");
		}
		fprintf(fp, "%s", line.c_str());
	}
}

void tprint(const char* str) {
	tprint("%s", str);
}

constexpr int nchains = 3;
static int current_chain;
static std::vector<std::vector<std::string>> inschains(nchains);

void tprint_new_chain() {
	int best = -1, smallest = 1000000000;
	for (int i = 0; i < inschains.size(); i++) {
		if (inschains[i].size() < smallest) {
			smallest = inschains[i].size();
			best = i;
		}
	}
	current_chain = best;
}

template<class ...Args>
void tprint_chain(const char* fstr, Args&&...args) {
	std::string str;
	for (int i = 0; i < ntab; i++) {
		str += "\t";
	}
	char* buf;
	ASPRINTF(&buf, fstr, std::forward<Args>(args)...)
	str += buf;
	free(buf);
//	printf( "%i\n", current_chain);
	inschains[current_chain].push_back(str);
}

void tprint_chain(const char* fstr) {
	std::string str;
	for (int i = 0; i < ntab; i++) {
		str += "\t";
	}
	str += fstr;
	inschains[current_chain].push_back(str);
}

void tprint_flush_chains() {
	int n[nchains];
	for (int i = 0; i < nchains; i++) {
		n[i] = 0;
	}
	const auto done = [&n]() {
		for( int i = 0; i < nchains; i++) {
			if( n[i] < inschains[i].size()) {
				return false;
			}
		}
		return true;
	};
	while (!done()) {
		int best = -1;
		double largest = 0;
		for (int i = 0; i < inschains.size(); i++) {
			if (n[i] < inschains[i].size()) {
				double value = (double) (inschains[i].size() - n[i]) / (inschains[i].size() + 1);
				if (value >= largest) {
					largest = value;
					best = i;
				}
			}
		}
		if (best == -1) {
			abort();
		}
		if (fp) {
			running_flops += parse_flops(inschains[best][n[best]].c_str());
			fprintf(fp, "%s", inschains[best][n[best]].c_str());
		}
		n[best]++;
	}
	inschains = decltype(inschains)(nchains);
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

flops_t parse_flops(const char* line) {
	flops_t fps0;
	int j = 0;
	while (j < strlen(line)) {
		if (strncmp(line + j, " += ", 4) == 0) {
			fps0.r++;
			j += 4;
		} else if (strncmp(line + j, " -= ", 4) == 0) {
			fps0.r++;
			j += 4;
		} else if (strncmp(line + j, " = ", 3) == 0) {
			fps0.asgn++;
			j += 2;
		} else if (strncmp(line + j, " *= ", 4) == 0) {
			fps0.r++;
			j += 4;
		} else if (strncmp(line + j, " /= ", 4) == 0) {
			fps0.rdiv++;
			j += 4;
		} else if (strncmp(line + j, " + ", 3) == 0) {
			fps0.r++;
			j += 3;
		} else if (strncmp(line + j, " - ", 3) == 0) {
			fps0.r++;
			j += 3;
		} else if (strncmp(line + j, " -", 2) == 0) {
			fps0.r++;
			j += 2;
		} else if (strncmp(line + j, " * ", 3) == 0) {
			fps0.r++;
			j += 3;
		} else if (strncmp(line + j, " / ", 3) == 0) {
			fps0.rdiv++;
			j += 3;
		} else if (strncmp(line + j, " < ", 3) == 0) {
			fps0.rcmp++;
			j += 3;
		} else if (strncmp(line + j, " > ", 3) == 0) {
			fps0.rcmp++;
			j += 3;
		} else if (strncmp(line + j, "CONVERT", 7) == 0) {
			fps0.con++;
			j += 7;
		} else if (strncmp(line + j, "fma", 3) == 0) {
			fps0.fma++;
			j += 3;
		} else if (strncmp(line + j, "rsqrt", 5) == 0) {
			fps0 += rsqrt_flops();
			j += 5;
		} else if (strncmp(line + j, "sqrt", 4) == 0) {
			fps0 += sqrt_flops();
			j += 4;
		} else if (strncmp(line + j, "erfcexp", 7) == 0) {
			fps0 += erfcexp_flops();
			j += 7;
		} else if (strncmp(line + j, "sincos", 6) == 0) {
			fps0 += sincos_flops();
			j += 6;
		} else if (strncmp(line + j, "safe_mul", 8) == 0) {
			fps0.i += 5;
			fps0.icmp++;
			fps0.con++;
			fps0.r += 2;
			j += 8;
		} else if (strncmp(line + j, "safe_add", 8) == 0) {
			fps0.i += 4;
			fps0.icmp += 2;
			fps0.con += 2;
			fps0.fma++;
			j += 8;
		} else if (strncmp(line + j, "greens_ewald_real", 17) == 0) {
			fps0 += greens_ewald_real_flops;
			j += 17;
		} else {
			j++;
		}
	}
	if (fps0.asgn) {
		if (fps0.con || fps0.fma || fps0.i || fps0.icmp || fps0.idiv || fps0.r || fps0.rcmp || fps0.rdiv) {
			fps0.asgn--;
		}
	}
	return fps0;
}

enum arg_type {
	LIT, PTR, CPTR, EXP, HEXP, XYEXP, MUL, CEXP, CMUL, FORCE, VEC3
};

void init_real(std::string var) {
	fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
	if (simd[typenum]) {
		tprint("T %s;\n", var.c_str());
		tprint("%s.set_NaN();\n", var.c_str());
	} else {
		tprint("T %s(std::numeric_limits<%s>::signaling_NaN());\n", var.c_str(), base_rtype[typenum].c_str());
	}
	fprintf(fp, "#else /* NDEBUG */ \n");
	tprint("T %s;\n", var.c_str());
	fprintf(fp, "#endif /* NDEBUG */ \n");
	if (var == "tmp1") {
		for (int i = 2; i < nchains; i++) {
			init_real(std::string("tmp") + std::to_string(i));
		}
	}
}

void init_reals(std::string var, int cnt) {
	fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
	if (simd[typenum]) {
		tprint("T %s [%i];\n", var.c_str(), cnt);
		for (int i = 0; i < cnt; i++) {
			tprint("%s[%i].set_NaN();\n", var.c_str(), i, base_rtype[typenum].c_str());
		}
	} else {
		tprint("T %s [%i]={", var.c_str(), cnt);
		int ontab = ntab;
		ntab = 0;
		for (int n = 0; n < cnt; n++) {
			tprint("std::numeric_limits<%s>::signaling_NaN()", base_rtype[typenum].c_str());
			if (n != cnt - 1) {
				tprint(",");
			}
		}
		tprint("};\n");
		ntab = ontab;
	}
	fprintf(fp, "#else /* NDEBUG */ \n");
	tprint("T %s [%i];\n", var.c_str(), cnt);
	fprintf(fp, "#endif /* NDEBUG */ \n");
//			" = {";
	/*	for (int n = 0; n < cnt - 1; n++) {
	 str += "std::numeric_limits<T>::signaling_NaN(), ";
	 }
	 str += "std::numeric_limits<T>::signaling_NaN()};";*/
}

template<class ...Args>
std::string func_args(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == VEC3) {
		str += std::string("vec3<") + type + "> " + arg;
	} else if (atype == EXP) {
		str += std::string("expansion") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == XYEXP) {
		str += std::string("detail::expansion_xy") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == HEXP) {
		str += std::string("detail::expansion_xz") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == MUL) {
		str += std::string("multipole") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CEXP) {
		str += std::string("const expansion") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CMUL) {
		str += std::string("const multipole") + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == FORCE) {
		str += std::string("force_type<") + type + ">& " + arg;
	} else {
		if (atype == CPTR) {
			str += std::string("const ");
		}
		str += type + " ";
		if (atype == PTR || atype == CPTR) {
			str += "*";
		}
		str += std::string(arg);
	}
	return str;
}

template<class ...Args>
std::string func_args(int P, const char* arg, arg_type atype, Args&& ...args) {
	auto str = func_args(P, arg, atype, 1);
	str += std::string(", ");
	str += func_args(P, std::forward<Args>(args)...);
	return str;
}
template<class ...Args>
std::string func_args_sig(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == VEC3) {
		str += std::string("vec3<") + type + ">";
	} else if (atype == EXP) {
		str += std::string("expansion") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == XYEXP) {
		str += std::string("detail::expansion_xy") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == HEXP) {
		str += std::string("detail::expansion_xz") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == MUL) {
		str += std::string("multipole") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == CEXP) {
		str += std::string("const expansion") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == CMUL) {
		str += std::string("const multipole") + "<" + type + ", " + std::to_string(P) + ">&";
	} else if (atype == FORCE) {
		str += std::string("force_type<") + type + ">&";
	} else {
		if (atype == CPTR) {
			str += std::string("const ");
		}
		str += type + "";
		if (atype == PTR || atype == CPTR) {
			str += "*";
		}
	}
	return str;
}

template<class ...Args>
std::string func_args_sig(int P, const char* arg, arg_type atype, Args&& ...args) {
	auto str = func_args_sig(P, arg, atype, 1);
	str += std::string(", ");
	str += func_args_sig(P, std::forward<Args>(args)...);
	return str;
}

template<class ...Args>
std::string func_args_call(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		str += std::string(arg) + "_st";
	} else if (atype == HEXP) {
		str += std::string(arg) + "_st";
	} else if (atype == XYEXP) {
		str += std::string(arg) + "_st";
	} else if (atype == MUL) {
		str += std::string(arg) + "_st";
	} else if (atype == CEXP) {
		str += std::string(arg) + "_st";
	} else if (atype == CMUL) {
		str += std::string(arg) + "_st";
	} else if (atype == FORCE) {
		str += std::string(arg);
	} else {
		str += std::string(arg);
	}
	return str;
}

template<class ...Args>
std::string func_args_call(int P, const char* arg, arg_type atype, Args&& ...args) {
	auto str = func_args_call(P, arg, atype, 1);
	str += std::string(", ");
	str += func_args_call(P, std::forward<Args>(args)...);
	return str;
}

template<class ...Args>
void func_args_cover(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == HEXP) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == XYEXP) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == CEXP) {
		tprint("const T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == MUL) {
		tprint("T* %s(%s_st.data());\n", arg, arg);
	} else if (atype == CMUL) {
		tprint("const T* %s(%s_st.data());\n", arg, arg);
	}
}

template<class ...Args>
void func_args_cover(int P, const char* arg, arg_type atype, Args&& ...args) {
	func_args_cover(P, arg, atype, 1);
	func_args_cover(P, std::forward<Args>(args)...);
}

template<class ...Args>
void func_args_dummies(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		tprint("static expansion<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == HEXP) {
		tprint("static expansion_xz<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == XYEXP) {
		tprint("static expansion_xy<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == CEXP) {
		tprint("static expansion<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == MUL) {
		tprint("static multipole<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == CMUL) {
		tprint("static multipole<%s, %i> %s_dummy;\n", type.c_str(), P, arg);
	} else if (atype == FORCE) {
		tprint("static force_type<%s> %s_dummy;\n", type.c_str(), arg);
	} else if (atype == VEC3) {
		tprint("static vec3<%s> %s_dummy;\n", type.c_str(), arg);
	}
}

template<class ...Args>
void func_args_dummies(int P, const char* arg, arg_type atype, Args&& ...args) {
	func_args_dummies(P, arg, atype, 1);
	func_args_dummies(P, std::forward<Args>(args)...);
}

void include(std::string filename) {
	constexpr int N = 1024;
	char buffer[N];
	filename = root_dir + filename;
	FILE* fp0 = fopen(filename.c_str(), "rt");
	if (!fp0) {
		printf("Unable to open %s\n", filename.c_str());
		abort();
	}
	while (!feof(fp0)) {
		if (!fgets(buffer, N, fp0)) {
			break;
		}
		fprintf(fp, "%s", buffer);
	}
	fclose(fp0);
}

std::string timing_body;
std::string current_sig;
int timing_cnt = 0;

template<class ... Args>
std::string func_header(const char* func, int P, bool pub, bool calcpot, bool timing, bool flops, bool vec, std::string head, Args&& ...args) {
	static std::set<std::string> igen;
	reset_running_flops();
	pub = pub && !nopot;
	std::string func_name = std::string(func);
	if (nopot && calcpot) {
		func_name += "_wo_potential";
	}
	std::string func_ptr = print2str("int(*)(");
	func_ptr += func_args_sig(P, std::forward<Args>(args)..., 0);
	func_ptr += ", int";
	func_ptr += ")";
	current_sig = func_ptr;
	if (timing) {
		if (timing_body != "") {
			timing_body += ",\n";
		}
		timing_body += print2str("\t{(void*)((%s) &%s), \"%s\", ", func_ptr.c_str(), func_name.c_str(), type.c_str());
		timing_cnt++;
	}
	std::string file_name = func_name + ".cpp";
	func_name = "int " + func_name;
	func_name += "(" + func_args(P, std::forward<Args>(args)..., 0);
	auto func_name2 = func_name;
	func_name += ", int flags = sfmmDefaultFlags)";
	func_name2 += ", int flags)";
	if (nopot && !calcpot) {
		set_file("/dev/null");
	} else {
		set_file(full_header.c_str());
	}
	static std::set<std::string> already_printed;
	if (already_printed.find(func_name) == already_printed.end()) {
		already_printed.insert(func_name);
		if (prefix[0]) {
			func_name = std::string(prefix) + " " + func_name;
			func_name2 = std::string(prefix) + " " + func_name2;
		}
		if (pub) {
			tprint("%s;\n", func_name.c_str());
		} else {
			if (is_vec(type)) {
				detail_header_vec += func_name + ";\n";
			} else {
				detail_header += func_name + ";\n";
			}
		}
	} else {
		if (prefix[0]) {
			func_name = std::string(prefix) + " " + func_name;
			func_name2 = std::string(prefix) + " " + func_name2;
		}
	}
	std::string func1 = std::string(func) + std::string("_") + type;
	auto dir = std::string("./generated_code/src/") + type;
	if (cuda) {
		dir += "_cuda";
	}
	std::string cmd = "mkdir -p " + dir;
	SYSTEM(cmd.c_str());
	dir += "/P" + std::to_string(P);
	cmd = "mkdir -p " + dir;
	SYSTEM(cmd.c_str());
	file_name = dir + "/" + file_name;
//		printf("%s ", file_name.c_str());
	if (nopot && !calcpot) {
		set_file("/dev/null");
	} else {
		set_file(file_name);
	}
	tprint("#include \"%s\"\n", header.c_str());
//	tprint("#include \"detail/%s\"\n", header.c_str());
	tprint("#include \"typecast_%s.hpp\"\n", type.c_str());
	tprint("\n");
	tprint("namespace sfmm {\n");
	if (!pub) {
		tprint("namespace detail {\n");
	}
	if (head != "") {
		tprint("\n");
		tprint("%s\n", head.c_str());
	}
	tprint("\n");
	tprint("%s {\n", func_name2.c_str());
	indent();
	if (calcpot && !nopot) {
		tprint("if( flags & sfmmCalculateWithoutPotential ) {\n ");
		indent();
		std::string str = std::string("return detail::") + std::string(func) + std::string("_wo_potential(");
		str += func_args_call(P, std::forward<Args>(args)..., 0);
		str += ", flags);\n";
		tprint("%s", str.c_str());
		deindent();
		tprint("}\n");
	}
	if (flops) {
		tprint("if( !(flags & sfmmFLOPsOnly) ) {\n");
		indent();
	}
	func_args_cover(P, std::forward<Args>(args)..., 0);
	if (vec) {
		tprint("T& x=dx[0];\n");
		tprint("T& y=dx[1];\n");
		tprint("T& z=dx[2];\n");
	}
	return file_name;
}

void create_func_data_ptr(std::string fname) {
	tprint("static auto* const func_data_ptr = detail::operator_initialize((void*)((%s) &%s%s));\n", current_sig.c_str(), fname.c_str(), pot_name());
}

void open_timer(std::string fname) {
	tprint("static bool initialized = false;\n");
	tprint("detail::func_data_t* func_data_ptr;\n");
	tprint("timer tm;\n");
	tprint("if(!initialized) {\n");
	indent();
	tprint("initialized = true;\n");
	tprint("func_data_ptr = detail::operator_initialize((void*)((%s) &%s%s));\n", current_sig.c_str(), fname.c_str(), pot_name());
	deindent();
	tprint("}\n");
	tprint("if( flags & sfmmProfilingOn ) {\n");
	indent();
	tprint("tm.start();\n");
	deindent();
	tprint("}\n");
}

void close_timer() {
	tprint("if( flags & sfmmProfilingOn ) {\n");
	indent();
	tprint("tm.stop();\n");
	tprint("detail::operator_update_timing(func_data_ptr, tm.read());\n");
	deindent();
	tprint("}\n");
}

void set_tprint(bool c) {
	tprint_on = c;
}

int lindex(int l, int m) {
	return l * (l + 1) + m;
}

int xyindex(int l0, int m0) {
	int index = 0;
	for (int l = 0;; l++) {
		for (int m = -l; m <= l; m++) {
			if (l % 2 != abs(m) % 2) {
				if (l == l0 && m == m0) {
					abort();
				} else {
					continue;
				}
			}
			if (l == l0 && m == m0) {
				return index;
			}
			index++;
		}
	}
}

int mindex(int l, int m) {
	if (nodip) {
		if (l == 1) {
			abort();
		}
		return std::max(0, lindex(l, m) - 3);
	} else {
		return lindex(l, m);
	}
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

void z_rot(int P, const char* dst, const char* src, stage_t stage, std::string opname = "") {
	tprint("rx[0] = cosphi;\n");
	tprint("ry[0] = sinphi;\n");
	for (int m = 1; m < P; m++) {
		tprint("rx[%i] = rx[%i] * cosphi - ry[%i] * sinphi;\n", m, m - 1, m - 1);
		tprint("ry[%i] = fma(rx[%i], sinphi, ry[%i] * cosphi);\n", m, m - 1, m - 1);
	}
	int mmin = 1;
	bool initR = true;
	std::function<int(int, int)> index;
	if (dst[0] == 'M') {
		index = mindex;
	} else {
		index = lindex;
	}
	bool same = strcmp(dst, src) == 0;
	std::vector<int> set_nan;
	if (!same) {
		tprint_chain("%s[0] = %s[0];\n", dst, src);
	}
	for (int m = 1; m <= P; m++) {
		if (!same) {
			tprint_new_chain();
			tprint_chain("%s[%i] = %s[%i];\n", dst, index(m, 0), src, index(m, 0));
		}
		for (int l = m; l <= P; l++) {
			if (dst[0] == 'M' && nodip && l == 1) {
				continue;
			}
			if (stage == POST1) {
				if (l == P) {
					if (l % 2 != m % 2) {
						continue;
					}
				} else if (nodip && l == P - 1) {
					if (l % 2 != m % 2) {
						continue;
					}
				}
			}
			bool read_ionly = false;
			bool read_ronly = false;
			bool write_ronly = false;
			if (stage == POST2) {
				if (l == P) {
					read_ionly = m % 2;
				} else if (nodip && l == P - 1) {
					read_ionly = m % 2;
				}
				if (l == P) {
					read_ronly = !(m % 2);
				} else if (nodip && l == P - 1) {
					read_ronly = !(m % 2);
				}
			} else if (stage == PRE2) {
				if (l == P || opname == "M2P" || opname == "L2P") {
					write_ronly = m % 2 != l % 2;
				}
			}
			read_ronly = read_ronly || (stage == XZ1 && l >= P);
			read_ronly = read_ronly || (stage == XZ2 && l >= P - 1);
			bool sw = false;
			if (stage == POST1 && (l >= P - 1)) {
				if (nodip) {
					sw = true;
				} else {
					sw = m % 2 == P % 2;
				}
			}
			read_ronly = read_ronly || sw;
			if (read_ionly) {
				tprint_new_chain();
				tprint_chain("%s[%i] = -%s[%i] * ry[%i];\n", dst, index(l, m), src, index(l, -m), m - 1);
				tprint_chain("%s[%i] = %s[%i] * rx[%i];\n", dst, index(l, -m), src, index(l, -m), m - 1);
			} else if (read_ronly) {
				tprint_new_chain();
				tprint_chain("%s[%i] = %s[%i] * ry[%i];\n", dst, index(l, -m), src, index(l, m), m - 1);
				tprint_chain("%s[%i] = %s[%i] * rx[%i];\n", dst, index(l, m), src, index(l, m), m - 1);
			} else if (write_ronly) {
				tprint_new_chain();
				tprint_chain("%s[%i] = %s[%i] * rx[%i] - %s[%i] * ry[%i];\n", dst, index(l, m), src, index(l, m), m - 1, src, index(l, -m), m - 1);
				set_nan.push_back(index(l, -m));
			} else {
				if (same) {
					tprint_new_chain();
					tprint_chain("tmp%i = %s[%i];\n", current_chain, dst, index(l, m));
					tprint_chain("%s[%i] = %s[%i] * rx[%i] - %s[%i] * ry[%i];\n", dst, index(l, m), src, index(l, m), m - 1, src, index(l, -m), m - 1);
					tprint_chain("%s[%i] = fma(tmp%i, ry[%i], %s[%i] * rx[%i]);\n", dst, index(l, -m), current_chain, m - 1, src, index(l, -m), m - 1);
				} else {
					tprint_new_chain();
					tprint_chain("%s[%i] = %s[%i] * rx[%i] - %s[%i] * ry[%i];\n", dst, index(l, m), src, index(l, m), m - 1, src, index(l, -m), m - 1);
					tprint_chain("%s[%i] = fma(%s[%i], ry[%i], %s[%i] * rx[%i]);\n", dst, index(l, -m), src, index(l, m), m - 1, src, index(l, -m), m - 1);
				}
			}

		}
	}
	tprint_flush_chains();
	if (set_nan.size()) {
		fprintf(fp, "#ifndef NDEBUG\n");
		for (auto i : set_nan) {
			tprint("%s[%i]=std::numeric_limits<%s>::signaling_NaN();\n", dst, i, type.c_str());
		}
		fprintf(fp, "#endif /* NDEBUG */ \n");
	}
}

void z_rot(int P, const char* name, stage_t stage, std::string opname = "") {
	z_rot(P, name, name, stage, opname);
}

void z_rot2(int P, const char* dst, const char* src, stage_t stage, std::string opname, bool init) {
	if (init) {
		tprint("r0[0] = cosphi;\n");
		tprint("ip[0] = sinphi;\n");
		tprint("in[0] = -sinphi;\n");
		for (int m = 1; m < P; m++) {
			tprint("r0[%i] = in[%i] * sinphi;\n", m, m - 1);
			tprint("ip[%i] = ip[%i] * cosphi;\n", m, m - 1);
			tprint("r0[%i] = fma(r0[%i], cosphi, r0[%i]);\n", m, m - 1, m);
			tprint("ip[%i] = fma(r0[%i], sinphi, ip[%i]);\n", m, m - 1, m);
			tprint("in[%i] = -ip[%i];\n", m, m);
		}
	}
	int mmin = 1;
	bool initR = true;
	std::function<int(int, int)> index;
	if (dst[0] == 'M') {
		index = mindex;
	} else {
		index = lindex;
	}
	std::vector<int> set_nan;
	using cmd_t = std::pair<int,std::string>;
	std::vector<cmd_t> cmds;
	cmds.push_back(std::make_pair(0, print2str("%s[0] = %s[0];\n", dst, src)));
	for (int m = 1; m <= P; m++) {
		if (!(nodip && m == 1 && dst[0] == 'M')) {
			cmds.push_back(std::make_pair(index(m, 0), print2str("%s[%i] = %s[%i];\n", dst, index(m, 0), src, index(m, 0))));
		}
		for (int l = m; l <= P; l++) {
			if (dst[0] == 'M' && nodip && l == 1) {
				continue;
			}
			if (stage == POST1) {
				if (l == P) {
					if (l % 2 != m % 2) {
						continue;
					}
				} else if (nodip && l == P - 1) {
					if (l % 2 != m % 2) {
						continue;
					}
				}
			}
			bool read_ionly = false;
			bool read_ronly = false;
			bool write_ronly = false;
			if (stage == POST2) {
				if (l == P) {
					read_ionly = m % 2;
				} else if (nodip && l == P - 1) {
					read_ionly = m % 2;
				}
				if (l == P) {
					read_ronly = !(m % 2);
				} else if (nodip && l == P - 1) {
					read_ronly = !(m % 2);
				}
			} else if (stage == PRE2) {
				if (l == P || opname == "M2P" || opname == "L2P") {
					write_ronly = m % 2 != l % 2;
				}
			}
			read_ronly = read_ronly || (stage == XZ1 && l >= P);
			read_ronly = read_ronly || (stage == XZ2 && l >= P - 1);
			bool sw = false;
			if (stage == POST1 && (l >= P - 1)) {
				if (nodip) {
					sw = true;
				} else {
					sw = m % 2 == P % 2;
				}
			}
			read_ronly = read_ronly || sw;
			if (read_ionly) {
				cmds.push_back(std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * in[%i];\n", dst, index(l, m), src, index(l, -m), m - 1)));
				cmds.push_back(std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * r0[%i];\n", dst, index(l, -m), src, index(l, -m), m - 1)));
			} else if (read_ronly) {
				cmds.push_back(std::make_pair(index(l, m), print2str("%s[%i] = %s[%i] * ip[%i];\n", dst, index(l, -m), src, index(l, m), m - 1)));
				cmds.push_back(std::make_pair(index(l, m), print2str("%s[%i] = %s[%i] * r0[%i];\n", dst, index(l, m), src, index(l, m), m - 1)));
			} else if (write_ronly) {
				cmds.push_back(std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * in[%i];\n", dst, index(l, m), src, index(l, -m), m - 1)));
				cmds.push_back(
						std::make_pair(index(l, m),
								print2str("%s[%i] = fma(%s[%i], r0[%i], %s[%i]);\n", dst, index(l, m), src, index(l, m), m - 1, dst, index(l, m))));
				set_nan.push_back(index(l, -m));
			} else {
				cmds.push_back(std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * in[%i];\n", dst, index(l, m), src, index(l, -m), m - 1)));
				cmds.push_back(std::make_pair(index(l, -m), print2str("%s[%i] = %s[%i] * r0[%i];\n", dst, index(l, -m), src, index(l, -m), m - 1)));
				cmds.push_back(
						std::make_pair(index(l, m),
								print2str("%s[%i] = fma(%s[%i], r0[%i], %s[%i]);\n", dst, index(l, m), src, index(l, m), m - 1, dst, index(l, m))));
				cmds.push_back(
						std::make_pair(index(l, m),
								print2str("%s[%i] = fma(%s[%i], ip[%i], %s[%i]);\n", dst, index(l, -m), src, index(l, m), m - 1, dst, index(l, -m))));
			}

		}
	}
	std::sort(cmds.begin(), cmds.end(), [](const cmd_t& a, const cmd_t& b) {
		return a.first < b.first;
	});
	for (auto cmd : cmds) {
		tprint("%s", cmd.second.c_str());
	}
	if (set_nan.size()) {
		fprintf(fp, "#ifndef NDEBUG\n");
		for (auto i : set_nan) {
			tprint("%s[%i]=std::numeric_limits<%s>::signaling_NaN();\n", dst, i, type.c_str());
		}
		fprintf(fp, "#endif /* NDEBUG */ \n");
	}
}
void xz_swap(int P, const char* name, bool inv, stage_t stage, const char* opname = "") {
	auto brot = [inv](int n, int m, int l) {
		if( inv ) {
			return Brot(n,m,l);
		} else {
			return Brot(n,l,m);
		}
	};
	std::function<int(int, int)> index;
	if (name[0] == 'M') {
		index = mindex;
	} else {
		index = lindex;
	}
	std::vector<int> set_nan;
	for (int n = 1; n <= P; n++) {
		if (name[0] == 'M' && nodip && n == 1) {
			continue;
		}
		int lmax = n;
		if (stage == POST1) {
			lmax = std::min(n, P - n);
			if (nodip && n == P - 1) {
				lmax = 0;
			}
		}
		for (int m = -lmax; m <= lmax; m++) {
			bool flag = false;
			if (stage == POST2) {
				if (P == n && n % 2 != abs(m) % 2) {
					continue;
				} else if (nodip && n == P - 1 && n % 2 != abs(m) % 2) {
					continue;
				}
			}
			tprint("A[%i] = %s[%i];\n", m + P, name, index(n, m));
		}
		std::vector<std::vector<std::pair<double, int>>>ops(2 * n + 1);
		int mmax = n;
		if (stage == PRE2) {
			mmax = std::min((P + 1) - n, n);
			if (std::string(opname) == std::string("M2P") || std::string(opname) == std::string("L2P")) {
				mmax = std::min(1, mmax);
			}
		}
		for (int m = 0; m <= n; m++) {
			if (m <= mmax) {
				for (int l = 0; l <= lmax; l++) {
					bool flag = false;
					if (stage == POST2) {
						if (P == n && n % 2 != abs(l) % 2) {
							continue;
						} else if (nodip && P - 1 == n && n % 2 != abs(l) % 2) {
							continue;
						}
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
			} else {
				set_nan.push_back(index(n, m));
				set_nan.push_back(index(n, -m));
			}
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			std::sort(ops[m].begin(), ops[m].end(), [](std::pair<double,int> a, std::pair<double,int> b) {
				return a.first < b.first;
			});
		}

		for (int m = 0; m < 2 * n + 1; m++) {
			tprint_new_chain();
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
						tprint_chain("%s[%i] %s= A[%i];\n", name, index(n, m - n), l == 0 ? "" : "+", ops[m][l].second);
					} else {
						if (l == 0) {
							tprint_chain("%s[%i] = TCAST(%.20e) * A[%i];\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second);
						} else {
							tprint_chain("%s[%i] = fma(TCAST(%.20e), A[%i], %s[%i]);\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second, name,
									index(n, m - n));
						}
					}
				} else {
					tprint_chain("tmp%i = A[%i];\n", current_chain, ops[m][l].second);
					for (int p = 1; p < len; p++) {
						tprint_chain("tmp%i += A[%i];\n", current_chain, ops[m][l + p].second);
					}
					if (l == 0) {
						tprint_chain("%s[%i] = TCAST(%.20e) * tmp%i;\n", name, index(n, m - n), ops[m][l].first, current_chain);
					} else {
						tprint_chain("%s[%i] = fma(TCAST(%.20e), tmp%i, %s[%i]);\n", name, index(n, m - n), ops[m][l].first, current_chain, name, index(n, m - n));
					}
				}
				l += len - 1;
			}
		}
		tprint_flush_chains();
	}
	if (set_nan.size()) {
		fprintf(fp, "#ifndef NDEBUG\n");
		for (auto i : set_nan) {
			tprint("%s[%i]=std::numeric_limits<%s>::signaling_NaN();\n", name, i, type.c_str());
		}
		fprintf(fp, "#endif /* NDEBUG */ \n");
	}
}

void xz_swap2(int P, const char* dst, const char* src, bool inv, stage_t stage, const char* opname = "") {
	auto brot = [inv](int n, int m, int l) {
		if( inv ) {
			return Brot(n,m,l);
		} else {
			return Brot(n,l,m);
		}
	};
	std::function<int(int, int)> index;
	if (dst[0] == 'M') {
		index = mindex;
	} else {
		index = lindex;
	}
	tprint("%s[0] = %s[0];\n", dst, src);
	std::vector<int> set_nan;
	for (int n = 1; n <= P; n++) {
		if (dst[0] == 'M' && nodip && n == 1) {
			continue;
		}
		int lmax = n;
		if (stage == POST1) {
			lmax = std::min(n, P - n);
			if (nodip && n == P - 1) {
				lmax = 0;
			}
		}
		std::vector<std::vector<std::pair<double, int>>>ops(2 * n + 1);
		int mmax = n;
		if (stage == PRE2) {
			mmax = std::min((P + 1) - n, n);
			if (std::string(opname) == std::string("M2P") || std::string(opname) == std::string("L2P")) {
				mmax = std::min(1, mmax);
			}
		}
		for (int m = 0; m <= n; m++) {
			if (m <= mmax) {
				for (int l = 0; l <= lmax; l++) {
					bool flag = false;
					if (stage == POST2) {
						if (P == n && n % 2 != abs(l) % 2) {
							continue;
						} else if (nodip && P - 1 == n && n % 2 != abs(l) % 2) {
							continue;
						}
					}
					double r = l == 0 ? brot(n, m, 0) : brot(n, m, l) + nonepow<double>(l) * brot(n, m, -l);
					double i = l == 0 ? 0.0 : brot(n, m, l) - nonepow<double>(l) * brot(n, m, -l);
					if (r != 0.0) {
						ops[n + m].push_back(std::make_pair(r, l));
					}
					if (i != 0.0 && m != 0) {
						ops[n - m].push_back(std::make_pair(i, -l));
					}
				}
			} else {
				set_nan.push_back(index(n, m));
				set_nan.push_back(index(n, -m));
			}
		}
		for (int m = 0; m < 2 * n + 1; m++) {
			std::sort(ops[m].begin(), ops[m].end(), [n, index](std::pair<double,int> a, std::pair<double,int> b) {
				return index(n, a.second) < index(n, b.second);
			});
		}
		for (int ppp = 0; ppp < (P + 1) * (P + 1); ppp++) {
			for (int m = 0; m < 2 * n + 1; m++) {
				for (int l = 0; l < ops[m].size(); l++) {
					int len = 1;
					while (len + l < ops[m].size()) {
						if (ops[m][len + l].first == ops[m][l].first && !close21(ops[m][l].first)) {
							len++;
						} else {
							break;
						}
					}
					if (index(n, ops[m][l].second) != ppp) {
						continue;
					}
					if (close21(ops[m][l].first)) {
						tprint("%s[%i] %s= %s[%i];\n", dst, index(n, m - n), l == 0 ? "" : "+", src, index(n, ops[m][l].second));
					} else {
						if (l == 0) {
							tprint("%s[%i] = TCAST(%.20e) * %s[%i];\n", dst, index(n, m - n), ops[m][l].first, src, index(n, ops[m][l].second));
						} else {
							tprint("%s[%i] = fma(TCAST(%.20e), %s[%i], %s[%i]);\n", dst, index(n, m - n), ops[m][l].first, src, index(n, ops[m][l].second), dst,
									index(n, m - n));
						}
					}
					l += len - 1;
				}
			}
		}
	}
	if (set_nan.size()) {
		fprintf(fp, "#ifndef NDEBUG\n");
		for (auto i : set_nan) {
			tprint("%s[%i]=std::numeric_limits<%s>::signaling_NaN();\n", dst, i, type.c_str());
		}
		fprintf(fp, "#endif /* NDEBUG */ \n");
	}
}

void m2l(int P, int Q, const char* mname, const char* lname) {
	tprint("A[0] = rinv;\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = rinv * A[%i];\n", n, n - 1);
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, factorial(n));
	}
	bool first[(Q + 1)][(2 * Q + 1)];
	for (int n = 0; n <= Q; n++) {
		for (int m = -n; m <= n; m++) {
			first[n][n + m] = true;
		}
	}
	for (int n = nopot; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			tprint_new_chain();
			const int maxk = std::min(P - n, P - 1);
			for (int k = m; k <= maxk; k++) {
				if (nodip && k == 1) {
					continue;
				}
				if (first[n][n + m]) {
					first[n][n + m] = false;
					tprint_chain("%s[%i] = %s[%i] * A[%i];\n", lname, lindex(n, m), mname, mindex(k, m), n + k);
				} else {
					tprint_chain("%s[%i] = fma(%s[%i], A[%i], %s[%i]);\n", lname, lindex(n, m), mname, mindex(k, m), n + k, lname, lindex(n, m));
				}
				if (m != 0) {
					if (first[n][n - m]) {
						first[n][n - m] = false;
						tprint_chain("%s[%i] = %s[%i] * A[%i];\n", lname, lindex(n, -m), mname, mindex(k, -m), n + k);
					} else {
						tprint_chain("%s[%i] = fma(%s[%i], A[%i], %s[%i]);\n", lname, lindex(n, -m), mname, mindex(k, -m), n + k, lname, lindex(n, -m));
					}
				}
			}
			if (m % 2 != 0) {
				if (!first[n][n + m]) {
					tprint_chain("%s[%i] = -%s[%i];\n", lname, lindex(n, m), lname, lindex(n, m));
				}
				if (!first[n][n - m]) {
					tprint_chain("%s[%i] = -%s[%i];\n", lname, lindex(n, -m), lname, lindex(n, -m));
				}
			}
		}
	}
	for (int n = nopot; n <= Q; n++) {
		for (int m = -n; m <= n; m++) {
			if (first[n][n + m]) {
				//	printf("---- %i %i %i %i %i\n", nodip, P, Q, n, m);
				//			tprint("L[%i] = TCAST(0);\n", lindex(n, m));
			}
		}
	}

	tprint_flush_chains();

}

void greens_body(int P, const char* M = nullptr) {
	tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("r2inv = TCAST(1) / r2;\n");
	tprint("O[0] = rsqrt(r2);\n");
	if (M) {
		tprint("O[0] *= %s;\n", M);
	}
	tprint("x *= r2inv;\n");
	tprint("y *= r2inv;\n");
	tprint("z *= r2inv;\n");
	tprint("zx1 = z;\n");
	for (int m = 1; m < P; m++) {
		tprint("zx%i = TCAST(%i) * z;\n", 2 * m + 1, 2 * m + 1);
	}
	auto index = lindex;
	tprint("O[%i] = x * O[0];\n", index(1, 1));
	tprint("O[%i] = y * O[0];\n", index(1, -1));
	for (int m = 2; m <= P; m++) {
		tprint("ax0 = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
		tprint("ay0 = O[%i] * TCAST(%i);\n", index(m - 1, -(m - 1)), 2 * m - 1);
		tprint("O[%i] = x * ax0 - y * ay0;\n", index(m, m));
		tprint("O[%i] = fma(y, ax0, x * ay0);\n", index(m, -m));
	}
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		const int c = current_chain;
		if (m + 1 <= P) {
			tprint_chain("O[%i] = zx%i * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			if (m != 0) {
				tprint_chain("O[%i] = zx%i * O[%i];\n", index(m + 1, -m), 2 * m + 1, index(m, -m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint_chain("ay%i = TCAST(-%i) * r2inv;\n", c, (n - 1) * (n - 1) - m * m);
				tprint_chain("O[%i] = fma(zx%i, O[%i], ay%i * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), c, index(n - 2, m));
				tprint_chain("O[%i] = fma(zx%i, O[%i], ay%i * O[%i]);\n", index(n, -m), 2 * n - 1, index(n - 1, -m), c, index(n - 2, -m));
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint_chain("O[%i] = (zx%i * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
				} else {
					tprint_chain("O[%i] = fma(zx%i, O[%i], TCAST(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), -((n - 1) * (n - 1) - m * m),
							index(n - 2, m));
				}

			}
		}
	}
	tprint_flush_chains();
}

std::string greens_safe(int P) {
	TAB0();
	reset_running_flops();
	auto fname = func_header("greens_safe", P, true, false, false, true, true, "", "O", EXP, "dx", VEC3);
	const auto mul = [](std::string a, std::string b, std::string c, int l) {
		tprint( "sw[%i] *= safe_mul(%s, %s, %s);\n", l, a.c_str(), b.c_str(), c.c_str());
	};
	const auto mul2 = [](std::string a, std::string b, std::string c, int l) {
		tprint( "sw[%i] *= safe_mul(%s, %s, %s);\n", l, a.c_str(), b.c_str(), c.c_str());
	};
	const auto add = [](std::string a, std::string b, std::string c, int l) {
		tprint( "sw[%i] *= safe_add(%s, %s, %s);\n", l, a.c_str(), b.c_str(), c.c_str());
	};
	tprint("T sw[]={");
	int otab = ntab;
	ntab = 0;
	for (int i = 0; i <= P; i++) {
		tprint("TCAST(1)%s", i != exp_sz(P) - 1 ? "," : "");
	}
	tprint("};\n");
	ntab = otab;
	init_real("flag");
	init_real("tmp0");
	init_real("tmp1");
	init_real("tmp3");
	init_real("rinv");
	init_real("r2");
	init_real("r2inv");
	init_real("ax");
	init_real("ay");
	tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("r2inv = TCAST(1) / r2;\n");
	tprint("O[0] = rsqrt(r2);\n");
	tprint("x *= r2inv;\n");
	tprint("y *= r2inv;\n");
	tprint("z *= r2inv;\n");
	auto index = lindex;
	const auto O = [index](int n, int m) {
		return std::string("O[") + std::to_string(index(n, m)) + "]";
	};
	const auto tcast = []( int n ) {
		return std::string("TCAST(") + std::to_string(n) + ")";
	};
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			mul(O(m, m), "x", "O[0]", m);
			mul(O(m, -m), "y", "O[0]", m);
		} else if (m > 0) {
			mul2("ax", O(m - 1, m - 1), tcast(2 * m - 1), m);
			mul2("ay", O(m - 1, -(m - 1)), tcast(2 * m - 1), m);
			mul("tmp2", "x", "ax", m);
			mul("tmp3", "y", "ay", m);
			tprint("tmp3 = -tmp3;\n");
			add(O(m, m), "tmp2", "tmp3", m);
			mul("tmp2", "x", "ay", m);
			mul("tmp3", "y", "ax", m);
			add(O(m, -m), "tmp2", "tmp3", m);
		}
		if (m + 1 <= P) {
			//	tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			mul("tmp2", "z", O(m, m), m + 1);
			mul2(O(m + 1, m), "tmp2", tcast(2 * m + 1), m + 1);
			if (m != 0) {
				mul("tmp2", "z", O(m, -m), m + 1);
				mul2(O(m + 1, -m), "tmp2", tcast(2 * m + 1), m + 1);
			}
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				mul2("ax", "z", tcast(2 * n - 1), n);
				mul2("ay", "r2inv", tcast(-((n - 1) * (n - 1) - m * m)), n);
				mul("tmp2", O(n - 1, m), "ax", n);
				mul("tmp3", O(n - 2, m), "ay", n);
				add(O(n, m), "tmp2", "tmp3", n);
				mul("tmp2", O(n - 1, -m), "ax", n);
				mul("tmp3", O(n - 2, -m), "ay", n);
				add(O(n, -m), "tmp2", "tmp3", n);
			} else {
				mul("tmp2", "z", O(n - 1, 0), n);
				mul2("tmp2", "tmp2", tcast(2 * n - 1), n);
				mul("tmp3", "r2inv", O(n - 2, 0), n);
				if ((n - 1) * (n - 1) != 1) {
					mul2("tmp3", "tmp3", tcast((n - 1) * (n - 1)), n);
				}
				tprint("tmp3 = -tmp3;\n");
				add(O(n, 0), "tmp2", "tmp3", n);
			}
		}
	}
	tprint("O[0] *= sw[0];\n");
	for (int n = 1; n <= P; n++) {
		tprint("sw[%i] *= sw[%i];\n", n, n - 1);
		tprint("flag = sw[%i];\n", n);
		for (int m = -n; m <= n; m++) {
			tprint("O[%i] *= flag;\n", index(n, m), n);
		}
	}
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "greens_safe"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

void m2lg_body(int P, int Q, bool ronly = false, std::function<int(int, int)> oindex = lindex) {
	struct entry_t {
		int l;
		int m;
		int o;
	};
	std::vector<entry_t> pos;
	std::vector<entry_t> neg;
	for (int n = nopot; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			bool iload = true;
			bool rload = true;
			tprint_new_chain();
			const int kmax = std::min(P - n, P - 1);
			for (int k = 0; k <= kmax; k++) {
				if (nodip && k == 1) {
					continue;
				}
				const int lmin = std::max(-k, -n - k - m);
				const int lmax = std::min(k, n + k - m);
				for (int l = lmin; l <= lmax; l++) {
					bool greal = false;
					bool mreal = false;
					int gxsgn = 1;
					int gysgn = 1;
					int mxsgn = 1;
					int mysgn = 1;
					int gxstr;
					int gystr;
					int mxstr;
					int mystr;
					if (m + l > 0) {
						gxstr = oindex(n + k, m + l);
						gystr = oindex(n + k, -m - l);
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							gxstr = oindex(n + k, -m - l);
							gystr = oindex(n + k, m + l);
							gysgn = -1;
						} else {
							gxstr = oindex(n + k, -m - l);
							gystr = oindex(n + k, m + l);
							gxsgn = -1;
						}
					} else {
						greal = true;
						gxstr = oindex(n + k, 0);
					}
					if (l > 0) {
						mxstr = mindex(k, l);
						mystr = mindex(k, -l);
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							mxstr = mindex(k, -l);
							mystr = mindex(k, l);
						} else {
							mxstr = mindex(k, -l);
							mystr = mindex(k, l);
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						mxstr = mindex(k, 0);
					}
					const auto csgn = [](int i) {
						return i > 0 ? '+' : '-';
					};
					const auto add_work = [n,ronly,&gystr,&pos,&neg](int sgn, int m, int mstr, int gstr) {
						if( ronly && gystr == gstr) {
							return;
						}
						entry_t entry;
						entry.l = lindex(n, m);
						entry.m = mstr;
						entry.o = gstr;
						if( sgn == 1) {
							pos.push_back(entry);
						} else {
							neg.push_back(entry);
						}
					};
					if (!mreal) {
						if (!greal) {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
							add_work(-mysgn * gysgn, m, mystr, gystr);
							if (m > 0) {
								add_work(mysgn * gxsgn, -m, mystr, gxstr);
								add_work(mxsgn * gysgn, -m, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
							if (m > 0) {
								add_work(mysgn * gxsgn, -m, mystr, gxstr);
							}
						}
					} else {
						if (!greal) {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
							if (m > 0) {
								add_work(mxsgn * gysgn, -m, mxstr, gystr);
							}
						} else {
							add_work(mxsgn * gxsgn, m, mxstr, gxstr);
						}
					}
				}
			}
		}
	}
	const auto cmp = [](entry_t a, entry_t b) {
		if( a.m < b.m ) {
			return true;
		} else if( a.m > b.m ) {
			return false;
		} else {
			return a.l < b.l;
		}
	};
	std::sort(neg.begin(), neg.end(), cmp);
	std::sort(pos.begin(), pos.end(), cmp);
	bool first[(P + 1) * (P + 1)];
	for (int n = 0; n < (P + 1) * (P + 1); n++) {
		first[n] = true;
	}
	for (int i = 0; i < neg.size(); i++) {
		if (first[neg[i].l]) {
			tprint("L[%i] = M[%i] * O[%i];\n", neg[i].l, neg[i].m, neg[i].o);
			first[neg[i].l] = false;
		} else {
			tprint("L[%i] = fma(M[%i], O[%i], L[%i]);\n", neg[i].l, neg[i].m, neg[i].o, neg[i].l);
		}
	}
	for (int n = 0; n < (P + 1) * (P + 1); n++) {
		if (!first[n]) {
			tprint("L[%i] = -L[%i];\n", n, n);
		}
	}
	for (int i = 0; i < pos.size(); i++) {
		if (first[pos[i].l]) {
			first[pos[i].l] = false;
			tprint("L[%i] = M[%i] * O[%i];\n", pos[i].l, pos[i].m, pos[i].o);
		} else {
			tprint("L[%i] = fma(M[%i], O[%i], L[%i]);\n", pos[i].l, pos[i].m, pos[i].o, pos[i].l);
		}
	}
}

std::vector<complex> spherical_singular_harmonic(int P, double x, double y, double z) {
	const double r2 = x * x + y * y + z * z;
	const double r2inv = double(1) / r2;
	complex R = complex(x, y);
	std::vector<complex> O(exp_sz(P));
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
	return std::abs(a) < 1e-10;
}

std::string P2L(int P) {
	auto fname = func_header("P2L", P, true, true, true, true, true, "", "L", EXP, "M", LIT, "dx", VEC3);
	init_reals("O", exp_sz(P));
	init_real("tmp1");
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
	init_real("zx1");
	for (int m = 1; m < P; m++) {
		init_real(std::string("zx") + std::to_string(2 * m + 1));
	}
	reset_running_flops();
	if (scaled) {
		tprint("tmp1 = TCAST(1) / L_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	greens_body(P, "M");
	for (int n = nopot; n < exp_sz(P); n++) {
		tprint("L[%i] += O[%i];\n", n, n);
	}
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	timing_body += print2str("\"P2L\", %i, %i, 0, %i, 0.0, 0}", P, nopot, get_running_flops(false).load() + flops_map[P][type].load());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	if (nopot) {
		tprint("}\n");
	}
	tprint("\n");
	TAB0();
	return fname;
}

std::string greens(int P) {
	auto fname = func_header("greens", P, true, false, false, true, true, "", "O", EXP, "dx", VEC3);
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
	init_real("zx1");
	for (int m = 1; m < P; m++) {
		init_real(std::string("zx") + std::to_string(2 * m + 1));
	}
	reset_running_flops();
	greens_body(P);
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "greens"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

std::string greens_xz(int P) {
	auto fname = func_header("greens_xz", P, false, false, false, true, false, "", "O", HEXP, "x", LIT, "z", LIT, "r2inv", LIT);
	init_real("ax0");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
	reset_running_flops();
	tprint("O[0] = sqrt(r2inv);\n");
	if (periodic && P > 1) {
		tprint("O_st.trace2() = TCAST(0);\n");
	}
	tprint("x *= r2inv;\n");
	tprint("z *= r2inv;\n");
	tprint("const T& zx1 = z;\n");
	for (int m = 1; m < P; m++) {
		tprint("const T zx%i = TCAST(%i) * z;\n", 2 * m + 1, 2 * m + 1);
	}
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	tprint("O[%i] = x * O[0];\n", index(1, 1));
	for (int m = 2; m <= P; m++) {
		tprint("O[%i] = x * O[%i] * TCAST(%i);\n", index(m, m), index(m - 1, m - 1), 2 * m - 1);
	}
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		if (m + 1 <= P) {
			tprint_chain("O[%i] = zx%i * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint_chain("ay%i = TCAST(-%i) * r2inv;\n", current_chain, (n - 1) * (n - 1) - m * m);
				tprint_chain("O[%i] = fma(zx%i, O[%i], ay%i * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), current_chain, index(n - 2, m));
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint_chain("O[%i] = (zx%i * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));

				} else {
					tprint_chain("O[%i] = fma(zx%i, O[%i], TCAST(-%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
				}
			}
		}
	}
	tprint_flush_chains();
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "greens_xz"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

std::string M2LG(int P, int Q) {
	auto fname = func_header("M2LG", P, true, true, false, true, false, "", "L", EXP, "M", CMUL, "O", EXP);
	reset_running_flops();
	m2lg_body(P, Q);
	if (!nopot && P > 2 && periodic) {
		tprint("L[%i] = fma(TCAST(-0.5) * O_st.trace2(), M_st.trace2(), L[%i]);\n", lindex(0, 0), lindex(0, 0));
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", lindex(1, -1), lindex(1, -1), lindex(1, -1));
		tprint("L[%i] -= O_st.trace2() * M[%i];\n", lindex(1, +0), lindex(1, +0), lindex(1, +0));
		tprint("L[%i] = fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", lindex(1, +1), lindex(1, +1), lindex(1, +1));
		tprint("L_st.trace2() = fma(TCAST(-0.5) * O_st.trace2(), M[%i], L_st.trace2());\n", lindex(0, 0));
	}
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "M2LG"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	TAB0();
	return fname;
}

std::string greens_ewald(int P, double alpha) {
	auto index = lindex;
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
	auto fname = func_header("greens_ewald", P, true, true, false, true, true, "", "G", EXP, "dx", VEC3);
	reset_running_flops();
	const auto c = tprint_on;
	tprint("expansion<%s, %i> Gr_st;\n", type.c_str(), P);
	tprint("T* Gr(Gr_st.data());\n", type.c_str(), P);
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
	bool first = true;
	init_real("sw");
	init_real("r");
	init_real("r2");
	init_real("xxx");
	init_real("tmp0");
	init_real("tmp1");
	init_real("gam1");
	init_real("exp0");
	init_real("xfac");
	init_real("xpow");
	init_real("gam");
	init_real("x2");
	init_real("x2y2");
	init_real("hdotx");
	init_real("phi");
	init_real("rzero");
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
				s += "x";
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
	for (int hx = -H; hx <= H; hx++) {
		for (int hy = -H; hy <= H; hy++) {
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= H2 && h2 > 0) {
					init_real(cosname(hx, hy, hz));
					init_real(sinname(hx, hy, hz));
				}
			}
		}
	}
	tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("rzero = TCONVERT(r2 < TCAST(%0.20e));\n", tiny());
	tprint("r = sqrt(r2) + rzero;\n");
	tprint("greens_safe(Gr_st, vec3<T>(x + rzero, y, z));\n");
	tprint("xxx = TCAST(%.20e) * r;\n", alpha);
	tprint("erfcexp(xxx, &gam1, &exp0);\n");
	tprint("gam1 *= TCAST(%.20e);\n", sqrt(M_PI));
	tprint("xfac = TCAST(%.20e) * r2;\n", alpha * alpha);
	tprint("xpow = TCAST(%.20e) * r;\n", alpha);
	double gam0inv = 1.0 / sqrt(M_PI);
	tprint("sw = TCAST(1) - rzero;\n");
	for (int l = 0; l <= P; l++) {
		tprint("gam = gam1 * TCAST(%.20e);\n", gam0inv);
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = sw * (TCAST(%.1e) - gam) * Gr[%i];\n", lindex(l, m), nonepow<double>(l), lindex(l, m));
		}
		if (l == 0) {
			tprint("G[%i] += rzero * TCAST(%.20e);\n", lindex(0, 0), (2) * alpha / sqrt(M_PI));
		}
		gam0inv *= 1.0 / -(l + 0.5);
		if (l != P) {
			tprint("gam1 = fma(TCAST(%.20e), gam1, xpow * exp0);\n", l + 0.5);
			if (l != P - 1) {
				tprint("xpow *= xfac;\n");
			}
		}
	}
	for (int ix = -R; ix <= R; ix++) {
		for (int iy = -R; iy <= R; iy++) {
			for (int iz = -R; iz <= R; iz++) {
				int ii = ix * ix + iy * iy + iz * iz;
				if (ii > R2 || ii == 0) {
					continue;
				}
				std::string xstr = "x";
				if (ix != 0) {
					xstr += std::string(" ") + (ix < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(ix)) + ")";
				}
				std::string ystr = "y";
				if (iy != 0) {
					ystr += std::string(" ") + (iy < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(iy)) + ")";
				}
				std::string zstr = "z";
				if (iz != 0) {
					zstr += std::string(" ") + (iz < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(iz)) + ")";
				}
				tprint("detail::greens_ewald_real<%s, %i, %i>(G_st, %s, %s, %s);\n", type.c_str(), P, lround(alpha * 100), xstr.c_str(), ystr.c_str(),
						zstr.c_str());
			}
		}
	}

	for (int hx = -H; hx <= H; hx++) {
		if (hx) {
			if (abs(hx) == 1) {
				tprint("x2 = %cx;\n", hx < 0 ? '-' : ' ');
			} else {
				tprint("x2 = TCAST(%i) * x;\n", hx);
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
					tprint("x2y2 = x2 %c y;\n", hy > 0 ? '+' : '-');
				} else {
					tprint("x2y2 = fma(TCAST(%i), y, x2);\n", hy);
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
						tprint("hdotx = x2y2 %c z;\n", hz > 0 ? '+' : '-');
					} else {
						tprint("hdotx = fma(TCAST(%i), z, x2y2);\n", hz);
					}
				} else {
					tprint("hdotx = x2y2;\n", hz);
				}
				tprint("phi = TCAST(%.20e) * hdotx;\n", 2.0 * M_PI);
				tprint("sincos(phi, &%s, &%s);\n", sinname(hx, hy, hz).c_str(), cosname(hx, hy, hz).c_str());
			}
		}
	}
	using table_type = std::unordered_map<double, std::vector<std::pair<int, std::string>>>;
	table_type ops[exp_sz(P)];
	for (int hx = -H; hx <= H; hx++) {
		for (int hy = -H; hy <= H; hy++) {
			for (int hz = -H; hz <= H; hz++) {
				const int h2 = hx * hx + hy * hy + hz * hz;
				if (h2 <= H2 && h2 > 0) {
					const double h = sqrt(h2);
					bool init = false;
					const auto G0 = spherical_singular_harmonic(P, (double) hx, (double) hy, (double) hz);
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
	for (int ii = 0; ii < exp_sz(P); ii++) {
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
	for (int ii = nopot; ii < exp_sz(P); ii++) {
		std::vector<std::pair<double, std::vector<std::pair<int, std::string> > > > sorted_ops(ops[ii].begin(), ops[ii].end());
		std::sort(sorted_ops.begin(), sorted_ops.end(),
				[](const std::pair<double,std::vector<std::pair<int, std::string> > >& a, const std::pair<double,std::vector<std::pair<int, std::string> > > & b ) {
					return fabs(a.first) * sqrt(a.second.size()) < fabs(b.first) * sqrt(b.second.size());
				});
		tprint_new_chain();
		for (auto j = sorted_ops.begin(); j != sorted_ops.end(); j++) {
			auto op = j->second;
			if (op.size()) {
				int sgn = op[0].first > 0 ? 1 : -1;
				for (int k = 0; k < op.size(); k++) {
					tprint_chain("tmp%i %c= %s;\n", current_chain, k == 0 ? ' ' : (sgn * op[k].first > 0 ? '+' : '-'), op[k].second.c_str());
				}
				if (sgn > 0) {
					tprint_chain("G[%i] = fma(TCAST(+%.20e), tmp%i, G[%i]);\n", ii, sgn * j->first, current_chain, ii);
				} else {
					tprint_chain("G[%i] = fma(TCAST(%.20e), tmp%i, G[%i]);\n ", ii, sgn * j->first, current_chain, ii);
				}
			}
		}
	}
	tprint_flush_chains();
	if (P > 1) {
		tprint("G_st.trace2() = TCAST(%.20e);\n", (4.0 * M_PI / 3.0));
	}
	if (!nopot) {
		tprint("G[%i] += TCAST(%.20e);\n", index(0, 0), M_PI / (alpha * alpha));
	}
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "greens_ewald"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	TAB0();
	return fname;

}

const char* boolstr(bool b) {
	return b ? "true" : "false";
}

void greens_xz_body(int P) {
	tprint("O[0] = sqrt(r2inv);\n");
	if (periodic && P > 1) {
		tprint("O_st.trace2() = TCAST(0);\n");
	}
	tprint("x *= r2inv;\n");
	tprint("z *= r2inv;\n");
	tprint("zx1 = z;\n");
	for (int m = 1; m < P; m++) {
		tprint("zx%i = TCAST(%i) * z;\n", 2 * m + 1, 2 * m + 1);
	}
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	tprint("O[%i] = x * O[0];\n", index(1, 1));
	for (int m = 2; m <= P; m++) {
		tprint("O[%i] = x * O[%i] * TCAST(%i);\n", index(m, m), index(m - 1, m - 1), 2 * m - 1);
	}
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		if (m + 1 <= P) {
			tprint_chain("O[%i] = zx%i * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint_chain("ay%i = TCAST(-%i) * r2inv;\n", current_chain, (n - 1) * (n - 1) - m * m);
				tprint_chain("O[%i] = fma(zx%i, O[%i], ay%i * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), current_chain, index(n - 2, m));
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint_chain("O[%i] = (zx%i * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));

				} else {
					tprint_chain("O[%i] = fma(zx%i, O[%i], TCAST(-%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
				}
			}
		}
	}
	tprint_flush_chains();
}

void M2L_allrot(int P, int Q, int rot) {
	std::string name = print2str("M2%cr%i", P == Q ? 'L' : 'P', rot);
	if (Q > 1) {
		func_header(name.c_str(), P, true, true, true, true, true, "", "L", EXP, rot == 0 ? "M" : "Min", CMUL, "dx", VEC3);
	} else {
		func_header(name.c_str(), P, true, true, true, true, true, "", "f", FORCE, rot == 0 ? "M" : "Min", CMUL, "dx", VEC3);
	}
	open_timer(name);
	init_real("tmp1");
	bool minit = false;
	if (rot != 0) {
		tprint("multipole<%s, %i> M_st;\n", type.c_str(), P);
		tprint("multipole<%s, %i> M0_st;\n", type.c_str(), P);
		tprint("T* M(M_st.data());\n");
		tprint("T* M0(M0_st.data());\n");
		if (Q == P) {
			tprint("expansion<%s, %i> L0_st;\n", type.c_str(), P);
			tprint("T* L0(L0_st.data());\n");
		}
		tprint("T* ptr;\n");
	}
	if (Q == 1) {
		init_reals("La", 4);
		init_reals("Lb", 4);
		tprint("T* L=La;\n");
		tprint("T* L0=Lb;\n");
	}
	if (rot == 1) {
		tprint("detail::expansion_xz<%s, %i> O_st;\n", type.c_str(), P);
	} else if (rot == 0) {
		tprint("expansion<%s, %i> O_st;\n", type.c_str(), P);
	}
	if (rot != 2) {
		tprint("T* O(O_st.data());\n", type.c_str(), P);
	}
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ax1");
	init_real("ax2");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
	init_real("R2");
	int N = std::max(P - 1, Q);
	if (rot > 0) {
		if (rot == 2) {
			init_reals("r0A", N);
			init_reals("ipA", N);
			init_reals("inA", N);
			init_reals("r0B", N);
			init_reals("ipB", N);
			init_reals("inB", N);
			tprint("T* r0=r0A;\n");
			tprint("T* ip=ipA;\n");
			tprint("T* in=inA;\n");
		} else {
			init_reals("r0", N);
			init_reals("ipA", N);
			init_reals("inA", N);
			tprint("T* ip=ipA;\n");
			tprint("T* in=inA;\n");
		}
		init_real("Rinv");
		init_real("R");
		init_real("Rzero");
		init_real("rzero");
		init_real("sinphi");
		init_real("cosphi");
		init_real("tmp0");
		init_real("r2przero");
		if (rot == 2) {
			init_real("rinv");
			init_reals("A", P + 1);
		}
	}
	if (rot != 2) {
		init_real("zx1");
		for (int m = 1; m < P; m++) {
			init_real(std::string("zx") + std::to_string(2 * m + 1));
		}
	}
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M0_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	if (periodic && P > 2) {
		tprint("M_st.t = M0_st.t;\n");
	}
	if (scaled) {
		tprint("M_st.r = M0_st.r;\n");
	}
	if (rot > 0) {
		tprint("R2 = fma(x, x, y * y);\n");
		tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
		tprint("r2 = fma(z, z, R2);\n");
		tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
		tprint("tmp1 = R2 + Rzero;\n");
		tprint("Rinv = rsqrt(tmp1);\n");
		tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
		tprint("r2przero = (r2 + rzero);\n");
		if (rot == 1) {
			tprint("r2inv = TCAST(1) / (r2przero);\n");
			tprint("cosphi = fma(x, Rinv, Rzero);\n");
			tprint("sinphi = -y * Rinv;\n");
		} else {
			tprint("rinv = rsqrt(r2przero);\n");
			tprint("cosphi = y * Rinv;\n");
			tprint("sinphi = fma(x, Rinv, Rzero);\n");
		}
	}
	if (rot == 1) {
		z_rot2(P - 1, "M", "Min", FULL, P != Q ? "M2P" : "M2L", true);
	} else if (rot == 2) {
		tprint("r0[0] = cosphi;\n");
		tprint("ip[0] = sinphi;\n");
		tprint("in[0] = -sinphi;\n");
		for (int m = 1; m < N; m++) {
			tprint("r0[%i] = in[%i] * sinphi;\n", m, m - 1);
			tprint("ip[%i] = ip[%i] * cosphi;\n", m, m - 1);
			tprint("r0[%i] = fma(r0[%i], cosphi, r0[%i]);\n", m, m - 1, m);
			tprint("ip[%i] = fma(r0[%i], sinphi, ip[%i]);\n", m, m - 1, m);
			tprint("in[%i] = -ip[%i];\n", m, m);
		}
		z_rot2(P - 1, "M0", "Min", PRE1, P != Q ? "M2P" : "M2L", false);
		xz_swap2(P - 1, "M", "M0", false, PRE1, P != Q ? "M2P" : "M2L");
		tprint("cosphi = fma(z, rinv, rzero);\n");
		tprint("sinphi = -R * rinv;\n");
		tprint("r0=r0B;\n");
		tprint("in=inB;\n");
		tprint("ip=ipB;\n");
		tprint("r0[0] = cosphi;\n");
		tprint("ip[0] = sinphi;\n");
		tprint("in[0] = -sinphi;\n");
		for (int m = 1; m < N; m++) {
			tprint("r0[%i] = in[%i] * sinphi;\n", m, m - 1);
			tprint("ip[%i] = ip[%i] * cosphi;\n", m, m - 1);
			tprint("r0[%i] = fma(r0[%i], cosphi, r0[%i]);\n", m, m - 1, m);
			tprint("ip[%i] = fma(r0[%i], sinphi, ip[%i]);\n", m, m - 1, m);
			tprint("in[%i] = -ip[%i];\n", m, m);
		}
		z_rot2(P - 1, "M0", "M", PRE2, P != Q ? "M2P" : "M2L", false);
		tprint("in=ipB;\n");
		tprint("ip=inB;\n");
		xz_swap2(P - 1, "M", "M0", false, PRE2, P != Q ? "M2P" : "M2L");
	}

	if (rot == 0) {
		greens_body(P);
	} else if (rot == 1) {
		tprint("O[0] = sqrt(r2inv);\n");
		if (periodic && P > 1) {
			tprint("O_st.trace2() = TCAST(0);\n");
		}
		tprint("R *= r2inv;\n");
		tprint("z *= r2inv;\n");
		tprint("zx1 = z;\n");
		for (int m = 1; m < P; m++) {
			tprint("zx%i = TCAST(%i) * z;\n", 2 * m + 1, 2 * m + 1);
		}
		const auto index = [](int l, int m) {
			return l*(l+1)/2+m;
		};
		tprint("O[%i] = R * O[0];\n", index(1, 1));
		for (int m = 2; m <= P; m++) {
			tprint("O[%i] = R * O[%i] * TCAST(%i);\n", index(m, m), index(m - 1, m - 1), 2 * m - 1);
		}
		for (int m = 0; m <= P; m++) {
			tprint_new_chain();
			if (m + 1 <= P) {
				tprint_chain("O[%i] = zx%i * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			}
			for (int n = m + 2; n <= P; n++) {
				if (m != 0) {
					tprint_chain("ay%i = TCAST(-%i) * r2inv;\n", current_chain, (n - 1) * (n - 1) - m * m);
					tprint_chain("O[%i] = fma(zx%i, O[%i], ay%i * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), current_chain, index(n - 2, m));
				} else {
					if ((n - 1) * (n - 1) - m * m == 1) {
						tprint_chain("O[%i] = (zx%i * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));

					} else {
						tprint_chain("O[%i] = fma(zx%i, O[%i], TCAST(-%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
								index(n - 2, m));
					}
				}
			}
		}
		tprint_flush_chains();
	}
	if (rot == 1 && P == Q) {
		tprint("ptr=L0;\n");
		tprint("L0=L;\n");
		tprint("L=ptr;\n");
	}
	if (rot != 2) {
		m2lg_body(P, Q, rot == 1, rot == 0 ? lindex : cindex);
	} else {
		tprint("A[0] = rinv;\n");
		for (int n = 1; n <= P; n++) {
			tprint("A[%i] = rinv * A[%i];\n", n, n - 1);
		}
		for (int n = 2; n <= P; n++) {
			tprint("A[%i] *= TCAST(%.20e);\n", n, factorial(n));
		}
		bool first[(Q + 1) * (Q + 1)];
		for (int n = 0; n < (Q + 1) * (Q + 1); n++) {
			first[n] = true;
		}
		struct entry_t {
			int l;
			int m;
			int r;
		};
		std::vector<entry_t> cmds;
		for (int n = nopot; n <= Q; n++) {
			for (int m = 0; m <= n; m++) {
				const int maxk = std::min(P - n, P - 1);
				for (int k = m; k <= maxk; k++) {
					if (nodip && k == 1) {
						continue;
					}
					entry_t cmd;
					cmd.l = lindex(n, m);
					cmd.m = mindex(k, m);
					cmd.r = n + k;
					cmds.push_back(cmd);
					if (m != 0) {
						cmd.l = lindex(n, -m);
						cmd.m = mindex(k, -m);
						cmds.push_back(cmd);
					}
				}
			}
		}
		std::sort(cmds.begin(), cmds.end(), [](entry_t a, entry_t b) {
			if( a.m < b.m ) {
				return true;
			} else if( a.m > b.m ) {
				return false;
			} else {
				return a.l < b.l;
			}
		});
		for (auto cmd : cmds) {
			if (first[cmd.l]) {
				first[cmd.l] = false;
				tprint("L[%i] = M[%i] * A[%i];\n", cmd.l, cmd.m, cmd.r);
			} else {
				tprint("L[%i] = fma(M[%i], A[%i], L[%i]);\n", cmd.l, cmd.m, cmd.r, cmd.l);
			}
		}
		for (int n = nopot; n <= Q; n++) {
			for (int m = -n; m <= n; m++) {
				if (abs(m) % 2 != 0) {
					if (!first[lindex(n, m)]) {
						tprint("L[%i] = -L[%i];\n", lindex(n, m), lindex(n, m));
					}
				}
			}
		}
	}
	if (rot == 1) {
		tprint("ip=inA;\n");
		tprint("in=ipA;\n");
		if (nodip) {
			z_rot2(Q, "L0", "L", ((Q == P) || (Q == 1 && P == 2)) ? XZ2 : FULL, P != Q ? "M2P" : "M2L", false);
		} else {
			z_rot2(Q, "L0", "L", Q == P ? XZ1 : FULL, P != Q ? "M2P" : "M2L", false);
		}
	} else if (rot == 2) {
		xz_swap2(Q, "L0", "L", true, P == Q ? POST1 : FULL, P != Q ? "M2P" : "M2L");
		z_rot2(Q, "L", "L0", P == Q ? POST1 : FULL, P != Q ? "M2P" : "M2L", false);
		xz_swap2(Q, "L0", "L", true, P == Q ? POST2 : FULL, P != Q ? "M2P" : "M2L");
		tprint("r0=r0A;\n");
		tprint("in=ipA;\n");
		tprint("ip=inA;\n");
		z_rot2(Q, "L", "L0", P == Q ? POST2 : FULL, P != Q ? "M2P" : "M2L", false);
	}
	if (Q == 1) {
		if (rot == 1) {
			tprint("L=L0;\n");
		}
		if (scaled) {
			tprint("rinv = TCAST(1) / M_st.scale();\n");
			tprint("r2inv = rinv * rinv;\n");
			if (!nopot) {
				tprint("f.potential = L[0] * rinv;\n");
			}
			tprint("f.force[0] = -L[3] * r2inv;\n");
			tprint("f.force[1] = -L[1] * r2inv;\n");
			tprint("f.force[2] = -L[2] * r2inv;\n");
		} else {
			if (!nopot) {
				tprint("f.potential = L[0];\n");
			}
			tprint("f.force[0] = -L[3];\n");
			tprint("f.force[1] = -L[1];\n");
			tprint("f.force[2] = -L[2];\n");
		}
	}
	close_timer();
	deindent();
	tprint("}");

	tprint("return %i;\n", get_running_flops().load());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	timing_body += print2str("\"M2%c\", %i, %i, %i, %i, 0.0, 0}", Q == P ? 'L' : 'P', P, nopot, rot, get_running_flops(false).load());
	TAB0();
}

std::string flags_header(const char* op, int P) {
	std::string str;
	str += print2str("static const int best_rot = operator_best_rotation(%i, %i, \"%s\", \"%s\");\n", P, nopot, type.c_str(), op);
	str += "\tint rot = -1;\n";
	str += "\tif( flags & sfmmWithRandomOptimization ) {\n";
	str += "\t\tflags &= ~sfmmWithRandomOptimization;\n";
	str += "\t\tconstexpr unsigned a = 1664525;\n";
	str += "\t\tconstexpr unsigned c = 1013904223;\n";
	str += "\t\tstatic thread_local unsigned num = 1;\n";
	str += "\t\tnum = a * num + c;\n";
	str += "\t\trot = num % 3;\n";
	str += "\t} else if( flags & sfmmWithBestOptimization ) {\n";
	str += "\t\tflags &= ~sfmmWithBestOptimization;\n";
	str += "\t\trot = best_rot;\n";
	str += "\t}\n";
	str += "\tif( rot >= 0 ) {\n";
	str += "\t\tflags |= (rot == 1 ? sfmmWithSingleRotationOptimization : (rot == 2 ? sfmmWithDoubleRotationOptimization : 0));\n";
	str += "\t}\n";
	return str;
}

std::string flags_choose3(std::string name, std::string name2) {
	std::string str;
	str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
	str += print2str("\t\treturn %sr1(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
	str += print2str("\t\treturn %sr2(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t} else {\n";
	str += print2str("\t\treturn %sr0(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t}\n";
	return str;
}
std::string flags_choose2(std::string name, std::string name2) {
	std::string str;
	str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
	str += print2str("\t\treturn %sr1(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t} else {\n";
	str += print2str("\t\treturn %sr0(%s, M0_st, dx, flags);\n", name.c_str(), name2.c_str());
	str += "\t}\n";
	return str;
}

std::string M2L(int P, int Q) {
	if (nopot) {
		return "";
	}
	std::string fname;
	std::string str;
	if (Q > 1) {
		fname = func_header("M2L", P, true, false, false, false, true, "", "L0", EXP, "M0", CMUL, "dx", VEC3);
		str = flags_header("M2L", P);
		str += flags_choose3("M2L", "L0_st");
	} else {
		fname = func_header("M2P", P, true, false, false, false, true, "", "f", FORCE, "M0", CMUL, "dx", VEC3);
		str += print2str("static const int best_rot = operator_best_rotation(%i, %i, \"%s\", \"%s\");\n", P, nopot, type.c_str(), "M2P");
		str += "\tint rot = -1;\n";
		str += "\tif( flags & sfmmWithRandomOptimization ) {\n";
		str += "\t\tflags &= ~sfmmWithRandomOptimization;\n";
		str += "\t\tconstexpr unsigned a = 1664525;\n";
		str += "\t\tconstexpr unsigned c = 1013904223;\n";
		str += "\t\tstatic thread_local unsigned num = 1;\n";
		str += "\t\tnum = a * num + c;\n";
		str += "\t\trot = num % 3;\n";
		str += "\t} else if( flags & sfmmWithBestOptimization ) {\n";
		str += "\t\tflags &= ~sfmmWithBestOptimization;\n";
		str += "\t\trot = best_rot;\n";
		str += "\t}\n";
		str += "\tif( rot >= 0 ) {\n";
		str += "\t\tflags |= (rot == 1 ? sfmmWithSingleRotationOptimization : (rot == 2 ? sfmmWithDoubleRotationOptimization : 0));\n";
		str += "\t}\n";
		str += flags_choose3("M2P", "f");
	}
	tprint(str.c_str());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	TAB0();
	return fname;
}

std::string M2L_ewald(int P) {
	auto fname = func_header("M2L_ewald", P, true, true, false, true, true, "", "L0", EXP, "M0", CMUL, "dx", VEC3);
	reset_running_flops();
	tprint("expansion<%s, %i> G_st;\n", type.c_str(), P);
	tprint("expansion<%s,%i> L_st;\n", type.c_str(), P);
	tprint("multipole<%s,%i> M_st;\n", type.c_str(), P);
	tprint("T* G(G_st.data());\n", type.c_str(), P);
	tprint("T* M(M_st.data());\n");
	tprint("T* L(L_st.data());\n");
	if (scaled) {
		init_real("a");
		init_real("b");
	}
	reset_running_flops();
	for (int n = nopot; n < exp_sz(P); n++) {
		tprint("L[%i] = TCAST(0);\n", n);
	}
	if (scaled) {
		tprint("L_st.r = %s(1);\n", base_rtype[typenum].c_str());
	}
	if (periodic && P > 1) {
		tprint("L_st.trace2() = TCAST(0);\n");
	}
	if (scaled) {
		tprint("a = M0_st.r;\n");
		tprint("b = a;\n");
		tprint("M_st.r = %s(1);\n", base_rtype[typenum].c_str());
		for (int n = 1; n < P; n++) {
			if (!(n == 1 && nodip)) {
				for (int m = -n; m <= n; m++) {
					tprint("M_st.o[%i] = M0_st.o[%i] * b;\n", mindex(n, m), mindex(n, m));
				}
			}
			if (periodic && P > 2 && n == 2) {
				tprint("M_st.t = M0_st.t * b;\n");
			}
			if (n != P - 1) {
				tprint("b *= a;\n");
			}
		}
		tprint("M_st.o[0] = M0_st.o[0];\n");
	} else {
		for (int n = 0; n < mul_sz(P); n++) {
			tprint("M_st.o[%i] = M0_st.o[%i];\n", n, n);
		}
		if (P > 2) {
			tprint("M_st.t = M0_st.t;\n");
		}
		if (scaled) {
			tprint("M_st.r = M0_st.r;\n");
		}
	}
	tprint("int flops = greens_ewald%s(G_st, vec3<T>(x, y, z));\n", nopot ? "_wo_potential" : "");
	tprint("flops += M2LG%s(L_st, M_st, G_st);\n", nopot ? "_wo_potential" : "");
	if (scaled) {
		tprint("a = L0_st.scale() / M_st.scale();\n");
		tprint("b = a;\n");
		tprint("L_st.r = L0_st.scale();\n");
		for (int n = 0; n <= P; n++) {
			if (!nopot || n > 0) {
				for (int m = -n; m <= n; m++) {
					tprint("L_st.o[%i] *= b;\n", lindex(n, m));
				}
				if (periodic && P > 1 && n == 2) {
					tprint("L_st.t *= b;\n", exp_sz(P));
				}
			}
			if (n != P) {
				tprint("b *= a;\n");
			}
		}
	}

	for (int n = nopot; n < exp_sz(P); n++) {
		tprint("L0[%i] += L[%i];\n", n, n);
	}
	if (periodic && P > 1) {
		tprint("L0_st.trace2() += L_st.trace2();\n");
	}
	tprint("return flops+%i;\n", get_running_flops().load());
	deindent();
	tprint("} else {\n");
	indent();
	tprint("expansion<%s, %i> O_st;\n", type.c_str(), P);
	tprint("return greens_ewald(O_st, vec3<T>(T(0), T(0), T(0)), flags) + M2LG%s(L_st, M_st, G_st) + %i;\n", pot_name(),
			get_running_flops(false).load() + flops_map[P][type + "greens_ewald_real"].load());
	deindent();
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	TAB0();
	return fname;
}

void func_closer(int P, std::string name, bool pub) {
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + name] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	if (!pub) {
		tprint("}\n");
	}
	tprint("\n");

}

void M2M_z(int P, int dir) {
	tprint("Y[0] = z;\n");
	for (int i = 1; i < P - 1; i++) {
		tprint("Y[%i] = z * Y[%i];\n", i, i - 1);
	}
	for (int i = 1; i < P - 1; i++) {
		tprint("Y[%i] *= TCAST(%0.20e);\n", i, 1.0 / factorial(i + 1));
	}
	for (int n = P - 1; n >= 0; n--) {
		for (int m = -n; m <= n; m++) {
			for (int k = 1; k <= n; k++) {
				if (abs(m) > n - k) {
					continue;
				}
				if (nodip && dir > 0) {
					if (n - k == 1) {
						continue;
					} else if (n == 1) {
						tprint("Md = Y[%i] * M[%i];\n", k - 1, mindex(n - k, m));
						continue;
					}
				}
				if (nodip && dir < 0) {
					if (n - k == 1) {
						if (m == 1) {
							tprint("M[%i] = fma(Y[%i], Md, M[%i]);\n", mindex(n, m), k - 1, mindex(n, m));
							continue;
						} else {
							continue;
						}
					} else if (n == 1) {
						continue;
					}
				}
				tprint("M[%i] = fma(Y[%i], M[%i], M[%i]);\n", mindex(n, m), k - 1, mindex(n - k, m), mindex(n, m));
			}
		}
	}
}

void regular_harmonic_full(int P) {
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			if (m - 1 > 0) {
				tprint("ax0 = Y[%i] * TCAST(%.20e);\n", lindex(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay0 = Y[%i] * TCAST(%.20e);\n", lindex(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax0 - y * ay0;\n", lindex(m, m));
				tprint("Y[%i] = fma(y, ax0, x * ay0);\n", lindex(m, -m));
			} else {
				tprint("Y[%i] = x * TCAST(%.20e);\n", lindex(m, m), 1.0 / (2.0 * m));
				tprint("Y[%i] = y * TCAST(%.20e);\n", lindex(m, -m), 1.0 / (2.0 * m));
			}
		}
	}
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		if (m + 1 <= P) {
			if (m == 0) {
				tprint_chain("Y[%i] = z;\n", lindex(m + 1, m));
			} else {
				tprint_chain("Y[%i] = z * Y[%i];\n", lindex(m + 1, m), lindex(m, m));
				tprint_chain("Y[%i] = z * Y[%i];\n", lindex(m + 1, -m), lindex(m, -m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			if (n == 2 && m == 0) {
				tprint_chain("ax%i = TCAST(%.20e) * z;\n", current_chain, inv * double(2 * n - 1));
				tprint_chain("ay%i = TCAST(%.20e) * R2;\n", current_chain, -(double) inv);
				tprint_chain("Y[%i] = fma(ax%i, Y[%i], ay%i);\n", lindex(n, m), current_chain, lindex(n - 1, m), current_chain);
			} else {
				tprint_chain("ax%i = TCAST(%.20e) * z;\n", current_chain, inv * double(2 * n - 1));
				tprint_chain("ay%i = TCAST(%.20e) * R2;\n", current_chain, -(double) inv);
				tprint_chain("Y[%i] = fma(ax%i, Y[%i], ay%i * Y[%i]);\n", lindex(n, m), current_chain, lindex(n - 1, m), current_chain, lindex(n - 2, m));
				if (m != 0) {
					tprint_chain("Y[%i] = fma(ax%i, Y[%i], ay%i * Y[%i]);\n", lindex(n, -m), current_chain, lindex(n - 1, -m), current_chain, lindex(n - 2, -m));
				}
			}
		}
	}
	tprint_flush_chains();
}

void regular_harmonic_xy(int P) {
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			if (m > 1) {
				tprint("ax0 = Y[%i] * TCAST(%.20e);\n", xyindex(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay0 = Y[%i] * TCAST(%.20e);\n", xyindex(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax0 - y * ay0;\n", xyindex(m, m));
				tprint("Y[%i] = fma(y, ax0, x * ay0);\n", xyindex(m, -m));
			} else {
				tprint("Y[%i] = x * TCAST(%.20e);\n", xyindex(m, m), 1.0 / (2.0 * m));
				tprint("Y[%i] = y * TCAST(%.20e);\n", xyindex(m, -m), 1.0 / (2.0 * m));
			}
		}
	}
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		for (int n = m + 2; n <= P; n += 2) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			if (n == 2 && m == 0) {
				tprint_chain("Y[%i] = TCAST(%.20e) * R2;\n", xyindex(n, m), -(double) inv);
			} else {
				tprint_chain("ay%i = TCAST(%.20e) * R2;\n", current_chain, -(double) inv);
				tprint_chain("Y[%i] = ay%i * Y[%i];\n", xyindex(n, m), current_chain, xyindex(n - 2, m));
				if (m != 0) {
					tprint_chain("Y[%i] = ay%i * Y[%i];\n", xyindex(n, -m), current_chain, xyindex(n - 2, -m));
				}
			}
		}
	}
	tprint_flush_chains();
}

std::string M2M_allrot(int P, int rot) {
	auto index = mindex;
	int flops = 0;
	const auto name = std::string("M2Mr") + std::to_string(rot);
	auto fname = func_header(name.c_str(), P, true, true, true, true, true, "", "M", MUL, "dx", VEC3);
	open_timer(name);
	init_real("tmp1");
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	init_real("R2");
	if (nodip) {
		init_real("Md");
	}
	if (rot == 0) {
		init_reals("Y", exp_sz(P - 1));
	} else if (rot == 1) {
		init_reals("Y", std::max(P - 1, xyexp_sz(P - 1)));
	} else if (rot == 2) {
		init_real("tmp0");
		init_real("R");
		init_real("Rzero");
		init_real("Rinv");
		init_real("cosphi");
		init_real("sinphi");
		init_reals("ry", P - 1);
		init_reals("A", 2 * (P - 1) + 1);
		tprint("T* const Y=A;\n");
		tprint("T* const rx=A;\n");
	}
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	reset_running_flops();
	if (P > 2 && !nopot && periodic) {
		tprint("M_st.trace2() = fma(TCAST(-4) * x, M[%i], M_st.trace2());\n", mindex(1, 1));
		tprint("M_st.trace2() = fma(TCAST(-4) * y, M[%i], M_st.trace2());\n", mindex(1, -1));
		tprint("M_st.trace2() = fma(TCAST(-2) * z, M[%i], M_st.trace2());\n", mindex(1, 0));
	}
	tprint("x = -x;\n");
	tprint("y = -y;\n");
	tprint("z = -z;\n");
	std::function<int(int, int)> yindex;
	if (rot == 0) {
		tprint("R2 = fma(x, x, fma(y, y, z * z));\n");
		if (P > 2 && !nopot && periodic) {
			tprint("M_st.trace2() = fma(R2, M[%i], M_st.trace2());\n", mindex(0, 0));
		}
		regular_harmonic_full(P - 1);
		yindex = lindex;
	} else if (rot == 1) {
		tprint("R2 = fma(x, x, y * y);\n");
		if (P > 2 && !nopot && periodic) {
			tprint("M_st.trace2() = fma(R2, M[%i], M_st.trace2());\n", mindex(0, 0));
			tprint("M_st.trace2() = fma(z * z, M[%i], M_st.trace2());\n", mindex(0, 0));
		}
		M2M_z(P, +1);
		regular_harmonic_xy(P - 1);
		yindex = xyindex;
	} else if (rot == 2) {
		tprint("R2 = fma(x, x, y * y);\n");
		if (P > 2 && !nopot && periodic) {
			tprint("M_st.trace2() = fma(R2, M[%i], M_st.trace2());\n", mindex(0, 0));
			tprint("M_st.trace2() = fma(z * z, M[%i], M_st.trace2());\n", mindex(0, 0));
		}
	}
	flops += get_running_flops().load();
	reset_running_flops();
	if (rot == 2) {
		tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
		tprint("tmp1 = R2 + Rzero;\n");
		tprint("Rinv = rsqrt(tmp1);\n");
		tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
		tprint("cosphi = fma(x, Rinv, Rzero);\n");
		tprint("sinphi = -y * Rinv;\n");
		flops += get_running_flops().load();
		reset_running_flops();
		tprint("const auto z_rotate=[&]() {\n");
		indent();
		z_rot(P - 1, "M", FULL);
		deindent();
		tprint("};\n");
		tprint("const auto xz_swap=[&]() {\n");
		indent();
		xz_swap(P - 1, "M", false, FULL);
		deindent();
		tprint("};\n");
		flops += 2 * get_running_flops().load();
		reset_running_flops();
		M2M_z(P, +1);
		tprint("z_rotate();\n");
		tprint("xz_swap();\n");
		if (nodip) {
			tprint("Md *= TCAST(0.5);\n");
		}
		tprint("z = R;\n");
		M2M_z(P, -1);
		tprint("xz_swap();\n");
		tprint("sinphi = -sinphi;\n");
		tprint("z_rotate();\n");
	} else {
		for (int n = P - 1; n >= 0; n--) {
			if (nodip && n == 1) {
				continue;
			}
			for (int m = 0; m <= n; m++) {
				tprint_new_chain();
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
					if (nodip && n - k == 1 && rot == 0) {
						continue;
					}
					const int lmin = std::max(-k, m - n + k);
					const int lmax = std::min(k, m + n - k);
					for (int l = -k; l <= k; l++) {
						if (rot == 1 && abs(l) % 2 != k % 2) {
							continue;
						}
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
						if (n - k == 1 && nodip) {
							if (m - l > 0) {
								continue;
							} else if (m - l < 0) {
								continue;
							} else {
								ASPRINTF(&mxstr, "Md");
							}
						} else {
							if (m - l > 0) {
								ASPRINTF(&mxstr, "M[%i]", mindex(n - k, abs(m - l)));
								ASPRINTF(&mystr, "M[%i]", mindex(n - k, -abs(m - l)));
							} else if (m - l < 0) {
								if (abs(m - l) % 2 == 0) {
									ASPRINTF(&mxstr, "M[%i]", mindex(n - k, abs(m - l)));
									ASPRINTF(&mystr, "M[%i]", mindex(n - k, -abs(m - l)));
									mysgn = -1;
								} else {
									ASPRINTF(&mxstr, "M[%i]", mindex(n - k, abs(m - l)));
									ASPRINTF(&mystr, "M[%i]", mindex(n - k, -abs(m - l)));
									mxsgn = -1;
								}
							} else {
								ASPRINTF(&mxstr, "M[%i]", mindex(n - k, 0));
							}
						}
						if (l > 0) {
							ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", yindex(k, -abs(l)));
						} else if (l < 0) {
							if (abs(l) % 2 == 0) {
								ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
								ASPRINTF(&gystr, "Y[%i]", yindex(k, -abs(l)));
								gysgn = -1;
							} else {
								ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
								ASPRINTF(&gystr, "Y[%i]", yindex(k, -abs(l)));
								gxsgn = -1;
							}
						} else {
							ASPRINTF(&gxstr, "Y[%i]", yindex(k, 0));
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
					tprint_chain("M[%i] = -M[%i];\n", mindex(n, m), mindex(n, m));
					for (int i = 0; i < neg_real.size(); i++) {
						tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", mindex(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), mindex(n, m));
					}
					tprint_chain("M[%i] = -M[%i];\n", mindex(n, m), mindex(n, m));
				} else {
					for (int i = 0; i < neg_real.size(); i++) {
						tprint_chain("M[%i] -= %s * %s;\n", mindex(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					}
				}
				for (int i = 0; i < pos_real.size(); i++) {
					tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", mindex(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), mindex(n, m));
				}
				if (fmaops && neg_imag.size() >= 2) {
					tprint_chain("M[%i] = -M[%i];\n", mindex(n, -m), mindex(n, -m));
					for (int i = 0; i < neg_imag.size(); i++) {
						tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", mindex(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), mindex(n, -m));
					}
					tprint_chain("M[%i] = -M[%i];\n", mindex(n, -m), mindex(n, -m));
				} else {
					for (int i = 0; i < neg_imag.size(); i++) {
						tprint_chain("M[%i] -= %s * %s;\n", mindex(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					}
				}
				for (int i = 0; i < pos_imag.size(); i++) {
					tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", mindex(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), mindex(n, -m));
				}
			}
			tprint_flush_chains();
		}
	}
	close_timer();
	deindent();
	tprint("}\n");
	flops += get_running_flops().load();
	reset_running_flops();
	tprint("return %i;\n", flops);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	timing_body += print2str("\"M2M\", %i, %i, %i, %i, 0.0, 0}", P, nopot, rot, flops);
	TAB0();
	return fname;
}

void L2L_z(int P, int Q, const char* var = "z") {
	const char* two = P == Q ? "" : "2";
	tprint("Y[0] = %s;\n", var);
	for (int i = 1; i < P; i++) {
		tprint("Y[%i] = %s * Y[%i];\n", i, var, i - 1);
	}
	for (int i = 1; i < P; i++) {
		tprint("Y[%i] *= TCAST(%0.20e);\n", i, 1.0 / factorial(i + 1));
	}
	for (int n = nopot; n <= Q; n++) {
		for (int m = -n; m <= n; m++) {
			tprint_new_chain();
			for (int k = 1; k <= P - n; k++) {
				if (abs(m) > n + k) {
					continue;
				}
				tprint("L%s[%i] = fma(Y[%i], L[%i], L%s[%i]);\n", two, lindex(n, m), k - 1, lindex(n + k, m), two, lindex(n, m));
			}
		}
		tprint_flush_chains();
	}
}

void L2L_allrot(int P, int Q, int rot) {
	auto index = lindex;
	int flops = 0;
	const auto name = std::string("L2") + (P == Q ? "L" : "P") + "r" + std::to_string(rot);
	if (P == Q) {
		func_header(name.c_str(), P, true, true, true, true, true, "", "L0", EXP, "dx", VEC3);
	} else {
		func_header(name.c_str(), P, true, true, true, true, true, "", "f", FORCE, "L0", CEXP, "dx", VEC3);
	}
	const char* two = P == Q ? "" : "2";
	open_timer(name);
	init_real("tmp1");
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	init_real("R2");
	if (Q == 1) {
		init_reals("L2", 4);
	}
	if (rot == 0) {
		init_reals("Y", exp_sz(P));
	} else if (rot == 1) {
		init_reals("Y", std::max(P, xyexp_sz(P)));
	} else if (rot == 2) {
		init_real("tmp0");
		init_real("R");
		init_real("Rzero");
		init_real("Rinv");
		init_real("cosphi");
		init_real("sinphi");
		init_reals("ry", P);
		init_reals("A", 2 * P + 1);
		tprint("T* const Y=A;\n");
		tprint("T* const rx=A;\n");
	}
	if (P == Q) {
		tprint("auto& L_st=L0_st;\n");
		tprint("auto* L=L0;\n");
	} else {
		tprint("expansion<%s,%i> L_st;\n", type.c_str(), P);
		tprint("T* L=L_st.data();\n");
		for (int i = 0; i < exp_sz(P); i++) {
			tprint("L[%i] = L0[%i];\n", i, i);
		}
		if (periodic) {
			tprint("L_st.trace2() = L0_st.trace2();\n");
		}
	}
	const auto init_L2 = [Q]() {
		if (Q == 1) {
			if (!nopot) {
				tprint("L2[0] = L[0];\n");
			}
			tprint("L2[3] = L[3];\n");
			tprint("L2[1] = L[1];\n");
			tprint("L2[2] = L[2];\n");
		}};
	if (scaled) {
		tprint("tmp1 = TCAST(1) / L_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	reset_running_flops();
	tprint("x = -x;\n");
	tprint("y = -y;\n");
	tprint("z = -z;\n");
	std::function<int(int, int)> yindex;
	if (rot == 0) {
		init_L2();
		tprint("R2 = fma(x, x, fma(y, y, z * z));\n");
		regular_harmonic_full(P);
		yindex = lindex;
	} else if (rot == 1) {
		tprint("R2 = fma(x, x, y * y);\n");
		L2L_z(P, P);
		init_L2();
		regular_harmonic_xy(P);
		yindex = xyindex;
	} else if (rot == 2) {
		tprint("R2 = fma(x, x, y * y);\n");
	}
	flops += get_running_flops().load();
	reset_running_flops();
	if (rot == 2) {
		tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
		tprint("tmp1 = R2 + Rzero;\n");
		tprint("Rinv = rsqrt(tmp1);\n");
		tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
		tprint("cosphi = fma(x, Rinv, Rzero);\n");
		tprint("sinphi = -y * Rinv;\n");
		flops += get_running_flops().load();
		reset_running_flops();
		if (P == Q) {
			tprint("const auto z_translate=[&]() {\n");
			indent();
			L2L_z(P, P);
			deindent();
			tprint("};\n");
			tprint("const auto z_rotate=[&]() {\n");
			indent();
			z_rot(P, "L", FULL);
			deindent();
			tprint("};\n");
			tprint("const auto xz_swap=[&]() {\n");
			indent();
			xz_swap(P, "L", true, FULL);
			deindent();
			tprint("};\n");
			flops += 2 * get_running_flops().load();
			reset_running_flops();
			tprint("z_translate();\n");
			tprint("z_rotate();\n");
			tprint("xz_swap();\n");
			tprint("z = R;\n");
			tprint("z_translate();\n");
			tprint("xz_swap();\n");
			tprint("sinphi = -sinphi;\n");
			tprint("z_rotate();\n");
		} else {
			L2L_z(P, P);
			z_rot(P, "L", FULL);
			xz_swap(P, "L", true, FULL);
			init_L2();
			L2L_z(P, Q, "R");
			xz_swap(Q, "L2", true, FULL);
			tprint("sinphi = -sinphi;\n");
			z_rot(Q, "L2", FULL);
			flops += get_running_flops().load();
			reset_running_flops();
		}
	} else {
		for (int n = nopot; n <= Q; n++) {
			for (int m = 0; m <= n; m++) {
				tprint_new_chain();
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
					for (int l = -k; l <= k; l++) {
						if (rot == 1 && abs(l) % 2 != k % 2) {
							continue;
						}
						char* mxstr = nullptr;
						char* mystr = nullptr;
						char* gxstr = nullptr;
						char* gystr = nullptr;
						int mxsgn = 1;
						int mysgn = 1;
						int gxsgn = 1;
						int gysgn = 1;
						if (abs(m + l) > n + k) {
							continue;
						}
						if (-abs(m + l) < -(k + n)) {
							continue;
						}
						if (m + l > 0) {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
							ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
						} else if (m + l < 0) {
							if (abs(m + l) % 2 == 0) {
								ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
								ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
								mysgn = -1;
							} else {
								ASPRINTF(&mxstr, "L[%i]", index(n + k, abs(m + l)));
								ASPRINTF(&mystr, "L[%i]", index(n + k, -abs(m + l)));
								mxsgn = -1;
							}
						} else {
							ASPRINTF(&mxstr, "L[%i]", index(n + k, 0));
						}
						if (l > 0) {
							ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", yindex(k, -abs(l)));
							gysgn = -1;
						} else if (l < 0) {
							if (abs(l) % 2 == 0) {
								ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
								ASPRINTF(&gystr, "Y[%i]", yindex(k, -abs(l)));
							} else {
								ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
								ASPRINTF(&gystr, "Y[%i]", yindex(k, -abs(l)));
								gysgn = -1;
								gxsgn = -1;
							}
						} else {
							ASPRINTF(&gxstr, "Y[%i]", yindex(k, 0));
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
					tprint_chain("L%s[%i] = -L%s[%i];\n", two, index(n, m), two, index(n, m));
					for (int i = 0; i < neg_real.size(); i++) {
						tprint_chain("L%s[%i] = fma(%s, %s, L%s[%i]);\n", two, index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), two, index(n, m));
					}
					tprint_chain("L%s[%i] = -L%s[%i];\n", two, index(n, m), two, index(n, m));
				} else {
					for (int i = 0; i < neg_real.size(); i++) {
						tprint_chain("L%s[%i] -= %s * %s;\n", two, index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
					}
				}
				for (int i = 0; i < pos_real.size(); i++) {
					tprint_chain("L%s[%i] = fma(%s, %s, L%s[%i]);\n", two, index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), two, index(n, m));
				}
				if (fmaops && neg_imag.size() >= 2) {
					tprint_chain("L%s[%i] = -L%s[%i];\n", two, index(n, -m), two, index(n, -m));
					for (int i = 0; i < neg_imag.size(); i++) {
						tprint_chain("L%s[%i] = fma(%s, %s, L%s[%i]);\n", two, index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), two,
								index(n, -m));
					}
					tprint_chain("L%s[%i] = -L%s[%i];\n", two, index(n, -m), two, index(n, -m));
				} else {
					for (int i = 0; i < neg_imag.size(); i++) {
						tprint_chain("L%s[%i] -= %s * %s;\n", two, index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
					}
				}
				for (int i = 0; i < pos_imag.size(); i++) {
					tprint_chain("L%s[%i] = fma(%s, %s, L%s[%i]);\n", two, index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), two, index(n, -m));
				}
			}
			tprint_flush_chains();
		}
	}

	if (P > 1 && periodic) {
		tprint("L%s[%i] = fma(TCAST(2) * x, L_st.trace2(), L%s[%i]);\n", index(1, 1), index(1, 1));
		tprint("L%s[%i] = fma(TCAST(2) * y, L_st.trace2(), L%s[%i]);\n", index(1, -1), index(1, -1));
		tprint("L%s[%i] = fma(TCAST(2) * z, L_st.trace2(), L%s[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L%s[%i] = fma(R2, L_st.trace2(), L%s[%i]);\n", index(0, 0), index(0, 0));
			if (rot != 0) {
				tprint("L%s[%i] = fma(z * z, L_st.trace2(), L%s[%i]);\n", index(0, 0), index(0, 0));
			}
		}
	}
	if (Q == 1) {
		if (!nopot) {
			tprint("f.potential += L2[0];\n");
		}
		tprint("f.force[0] -= L2[3];\n");
		tprint("f.force[1] -= L2[1];\n");
		tprint("f.force[2] -= L2[2];\n");
	}
	close_timer();
	deindent();
	tprint("}\n");
	flops += get_running_flops().load();
	reset_running_flops();
	tprint("return %i;\n", flops);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	timing_body += print2str("\"L2%s\", %i, %i, %i, %i, 0.0, 0}", (P == Q ? "L" : "P"), P, nopot, rot, flops);
	TAB0();
}

std::string regular_harmonic(int P) {
	auto fname = func_header("regular_harmonic", P, false, false, false, true, true, "", "Y", EXP, "dx", VEC3);
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	reset_running_flops();
	if (P > 1) {
		init_real("r2");
		tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	}
	tprint("Y[0] = TCAST(1);\n");
	if (periodic && P > 1) {
		tprint("Y_st.trace2() = r2;\n");
	}
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			if (m - 1 > 0) {
				tprint("ax0 = Y[%i] * TCAST(%.20e);\n", lindex(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay0 = Y[%i] * TCAST(%.20e);\n", lindex(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax0 - y * ay0;\n", lindex(m, m));
				tprint("Y[%i] = fma(y, ax0, x * ay0);\n", lindex(m, -m));
			} else {
				tprint("ax0 = Y[%i] * TCAST(%.20e);\n", lindex(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax0;\n", lindex(m, m));
				tprint("Y[%i] = y * ax0;\n", lindex(m, -m));
			}
		}
	}
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		if (m + 1 <= P) {
			if (m == 0) {
				tprint_chain("Y[%i] = z * Y[%i];\n", lindex(m + 1, m), lindex(m, m));
			} else {
				tprint_chain("Y[%i] = z * Y[%i];\n", lindex(m + 1, m), lindex(m, m));
				tprint_chain("Y[%i] = z * Y[%i];\n", lindex(m + 1, -m), lindex(m, -m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			tprint_chain("ax%i = TCAST(%.20e) * z;\n", current_chain, inv * double(2 * n - 1));
			tprint_chain("ay%i = TCAST(%.20e) * r2;\n", current_chain, -(double) inv);
			tprint_chain("Y[%i] = fma(ax%i, Y[%i], ay%i * Y[%i]);\n", lindex(n, m), current_chain, lindex(n - 1, m), current_chain, lindex(n - 2, m));
			if (m != 0) {
				tprint_chain("Y[%i] = fma(ax%i, Y[%i], ay%i * Y[%i]);\n", lindex(n, -m), current_chain, lindex(n - 1, -m), current_chain, lindex(n - 2, -m));
			}
		}
	}
	tprint_flush_chains();
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "regular_harmonic"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

void cuda_header() {
	tprint("const int& tid = threadIdx.x;\n");
	tprint("const int& bsz = blockDim.x;\n");
}

std::string regular_harmonic_xz(int P) {
	auto fname = func_header("regular_harmonic_xz", P, false, false, false, true, false, "", "Y", HEXP, "x", LIT, "z", LIT);
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	init_real("r2");
	reset_running_flops();
	tprint("r2 = fma(x, x, z * z);\n");
	tprint("Y[0] = TCAST(1);\n");
	if (periodic && P > 1) {
		tprint("Y_st.trace2() = r2;\n");
	}
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			if (m - 1 > 0) {
				tprint("ax0 = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax0;\n", index(m, m));
			} else {
				tprint("ax0 = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax0;\n", index(m, m));
			}
		}
	}
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		if (m + 1 <= P) {
			if (m == 0) {
				tprint_chain("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
			} else {
				tprint_chain("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			tprint_chain("ax%i = TCAST(%.20e) * z;\n", current_chain, inv * double(2 * n - 1));
			tprint_chain("ay%i = TCAST(%.20e) * r2;\n", current_chain, -(double) inv);
			tprint_chain("Y[%i] = fma(ax%i, Y[%i], ay%i * Y[%i]);\n", index(n, m), current_chain, index(n - 1, m), current_chain, index(n - 2, m));
		}
	}
	tprint_flush_chains();
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	flops_map[P][type + "regular_harmonic_xz"] = get_running_flops();
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	TAB0();
	return fname;
}

std::string M2M(int P) {
	if (nopot) {
		return "";
	}
	std::string str;
	auto fname = func_header("M2M", P, true, false, false, false, true, "", "M", MUL, "dx", VEC3);
	str = flags_header("M2M", P);
	str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
	str += print2str("\t\treturn M2Mr1(M_st, dx, flags);\n");
	str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
	str += print2str("\t\treturn M2Mr2(M_st, dx, flags);\n");
	str += "\t} else {\n";
	str += print2str("\t\treturn M2Mr0(M_st, dx, flags);\n");
	str += "\t}\n";
	tprint(str.c_str());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	TAB0();
	return fname;
}

std::string P2M(int P) {
	tprint("\n");
	auto fname = func_header("P2M", P, true, true, true, true, true, "", "M", MUL, "m", LIT, "dx", VEC3);
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	init_real("r2");
	init_real("tmp1");
	if (nodip) {
		init_real("Mdx");
		init_real("Mdy");
		init_real("Mdz");
//		tprint("M[1] = TCAST(0);\n");
//		tprint("M[2] = TCAST(0);\n");
//		tprint("M[3] = TCAST(0);\n");
	}
	reset_running_flops();
	tprint("x = -x;\n");
	tprint("y = -y;\n");
	tprint("z = -z;\n");
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("M[0] = m;\n");
	if (!nopot && periodic & P > 2) {
		tprint("M_st.trace2() = m * r2;\n");
	}
	for (int m = 0; m < P; m++) {
		if (m > 0) {
			if (m - 1 > 0) {
				if (m - 1 == 1 && nodip) {
					tprint("ax0 = Mdx * TCAST(%.20e);\n", 1.0 / (2.0 * m));
					tprint("ay0 = Mdy * TCAST(%.20e);\n", 1.0 / (2.0 * m));
				} else {
					tprint("ax0 = M[%i] * TCAST(%.20e);\n", mindex(m - 1, m - 1), 1.0 / (2.0 * m));
					tprint("ay0 = M[%i] * TCAST(%.20e);\n", mindex(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				}
				if (m == 1 && nodip) {
					tprint("Mdx = x * ax0 - y * ay0;\n");
					tprint("Mdy = fma(y, ax0, x * ay0);\n");
				} else {
					tprint("M[%i] = x * ax0 - y * ay0;\n", mindex(m, m));
					tprint("M[%i] = fma(y, ax0, x * ay0);\n", mindex(m, -m));
				}

			} else {
				tprint("ax0 = M[%i] * TCAST(%.20e);\n", mindex(m - 1, m - 1), 1.0 / (2.0 * m));
				if (nodip) {
					tprint("Mdx = x * ax0;\n");
					tprint("Mdy = y * ax0;\n");
				} else {
					tprint("M[%i] = x * ax0;\n", mindex(m, m));
					tprint("M[%i] = y * ax0;\n", mindex(m, -m));
				}
			}
		}
	}
	for (int m = 0; m < P; m++) {
		tprint_new_chain();
		if (m + 1 < P) {
			if (m == 0) {
				if (nodip) {
					tprint_chain("Mdz = z * M[0];\n");
				} else {
					tprint_chain("M[%i] = z * M[%i];\n", mindex(1, 0), mindex(0, 0));
				}
			} else {
				if (nodip && m == 1) {
					tprint_chain("M[%i] = z * Mdx;\n", mindex(m + 1, m));
					tprint_chain("M[%i] = z * Mdy;\n", mindex(m + 1, -m));
				} else {
					tprint_chain("M[%i] = z * M[%i];\n", mindex(m + 1, m), mindex(m, m));
					tprint_chain("M[%i] = z * M[%i];\n", mindex(m + 1, -m), mindex(m, -m));
				}
			}
		}
		for (int n = m + 2; n < P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			tprint_chain("ax%i = TCAST(%.20e) * z;\n", current_chain, inv * double(2 * n - 1));
			tprint_chain("ay%i = TCAST(%.20e) * r2;\n", current_chain, -(double) inv);
			if (nodip && n - 2 == 1) {
				if (m == 0) {
					tprint_chain("M[%i] = fma(ax%i, M[%i], ay%i * Mdz);\n", mindex(n, m), current_chain, mindex(n - 1, m), current_chain);
				} else {
					tprint_chain("M[%i] = fma(ax%i, M[%i], ay%i * Mdx);\n", mindex(n, m), current_chain, mindex(n - 1, m), current_chain);
					tprint_chain("M[%i] = fma(ax%i, M[%i], ay%i * Mdy);\n", mindex(n, -m), current_chain, mindex(n - 1, -m), current_chain);
				}
			} else if (nodip && n - 1 == 1) {
				if (m == 0) {
					tprint_chain("M[%i] = fma(ax%i, Mdz, ay%i * M[%i]);\n", mindex(n, m), current_chain, mindex(n - 2, m), current_chain);
				} else {
					tprint_chain("M[%i] = fma(ax%i, Mdx, ay%i * M[%i]);\n", mindex(n, m), current_chain, mindex(n - 2, m));
					tprint_chain("M[%i] = fma(ax%i, Mdy, ay%i * M[%i]);\n", mindex(n, -m), current_chain, mindex(n - 2, -m));
				}
			} else {
				tprint_chain("M[%i] = fma(ax%i, M[%i], ay%i * M[%i]);\n", mindex(n, m), current_chain, mindex(n - 1, m), current_chain, mindex(n - 2, m));
				if (m != 0) {
					tprint_chain("M[%i] = fma(ax%i, M[%i], ay%i * M[%i]);\n", mindex(n, -m), current_chain, mindex(n - 1, -m), current_chain, mindex(n - 2, -m));
				}
			}
		}
	}
	tprint_flush_chains();
	deindent();
	tprint("}\n");
	tprint("return %i;\n", get_running_flops().load());
	timing_body += print2str("\"P2M\", %i, %i, 0, %i, 0.0, 0}", P, nopot, get_running_flops(false).load());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	TAB0();
	return fname;
}

std::string L2L(int P, int Q) {
	if (nopot) {
		return "";
	}
	std::string str;
	std::string fname;
	if (Q == P) {
		fname = func_header("L2L", P, true, false, false, false, true, "", "L", EXP, "dx", VEC3);
		str = flags_header("L2L", P);
		str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Lr1(L_st, dx, flags);\n");
		str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Lr2(L_st, dx, flags);\n");
		str += "\t} else {\n";
		str += print2str("\t\treturn L2Lr0(L_st, dx, flags);\n");
		str += "\t}\n";
	} else {
		fname = func_header("L2P", P, true, false, false, false, true, "", "f", FORCE, "L", CEXP, "dx", VEC3);
		str = flags_header("L2P", P);
		str += "\tif( flags & sfmmWithSingleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Pr1(f, L_st, dx, flags);\n");
		str += "\t} else if( flags & sfmmWithDoubleRotationOptimization ) {\n";
		str += print2str("\t\treturn L2Pr2(f, L_st, dx, flags);\n");
		str += "\t} else {\n";
		str += print2str("\t\treturn L2Pr0(f, L_st, dx, flags);\n");
		str += "\t}\n";
	}
	tprint(str.c_str());
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	if (nopot) {
		tprint("}\n");
	}
	TAB0();
	return fname;
}

void safe_math_float() {

	tprint("T safe_mul(T& a, T b, T c) {\n");
	indent();
	tprint("const V be = ((V&)b & VCAST(0x7F800000)) >> VCAST(23);\n");
	tprint("const V ce = ((V&)c & VCAST(0x7F800000)) >> VCAST(23);\n");
	tprint("const T flag = T(be + ce < V(381));\n");
	tprint("a = (flag * b) * c;\n");
	tprint("return flag;\n");
	deindent();
	tprint("}\n\n");
	tprint("T safe_add(T& a, T b, T c) {\n");
	indent();
	tprint("const V be = ((V&)b & VCAST(0x7F800000)) >> VCAST(23);\n");
	tprint("const V ce = ((V&)c & VCAST(0x7F800000)) >> VCAST(23);\n");
	tprint("const T flag = T(be < V(254)) * T(ce < V(254));\n");
	tprint("a = fma(flag, b, flag * c);\n");
	tprint("return flag;\n");
	deindent();
	tprint("}\n\n");
	tprint("\n");
}

void safe_math_double() {

	tprint("T safe_mul(T& a, T b, T c) {\n");
	indent();
	tprint("const V be = ((V&)b & VCAST(0x7FF0000000000000LL)) >> VCAST(52);\n");
	tprint("const V ce = ((V&)c & VCAST(0x7FF0000000000000LL)) >> VCAST(52);\n");
	tprint("const T flag = T(be + ce < V(3069));\n");
	tprint("a = (flag * b) * c;\n");
	tprint("return flag;\n");
	deindent();
	tprint("}\n\n");
	tprint("T safe_add(T& a, T b, T c) {\n");
	indent();
	tprint("const V be = ((V&)b & VCAST(0x7FF0000000000000LL)) >> VCAST(52);\n");
	tprint("const V ce = ((V&)c & VCAST(0x7FF0000000000000LL)) >> VCAST(52);\n");
	tprint("const T flag = T(be < V(2046)) * T(ce < V(2046));\n");
	tprint("a = fma(flag, b, c);\n");
	tprint("return flag;\n");
	deindent();
	tprint("}\n\n");
	tprint("\n");
}

int flops_t::load() const {
	return r + 2 * fma + 4 * rdiv;				// + con + rcmp;
//	return r + i + fma + 4 * (rdiv + idiv) + con + asgn + icmp + rcmp;
}

void math_float(std::string _type) {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
	const char* type = _type.c_str();
	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	if (is_vec(_type)) {
		fprintf(fp, "#ifndef __CUDACC__\n");
	}
	tprint("inline %s fma(%s a, %s b, %s c) {\n", type, type, type, type);
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("%s rsqrt(%s);\n", type, type);
	tprint("%s sqrt(%s);\n", type, type);
	tprint("void sincos(%s, %s*, %s*);\n", type, type, type, type);
	tprint("void erfcexp(%s, %s*, %s*);\n", type, type, type, type);
	tprint("%s safe_mul(%s&, %s, %s);\n", type, type, type, type);
	tprint("%s safe_add(%s&, %s, %s);\n", type, type, type, type);
	if (is_vec(_type)) {
		tprint("#endif /* __CUDACC__ */ \n");
	}
	tprint("\n");
	fclose(fp);
	fp = fopen(print2str("./generated_code/src/math/math_%s.cpp", type).c_str(), "wt");
	tprint("\n");
	tprint("#include \"%s\"\n", header.c_str());
	tprint("#include \"typecast_%s.hpp\"\n", type);
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("\n");
	tprint("T rsqrt(T x) {\n");
	indent();
	tprint("V i;\n");
	tprint("T y;\n");
	tprint("x += TCAST(%.20e);\n", std::numeric_limits<float>::min());
	tprint("i = *((V*) &x);\n");
	tprint("x *= TCAST(0.5);\n");
	tprint("i >>= VCAST(1);\n");
	tprint("i = VCAST(0x5F3759DF) - i;\n");
	tprint("y = *((T*) &i);\n");
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= TCAST(1.5) - x * y * y;\n");
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T sqrt(T x) {\n");
	indent();
	tprint("return TCAST(1) / rsqrt(x);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("void sincos(T x, T* s, T* c) {\n");
	indent();
	tprint("V ssgn, j, i, k;\n");
	tprint("T x2;\n");
	tprint("ssgn = V(((*((V*) &x) & VCAST(0x80000000)) >> VCAST(30)) - VCAST(1));\n");
	tprint("j = V((*((V*) &x) & VCAST(0x7FFFFFFF)));\n");
	tprint("x = *((T*) &j);\n");
	tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
	tprint("x -= T(i) * TCAST(%.20e);\n", M_PI);
	tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
	tprint("x2 = x * x;\n");
	{
		constexpr int N = 11;
		tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
		tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
		for (int n = N - 2; n >= 0; n -= 2) {
			tprint("%s = fma(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = fma(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
		}
	}
	tprint("%s *= x;\n", cout);
	tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
	tprint("%s *= T(ssgn * k);\n", sout);
	tprint("%s *= T(k);\n", cout);
	deindent();
	tprint("}\n");
	tprint("\n");

	tprint("T exp(T x) {\n");
	{
		indent();
		constexpr int N = 7;
		tprint("V k;\n");
		tprint("T y, xxx;\n");
		tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
		tprint("k -= x < TCAST(0);\n");
		tprint("xxx = x - T(k) * TCAST(0.6931471805599453094172);\n");
		tprint("y = TCAST(%.20e);\n", 1.0 / factorial(N));
		for (int i = N - 1; i >= 0; i--) {
			tprint("y = fma(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-fmaops);
		}
		tprint("k = (k + VCAST(127)) << VCAST(23);\n");
		tprint("y *= *((T*) (&k));\n"); //1
		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	{

		tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
		indent();
		tprint("T x2, nx2, q0, q1, sw0, sw1, tmp1, tmp0;\n");
		tprint("x2 = x * x;\n");
		tprint("nx2 = -x2;\n");
		tprint("*exp0 = exp(nx2);\n");

		constexpr double x0 = 2.75;
		tprint("sw0 = (x < TCAST(%.20e));\n", x0);
		tprint("sw1 = TCAST(1) - sw0;\n", x0);
		constexpr int N0 = 25;
		constexpr int N1 = x0 * x0 + 0.5;
		tprint("q0 = TCAST(2) * x * x;\n");
		tprint("q1 = TCAST(1) / (TCAST(2) * x * x);\n");
		tprint("tmp0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N0 + 1));
		tprint("tmp1 = TCAST(%.20e);\n", dfactorial(2 * N1 - 1) * nonepow(N1));
		int n1 = N1 - 1;
		int n0 = N0 - 1;
		while (n1 >= 1 || n0 >= 0) {
			if (n0 >= 0) {
				tprint("tmp0 = fma(tmp0, q0, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n0 + 1));
			}
			if (n1 >= 1) {
				tprint("tmp1 = fma(tmp1, q1, TCAST(%.20e));\n", dfactorial(2 * n1 - 1) * nonepow(n1));
			}
			n1--;
			n0--;
		}
		tprint("tmp0 *= TCAST(%.20e) * x * *exp0;\n", 2.0 / sqrt(M_PI));
		tprint("tmp1 = fma(tmp1, q1, TCAST(1));\n");
		tprint("tmp0 = TCAST(1) - tmp0;\n");
		tprint("tmp1 *= *exp0 * TCAST(%.20e) / x;\n", 1.0 / sqrt(M_PI));
		tprint("*erfc0 = fma(sw0, tmp0, sw1 * tmp1);\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	safe_math_float();
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);

	fp = nullptr;
}

void math_double(std::string _str) {
	const char* type = _str.c_str();
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
	constexpr double x0 = 1.0;
	constexpr double x1 = 4.7;

	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	if (is_vec(_str)) {
		tprint("#ifndef __CUDACC__\n");
	}
	tprint("\n");
	tprint("inline %s fma(%s a, %s b, %s c) {\n", type, type, type, type);
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("%s rsqrt(%s);\n", type, type);
	tprint("%s sqrt(%s);\n", type, type);
	tprint("void sincos(%s, %s*, %s*);\n", type, type, type);
	tprint("void erfcexp(%s, %s*, %s*);\n", type, type, type);
	tprint("%s safe_mul(%s&, %s, %s);\n", type, type, type, type);
	tprint("%s safe_add(%s&, %s, %s);\n", type, type, type, type);
	if (is_vec(_str)) {
		tprint("#endif /* __CUDACC__ */\n");
	}
	tprint("\n");
	fclose(fp);
	fp = fopen(print2str("./generated_code/src/math/math_%s.cpp", rtype[typenum].c_str()).c_str(), "wt");
	tprint("\n");
	tprint("#include \"%s\"\n", header.c_str());
	tprint("#include \"typecast_%s.hpp\"\n", type);
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("\n");
	tprint("T rsqrt(T x) {\n");
	indent();
	tprint("V i;\n");
	tprint("T y;\n");
	tprint("x += TCAST(%.20e);\n", std::numeric_limits<double>::min());
	tprint("i = *((V*) &x);\n");
	tprint("x *= TCAST(0.5);\n");
	tprint("i >>= VCAST(1);\n");
	tprint("i = VCAST(0x5FE6EB50C7B537A9) - i;\n");
	tprint("y = *((T*) &i);\n");
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
	tprint("return y;\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T sqrt(T x) {\n");
	indent();
	tprint("return TCAST(1) / rsqrt(x);\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("void sincos(T x, T* s, T* c) {\n");
	indent();
	tprint("V ssgn, j, i, k;\n");
	tprint("T x2;\n");
	tprint("ssgn = V(((*((V*) &x) & VCAST(0x8000000000000000LL)) >> VCAST(62LL)) - VCAST(1LL));\n");
	tprint("j = V((*((V*) &x) & VCAST(0x7FFFFFFFFFFFFFFFLL)));\n");
	tprint("x = *((T*) &j);\n");
	tprint("i = x * TCAST(%.20e);\n", 1.0 / M_PI);
	tprint("x -= T(i) * TCAST(%.20e);\n", M_PI);
	tprint("x -= TCAST(%.20e);\n", 0.5 * M_PI);
	tprint("x2 = x * x;\n");
	{
		constexpr int N = 21;
		tprint("%s = TCAST(%.20e);\n", cout, nonepow(N / 2) / factorial(N));
		tprint("%s = TCAST(%.20e);\n", sout, cout, nonepow((N - 1) / 2) / factorial(N - 1));
		for (int n = N - 2; n >= 0; n -= 2) {
			tprint("%s = fma(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = fma(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
		}
	}
	tprint("%s *= x;\n", cout);
	tprint("k = (((i & VCAST(1)) << VCAST(1)) - VCAST(1));\n");
	tprint("%s *= T(ssgn * k);\n", sout);
	tprint("%s *= T(k);\n", cout);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("T exp(T x) {\n");
	{
		indent();
		constexpr int N = 18;
		tprint("V k;\n");
		tprint("T xxx, y;\n");
		tprint("k =  x / TCAST(0.6931471805599453094172) + TCAST(0.5);\n"); // 1 + divops
		tprint("k -= x < TCAST(0);\n");
		tprint("xxx = x - T(k) * TCAST(0.6931471805599453094172);\n");
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
	tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
	indent();
	tprint("T x2, nx2, q0, q2, x0, x1, sw0, sw1, sw2, res0, res1, res2;\n");
	tprint("x2 = x * x;\n");
	tprint("nx2 = -x2;\n");
	tprint("*exp0 = exp(nx2);\n");

	tprint("sw0 = T(x < TCAST(%.20e));\n", x0);
	tprint("sw1 = T(x >= TCAST(%.20e)) * T(x <= TCAST(%.20e));\n", x0, x1);
	tprint("sw2 = T(x > TCAST(%.20e));\n", x1);
	{
		constexpr int N0 = 17;
		tprint("x0 = x;\n");
		tprint("q0 = TCAST(2) * x0 * x0;\n");
		tprint("res0 = TCAST(%.20e);\n", 1.0 / dfactorial(2 * N0 + 1));
		constexpr int N1 = 35;
		constexpr double a = (x1 - x0) * 0.5 + x0;
		static double c0[N1 + 1];
		static bool init = false;
		double q[N1 + 1];
		init = true;
		c0[0] = (a) * exp(a * a) * erfc(a);
		q[0] = exp(a * a) * erfc(a);
		q[1] = -2.0 / sqrt(M_PI) + 2 * a * exp(a * a) * erfc(a);
		for (int n = 2; n <= N1; n++) {
			q[n] = 2 * (a * q[n - 1] + q[n - 2]) / n;
		}
		for (int n = 1; n <= N1; n++) {
			c0[n] = q[n - 1] + (a) * q[n];
		}
		tprint("x1 = x0 - TCAST(%.20e);\n", a);
		tprint("res1 = TCAST(%.20e);\n", c0[N1]);
		constexpr int N2 = x1 * x1 + 0.5;
		tprint("x2 = x0;\n");
		tprint("q2 = TCAST(1) / (TCAST(2) * x2 * x2);\n");
		tprint("res2 = TCAST(%.20e);\n", dfactorial(2 * N2 - 1) * nonepow(N2));
		int n0 = N0 - 1;
		int n1 = N1 - 1;
		int n2 = N2 - 1;
		while (n0 >= 0 || n1 >= 0 || n2 >= 1) {
			if (n0 >= 0) {
				tprint("res0 = fma(res0, q0, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n0 + 1));
			}
			if (n1 >= 0) {
				tprint("res1 = fma(res1, x1, TCAST(%.20e));\n", c0[n1]);
			}
			if (n2 >= 1) {
				tprint("res2 = fma(res2, q2, TCAST(%.20e));\n", dfactorial(2 * n2 - 1) * nonepow(n2));
			}
			n0--;
			n1--;
			n2--;
		}
		tprint("res0 *= TCAST(%.20e) * x0 * *exp0;\n", 2.0 / sqrt(M_PI));
		tprint("res0 = TCAST(1) - res0;\n");
		tprint("res1 *= *exp0 / x0;\n");
		tprint("res2 = fma(res2, q2, TCAST(1));\n");
		tprint("res2 *= *exp0 * TCAST(%.20e) / x2;\n", 1.0 / sqrt(M_PI));
	}
	tprint("*erfc0 = fma(sw0, res0, fma(sw1, res1, sw2 * res2));\n");

	deindent();
	tprint("}\n");
	tprint("\n");
	safe_math_double();
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);
	fp = nullptr;
}

void typecast_functions() {
	printf("Doing type headers\n");
	if (fp) {
		fclose(fp);
	}
	for (int i = 0; i < base_rtype.size(); i++) {
		fp = fopen(print2str("./generated_code/include/typecast_%s.hpp", rtype[i].c_str()).c_str(), "at");
		tprint("#pragma once\n\n");
		tprint("#include \"sfmm.hpp\"\n\n");
		tprint("\n");
		tprint("namespace sfmm {\n\n");
		tprint("\n");
		tprint("typedef %s T;\n", rtype[i].c_str());
		tprint("typedef %s U;\n\n", uitype[i].c_str());
		tprint("typedef %s V;\n", itype[i].c_str());
		tprint("\n");
		tprint("typedef T TCONVERT;\n");
		tprint("typedef U UCONVERT;\n");
		tprint("typedef V VCONVERT;\n");
		tprint("\n");
		tprint("#define TCAST(a) (%s(%s(a)))\n", rtype[i].c_str(), base_rtype[i].c_str());
		tprint("#define VCAST(a) (%s(%s(a)))\n", itype[i].c_str(), base_itype[i].c_str());
		tprint("#define UCAST(a) (%s(%s(a)))\n", uitype[i].c_str(), base_uitype[i].c_str());
		tprint("\n");
		tprint("}\n");
		tprint("\n");
		fclose(fp);
	}
	fp = nullptr;
}

int main() {
	printf("generation attributes: ");
#ifdef USE_FLOAT
	printf( "float, ");
#endif
#ifdef USE_DOUBLE
	printf( "double, ");
#endif
#ifdef USE_SIMD
	printf( "simd, ");
#endif
	printf("simd double width = %i, ", DOUBLE_SIMD_WIDTH);
	printf("simd float width = %i, ", FLOAT_SIMD_WIDTH);
	printf("simd m2m width = %i\n", M2M_SIMD_WIDTH);

#ifdef USE_FLOAT
	base_rtype.push_back("float");
	rtype.push_back("float");
	base_itype.push_back("int32_t");
	itype.push_back("int32_t");
	base_uitype.push_back("uint32_t");
	uitype.push_back("uint32_t");
	precision.push_back(1);
	m2monly.push_back(0);
	simd.push_back(0);
	simd_size.push_back(1);
#ifdef USE_SIMD
	base_rtype.push_back("float");
	rtype.push_back("simd_f32");
	base_itype.push_back("int32_t");
	itype.push_back("simd_i32");
	base_uitype.push_back("uint32_t");
	uitype.push_back("simd_ui32");
	precision.push_back(1);
	m2monly.push_back(0);
	simd.push_back(1);
	simd_size.push_back(FLOAT_SIMD_WIDTH);
	if( M2M_SIMD_WIDTH != FLOAT_SIMD_WIDTH) {
		base_rtype.push_back("float");
		rtype.push_back("m2m_simd_f32");
		base_itype.push_back("int32_t");
		itype.push_back("m2m_simd_i32");
		base_uitype.push_back("uint32_t");
		uitype.push_back("m2m_simd_ui32");
		precision.push_back(1);
		m2monly.push_back(1);
		simd.push_back(1);
		simd_size.push_back(M2M_SIMD_WIDTH);
	}
#endif
#endif
#ifdef USE_DOUBLE
	base_rtype.push_back("double");
	rtype.push_back("double");
	base_itype.push_back("int64_t");
	itype.push_back("int64_t");
	base_uitype.push_back("uint64_t");
	uitype.push_back("uint64_t");
	precision.push_back(2);
	m2monly.push_back(0);
	simd.push_back(0);
	simd_size.push_back(1);
#ifdef USE_SIMD
	base_rtype.push_back("double");
	rtype.push_back("simd_f64");
	base_itype.push_back("int64_t");
	itype.push_back("simd_i64");
	base_uitype.push_back("uint64_t");
	uitype.push_back("simd_ui64");
	precision.push_back(2);
	m2monly.push_back(0);
	simd.push_back(1);
	simd_size.push_back(DOUBLE_SIMD_WIDTH);
	if( M2M_SIMD_WIDTH != DOUBLE_SIMD_WIDTH) {
		base_rtype.push_back("double");
		rtype.push_back("m2m_simd_f64");
		base_itype.push_back("int64_t");
		itype.push_back("m2m_simd_i64");
		base_uitype.push_back("uint64_t");
		uitype.push_back("m2m_simd_ui64");
		precision.push_back(2);
		m2monly.push_back(1);
		simd.push_back(1);
		simd_size.push_back(M2M_SIMD_WIDTH);
	}
#endif
#endif
	if (rtype.size() == 0) {
		printf("WARNING: Code generator not given any types - enable at least one of SFMM_USE_FLOAT or SFMM_USE_DOUBLE.\n");
	}
	SYSTEM("mkdir -p generated_code\n");
	SYSTEM("mkdir -p ./generated_code/include\n");
	SYSTEM("mkdir -p ./generated_code/include/detail\n");
	SYSTEM("mkdir -p ./generated_code/src\n");
	SYSTEM("mkdir -p ./generated_code/src/math\n");
	tprint("\n");
	set_file(full_header.c_str());
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#ifdef __CUDA_ARCH__\n");
	tprint("#define SFMM_PREFIX __device__\n");
	tprint("#else\n");
	tprint("#define SFMM_PREFIX\n");
	tprint("#endif /* __CUDA_ARCH__ */\n");
	tprint("\n");
	tprint("#include <atomic>\n");
	tprint("#include <array>\n");
	tprint("#include <cmath>\n");
	tprint("#include <cstdint>\n");
	tprint("#include <limits>\n");
	tprint("#include <time.h>\n");
	tprint("#include <utility>\n");
	tprint("\n");
	tprint("#define sfmmCalculateWithPotential    0\n");
	tprint("#define sfmmProfilingOff 0\n");
	tprint("#define sfmmWithoutOptimization 0\n");
	tprint("#define sfmmCalculateWithoutPotential 1\n");
	tprint("#define sfmmWithSingleRotationOptimization 2\n");
	tprint("#define sfmmWithDoubleRotationOptimization 4\n");
	tprint("#define sfmmWithRandomOptimization 8\n");
	tprint("#define sfmmWithBestOptimization 16\n");
	tprint("#define sfmmProfilingOn 32\n");
	tprint("#define sfmmFLOPsOnly 64\n");
	tprint("#define sfmmDefaultFlags (sfmmCalculateWithPotential | sfmmWithBestOptimization)\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("\n");
	include("complex.hpp");
	nopot = 1;
	int ntypenames = rtype.size();

#ifdef USE_SIMD
	tprint("#ifndef __CUDACC__\n");
	include("vec_simd.hpp");
	tprint("#endif /* __CUDACC__ */ \n");
	tprint("\n");
	tprint("%s", vec_header().c_str());
#endif
	std::string str;
	str += "template<class T>\n";
	str += "struct type_traits {\n";
	str += "\tstatic constexpr int precision = 0;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = void;\n";
	str += "};\n\n";
#ifdef USE_FLOAT
	str += "template<>\n";
	str += "struct type_traits<float> {\n";
	str += "\tstatic constexpr int precision = 2;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = float;\n";
	str += "};\n\n";
#ifdef USE_SIMD
	str += "template<>\n";
	str += "struct type_traits<simd_f32> {\n";
	str += "\tstatic constexpr int precision = 1;\n";
	str += "\tstatic constexpr bool is_simd = true;\n";
	str += "\tusing type = float;\n";
	str += "};\n\n";
	if(M2M_SIMD_WIDTH != FLOAT_SIMD_WIDTH) {
		str += "template<>\n";
		str += "struct type_traits<m2m_simd_f32> {\n";
		str += "\tstatic constexpr int precision = 1;\n";
		str += "\tstatic constexpr bool is_simd = true;\n";
		str += "\tusing type = float;\n";
		str += "};\n\n";
	}
#endif
#endif
#ifdef USE_DOUBLE
	str += "template<>\n";
	str += "struct type_traits<double> {\n";
	str += "\tstatic constexpr int precision = 2;\n";
	str += "\tstatic constexpr bool is_simd = false;\n";
	str += "\tusing type = double;\n";
	str += "};\n\n";
#ifdef USE_SIMD
	str += "template<>\n";
	str += "struct type_traits<simd_f64> {\n";
	str += "\tstatic constexpr int precision = 2;\n";
	str += "\tstatic constexpr bool is_simd = true;\n";
	str += "\tusing type = double;\n";
	str += "};\n\n";
	if(M2M_SIMD_WIDTH != FLOAT_SIMD_WIDTH) {
		str += "template<>\n";
		str += "struct type_traits<m2m_simd_f64> {\n";
		str += "\tstatic constexpr int precision = 2;\n";
		str += "\tstatic constexpr bool is_simd = true;\n";
		str += "\tusing type = double;\n";
		str += "};\n\n";
	}
#endif
#endif
	tprint(str.c_str());
	include("vec3.hpp");

	std::string str1 = "\n#define SFMM_EXPANSION_MEMBERS(classname, type, ppp) \\\n"
			"\tclass reference { \\\n"
			"\t\tT* ax; \\\n"
			"\t\tT* ay; \\\n"
			"\t\tT rsgn; \\\n"
			"\t\tT isgn; \\\n"
			"\tpublic: \\\n"
			"\t\toperator complex<T>() const { \\\n"
			"\t\t\tif(ax != ay) { \\\n"
			"\t\t\t\treturn complex<T>(rsgn * *ax, isgn * *ay); \\\n"
			"\t\t\t} else { \\\n"
			"\t\t\t\treturn complex<T>(rsgn * *ax, T(0)); \\\n"
			"\t\t\t} \\\n"
			"\t\t} \\\n"
			"\t\treference& operator=(complex<T> other) { \\\n"
			"\t\t\t*ax = other.real() * rsgn; \\\n"
			"\t\t\tif(ax != ay) { \\\n"
			"\t\t\t\t*ay = other.imag() * isgn; \\\n"
			"\t\t\t} else { \\\n"
			"\t\t\t\t*ay = T(0); \\\n"
			"\t\t\t} \\\n"
			"\t\t\treturn *this; \\\n"
			"\t\t} \\\n"
			"\t\tfriend classname<type,ppp>; \\\n"
			"\t}; \\\n"
			"\tSFMM_PREFIX complex<T> operator()(int n, int m) const { \\\n"
			"\t\tcomplex<T> c; \\\n"
			"\t\tconst int n2n = n * n + n; \\\n"
			"\t\tconst int m0 = std::abs(m); \\\n"
#ifdef NO_DIPOLE
			"\tconst int ip = n == 0 ? 0 : n2n + m0 - 3; \\\n"
			"\tconst int im = n == 0 ? 0 : n2n - m0 - 3; \\\n"
#else
			"\t\tconst int ip = n2n + m0; \\\n"
			"\t\tconst int im = n2n - m0; \\\n"
#endif
			"\t\tc.real() = o[ip]; \\\n"
			"\t\tc.imag() = o[im]; \\\n"
			"\t\tif( m < 0 ) { \\\n"
			"\t\t\tif( m % 2 == 0 ) { \\\n"
			"\t\t\t\tc.imag() = -c.imag(); \\\n"
			"\t\t\t} else { \\\n"
			"\t\t\t\tc.real() = -c.real(); \\\n"
			"\t\t\t} \\\n"
			"\t\t} else if( m == 0 ) { \\\n"
			"\t\t\tc.imag() = T(0); \\\n"
			"\t\t} \\\n"
			"\t\treturn c; \\\n"
			"\t} \\\n"
			"\tSFMM_PREFIX reference operator()(int n, int m) { \\\n"
			"\t\treference ref; \\\n"
			"\t\tconst int n2n = n * n + n; \\\n"
			"\t\tconst int m0 = std::abs(m); \\\n"
#ifdef NO_DIPOLE
			"\tconst int ip = n == 0 ? 0 : n2n + m0 - 3; \\\n"
			"\tconst int im = n == 0 ? 0 : n2n - m0 - 3; \\\n"
#else
			"\t\tconst int ip = n2n + m0; \\\n"
			"\t\tconst int im = n2n - m0; \\\n"
#endif
			"\t\tref.ax = o + ip; \\\n"
			"\t\tref.ay = o + im; \\\n"
			"\t\tref.rsgn = ref.isgn = T(1); \\\n"
			"\t\tif( m < 0 ) { \\\n"
			"\t\t\tif( m % 2 == 0 ) { \\\n"
			"\t\t\t\tref.isgn = -ref.isgn; \\\n"
			"\t\t\t} else { \\\n"
			"\t\t\t\tref.rsgn = -ref.rsgn; \\\n"
			"\t\t\t} \\\n"
			"\t\t} \\\n"
			"\t\treturn ref; \\\n"
			"\t} \\\n"
			"\tSFMM_PREFIX classname(const classname& other) { \\\n"
			"\t\t*this = other; \\\n"
			"\t} \\\n"
			"\tSFMM_PREFIX T* data() { \\\n"
			"\t\treturn o; \\\n"
			"\t} \\\n"
			"\tSFMM_PREFIX const T* data() const { \\\n"
			"\t\treturn o; \\\n"
			"\t} \\\n"
			"\n"
			"";

	fprintf(fp, "%s", str1.c_str());

	set_file(full_header.c_str());
	tprint("\n");
	tprint("namespace detail {\n");
	tprint("template<class T, int P>\n");
	tprint("class expansion_xz {\n");
	tprint("};\n");
	tprint("\ntemplate<class T, int P>\n");
	tprint("class expansion_xy {\n");
	tprint("};\n");
	tprint("}\n");
	tprint("template<class T, int P>\n");
	tprint("class expansion {\n");
	tprint("};\n");
	tprint("\n");
	tprint("template<class T, int P>\n");
	tprint("class multipole {\n");
	tprint("};\n");
	tprint("\n");

	nopot = 0;
	typecast_functions();

	set_file(full_header.c_str());

	for (int ti = 0; ti < ntypenames; ti++) {
		typenum = ti;
		printf("%s\n", rtype[ti].c_str());
		if (precision[ti] == 1) {
			math_float(rtype[ti]);
		} else {
			math_double(rtype[ti]);
		}
		prefix = "SFMM_PREFIX";
		type = rtype[ti];
		if (is_vec(type)) {
			set_file(full_header.c_str());
			tprint("#ifndef __CUDACC__\n");
		}
		for (nopot = 0; nopot <= 1; nopot++) {
			for (int P = pmin - 1; P <= pmax; P++) {
				std::string fname;
				if (!nopot) {
					regular_harmonic(P);
					regular_harmonic_xz(P);
					if (simd[typenum] && !m2monly[typenum]) {
						greens(P);
						if (periodic) {
							greens_safe(P);
						}
						greens_xz(P);
					}
				}
			}
			for (int P = pmin; P <= pmax && !m2monly[ti]; P++) {
				flops_t flops0, flops1, flops2;
				std::string fname;
				flops_t fps;

				flops0.reset();
				flops0 += sqrt_flops();
				flops0 += erfcexp_flops();
				//fma+2+(P+1)*(1+(2*P+1)) r+5+(P+1)*(5)  rdiv+(P+1)
				flops0.fma += 2 + (P + 1) * (2 + 2 * (P + 1));
				flops0.r += 5 + 5 * (P + 1);
				flops0.rdiv += P + 1;
				flops_map[P][type + "greens_ewald_real"] = flops0;
				const double alpha = is_float(type) ? 2.4 : 2.25;
				M2LG(P, P);
				if (periodic) {
					greens_ewald(P, alpha);
				}
			}
			fflush(stdout);
		}
		if (is_vec(type)) {
			set_file(full_header.c_str());
			tprint("#endif /* __CUDACC__ */ \n");
		}
	}
	for (int ti = 0; ti < ntypenames; ti++) {
		typenum = ti;
		printf("%s\n", rtype[ti].c_str());
		prefix = "SFMM_PREFIX";
		type = rtype[ti];
		if (is_vec(type)) {
			set_file(full_header.c_str());
			tprint("#ifndef __CUDACC__\n");
		}
		for (nopot = 0; nopot <= 1; nopot++) {
			for (int P = pmin; P <= pmax; P++) {
				if (!simd[ti]) {
					L2L(P, P);
					L2L_allrot(P, P, 0);
					L2L_allrot(P, P, 1);
					L2L_allrot(P, P, 2);
				}
				if (!m2monly[typenum] && simd[typenum]) {
					L2L(P, 1);
					L2L_allrot(P, 1, 0);
					L2L_allrot(P, 1, 1);
					L2L_allrot(P, 1, 2);
					P2M(P);
					M2L(P, P);
					M2L_allrot(P, P, 0);
					M2L_allrot(P, P, 1);
					M2L_allrot(P, P, 2);
					M2L(P, 1);
					M2L_allrot(P, 1, 0);
					M2L_allrot(P, 1, 1);
					M2L_allrot(P, 1, 2);
					P2L(P);
				}
				if (periodic) {
					M2L_ewald(P);
				}
				if (m2monly[typenum]) {
					M2M(P);
					M2M_allrot(P, 0);
					M2M_allrot(P, 1);
					M2M_allrot(P, 2);
				}
			}
			fflush(stdout);
		}
		if (is_vec(type)) {
			set_file(full_header.c_str());
			tprint("#endif /* __CUDACC__ */ \n");
		}
	}
//	printf("./generated_code/include/sfmm.h");
	fflush(stdout);
	set_file(full_header.c_str());
	tprint("\n");
	set_file(full_header.c_str());
	tprint("namespace detail {\n");
	tprint("\n");
	tprint("%s", detail_header.c_str());
	tprint("#ifndef __CUDACC__\n");
	tprint("%s", detail_header_vec.c_str());
	tprint("#endif /* __CUDACC__ */\n\n");
	tprint("\n");
	tprint("template<class T, int P, int ALPHA100>\n");
	tprint("SFMM_PREFIX void greens_ewald_real(expansion<T, P>& G_st, T x, T y, T z) {\n");
	indent();
	tprint("constexpr double ALPHA = ALPHA100 / 100.0;\n");
	tprint("expansion<T, P> Gr_st;\n");
	tprint("const T r2 = fma(x, x, fma(y, y, z * z));\n");
	tprint("const T r = sqrt(r2);\n");
	tprint("greens(Gr_st, vec3<T>(x, y, z));\n");
	tprint("const T* Gr(Gr_st.data());\n");
	tprint("T* G (G_st.data());\n");
	tprint("const T xxx = T(ALPHA) * r;\n");
	tprint("T gam1, exp0;\n");
	tprint("erfcexp(xxx, &gam1, &exp0);\n");
	tprint("gam1 *= T(%.20e);\n", -sqrt(M_PI));
	tprint("const T xfac = T(ALPHA * ALPHA) * r2;\n");
	tprint("T xpow = T(ALPHA) * r;\n");
	tprint("T gam0inv = T(%.20e);\n", 1.0 / sqrt(M_PI));
	tprint("for (int l = 0; l <= P; l++) {\n");
	indent();
	tprint("const int l2 = l * (l + 1);\n");
	tprint("const T gam = gam1 * gam0inv;\n");
	tprint("for (int m = -l; m <= l; m++) {\n");
	indent();
	tprint("const int i = l2 + m;\n");
	tprint("G[i] = fma(gam, Gr[i], G[i]);\n");
	deindent();
	tprint("}\n");
	tprint("gam0inv /= -(T(l) + T(0.5));\n");
	tprint("gam1 = fma(T(l + 0.5), gam1, -xpow * exp0);\n");
	tprint("xpow *= xfac;\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");

	set_file(full_header.c_str());

	nopot = 1;
	for (int ti = 0; ti < ntypenames; ti++) {
		typenum = ti;
		type = rtype[ti];
		if (simd[ti]) {
			fprintf(fp, "#ifndef __CUDACC__\n");
		}
		for (int P = pmin - 1; P <= pmax; P++) {
			tprint("\n");
			tprint("template<>\n");
			tprint("class expansion<%s,%i> {\n", type.c_str(), P);
			indent();
			tprint("typedef %s T;\n", type.c_str());
			tprint("typedef typename type_traits<T>::type base_type;\n");
			tprint("T o[%i];\n", exp_sz(P));
			if (periodic && P > 1) {
				tprint("T t;\n");
			}
			if (scaled) {
				tprint("base_type r;\n");
			}
			deindent();
			tprint("public:\n");
			indent();
			tprint("typedef T type;\n");

			if (scaled) {
				tprint("expansion& load( expansion<base_type,%i> other, int index = -1) {\n", P);
				indent();
				tprint("other.rescale(r);\n");
			} else {
				tprint("expansion& load( const expansion<base_type,%i>& other, int index = -1) {\n", P);
				indent();
			}
			if (simd[typenum]) {
				tprint("if( index == -1 ) {\n");
				indent();
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = other.trace2();\n");
				}
				deindent();
				tprint("} else {\n");
				indent();
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i][index] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t[index] = other.trace2();\n");
				}
				deindent();
				tprint("}\n");
			} else {
				tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
				indent();
				tprint("o[i] = other[i];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = other.trace2();\n");
				}

			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_EXPANSION_MEMBERS(expansion, %s, %i);\n", type.c_str(), P);
			tprint("SFMM_PREFIX expansion& operator=(const expansion& other) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = other.o[n];\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = other.t;\n");
			}
			if (scaled) {
				tprint("r = other.r;\n");
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX expansion& operator+=(expansion other) {\n");
			indent();
			if (scaled) {
				tprint("other.rescale(r);\n");
			}
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] += other.o[n];\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t += other.t;\n");
			}
			if (scaled) {
				tprint("r = other.r;\n");
			}
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX expansion(base_type r0 = base_type(1)) {\n");
			indent();
			fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = std::numeric_limits<T>::signaling_NaN();\n");
			}
			fprintf(fp, "#endif\n");
			if (scaled) {
				tprint("r = r0;\n");
			}
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX void init(base_type r0 = base_type(1)) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = T(0);\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = T(0);\n");
			}
			if (scaled) {
				tprint("r = r0;\n");
			}
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX void rescale(base_type r0) {\n");
			indent();
			if (scaled) {
				tprint("const base_type a = r0 / r;\n");
				tprint("base_type b = a;\n");
				tprint("r = r0;\n");
				for (int n = 0; n <= P; n++) {
					for (int m = -n; m <= n; m++) {
						tprint("o[%i] *= T(b);\n", lindex(n, m));
					}
					if (periodic && P > 1 && n == 2) {
						tprint("t *= T(b);\n", exp_sz(P));
					}
					if (n != P) {
						tprint("b *= base_type(a);\n");
					}
				}
			}
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX base_type scale() const {\n");
			indent();
			if (scaled) {
				tprint("return r;\n");
			} else {
				tprint("return base_type(1);\n");
			}
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("SFMM_PREFIX T& trace2() {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX T trace2() const {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
			}
			tprint("SFMM_PREFIX static constexpr size_t size() {\n");
			indent();
			tprint("return %i;\n", exp_sz(P));
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX const T& operator[](int i) const {\n");
			indent();
			tprint("return o[i];\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX T& operator[](int i) {\n");
			indent();
			tprint("return o[i];\n");
			deindent();
			tprint("}\n");
			if (!m2monly[typenum] && simd[typenum]) {
				if (scaled && periodic && P >= pmin) {
					tprint("friend int M2L_ewald(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", P, P);
					tprint("friend int detail::M2L_ewald%s(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", pot_name(), P, P);
					tprint("friend int M2L_ewald(expansion<T, %i>&, const multipole_wo_dipole<T, %i>&, vec3<T>, int);\n", P, P);
					tprint("friend int detail::M2L_ewald%s(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", pot_name(), P, P);
				}
				if (scaled && P >= pmin) {
					for (int rot = 0; rot <= 2; rot++) {
						tprint("friend int M2Lr%i(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, P, P);
						tprint("friend int detail::M2Lr%i%s(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, pot_name(), P, P);
					}
				}
			}
			deindent();
			tprint("};\n");
			if (P > pmin - 1) {
				tprint("\n");
				tprint("template<>\n");
				tprint("class multipole<%s,%i> {\n", type.c_str(), P);
				indent();
				tprint("typedef %s T;\n", type.c_str());
				tprint("typedef typename type_traits<T>::type base_type;\n");
				tprint("T o[%i];\n", mul_sz(P));
				if (periodic && P > 2) {
					tprint("T t;\n");
				}
				if (scaled) {
					tprint("base_type r;\n");
				}
				deindent();
				tprint("public:\n");
				indent();
				tprint("typedef T type;\n");
				if (scaled) {
					tprint("multipole& load( multipole<base_type,%i> other, int index ) {\n", P);
					indent();
					tprint("other.rescale(r);\n");
				} else {
					tprint("multipole& load( const multipole<base_type,%i>& other, int index ) {\n", P);
					indent();
				}
				if (simd[typenum]) {
					tprint("if( index == -1 ) {\n");
					indent();
					tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
					indent();
					tprint("o[i] = other[i];\n");
					deindent();
					tprint("}\n");
					if (periodic && P > 2) {
						tprint("t = other.trace2();\n");
					}
					deindent();
					tprint("} else {\n");
					indent();
					tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
					indent();
					tprint("o[i][index] = other[i];\n");
					if (periodic && P > 2) {
						tprint("t[index] = other.trace2();\n");
					}
					deindent();
					tprint("}\n");
					deindent();
					tprint("}\n");
				} else {
					tprint("for( int i = 0; i < %i; i++ ) {\n", mul_sz(P));
					indent();
					tprint("o[i] = other[i];\n");
					deindent();
					tprint("}\n");
					if (periodic && P > 2) {
						tprint("t = other.trace2();\n");
					}
				}
				tprint("return *this;\n");
				deindent();
				tprint("}\n");

				tprint("SFMM_EXPANSION_MEMBERS(multipole, %s, %i);\n", type.c_str(), P);
				tprint("SFMM_PREFIX multipole& operator=(const multipole& other) {\n");
				indent();
				tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
				indent();
				tprint("o[n] = other.o[n];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = other.t;\n");
				}
				if (scaled) {
					tprint("r = other.r;\n");
				}
				tprint("return *this;\n");
				deindent();
				tprint("}\n");

				tprint("SFMM_PREFIX multipole& operator+=(multipole other) {\n");
				indent();
				if (scaled) {
					tprint("other.rescale(r);\n");
				}
				tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
				indent();
				tprint("o[n] += other.o[n];\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t += other.t;\n");
				}
				if (scaled) {
					tprint("r = other.r;\n");
				}
				tprint("return *this;\n");
				deindent();
				tprint("}\n");

				tprint("SFMM_PREFIX multipole(base_type r0 = base_type(1)) {\n");
				indent();
				fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
				tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
				indent();
				tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = std::numeric_limits<T>::signaling_NaN();\n");
				}
				fprintf(fp, "#endif\n");
				if (scaled) {
					tprint("r = r0;\n");
				}
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX void init(base_type r0 = base_type(1)) {\n");
				indent();
				tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
				indent();
				tprint("o[n] = T(0);\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 2) {
					tprint("t = T(0);\n");
				}
				if (scaled) {
					tprint("r = r0;\n");
				}
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX void rescale(base_type r0) {\n");
				indent();
				if (scaled) {
					tprint("const base_type a = r / r0;\n");
					tprint("base_type b = a;\n");
					tprint("r = r0;\n");
					for (int n = 1; n < P; n++) {
						if (!(nodip && n == 1)) {
							for (int m = -n; m <= n; m++) {
								tprint("o[%i] *= T(b);\n", mindex(n, m));
							}
							if (periodic && P > 2 && n == 2) {
								tprint("t *= T(b);\n");
							}
						}
						if (n != P - 1) {
							tprint("b *= base_type(a);\n");
						}
					}
				}
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX base_type scale() const {\n");
				indent();
				if (scaled) {
					tprint("return r;\n");
				} else {
					tprint("return base_type(1);\n");
				}
				deindent();
				tprint("}\n");
				if (periodic && P > 2 && P > 1) {
					tprint("SFMM_PREFIX T& trace2() {\n");
					indent();
					tprint("return t;\n");
					deindent();
					tprint("}\n");
					tprint("SFMM_PREFIX T trace2() const {\n");
					indent();
					tprint("return t;\n");
					deindent();
					tprint("}\n");
				}
				tprint("SFMM_PREFIX static constexpr size_t size() {\n");
				indent();
				tprint("return %i;\n", mul_sz(P));
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX const T& operator[](int i) const {\n");
				indent();
				tprint("return o[i];\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX T& operator[](int i) {\n");
				indent();
				tprint("return o[i];\n");
				deindent();
				tprint("}\n");
				if (!m2monly[typenum] && simd[typenum]) {
					if (periodic) {
						tprint("friend int M2L_ewald(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", P, P);
						tprint("friend int detail::M2L_ewald%s(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", pot_name(), P, P);
					}
					for (int rot = 0; rot <= 2; rot++) {
						tprint("friend int M2Lr%i(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, P, P);
						tprint("friend int detail::M2Lr%i%s(expansion<T, %i>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, pot_name(), P, P);
						//	if (rot < 2) {
						tprint("friend int M2Pr%i(force_type<T>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, P);
						tprint("friend int detail::M2Pr%i%s(force_type<T>&, const multipole<T, %i>&, vec3<T>, int);\n", rot, pot_name(), P);
						//		}
					}
				}
				deindent();
				tprint("};\n");
				tprint("\n");
			}
			tprint("\n");
			tprint("namespace detail {\n");
			tprint("template<>\n");
			tprint("class expansion_xz<%s,%i> {\n", type.c_str(), P);
			indent();
			tprint("typedef %s T;\n", type.c_str());
			tprint("T o[%i];\n", half_exp_sz(P));
			if (periodic && P > 1) {
				tprint("T t;\n");
			}
			deindent();
			tprint("public:\n");
			indent();
			tprint("SFMM_PREFIX expansion_xz() {\n");
			indent();
			fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", half_exp_sz(P));
			indent();
			tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = std::numeric_limits<T>::signaling_NaN();\n");
			}
			fprintf(fp, "#endif\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX T* data() {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX const T* data() const {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("SFMM_PREFIX T& trace2() {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX T trace2() const {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
			}
			deindent();
			tprint("};\n");
			tprint("\ntemplate<>\n");
			tprint("class expansion_xy<%s,%i> {\n", type.c_str(), P);
			indent();
			tprint("typedef %s T;\n", type.c_str());
			tprint("T o[%i];\n", xyexp_sz(P));
			if (periodic && P > 1) {
				tprint("T t;\n");
			}
			deindent();
			tprint("public:\n");
			indent();
			tprint("SFMM_PREFIX expansion_xy() {\n");
			indent();
			fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", half_exp_sz(P));
			indent();
			tprint("o[n] = std::numeric_limits<T>::signaling_NaN();\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("t = std::numeric_limits<T>::signaling_NaN();\n");
			}
			fprintf(fp, "#endif\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX T* data() {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX const T* data() const {\n");
			indent();
			tprint("return o;\n");
			deindent();
			tprint("}\n");
			if (periodic && P > 1) {
				tprint("SFMM_PREFIX T& trace2() {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX T trace2() const {\n");
				indent();
				tprint("return t;\n");
				deindent();
				tprint("}\n");
			}
			deindent();
			tprint("};\n");
			tprint("}\n");
		}
		if (simd[ti]) {
			fprintf(fp, "#endif /* __CUDACC__ */\n");
		}

	}
	nopot = 0;
	include("complex_impl.hpp");
	include("expansion.hpp");

	str = "template<class V, typename std::enable_if<is_compound_type<V>::value>::type* = nullptr>\n"
			"inline void apply_padding(V& A, int n) {\n"
			"\tfor (int i = 0; i < V::size(); i++) {\n"
			"\t\tapply_padding(A[i], n);\n"
			"\t}\n"
#ifdef PERIODIC
			"\tapply_padding(A.trace2(), n);\n"
#endif
			"}\n"
			"\n"
			"template<class V, typename std::enable_if<is_compound_type<V>::value>::type* = nullptr>\n"
			"inline void apply_mask(V& A, int n) {\n"
			"\tconst auto mask = create_mask<typename V::type>(n);\n"
			"\tfor (int i = 0; i < V::size(); i++) {\n"
			"\t\tA[i] *= mask;\n"
			"\t}\n"
#ifdef PERIODIC
			"\tA.trace2() *= mask;\n"
#endif
			"}\n"
			"\n"
			"";
	str += "template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>\n"
			"inline expansion<typename type_traits<T>::type, P> reduce_sum(const expansion<T, P>& A) {\n"
			"\tconstexpr int end = expansion<T, P>::size();\n"
			"\texpansion<typename type_traits<T>::type, P> B;\n"
			"\tfor (int i = 0; i < end; i++) {\n"
			"\t\tB[i] = reduce_sum(A[i]);\n"
			"\t}\n"
#ifdef PERIODIC
			"\tB.trace2() = reduce_sum(A.trace2());\n"
#endif
			"\treturn B;\n"
			"}\n"
			"\n"
			"";

	str += "template<class T, int P, typename std::enable_if<type_traits<T>::is_simd, T>* = nullptr>\n"
			"inline multipole<typename type_traits<T>::type, P> reduce_sum(const multipole<T, P>& A) {\n"
			"\tconstexpr int end = multipole<T, P>::size();\n"
			"\tmultipole<typename type_traits<T>::type, P> B;\n"
			"\tfor (int i = 0; i < end; i++) {\n"
			"\t\tB[i] = reduce_sum(A[i]);\n"
			"\t}\n"
#ifdef PERIODIC
			"\tB.trace2() = reduce_sum(A.trace2());\n"
#endif
			"\treturn B;\n"
			"}\n"
			"";
	fprintf(fp, "%s", str.c_str());
	include("timer.hpp");
	tprint("}\n");
	tprint("extern \"C\" {\n");
	tprint("void sfmm_detail_atomic_inc_dbl(double* num, double val);\n");
	tprint("void sfmm_detail_atomic_inc_int(unsigned long long* num, unsigned long long val);\n");
	tprint("}\n\n");
	set_file("./generated_code/src/database.cpp");
	timing_body = std::string("#include \"sfmm.hpp\"\n\nnamespace sfmm {\nnamespace detail {\nstatic func_data_t function_data[] = {") + timing_body;
	timing_body += "\n};\n";
	timing_body += "\nint operator_count() {\n";
	timing_body += print2str("\treturn %i;\n", timing_cnt);
	timing_body += print2str("}\n\n", timing_cnt);
	timing_body += "\nfunc_data_t* operator_data(int index) {\n";
	timing_body += print2str("\treturn function_data + index;\n");
	timing_body += print2str("}\n\n", timing_cnt);
	timing_body += "\n}\n}\n";
	tprint("%s\n", timing_body.c_str());
	set_file("./generated_code/src/timing.cpp");
	include("timing.cpp");
	set_file("./generated_code/src/atomic.c");
	include("atomic.c");

	/*set_file("./generated_code/src/flops.cpp");
	 tprint("#include \"sfmm.hpp\"\n");
	 tprint("#include <unordered_map>\n");
	 tprint("#include <mutex>\n");
	 tprint("\n");
	 tprint("namespace sfmm {\n");
	 tprint("namespace detail {\n");
	 tprint("static void initialize() { \n");
	 indent();
	 tprint("std::array<std::array<std::array<std::unordered_map<std::string, std::unordered_map<std::string, int>>,3>,2>,%i> flops;\n", PMAX + 1);
	 for (int ti = 0; ti < ntypenames; ti++) {
	 typenum = ti;
	 type = rtype[ti];
	 int nops = 7;
	 const char* opnames[nops] = { "M2L", "M2P", "P2L", "P2M", "M2M", "L2L", "L2P" };
	 for (nopot = 0; nopot <= 1; nopot++) {
	 for (int rot = 0; rot < 3; rot++) {
	 for (int P = pmin; P <= pmax; P++) {
	 for (int op = 0; op < nops; op++) {
	 auto& map = allflops[P][nopot][rot][rtype[ti]];
	 auto iter = map.find(opnames[op]);
	 if (iter != map.end()) {
	 tprint("flops[%i][%i][%i][\"%s\"][\"%s\"] = %i;\n", P, nopot, rot, rtype[ti].c_str(), opnames[op], iter->second);
	 }
	 }
	 }
	 }
	 }
	 }
	 std::string str2 = "";
	 str2 += "\tfor( int i = 0; i < operator_count(); i++) {\n";
	 str2 += "\t\tauto& entry = *operator_data(i);\n";
	 str2 += "\t\tentry.flops = flops[entry.P][entry.nopot][entry.nrot][entry.type][entry.name];\n";
	 str2 += "\t}\n";
	 fprintf(fp, "%s", str2.c_str());
	 deindent();
	 tprint("}\n\n");
	 tprint("void operator_flops_initialize() {\n");
	 indent();
	 tprint("static std::once_flag flag;\n");
	 tprint("std::call_once(flag, []() {detail::initialize();});\n");
	 deindent();
	 tprint("}\n\n");
	 tprint("}\n\n");
	 tprint("}\n");*/

	set_file("./generated_code/src/best_flops.cpp");
	str = print2str("#include \"sfmm.hpp\"\n");
	str += print2str("#include <unordered_map>\n");
	str += print2str("#include <mutex>\n");
	str += print2str("#include <stdio.h>\n");
	str += print2str("\n");
	str += print2str("namespace sfmm {\n");
	str += print2str("namespace detail {\n");
	str += print2str("\n");
	str += print2str("static constexpr int pmax = %i;\n", PMAX);
	str += print2str("static std::array<std::array<std::unordered_map<std::string, std::unordered_map<std::string, int>>,2>,%i> best_rotation;\n", PMAX + 1);
	str += print2str("static std::array<std::array<std::array<std::unordered_map<std::string, std::unordered_map<std::string, int>>,2>,3>,%i> opflops;\n",
			PMAX + 1);
	str += print2str("\n");
	str += "static void initialize() {\n"
			"\tstatic std::once_flag flag;\n"
			"\tstd::call_once(flag, []() {\n";
	str += "\t\tstd::array<std::array<std::unordered_map<std::string, std::unordered_map<std::string, int>>, 2>," + std::to_string(PMAX + 1) + "> best_flops;\n";
	str += "\t\tint best_rot = -1;\n"
			"\t\tint best_ops = 9999999;\n"
			"\t\tfor (int i = 0; i < detail::operator_count(); i++) {\n"
			"\t\t\tconst detail::func_data_t& func = *(detail::operator_data(i));\n"
			"\t\t\tbest_flops[func.P][func.nopot][func.type][func.name] = std::numeric_limits<int>::max();\n"
			"\t\t}\n"
			"\t\tfor (int i = 0; i < detail::operator_count(); i++) {\n"
			"\t\t\tconst detail::func_data_t& func = *(detail::operator_data(i));\n"
			"\t\t\tauto& flops = best_flops[func.P][func.nopot][func.type][func.name];\n"
			"\t\t\tif( func.flops < flops ) {\n"
			"\t\t\t\tflops = func.flops;\n"
			"\t\t\t\tdetail::best_rotation[func.P][func.nopot][func.type][func.name] = func.nrot;\n"
			"\t\t\t}\n"
			"\t\t\tdetail::opflops[func.P][func.nrot][func.nopot][func.type][func.name] = func.flops;\n"
			"\t\t}\n"
			"\t});\n"
			"}\n\n"
			"";
	str += "std::string operator_best_rotations() {\n";
	str += "\tdetail::initialize();\n";
	str += "\tstd::string str;\n";
	str += "\n\n\tstr += print2str( \"-----M2L---------\\n\" );\n";
	str += "\tstr += print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti] && !m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += print2str( \" %2i   %%1i   %%1i \\n\", operator_best_rotation(%i, 0, \"%s\", \"M2L\"), operator_best_rotation(%i, 1, \"%s\", \"M2L\") );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += print2str( \"-----M2P---------\\n\" );\n";
	str += "\tstr += print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti] && !m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += print2str( \" %2i   %%1i   %%1i \\n\", operator_best_rotation(%i, 0, \"%s\", \"M2P\"), operator_best_rotation(%i, 1, \"%s\", \"M2P\") );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += print2str( \"-----L2P---------\\n\" );\n";
	str += "\tstr += print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti] && !m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += print2str( \" %2i   %%1i   %%1i \\n\", operator_best_rotation(%i, 0, \"%s\", \"L2P\"), operator_best_rotation(%i, 1, \"%s\", \"L2P\") );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += print2str( \"-----L2L---------\\n\" );\n";
	str += "\tstr += print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (!simd[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += print2str( \" %2i   %%1i   %%1i \\n\", operator_best_rotation(%i, 0, \"%s\", \"L2L\"), operator_best_rotation(%i, 1, \"%s\", \"L2L\") );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += print2str( \"-----M2M---------\\n\" );\n";
	str += "\tstr += print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += print2str( \" %2i   %%1i   %%1i \\n\", operator_best_rotation(%i, 0, \"%s\", \"M2M\"), operator_best_rotation(%i, 1, \"%s\", \"M2M\") );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\treturn str;\n";
	str += ("}\n\n");
	str += print2str("}\n\n");

	str += "int operator_best_rotation(int P, int nopot, const char* type, const char* name) {\n"
			"\tdetail::initialize();\n"
			"\treturn detail::best_rotation[P][nopot][type][name];\n"
			"}\n"
			"\n"
			""
			"void operator_write_new_bestops_source() {\n"
			"}\n"
			"";
	str += "\nstd::string operator_show_flops() {\n";
	str += "detail::initialize();\n";
	str += "\tstd::string str;\n";
	str += "\n\n\tstr += detail::print2str( \"-----M2L---------\\n\" );\n";
	str += "\tstr += detail::print2str( \"P-w/pot----w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti] && !m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += detail::print2str( \"%2i %%4i/%%4i/%%4i (%%i) %%4i/%%4i/%%4i (%%i) \\n\", detail::opflops[%i][0][0][\"%s\"][\"M2L\"], detail::opflops[%i][1][0][\"%s\"][\"M2L\"], detail::opflops[%i][2][0][\"%s\"][\"M2L\"], detail::best_rotation[%i][0][\"%s\"][\"M2L\"], detail::opflops[%i][0][1][\"%s\"][\"M2L\"], detail::opflops[%i][1][1][\"%s\"][\"M2L\"], detail::opflops[%i][2][1][\"%s\"][\"M2L\"], detail::best_rotation[%i][1][\"%s\"][\"M2L\"] );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(),
								p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += detail::print2str( \"-----M2P---------\\n\" );\n";
	str += "\tstr += detail::print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti] && !m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += detail::print2str( \"%2i %%4i/%%4i/%%4i (%%i) %%4i/%%4i/%%4i (%%i) \\n\", detail::opflops[%i][0][0][\"%s\"][\"M2P\"], detail::opflops[%i][1][0][\"%s\"][\"M2P\"], detail::opflops[%i][2][0][\"%s\"][\"M2P\"], detail::best_rotation[%i][0][\"%s\"][\"M2P\"], detail::opflops[%i][0][1][\"%s\"][\"M2P\"], detail::opflops[%i][1][1][\"%s\"][\"M2P\"], detail::opflops[%i][2][1][\"%s\"][\"M2P\"], detail::best_rotation[%i][1][\"%s\"][\"M2P\"] );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(),
								p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += detail::print2str( \"-----L2P---------\\n\" );\n";
	str += "\tstr += detail::print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (simd[ti] && !m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += detail::print2str( \"%2i %%4i/%%4i/%%4i (%%i) %%4i/%%4i/%%4i (%%i) \\n\", detail::opflops[%i][0][0][\"%s\"][\"L2P\"], detail::opflops[%i][1][0][\"%s\"][\"L2P\"], detail::opflops[%i][2][0][\"%s\"][\"L2P\"], detail::best_rotation[%i][0][\"%s\"][\"L2P\"], detail::opflops[%i][0][1][\"%s\"][\"L2P\"], detail::opflops[%i][1][1][\"%s\"][\"L2P\"], detail::opflops[%i][2][1][\"%s\"][\"L2P\"], detail::best_rotation[%i][1][\"%s\"][\"L2P\"] );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(),
								p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += detail::print2str( \"-----L2L---------\\n\" );\n";
	str += "\tstr += detail::print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (!simd[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += detail::print2str( \"%2i %%4i/%%4i/%%4i (%%i) %%4i/%%4i/%%4i (%%i) \\n\", detail::opflops[%i][0][0][\"%s\"][\"L2L\"], detail::opflops[%i][1][0][\"%s\"][\"L2L\"], detail::opflops[%i][2][0][\"%s\"][\"L2L\"], detail::best_rotation[%i][0][\"%s\"][\"L2L\"], detail::opflops[%i][0][1][\"%s\"][\"L2L\"], detail::opflops[%i][1][1][\"%s\"][\"L2L\"], detail::opflops[%i][2][1][\"%s\"][\"L2L\"], detail::best_rotation[%i][1][\"%s\"][\"L2L\"] );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(),
								p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += detail::print2str( \"-----M2M---------\\n\" );\n";
	str += "\tstr += detail::print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (m2monly[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += detail::print2str( \"%2i %%4i/%%4i/%%4i (%%i) %%4i/%%4i/%%4i (%%i) \\n\", detail::opflops[%i][0][0][\"%s\"][\"M2M\"], detail::opflops[%i][1][0][\"%s\"][\"M2M\"], detail::opflops[%i][2][0][\"%s\"][\"M2M\"], detail::best_rotation[%i][0][\"%s\"][\"M2M\"], detail::opflops[%i][0][1][\"%s\"][\"M2M\"], detail::opflops[%i][1][1][\"%s\"][\"M2M\"], detail::opflops[%i][2][1][\"%s\"][\"M2M\"], detail::best_rotation[%i][1][\"%s\"][\"M2M\"] );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(), p, rtype[ti].c_str(),
								p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += detail::print2str( \"-----P2L---------\\n\" );\n";
	str += "\tstr += detail::print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (!m2monly[ti] && simd[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += detail::print2str( \"%2i %%4i              %%4i              \\n\", detail::opflops[%i][0][0][\"%s\"][\"P2L\"], detail::opflops[%i][0][1][\"%s\"][\"P2L\"] );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\n\tstr += detail::print2str( \"-----P2M---------\\n\" );\n";
	str += "\tstr += detail::print2str( \"--P-w/pot-w/o-pot-\\n\" );\n";
	for (int ti = 0; ti < ntypenames; ti++) {
		if (!m2monly[ti] && simd[ti]) {
			for (int p = pmin; p <= pmax; p++) {
				str +=
						print2str(
								"\tstr += detail::print2str( \"%2i %%4i              %%4i              \\n\", detail::opflops[%i][0][0][\"%s\"][\"P2M\"], detail::opflops[%i][0][1][\"%s\"][\"P2M\"] );\n",
								p, p, rtype[ti].c_str(), p, rtype[ti].c_str());
			}
		}
	}
	str += "\n\treturn str;\n";
	str += ("}\n\n");

	str += print2str("}\n");

	tprint("%s\n", str.c_str());
	return 0;
}
