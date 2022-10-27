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
	int load() const {
		return r + i + fma + 4 * (rdiv + idiv) + con + asgn + icmp + rcmp;
	}
};

using complex = std::complex<double>;

static int ntab = 0;
static int tprint_on = true;

//#define FLOAT
//#define DOUBLE
//#define CUDA_FLOAT
//#define CUDA_DOUBLE
//#define VEC_DOUBLE
//#define VEC_FLOAT
#define VEC_DOUBLE_SIZE 2
#define VEC_FLOAT_SIZE 8

#define ASPRINTF(...) if( asprintf(__VA_ARGS__) == 0 ) {printf( "ASPRINTF error %s %i\n", __FILE__, __LINE__); abort(); }
#define SYSTEM(...) if( system(__VA_ARGS__) != 0 ) {printf( "SYSTEM error %s %i\n", __FILE__, __LINE__); abort(); }

static int nopot = false;
static bool fmaops = true;
static int periodic = 0;
static int pmin = PMIN;
static int pmax = PMAX;
static std::string type = "float";
static std::string sitype = "int";
static std::string uitype = "unsigned";
static const int divops = 4;
static const char* prefix = "";
static std::string detail_header;
static std::string detail_header_vec;
static std::vector<std::string> lines[2];
static std::string vf = std::string("v") + std::to_string(VEC_FLOAT_SIZE) + "sf";
static std::string vd = std::string("v") + std::to_string(VEC_DOUBLE_SIZE) + "df";
static std::string vsi32 = std::string("v") + std::to_string(VEC_FLOAT_SIZE) + "si32";
static std::string vsi64 = std::string("v") + std::to_string(VEC_DOUBLE_SIZE) + "si64";
static std::string vui32 = std::string("v") + std::to_string(VEC_FLOAT_SIZE) + "ui32";
static std::string vui64 = std::string("v") + std::to_string(VEC_DOUBLE_SIZE) + "ui64";
#ifdef FLOAT
static std::string header = "sfmmf.hpp";
#endif
#ifdef CUDA_DOUBLE
static std::string header = "cusfmmd.hpp";
#endif
#ifdef CUDA_FLOAT
static std::string header = "cusfmmf.hpp";
#endif
#ifdef DOUBLE
static std::string header = "sfmmd.hpp";
#endif
#ifdef VEC_FLOAT
static std::string header = "sfmmvf.hpp";
#endif
#ifdef VEC_DOUBLE
static std::string header = "sfmmvd.hpp";
#endif
static std::string full_header = std::string("./generated_code/include/") + header;

static const char* period_name() {
	return periodic ? "_periodic" : "_nonperiodic";
}

static std::string vec_header() {
	std::string str = "\n";
	str += "#ifndef SFMM_VEC_HEADER42\n";
	str += "#define SFMM_VEC_HEADER42\n";
	str += "#define create_binary_op(vtype, type, op) \\\n";
	str += "   inline vtype operator op (const vtype& u ) const { \\\n";
	str += "      vtype w; \\\n";
	str += "      w.v = v op u.v; \\\n";
	str += "		  return w; \\\n";
	str += "   } \\\n";
	str += "	  inline vtype& operator op##= (const vtype& u ) { \\\n";
	str += "       *this = *this op u; \\\n";
	str += "	      return *this; \\\n";
	str += "   }\n";
	str += "\n";
	str += "#define create_unary_op(vtype, type, op) \\\n";
	str += "   inline vtype operator op () const { \\\n";
	str += "      vtype w; \\\n";
	str += "      w.v = op v; \\\n";
	str += "      return w; \\\n";
	str += "   }\n";
	str += "\n";
	str += "#define create_convert_op_prot(vtype,type,ovtype,otype) \\\n";
	str += "   inline vtype(const ovtype&); \\\n";
	str += "   inline vtype& operator=(const ovtype&); \\\n";
	str += "   inline vtype& operator=(const otype&)\n";
	str += "\n";
	str += "#define create_convert_op_def(vtype,type,ovtype,otype) \\\n";
	str += "   inline vtype::vtype(const ovtype& other) { \\\n";
	str += "	     v = __builtin_convertvector(other.v, simd_t); \\\n";
	str += "   } \\\n";
	str += "   inline vtype& vtype::operator=(const ovtype& other) { \\\n";
	str += "	     v = __builtin_convertvector(other.v, simd_t); \\\n";
	str += "	     return *this; \\\n";
	str += "   }\n";
	str += "\n";
	str += "#define create_broadcast_op(vtype,type) \\\n";
	str += "   inline vtype(const type& other) { \\\n";
	str += "	     v = other - simd_t{}; \\\n";
	str += "   } \\\n";
	str += "   inline vtype& operator=(const type& other) { \\\n";
	str += "	     v = other - simd_t{}; \\\n";
	str += "	     return *this; \\\n";
	str += "   }\n";
	str += "\n";
	str += "#define create_compare_op_prot(vtype,vstype,stype,op) \\\n";
	str += "   inline vstype operator op (const vtype&) const\n";
	str += "\n";
	str += "#define create_compare_op_def(vtype,vstype,sitype,op) \\\n";
	str += "   inline vstype vtype::operator op (const vtype& other) const { \\\n";
	str += "	     vstype w; \\\n";
	str += "      w.v = (-(v op other.v)); \\\n";
	str += "      return w; \\\n";
	str += "   }\n";
	str += "\n";
	str += "#define create_vec_types_fwd(vtype)              \\\n";
	str += "class vtype\n";
	str += "\n";
	str += "#define create_rvec_types(vtype, type, vstype, stype, vutype, utype, size)              \\\n";
	str += "   class vtype {                                           \\\n";
	str += "      typedef type simd_t __attribute__ ((vector_size(size*sizeof(type))));  \\\n";
	str += "      simd_t v;  \\\n";
	str += "   public: \\\n";
	str += "      inline constexpr vtype() : v() {} \\\n";
	str += "      inline type operator[](int i) const {  \\\n";
	str += "         return v[i]; \\\n";
	str += "      }\\\n";
	str += "	     inline type& operator[](int i) {  \\\n";
	str += "		     return v[i]; \\\n";
	str += "	     }\\\n";
	str += "      create_binary_op(vtype, type, +); \\\n";
	str += "      create_binary_op(vtype, type, -); \\\n";
	str += "      create_binary_op(vtype, type, *); \\\n";
	str += "      create_binary_op(vtype, type, /); \\\n";
	str += "      create_unary_op(vtype, type, +); \\\n";
	str += "      create_unary_op(vtype, type, -); \\\n";
	str += "      create_convert_op_prot(vtype,type,vstype,stype); \\\n";
	str += "      create_convert_op_prot(vtype,type,vutype,utype); \\\n";
	str += "      create_broadcast_op(vtype,type); \\\n";
	str += "      create_compare_op_prot(vtype, vstype, stype, <); \\\n";
	str += "      create_compare_op_prot(vtype, vstype, stype, >); \\\n";
	str += "      create_compare_op_prot(vtype, vstype, stype, <=); \\\n";
	str += "      create_compare_op_prot(vtype, vstype, stype, >=); \\\n";
	str += "      create_compare_op_prot(vtype, vstype, stype, ==); \\\n";
	str += "      create_compare_op_prot(vtype, vstype, stype, !=); \\\n";
	str += "      friend class vstype; \\\n";
	str += "      friend class vutype; \\\n";
	str += "   }\n";
	str += "\n";
	str += "#define create_ivec_types(vtype, type, votype, otype, vrtype, rtype, vstype, stype, size)              \\\n";
	str += "   class vtype {                                           \\\n";
	str += "      typedef type simd_t __attribute__ ((vector_size(size*sizeof(type))));  \\\n";
	str += "      simd_t v;  \\\n";
	str += "   public: \\\n";
	str += "      inline constexpr vtype() : v() {} \\\n";
	str += "      inline type operator[](int i) const {  \\\n";
	str += "         return v[i]; \\\n";
	str += "      }\\\n";
	str += "      inline type& operator[](int i) {  \\\n";
	str += "         return v[i]; \\\n";
	str += "      }\\\n";
	str += "      create_binary_op(vtype, type, +); \\\n";
	str += "      create_binary_op(vtype, type, -); \\\n";
	str += "      create_binary_op(vtype, type, *); \\\n";
	str += "      create_binary_op(vtype, type, /); \\\n";
	str += "      create_binary_op(vtype, type, &); \\\n";
	str += "      create_binary_op(vtype, type, ^); \\\n";
	str += "      create_binary_op(vtype, type, |); \\\n";
	str += "      create_binary_op(vtype, type, >>); \\\n";
	str += "      create_binary_op(vtype, type, <<); \\\n";
	str += "      create_unary_op(vtype, type, +); \\\n";
	str += "      create_unary_op(vtype, type, -); \\\n";
	str += "      create_unary_op(vtype, type, ~); \\\n";
	str += "      create_broadcast_op(vtype,type); \\\n";
	str += "      create_convert_op_prot(vtype,type,vrtype,rtype); \\\n";
	str += "      create_convert_op_prot(vtype,type,votype,otype); \\\n";
	str += "      create_compare_op_prot(vtype,vstype,stype,<); \\\n";
	str += "      create_compare_op_prot(vtype,vstype,stype,>); \\\n";
	str += "      create_compare_op_prot(vtype,vstype,stype,<=); \\\n";
	str += "      create_compare_op_prot(vtype,vstype,stype,>=); \\\n";
	str += "      create_compare_op_prot(vtype,vstype,stype,==); \\\n";
	str += "      create_compare_op_prot(vtype,vstype,stype,!=); \\\n";
	str += "      friend class vrtype; \\\n";
	str += "      friend class votype; \\\n";
	str += "   }\n";
	str += "\n";
	str += "#define create_rvec_types_def(vtype, type, vstype, stype, vutype, utype, size)\\\n";
	str += "   create_convert_op_def(vtype, type, vstype, stype); \\\n";
	str += "   create_convert_op_def(vtype, type, vutype, utype)\n";
	str += "\n";
	str += "#define create_ivec_types_def(vtype, type, votype, otype, vrtype, rtype, size)              \\\n";
	str += "   create_convert_op_def(vtype,type,vrtype,rtype); \\\n";
	str += "   create_convert_op_def(vtype,type,votype,otype)\n";
	str += "\n";
	str += "#define create_vec_types(vrtype,rtype,vstype,stype,vutype,utype,size) \\\n";
	str += "   create_vec_types_fwd(vrtype); \\\n";
	str += "   create_vec_types_fwd(vutype); \\\n";
	str += "   create_vec_types_fwd(vstype); \\\n";
	str += "   create_rvec_types(vrtype,rtype,vstype,stype,vutype,utype, size); \\\n";
	str += "   create_ivec_types(vutype,utype,vstype,stype,vrtype,rtype,vstype,stype,size); \\\n";
	str += "   create_ivec_types(vstype,stype,vutype,utype,vrtype,rtype,vstype,stype,size); \\\n";
	str += "   create_rvec_types_def(vrtype,rtype,vstype,stype,vutype,utype, size); \\\n";
	str += "   create_ivec_types_def(vutype,utype,vstype,stype,vrtype,rtype, size); \\\n";
	str += "   create_ivec_types_def(vstype,stype,vutype,utype,vrtype,rtype, size); \\\n";
	str += "   create_compare_op_def(vrtype,vstype,stype,<); \\\n";
	str += "   create_compare_op_def(vrtype,vstype,stype,>); \\\n";
	str += "   create_compare_op_def(vrtype,vstype,stype,<=); \\\n";
	str += "   create_compare_op_def(vrtype,vstype,stype,>=); \\\n";
	str += "   create_compare_op_def(vrtype,vstype,stype,==); \\\n";
	str += "   create_compare_op_def(vrtype,vstype,stype,!=); \\\n";
	str += "   create_compare_op_def(vutype,vstype,stype,<); \\\n";
	str += "   create_compare_op_def(vutype,vstype,stype,>); \\\n";
	str += "   create_compare_op_def(vutype,vstype,stype,<=); \\\n";
	str += "   create_compare_op_def(vutype,vstype,stype,>=); \\\n";
	str += "   create_compare_op_def(vutype,vstype,stype,==); \\\n";
	str += "   create_compare_op_def(vutype,vstype,stype,!=); \\\n";
	str += "   create_compare_op_def(vstype,vstype,stype,<); \\\n";
	str += "   create_compare_op_def(vstype,vstype,stype,>); \\\n";
	str += "   create_compare_op_def(vstype,vstype,stype,<=); \\\n";
	str += "   create_compare_op_def(vstype,vstype,stype,>=); \\\n";
	str += "   create_compare_op_def(vstype,vstype,stype,==); \\\n";
	str += "   create_compare_op_def(vstype,vstype,stype,!=)\n";
	str += "#endif\n";
	char* b;
#ifdef VEC_FLOAT
	str += "#ifndef SFMM_VEC_FLOAT42\n";
	str += "#define SFMM_VEC_FLOAT42\n";
	ASPRINTF(&b, "create_vec_types(%s, float, %s, int32_t, %s, uint32_t, %i);\n", vf.c_str(), vsi32.c_str(), vui32.c_str(), VEC_FLOAT_SIZE);
	str += b;
	free(b);
	ASPRINTF(&b, "inline float sum(v%isf v) {\n", VEC_FLOAT_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_FLOAT_SIZE; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttypedef float type%i __attribute__ ((vector_size(%i*sizeof(float))));\n", sz, sz);
		str += b;
		free(b);
	}
	ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&v));\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_FLOAT_SIZE / 2; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&a%i));\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\tconst type%i& b%i = *(((type%i*)(&a%i))+1);\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\ta%i += b%i;\n", sz, sz);
		str += b;
		free(b);
	}
	ASPRINTF(&b, "\treturn a2[0] + a2[1];\n");
	str += b;
	free(b);
	str += "}\n";
	ASPRINTF(&b, "inline int64_t sum(v%isi32 v) {\n", VEC_FLOAT_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_FLOAT_SIZE; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttypedef float type%i __attribute__ ((vector_size(%i*sizeof(int32_t))));\n", sz, sz);
		str += b;
		free(b);
	}
	str += "\n";
	ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&v));\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_FLOAT_SIZE / 2; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&a%i));\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\tconst type%i& b%i = *(((type%i*)(&a%i))+1);\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\ta%i += b%i;\n", sz, sz);
		str += b;
		free(b);
	}
	ASPRINTF(&b, "\treturn a2[0] + a2[1];\n");
	str += b;
	free(b);
	str += "}\n";
	str += "#endif\n";
#endif

#ifdef VEC_DOUBLE
	str += "#ifndef SFMM_VEC_DOUBLE42\n";
	str += "#define SFMM_VEC_DOUBLE42\n";
	ASPRINTF(&b, "create_vec_types(%s, double, %s, int64_t, %s, uint64_t, %i);\n", vd.c_str(), vsi64.c_str(), vui64.c_str(), VEC_DOUBLE_SIZE);
	str += b;
	free(b);
	ASPRINTF(&b, "inline double sum(v%idf v) {\n", VEC_DOUBLE_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_DOUBLE_SIZE; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttypedef double type%i __attribute__ ((vector_size(%i*sizeof(double))));\n", sz, sz);
		str += b;
		free(b);
	}
	ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&v));\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_DOUBLE_SIZE / 2; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&a%i));\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\tconst type%i& b%i = *(((type%i*)(&a%i))+1);\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\ta%i += b%i;\n", sz, sz);
		str += b;
		free(b);
	}
	ASPRINTF(&b, "\treturn a2[0] + a2[1];\n");
	str += b;
	free(b);
	str += "}\n";
	str += "\n";
	ASPRINTF(&b, "inline double sum(v%isi64 v) {\n", VEC_DOUBLE_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_DOUBLE_SIZE; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttypedef double type%i __attribute__ ((vector_size(%i*sizeof(int64_t))));\n", sz, sz);
		str += b;
		free(b);
	}
	ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&v));\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	str += b;
	free(b);
	for (int sz = VEC_DOUBLE_SIZE / 2; sz > 1; sz /= 2) {
		ASPRINTF(&b, "\ttype%i a%i = *((type%i*)(&a%i));\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\tconst type%i& b%i = *(((type%i*)(&a%i))+1);\n", sz, sz, sz, 2 * sz);
		str += b;
		free(b);
		ASPRINTF(&b, "\ta%i += b%i;\n", sz, sz);
		str += b;
		free(b);
	}
	ASPRINTF(&b, "\treturn a2[0] + a2[1];\n");
	str += b;
	free(b);
	str += "}\n";
	str += "#endif\n";
#endif

	return str;
}

static int cuda = 0;

static bool is_float(std::string str) {
	return str == "float" || str == vf;
}

static bool is_double(std::string str) {
	return str == "double" || str == vd;
}

static bool is_vec(std::string str) {
	return str == vf || str == vd;
}

const double ewald_r2 = (2.6 + 0.5 * sqrt(3));
const int ewald_h2 = 8;

static FILE* fp = nullptr;
int exp_sz(int P) {
	return (P + 1) * (P + 1);
}

int half_exp_sz(int P) {
	return (P + 2) * (P + 1) / 2;
}

int mul_sz(int P) {
	return P * P;
}

static flops_t greens_ewald_real_flops;

double tiny() {
	if (is_float(type)) {
		return 2.0 / std::numeric_limits<float>::max();
	} else {
		return 2.0 / std::numeric_limits<double>::max();
	}
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
	fps.r += 10;
	fps.i += 2;
	fps.asgn += 2;
	fps.fma += 2;
	fps.rdiv++;
	if (!is_float(type)) {
		fps.fma += 2;
	}
	return fps;
}

flops_t rsqrt_flops() {
	flops_t fps;
	fps.r += 10;
	fps.i += 2;
	fps.asgn += 2;
	fps.fma += 2;
	if (!is_float(type)) {
		fps.fma += 2;
	}
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
			fps.r += 16;
			fps.rcmp++;
			fps.con++;
			fps.rdiv += 2;
			fps.asgn += 2;
			fps.fma += 33;
		} else {
			fps.r += 7;
			fps.rcmp++;
			fps.asgn++;
			fps.fma += 25;
		}
	} else {
		if (is_vec(type)) {
			fps.r += 20;
			fps.rcmp += 4;
			fps.asgn += 5;
			fps.rdiv += 2;
			fps.fma += 74;
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
			flops_t fps0;
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
			fps += fps0;
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

template<class ...Args>
void tprint(int index, const char* fstr, Args&&...args) {
	if (fp == nullptr) {
		return;
	}
	std::string str;
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			str += "\t";
		}
		char* buf;
		ASPRINTF(&buf, fstr, std::forward<Args>(args)...)
		str += buf;
		free(buf);
		lines[index].push_back(str);
	}
}

void tprint(int index, const char* fstr) {
	if (fp == nullptr) {
		return;
	}
	std::string str;
	if (tprint_on) {
		for (int i = 0; i < ntab; i++) {
			str += "\t";
		}
		str += fstr;
		lines[index].push_back(str);
	}
}

void tprint_flush() {
	int n0 = 0;
	int n1 = 0;
	while (n0 < lines[0].size() || n1 < lines[1].size()) {
		if (n0 < lines[0].size() && n1 < lines[1].size()) {
			if ((double) n0 / (lines[0].size() + 1) < (double) n1 / (lines[1].size() + 1)) {
				fprintf(fp, "%s", lines[0][n0].c_str());
				n0++;
			} else {
				fprintf(fp, "%s", lines[1][n1].c_str());
				n1++;
			}
		} else if (n0 < lines[0].size()) {
			fprintf(fp, "%s", lines[0][n0].c_str());
			n0++;
		} else {
			fprintf(fp, "%s", lines[1][n1].c_str());
			n1++;
		}
	}
	lines[0].resize(0);
	lines[1].resize(0);
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
			fprintf(fp, "%s", inschains[best][n[best]].c_str());
		}
		n[best]++;
	}
	inschains = decltype(inschains)(nchains);
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
	LIT, PTR, CPTR, EXP, MUL, CEXP, CMUL, FORCE
};

void init_real(std::string var) {
	tprint("T %s;\n", var.c_str());
	if (var == "tmp1") {
		for (int i = 2; i < nchains; i++) {
			init_real(std::string("tmp") + std::to_string(i));
		}
	}
}

void init_reals(std::string var, int cnt) {
	std::string str;
	str = "T " + var + "[" + std::to_string(cnt) + "];";
//			" = {";
	/*	for (int n = 0; n < cnt - 1; n++) {
	 str += "std::numeric_limits<T>::signaling_NaN(), ";
	 }
	 str += "std::numeric_limits<T>::signaling_NaN()};";*/
	tprint("%s\n", str.c_str());
}

template<class ...Args>
std::string func_args(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		str += std::string("expansion") + period_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == MUL) {
		str += std::string("multipole") + period_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CEXP) {
		str += std::string("const expansion") + period_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CMUL) {
		str += std::string("const multipole") + period_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == FORCE) {
		str += std::string("force_type<") + type + ">& " + arg;
	} else {
		if (atype == CPTR) {
			str += std::string("const ");
		}
		str += std::string(type) + " ";
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
void func_args_cover(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		tprint("T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	} else if (atype == CEXP) {
		tprint("const T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	} else if (atype == MUL) {
		tprint("T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	} else if (atype == CMUL) {
		tprint("const T* %s = %s_st.data();\n", arg, arg);
		tprint("T %s_r = %s_st.scale();\n", arg, arg);
	}
}

template<class ...Args>
void func_args_cover(int P, const char* arg, arg_type atype, Args&& ...args) {
	func_args_cover(P, arg, atype, 1);
	func_args_cover(P, std::forward<Args>(args)...);
}

template<class ... Args>
std::string func_header(const char* func, int P, bool pub, bool calcpot, bool scale, bool flags, std::string head, Args&& ...args) {
	static std::set<std::string> igen;
	std::string func_name = std::string(func);
	if (nopot && calcpot) {
		func_name += "_nopot";
	}
	std::string file_name = func_name + "_" + type + "_P" + std::to_string(P) + period_name() + (cuda ? ".cu" : ".cpp");
	func_name = "void " + func_name;
	func_name += "(" + func_args(P, std::forward<Args>(args)..., 0);
	auto func_name2 = func_name;
	if (flags) {
		func_name += ", int flags = 0)";
		func_name2 += ", int flags)";
		if (!pub) {
			func_name = func_name2;
		}
	} else {
		func_name += ")";
		func_name2 += ")";
	}
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
	tprint("#include <stdio.h>\n");
	tprint("#include \"%s\"\n", header.c_str());
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
	if (flags && scale) {
		tprint("const bool scaling = !(flags & SFMM_IGNORE_SCALING);\n");
	}
	func_args_cover(P, std::forward<Args>(args)..., 0);
	return file_name;
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

void z_rot(int P, const char* name, bool noevenhi, bool exclude, bool noimaghi) {
	tprint("rx[0] = cosphi;\n");
	tprint("ry[0] = sinphi;\n");
	for (int m = 1; m < P; m++) {
		tprint("rx[%i] = rx[%i] * cosphi - ry[%i] * sinphi;\n", m, m - 1, m - 1);
		tprint("ry[%i] = detail::fma(rx[%i], sinphi, ry[%i] * cosphi);\n", m, m - 1, m - 1);
	}
	int mmin = 1;
	bool initR = true;

	for (int m = 1; m <= P; m++) {
		tprint_new_chain();
		for (int l = m; l <= P; l++) {
			if (noevenhi && l == P) {
				if ((((P + l) / 2) % 2 == 1) ? m % 2 == 0 : m % 2 == 1) {
					continue;
				}
			}
			if (exclude && l == P && m % 2 == 1) {
				tprint_chain("%s[%i] = -%s[%i] * ry[%i];\n", name, index(l, m), name, index(l, -m), m - 1);
				tprint_chain("%s[%i] *= rx[%i];\n", name, index(l, -m), m);
			} else if ((exclude && l == P && m % 2 == 0) || (noimaghi && l == P)) {
				tprint_chain("%s[%i] = %s[%i] * ry[%i];\n", name, index(l, -m), name, index(l, m), m - 1);
				tprint_chain("%s[%i] *= rx[%i];\n", name, index(l, m), m - 1);
			} else {
				if (noevenhi && ((l >= P - 1 && m % 2 == P % 2))) {
					tprint_chain("%s[%i] = %s[%i] * ry[%i];\n", name, index(l, -m), name, index(l, m), m - 1);
					tprint_chain("%s[%i] *= rx[%i];\n", name, index(l, m), m - 1);
				} else {
					tprint_chain("tmp%i = %s[%i];\n", current_chain, name, index(l, m));
					tprint_chain("%s[%i] = %s[%i] * rx[%i] - %s[%i] * ry[%i];\n", name, index(l, m), name, index(l, m), m - 1, name, index(l, -m), m - 1);
					tprint_chain("%s[%i] = detail::fma(tmp%i, ry[%i], %s[%i] * rx[%i]);\n", name, index(l, -m), current_chain, m - 1, name, index(l, -m), m - 1);
				}
			}

		}
	}
	tprint_flush_chains();
}

void m2l(int P, int Q, const char* mname, const char* lname) {
	tprint("A[0] = rinv;\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = rinv * A[%i];\n", n, n - 1);
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, factorial(n));
	}
	for (int n = nopot; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			tprint_new_chain();
			bool pfirst = true;
			bool nfirst = true;
			const int maxk = std::min(P - n, P - 1);
			bool looped = true;
			for (int k = m; k <= maxk; k++) {
				looped = true;
				if (pfirst) {
					pfirst = false;
					tprint_chain("%s[%i] = %s[%i] * A[%i];\n", lname, index(n, m), mname, index(k, m), n + k);
				} else {
					tprint_chain("%s[%i] = detail::fma(%s[%i], A[%i], %s[%i]);\n", lname, index(n, m), mname, index(k, m), n + k, lname, index(n, m));
				}
				if (m != 0) {
					if (nfirst) {
						nfirst = false;
						tprint_chain("%s[%i] = %s[%i] * A[%i];\n", lname, index(n, -m), mname, index(k, -m), n + k);
					} else {
						tprint_chain("%s[%i] = detail::fma(%s[%i], A[%i], %s[%i]);\n", lname, index(n, -m), mname, index(k, -m), n + k, lname, index(n, -m));
					}
				}
			}
			if (m % 2 != 0) {
				if (!pfirst) {
					tprint_chain("%s[%i] = -%s[%i];\n", lname, index(n, m), lname, index(n, m));
				}
				if (!nfirst) {
					tprint_chain("%s[%i] = -%s[%i];\n", lname, index(n, -m), lname, index(n, -m));
				}
			}
		}
	}
	tprint_flush_chains();

}

void xz_swap(int P, const char* name, bool inv, bool m_restrict, bool l_restrict, bool noevenhi) {
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
		std::vector<std::vector<std::pair<double, int>>>ops(2 * n + 1);
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
							tprint_chain("%s[%i] = detail::fma(TCAST(%.20e), A[%i], %s[%i]);\n", name, index(n, m - n), ops[m][l].first, ops[m][l].second, name,
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
						tprint_chain("%s[%i] = detail::fma(TCAST(%.20e), tmp%i, %s[%i]);\n", name, index(n, m - n), ops[m][l].first, current_chain, name,
								index(n, m - n));
					}
				}
				l += len - 1;
			}
		}
		tprint_flush_chains();
	}
}

void greens_body(int P, const char* M = nullptr) {
	init_real("r2");
	init_real("r2inv");
	init_real("ax");
	init_real("ay");
	tprint("r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	if (M) {
		tprint("r2inv = %s / r2;\n", M);
	} else {
		tprint("r2inv = TCAST(1) / r2;\n");
	}
	tprint("O[0] = detail::rsqrt(r2);\n");
	if (M) {
		tprint("O[0] *= %s;\n", M);
	}
	if (periodic) {
		tprint("O_st.trace2() = TCAST(0);\n");
	}
	tprint("x *= r2inv;\n");
	tprint("y *= r2inv;\n");
	tprint("z *= r2inv;\n");
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
			tprint("O[%i] = y * O[0];\n", index(m, -m));
		} else if (m > 0) {
			tprint("ax = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("ay = O[%i] * TCAST(%i);\n", index(m - 1, -(m - 1)), 2 * m - 1);
			tprint("O[%i] = x * ax - y * ay;\n", index(m, m));
			tprint("O[%i] = detail::fma(y, ax, x * ay);\n", index(m, -m));
		}
		if (m + 1 <= P) {
			tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
			if (m != 0) {
				tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, -m), 2 * m + 1, index(m, -m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = TCAST(%i) * z;\n", 2 * n - 1);
				tprint("ay = TCAST(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = detail::fma(ax, O[%i], ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
				tprint("O[%i] = detail::fma(ax, O[%i], ay * O[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));
				} else {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - TCAST(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
				}

			}
		}
	}
}

void m2lg_body(int P, int Q) {

	for (int n = nopot; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			tprint_new_chain();
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
						ASPRINTF(&gxstr, "O[%i]", index(n + k, m + l));
						ASPRINTF(&gystr, "O[%i]", index(n + k, -m - l));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							ASPRINTF(&gxstr, "O[%i]", index(n + k, -m - l));
							ASPRINTF(&gystr, "O[%i]", index(n + k, m + l));
							gysgn = -1;
						} else {
							ASPRINTF(&gxstr, "O[%i]", index(n + k, -m - l));
							ASPRINTF(&gystr, "O[%i]", index(n + k, m + l));
							gxsgn = -1;
						}
					} else {
						greal = true;
						ASPRINTF(&gxstr, "O[%i]", index(n + k, 0));
					}
					if (l > 0) {
						ASPRINTF(&mxstr, "M[%i]", index(k, l));
						ASPRINTF(&mystr, "M[%i]", index(k, -l));
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							ASPRINTF(&mxstr, "M[%i]", index(k, -l));
							ASPRINTF(&mystr, "M[%i]", index(k, l));
						} else {
							ASPRINTF(&mxstr, "M[%i]", index(k, -l));
							ASPRINTF(&mystr, "M[%i]", index(k, l));
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						ASPRINTF(&mxstr, "M[%i]", index(k, 0));
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
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}

		}
	}
	tprint_flush_chains();
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
	return abs(a) < 1e-10;
}

std::string P2L(int P) {
	auto fname = func_header("P2L", P, true, false, true, true, "", "L", EXP, "M", LIT, "x", LIT, "y", LIT, "z", LIT);
	init_real("tmp1");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("expansion%s<%s,%i> O_st(L_st.scale());\n", period_name(), type.c_str(), P);
	tprint("T* O = O_st.data();\n");
	greens_body(P, "M");
	tprint("L_st += O_st;\n");
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string greens(int P) {
	auto fname = func_header("greens", P, true, false, false, true, "", "O", EXP, "x", LIT, "y", LIT, "z", LIT);
	greens_body(P);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}");
	tprint("\n");
	return fname;
}

std::string greens_xz(int P) {
	auto fname = func_header("greens_xz", P, false, false, false, true, "", "O", EXP, "x", LIT, "z", LIT, "r2inv", LIT);
	init_real("ax");
	init_real("ay");
	tprint("O[0] = detail::sqrt(r2inv);\n");
	if (periodic) {
		tprint("O_st.trace2() = TCAST(0);\n");
	}
	tprint("x *= r2inv;\n");
	tprint("z *= r2inv;\n");
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m == 1) {
			tprint("O[%i] = x * O[0];\n", index(m, m));
		} else if (m > 0) {
			tprint("ax = O[%i] * TCAST(%i);\n", index(m - 1, m - 1), 2 * m - 1);
			tprint("O[%i] = x * ax;\n", index(m, m));
		}
		if (m + 1 <= P) {
			tprint("O[%i] = TCAST(%i) * z * O[%i];\n", index(m + 1, m), 2 * m + 1, index(m, m));
		}
		for (int n = m + 2; n <= P; n++) {
			if (m != 0) {
				tprint("ax = TCAST(%i) * z;\n", 2 * n - 1);
				tprint("ay = TCAST(-%i) * r2inv;\n", (n - 1) * (n - 1) - m * m);
				tprint("O[%i] = detail::fma(ax, O[%i], ay * O[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
			} else {
				if ((n - 1) * (n - 1) - m * m == 1) {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), index(n - 2, m));

				} else {
					tprint("O[%i] = (TCAST(%i) * z * O[%i] - TCAST(%i) * r2inv * O[%i]);\n", index(n, m), 2 * n - 1, index(n - 1, m), (n - 1) * (n - 1) - m * m,
							index(n - 2, m));
				}
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2L_ewald(int P) {
	auto fname = func_header("M2L_ewald", P, true, false, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	tprint("expansion%s<%s, %i> G_st;\n", period_name(), type.c_str(), P);
	tprint("T* G = G_st.data();\n", type.c_str(), P);
	tprint("auto M_st = M0_st;\n");
	tprint("expansion%s<%s,%i> L_st;\n", period_name(), type.c_str(), P);
	tprint("auto* M = M_st.data();\n");
	tprint("auto* L = M_st.data();\n");
	tprint("L_st.init();\n");

	tprint("if( scaling ) {\n");
	indent();
	tprint("M_st.rescale(TCAST(1));\n");
	deindent();
	tprint("}\n");

	tprint("greens_ewald%s(G_st, x, y, z, flags);\n", nopot ? "_nopot" : "");
	tprint("M2LG%s(L_st, M_st, G_st, flags);\n", nopot ? "_nopot" : "");
	tprint("if( scaling ) {\n");
	indent();
	tprint("L_st.rescale(L0_st.scale());\n");
	deindent();
	tprint("}\n");
	tprint("L0_st += L_st;\n");
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2LG(int P, int Q) {
	auto fname = func_header("M2LG", P, true, true, false, true, "", "L", EXP, "M", CMUL, "O", EXP);
	m2lg_body(P, Q);
	if (!nopot && P > 2 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-0.5) * O_st.trace2(), M_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", index(1, -1), index(1, -1), index(1, -1));
		tprint("L[%i] -= O_st.trace2() * M[%i];\n", index(1, +0), index(1, +0), index(1, +0));
		tprint("L[%i] = detail::fma(TCAST(-2) * O_st.trace2(), M[%i], L[%i]);\n", index(1, +1), index(1, +1), index(1, +1));
		tprint("L_st.trace2() = detail::fma(TCAST(-0.5) * O_st.trace2(), M[%i], L_st.trace2());\n", index(0, 0));
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string greens_ewald(int P, double alpha) {
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
	auto fname = func_header("greens_ewald", P, true, true, false, true, "", "G", EXP, "x0", LIT, "y0", LIT, "z0", LIT);
	const auto c = tprint_on;
	tprint("expansion%s<%s, %i> Gr_st;\n", period_name(), type.c_str(), P);
	tprint("T* Gr = Gr_st.data();\n", type.c_str(), P);
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
	tprint("r2 = detail::fma(x0, x0, detail::fma(y0, y0, z0 * z0));\n");
	tprint("r = detail::sqrt(r2);\n");
	tprint("greens(Gr_st, x0, y0, z0, flags);\n");
	tprint("xxx = TCAST(%.20e) * r;\n", alpha);
	tprint("detail::erfcexp(xxx, &gam1, &exp0);\n");
	tprint("gam1 *= TCAST(%.20e);\n", sqrt(M_PI));
	tprint("xfac = TCAST(%.20e) * r2;\n", alpha * alpha);
	tprint("xpow = TCAST(%.20e) * r;\n", alpha);
	double gam0inv = 1.0 / sqrt(M_PI);
	tprint("sw = r2 > TCAST(0);\n");
	for (int l = 0; l <= P; l++) {
		tprint("gam = gam1 * TCAST(%.20e);\n", gam0inv);
		for (int m = -l; m <= l; m++) {
			tprint("G[%i] = sw * (TCAST(%.1e) - gam) * Gr[%i];\n", index(l, m), nonepow<double>(l), index(l, m));
		}
		if (l == 0) {
			tprint("G[%i] += (TCAST(1) - sw) * TCAST(%.20e);\n", index(0, 0), (2) * alpha / sqrt(M_PI));
		}
		gam0inv *= 1.0 / -(l + 0.5);
		if (l != P) {
			tprint("gam1 = detail::fma(TCAST(%.20e), gam1, xpow * exp0);\n", l + 0.5);
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
				std::string xstr = "x0";
				if (ix != 0) {
					xstr += std::string(" ") + (ix < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(ix)) + ")";
				}
				std::string ystr = "y0";
				if (iy != 0) {
					ystr += std::string(" ") + (iy < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(iy)) + ")";
				}
				std::string zstr = "z0";
				if (iz != 0) {
					zstr += std::string(" ") + (iz < 0 ? "+" : "-") + " TCAST(" + std::to_string(abs(iz)) + ")";
				}
				tprint("detail::greens_ewald_real<%s, %i, %i>(G_st, %s, %s, %s, flags);\n", type.c_str(), P, lround(alpha * 100), xstr.c_str(), ystr.c_str(),
						zstr.c_str());
			}
		}
	}

	for (int hx = -H; hx <= H; hx++) {
		if (hx) {
			if (abs(hx) == 1) {
				tprint("x2 = %cx0;\n", hx < 0 ? '-' : ' ');
			} else {
				tprint("x2 = TCAST(%i) * x0;\n", hx);
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
				} else {
					tprint("x2y2 = detail::fma(TCAST(%i), y0, x2);\n", hy);
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
					} else {
						tprint("hdotx = detail::fma(TCAST(%i), z0, x2y2);\n", hz);
					}
				} else {
					tprint("hdotx = x2y2;\n", hz);
				}
				tprint("phi = TCAST(%.20e) * hdotx;\n", 2.0 * M_PI);
				tprint("detail::sincos(phi, &%s, &%s);\n", sinname(hx, hy, hz).c_str(), cosname(hx, hy, hz).c_str());
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
					tprint_chain("G[%i] = detail::fma(TCAST(+%.20e), tmp%i, G[%i]);\n", ii, sgn * j->first, current_chain, ii);
				} else {
					tprint_chain("G[%i] = detail::fma(TCAST(%.20e), tmp%i, G[%i]);\n ", ii, sgn * j->first, current_chain, ii);
				}
			}
		}
	}
	tprint_flush_chains();
	tprint("G_st.trace2() = TCAST(%.20e);\n", (4.0 * M_PI / 3.0));
	if (!nopot) {
		tprint("G[%i] += TCAST(%.20e);\n", index(0, 0), M_PI / (alpha * alpha));
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;

}

std::string M2L_norot(int P, int Q) {
	std::string fname;
	if (Q > 1) {
		fname = func_header("M2L", P, true, true, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	} else {
		fname = func_header("M2P", P, true, true, true, true, "", "f", FORCE, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	}
	init_real("tmp1");
	if (Q == 1) {
		init_real("rinv");
		init_real("r2inv");
	}
	tprint("expansion%s<%s, %i> O_st;\n", period_name(), type.c_str(), P);
	tprint("T* O = O_st.data();\n", type.c_str(), P);
	tprint("auto M_st = M0_st;\n");
	tprint("auto* M = M_st.data();\n");
	if (Q > 1) {
		tprint("expansion%s<%s, %i>& L_st = L0_st;\n", period_name(), type.c_str(), P);
		tprint("T* L = L_st.data();\n", type.c_str(), P);
		tprint("if( scaling ) {\n");
		indent();
		tprint("M_st.rescale(L_st.scale());\n");
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
		deindent();
		tprint("}\n");
	} else {
		init_reals("L", exp_sz(Q));
		tprint("L[0] = TCAST(0);\n");
		tprint("L[1] = TCAST(0);\n");
		tprint("L[2] = TCAST(0);\n");
		tprint("L[3] = TCAST(0);\n");
		tprint("if( scaling ) {\n");
		indent();
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
		deindent();
		tprint("}\n");
	}
	tprint("greens(O_st, x, y, z, flags);\n");
	m2lg_body(P, Q);
	if (Q == 1) {
		tprint("rinv = TCAST(1) / M_st.scale();\n");
		tprint("r2inv = rinv * rinv;\n");
		if (!nopot) {
			tprint("f.potential = detail::fma(L[0], rinv, f.potential);\n");
		}
		tprint("f.force[0] -= L[3] * r2inv;\n");
		tprint("f.force[1] -= L[1] * r2inv;\n");
		tprint("f.force[2] -= L[2] * r2inv;\n");
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2L_rot1(int P, int Q) {
	std::string fname;
	if (Q > 1) {
		fname = func_header("M2L", P, true, true, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	} else {
		fname = func_header("M2P", P, true, true, true, true, "", "f", FORCE, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	}
	init_real("R2");
	init_real("Rzero");
	init_real("r2");
	init_real("rzero");
	init_real("tmp1");
	init_real("Rinv");
	init_real("r2inv");
	init_real("R");
	init_real("cosphi");
	init_real("sinphi");
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_reals("L", exp_sz(Q));
	tprint("expansion%s<%s, %i> O_st;\n", period_name(), type.c_str(), P);
	tprint("T* O = O_st.data();\n", type.c_str(), P);
	tprint("auto M_st = M0_st;\n");
	tprint("auto* M = M_st.data();\n");
	if (Q == 1) {
		init_real("rinv");
	}
	tprint("if( scaling ) {\n");
	indent();
	if (Q > 1) {
		tprint("M_st.rescale(L0_st.scale());\n");
	}
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");

	tprint("R2 = detail::fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("r2 = detail::fma(z, z, R2);\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	tprint("R = TCAST(1) / Rinv;\n");
	tprint("r2inv = TCAST(1) / (r2 + rzero);\n");
	tprint("cosphi = detail::fma(x, Rinv, Rzero);\n");
	tprint("sinphi = -y * Rinv;\n");
	z_rot(P - 1, "M", false, false, false);
	tprint("detail::greens_xz(O_st, R, z, r2inv, flags);\n");
	const auto oindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int n = nopot; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			tprint_new_chain();
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
							ASPRINTF(&gxstr, "O[%i]", oindex(n + k, m + l));
						} else if (m + l < 0) {
							if (abs(m + l) % 2 == 0) {
								ASPRINTF(&gxstr, "O[%i]", oindex(n + k, -m - l));
							} else {
								ASPRINTF(&gxstr, "O[%i]", oindex(n + k, -m - l));
								gxsgn = -1;
							}
						} else {
							ASPRINTF(&gxstr, "O[%i]", oindex(n + k, 0));
						}
						if (l > 0) {
							ASPRINTF(&mxstr, "M[%i]", index(k, l));
							ASPRINTF(&mystr, "M[%i]", index(k, -l));
							mysgn = -1;
						} else if (l < 0) {
							if (l % 2 == 0) {
								ASPRINTF(&mxstr, "M[%i]", index(k, -l));
								ASPRINTF(&mystr, "M[%i]", index(k, l));
							} else {
								ASPRINTF(&mxstr, "M[%i]", index(k, -l));
								ASPRINTF(&mystr, "M[%i]", index(k, l));
								mxsgn = -1;
								mysgn = -1;
							}
						} else {
							mreal = true;
							ASPRINTF(&mxstr, "M[%i]", index(k, 0));
						}
						if (!mreal) {
							if (mxsgn * gxsgn == sgn) {
								if (pfirst) {
									tprint_chain("L[%i] = %s * %s;\n", index(n, m), mxstr, gxstr);
									pfirst = false;
								} else {
									tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), mxstr, gxstr, index(n, m));
								}
							}
							if (mysgn * gxsgn == sgn) {
								if (m > 0) {
									if (nfirst) {
										tprint_chain("L[%i] = %s * %s;\n", index(n, -m), mystr, gxstr);
										nfirst = false;
									} else {
										tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), mystr, gxstr, index(n, -m));
									}
								}
							}
						} else {
							if (mxsgn * gxsgn == sgn) {
								if (pfirst) {
									tprint_chain("L[%i] = %s * %s;\n", index(n, m), mxstr, gxstr);
									pfirst = false;
								} else {
									tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), mxstr, gxstr, index(n, m));
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
					tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				}
				if (!nfirst && sgn == -1) {
					tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				}

			}
		}
	}
	tprint_flush_chains();
	tprint("sinphi = -sinphi;\n");
	z_rot(Q, "L", false, false, Q == P);
	if (Q > 1) {
		for (int n = 0; n < exp_sz(Q); n++) {
			tprint("L0[%i] += L[%i];\n", n, n);
		}
	} else {
		tprint("rinv = TCAST(1) / M_st.scale();\n");
		tprint("r2inv = rinv * rinv;\n");
		if (!nopot) {
			tprint("f.potential = detail::fma(L[0], rinv, f.potential);\n");
		}
		tprint("f.force[0] -= L[3] * r2inv;\n");
		tprint("f.force[1] -= L[1] * r2inv;\n");
		tprint("f.force[2] -= L[2] * r2inv;\n");
	}

	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2L_rot2(int P, int Q) {
	std::string fname;
	if (Q > 1) {
		fname = func_header("M2L", P, true, true, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	} else {
		fname = func_header("M2P", P, true, true, true, true, "", "f", FORCE, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	}
	init_reals("L", exp_sz(Q));
	init_reals("A", 2 * P + 1);
	init_real("R2");
	init_real("Rzero");
	init_real("r2");
	init_real("rzero");
	init_real("tmp1");
	init_real("Rinv");
	if (Q == 1) {
		init_real("r2inv");
	}
	init_real("R");
	init_real("cosphi");
	init_real("sinphi");
	init_real("cosphi0");
	init_real("sinphi0");
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("r2przero");
	init_real("rinv");
	tprint("auto M_st = M0_st;\n");
	tprint("auto* M = M_st.data();\n");
	tprint("if( scaling ) {\n");
	indent();
	if (Q > 1) {
		tprint("M_st.rescale(L0_st.scale());\n");
	}
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	tprint("R = TCAST(1) / Rinv;\n");
	tprint("r2 = (detail::fma(z, z, R2));\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("r2przero = (r2 + rzero);\n");
	tprint("rinv = detail::rsqrt(r2przero);\n");

	tprint("cosphi = y * Rinv;\n");
	tprint("sinphi = detail::fma(x, Rinv, Rzero);\n");
	z_rot(P - 1, "M", false, false, false);
	xz_swap(P - 1, "M", false, false, false, false);

	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = detail::fma(z, rinv, rzero);\n");
	tprint("sinphi = -R * rinv;\n");
	z_rot(P - 1, "M", false, false, false);
	xz_swap(P - 1, "M", false, true, false, false);
	m2l(P, Q, "M", "L");
	xz_swap(Q, "L", true, false, true, false);

	tprint("sinphi = -sinphi;\n");
	z_rot(Q, "L", true, false, false);
	xz_swap(Q, "L", true, false, false, true);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	z_rot(Q, "L", false, true, false);
	if (Q > 1) {
		for (int n = 0; n < exp_sz(Q); n++) {
			tprint("L0[%i] += L[%i];\n", n, n);
		}
	} else {
		tprint("rinv = TCAST(1) / M_st.scale();\n");
		tprint("r2inv = rinv * rinv;\n");
		if (!nopot) {
			tprint("f.potential = detail::fma(L[0], rinv, f.potential);\n");
		}
		tprint("f.force[0] -= L[3] * r2inv;\n");
		tprint("f.force[1] -= L[1] * r2inv;\n");
		tprint("f.force[2] -= L[2] * r2inv;\n");
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string regular_harmonic(int P) {
	auto fname = func_header("regular_harmonic", P, false, false, false, true, "", "Y", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("ax");
	init_real("ay");
	init_real("r2");
	tprint("r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	tprint("Y[0] = TCAST(1);\n");
	if (periodic) {
		tprint("Y_st.trace2() = r2;\n");
	}
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay = Y[%i] * TCAST(%.20e);\n", index(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax - y * ay;\n", index(m, m));
				tprint("Y[%i] = detail::fma(y, ax, x * ay);\n", index(m, -m));
			} else {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
				tprint("Y[%i] = y * ax;\n", index(m, -m));
			}
		}
		if (m + 1 <= P) {
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, -m), index(m, -m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			tprint("ax =  TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("Y[%i] = detail::fma(ax, Y[%i], ay * Y[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
			if (m != 0) {
				tprint("Y[%i] = detail::fma(ax, Y[%i], ay * Y[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
			}
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

void cuda_header() {
	tprint("const int& tid = threadIdx.x;\n");
	tprint("const int& bsz = blockDim.x;\n");
}

std::string regular_harmonic_xz(int P) {
	auto fname = func_header("regular_harmonic_xz", P, false, false, false, true, "", "Y", EXP, "x", LIT, "z", LIT);
	init_real("ax");
	init_real("ay");
	init_real("r2");
	tprint("r2 = detail::fma(x, x, z * z);\n");
	tprint("Y[0] = TCAST(1);\n");
	if (periodic) {
		tprint("Y_st.trace2() = r2;\n");
	}
	const auto index = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
			if (m - 1 > 0) {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
			} else {
				tprint("ax = Y[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("Y[%i] = x * ax;\n", index(m, m));
			}
		}
		if (m + 1 <= P) {
			if (m == 0) {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
			} else {
				tprint("Y[%i] = z * Y[%i];\n", index(m + 1, m), index(m, m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			tprint("ax =  TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("Y[%i] = detail::fma(ax, Y[%i], ay * Y[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2M_norot(int P) {
	auto fname = func_header("M2M", P + 1, true, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	init_real("tmp1");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("expansion%s<%s, %i> Y_st;\n", period_name(), type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z, flags);\n");
	if (P > 1 && !nopot && periodic) {
		tprint("M_st.trace2() = detail::fma(TCAST(-4) * x, M[%i], M_st.trace2());\n", index(1, 1));
		tprint("M_st.trace2() = detail::fma(TCAST(-4) * y, M[%i], M_st.trace2());\n", index(1, -1));
		tprint("M_st.trace2() = detail::fma(TCAST(-2) * z, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = detail::fma(x * x, M[%i], M_st.trace2());\n", index(0, 0));
		tprint("M_st.trace2() = detail::fma(y * y, M[%i], M_st.trace2());\n", index(0, 0));
		tprint("M_st.trace2() = detail::fma(z * z, M[%i], M_st.trace2());\n", index(0, 0));
	}
	for (int n = P; n >= 0; n--) {

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
						ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						ASPRINTF(&mxstr, "M[%i]", index(n - k, 0));
					}
					if (l > 0) {
						ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
						ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gysgn = -1;
						} else {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
						}
					} else {
						ASPRINTF(&gxstr, "Y[%i]", index(k, 0));
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
				tprint_chain("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("M[%i] = -M[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("M[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("M[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
		tprint_flush_chains();
	}
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2M_rot1(int P) {

	auto fname = func_header("M2M", P + 1, true, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	tprint("expansion%s<%s, %i> Y_st;\n", period_name(), type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("tmp1");
	init_real("R2");
	init_real("Rzero");
	init_real("cosphi");
	init_real("sinphi");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	tprint("R = TCAST(1) / Rinv;\n");
	tprint("cosphi = detail::fma(x, Rinv, Rzero);\n");
	tprint("sinphi = -y * Rinv;\n");
	z_rot(P, "M", false, false, false);
	if (P > 1 && !nopot && periodic) {
		tprint("M_st.trace2() = detail::fma(TCAST(-4) * R, M[%i], M_st.trace2());\n", index(1, 1));
		tprint("M_st.trace2() = detail::fma(TCAST(-2) * z, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = detail::fma(R * R, M[%i], M_st.trace2());\n", index(0, 0));
		tprint("M_st.trace2() = detail::fma(z * z, M[%i], M_st.trace2());\n", index(0, 0));
	}
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("detail::regular_harmonic_xz(Y_st, -R, -z, flags);\n");
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
						ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
						ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
					} else if (m - l < 0) {
						if (abs(m - l) % 2 == 0) {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mysgn = -1;
						} else {
							ASPRINTF(&mxstr, "M[%i]", index(n - k, abs(m - l)));
							ASPRINTF(&mystr, "M[%i]", index(n - k, -abs(m - l)));
							mxsgn = -1;
						}
					} else {
						ASPRINTF(&mxstr, "M[%i]", index(n - k, 0));
					}
					ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
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
					tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint("M[%i] = -M[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint("M[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint("M[%i] = detail::fma(%s, %s, M[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
	}
	tprint("sinphi = -sinphi;\n");
	z_rot(P, "M", false, false, false);
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2M_rot2(int P) {
	auto fname = func_header("M2M", P + 1, true, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	init_reals("A", 2 * P + 1);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("R2");
	init_real("Rzero");
	init_real("tmp1");
	init_real("r2");
	init_real("rzero");
	init_real("r");
	init_real("rinv");
	init_real("cosphi");
	init_real("cosphi0");
	init_real("sinphi");
	init_real("sinphi0");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	tprint("r2 = detail::fma(z, z, R2);\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = detail::rsqrt(tmp1);\n");
	tprint("R = TCAST(1) / Rinv;\n");
	tprint("r = TCAST(1) / rinv;\n");
	tprint("cosphi = y * Rinv;\n");
	tprint("sinphi = detail::fma(x, Rinv, Rzero);\n");
	z_rot(P, "M", false, false, false);
	xz_swap(P, "M", false, false, false, false);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = detail::fma(z, rinv, rzero);\n");
	tprint("sinphi = -R * rinv;\n");
	z_rot(P, "M", false, false, false);
	xz_swap(P, "M", false, false, false, false);
	if (P > 1 && !nopot && periodic) {
		tprint("M_st.trace2() = detail::fma(TCAST(-2) * r, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = detail::fma(r * r, M[%i], M_st.trace2());\n", index(0, 0));
	}
	tprint("A[0] = TCAST(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = -r * A[%i];\n", n, n - 1);
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, 1.0 / factorial(n));
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
				tprint("M[%i] = detail::fma(M[%i], A[%i], M[%i]);\n", index(n, m), index(n - k, m), k, index(n, m));
				if (m > 0) {
					tprint("M[%i] = detail::fma(M[%i], A[%i], M[%i]);\n", index(n, -m), index(n - k, -m), k, index(n, -m));
				}
			}

		}
	}
	xz_swap(P, "M", false, false, false, false);
	tprint("sinphi = -sinphi;\n");
	z_rot(P, "M", false, false, false);
	xz_swap(P, "M", false, false, false, false);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	z_rot(P, "M", false, false, false);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string L2L_norot(int P) {
	auto fname = func_header("L2L", P, true, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("tmp1");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("expansion%s<%s, %i> Y_st;\n", period_name(), type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z, flags);\n");

	for (int n = nopot; n <= P; n++) {
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
						ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
						ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						ASPRINTF(&gxstr, "Y[%i]", index(k, 0));
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
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
		tprint_flush_chains();
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * x, L_st.trace2(), L[%i]);\n", index(1, 1), index(1, 1));
		tprint("L[%i] = detail::fma(TCAST(-2) * y, L_st.trace2(), L[%i]);\n", index(1, -1), index(1, -1));
		tprint("L[%i] = detail::fma(TCAST(-2) * z, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L[%i] = detail::fma(x * x, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = detail::fma(y * y, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = detail::fma(z * z, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
		}
	}

	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string L2L_rot1(int P) {

	auto fname = func_header("L2L", P, true, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("R2");
	init_real("Rzero");
	init_real("tmp1");
	init_real("cosphi");
	init_real("sinphi");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");

	tprint("R2 = detail::fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	tprint("R = TCAST(1) / Rinv;\n");
	tprint("cosphi = detail::fma(x, Rinv, Rzero);\n");
	tprint("sinphi = -y * Rinv;\n");
	z_rot(P, "L", false, false, false);
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("expansion%s<%s, %i> Y_st;\n", period_name(), type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	tprint("detail::regular_harmonic_xz(Y_st, -R, -z, flags);\n");
	int sw = 1;
	for (int n = nopot; n <= P; n++) {
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
					ASPRINTF(&gxstr, "Y[%i]", yindex(k, abs(l)));
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
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("L[%i] = detail::fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
		tprint_flush_chains();
	}
	tprint("sinphi = -sinphi;\n");
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * R, L_st.trace2(), L[%i]);\n", index(1, 1), index(1, 1));
		tprint("L[%i] = detail::fma(TCAST(-2) * z, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L[%i] = detail::fma(R2, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = detail::fma(z * z, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
		}
	}
	z_rot(P, "L", false, false, false);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string L2L_rot2(int P) {
	auto fname = func_header("L2L", P, true, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_reals("A", 2 * P + 1);
	tprint("T rx[%i];\n", P);
	tprint("T ry[%i];\n", P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("R2");
	init_real("Rzero");
	init_real("tmp1");
	init_real("r2");
	init_real("rzero");
	init_real("r");
	init_real("rinv");
	init_real("cosphi");
	init_real("cosphi0");
	init_real("sinphi");
	init_real("sinphi0");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("R2 = detail::fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = detail::rsqrt(tmp1);\n");
	tprint("r2 = detail::fma(z, z, R2);\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = detail::rsqrt(tmp1);\n");
	tprint("R = TCAST(1) / Rinv;\n");
	tprint("r = TCAST(1) / rinv;\n");
	tprint("cosphi = y * Rinv;\n");
	tprint("sinphi = detail::fma(x, Rinv, Rzero);\n");
	z_rot(P, "L", false, false, false);
	xz_swap(P, "L", true, false, false, false);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = detail::fma(z, rinv, rzero);\n");
	tprint("sinphi = -R * rinv;\n");
	z_rot(P, "L", false, false, false);
	xz_swap(P, "L", true, false, false, false);
	tprint("A[0] = TCAST(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = -r * A[%i];\n", n, n - 1);
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, 1.0 / factorial(n));
	}
	int sw = 1;
	for (int n = nopot; n <= P; n++) {
		for (int m = 0; m <= n; m++) {
			tprint_new_chain();
			for (int k = 1; k <= P - n; k++) {
				if (abs(m) > n + k) {
					continue;
				}
				if (-abs(m) < -(k + n)) {
					continue;
				}
				tprint_chain("L[%i] = detail::fma(L[%i], A[%i], L[%i]);\n", index(n, m), index(n + k, m), k, index(n, m));
				if (m > 0) {
					tprint_chain("L[%i] = detail::fma(L[%i], A[%i], L[%i]);\n", index(n, -m), index(n + k, -m), k, index(n, -m));
				}
			}
		}
		tprint_flush_chains();
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = detail::fma(TCAST(-2) * r, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L[%i] = detail::fma(r * r, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
		}
	}
	xz_swap(P, "L", true, false, false, false);
	tprint("sinphi = -sinphi;\n");
	z_rot(P, "L", false, false, false);
	xz_swap(P, "L", true, false, false, false);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	z_rot(P, "L", false, false, false);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string L2P(int P) {
	const char* fstr[4] = { "f.potential", "f.force[2]", "f.force[0]", "f.force[1]" };
	auto fname = func_header("L2P", P, true, true, true, true, "", "f0", FORCE, "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("rinv");
	init_real("r2inv");
	init_real("tmp1");
	tprint("force_type<%s> f;\n", type.c_str());
	tprint("expansion%s<%s, %i> Y_st;\n", period_name(), type.c_str(), P);
	tprint("T* Y = Y_st.data();\n", type.c_str(), P);
	if (!nopot) {
		tprint("f.potential = TCAST(0);\n");
	}
	tprint("f.force[0] = TCAST(0);\n");
	tprint("f.force[1] = TCAST(0);\n");
	tprint("f.force[2] = TCAST(0);\n");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / L_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
//tprint("expansion<T,1> L1;\n");
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z, flags);\n");

	for (int n = nopot; n <= 1; n++) {
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
						ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
						ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						gysgn = -1;
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
						} else {
							ASPRINTF(&gxstr, "Y[%i]", index(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", index(k, -abs(l)));
							gxsgn = -1;
							gysgn = -1;
						}
					} else {
						ASPRINTF(&gxstr, "Y[%i]", index(k, 0));
					}
					if (n > 0) {
						mxsgn = -mxsgn;
						mysgn = -mysgn;
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
				tprint_chain("%s = -%s;\n", fstr[index(n, m)], fstr[index(n, m)]);
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("%s = detail::fma(%s, %s, %s);\n", fstr[index(n, m)], neg_real[i].first.c_str(), neg_real[i].second.c_str(), fstr[index(n, m)]);
				}
				tprint_chain("%s = -%s;\n", fstr[index(n, m)], fstr[index(n, m)]);
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("%s -= %s * %s;\n", fstr[index(n, m)], neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("%s = detail::fma(%s, %s, %s);\n", fstr[index(n, m)], pos_real[i].first.c_str(), pos_real[i].second.c_str(), fstr[index(n, m)]);
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("%s = -%s;\n", fstr[index(n, -m)], fstr[index(n, -m)]);
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("%s = detail::fma(%s, %s, %s);\n", fstr[index(n, -m)], neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), fstr[index(n, -m)]);
				}
				tprint_chain("%s = -%s;\n", fstr[index(n, -m)], fstr[index(n, -m)]);
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("%s -= %s * %s;\n", fstr[index(n, -m)], neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("%s = detail::fma(%s, %s, %s);\n", fstr[index(n, -m)], pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), fstr[index(n, -m)]);
			}
		}
	}
	tprint_flush_chains();
	if (P >= 1 && periodic) {
		tprint("%s = detail::fma(TCAST(2) * x, L_st.trace2(), %s);\n", fstr[index(1, 1)], fstr[index(1, 1)]);
		tprint("%s = detail::fma(TCAST(2) * y, L_st.trace2(), %s);\n", fstr[index(1, -1)], fstr[index(1, -1)]);
		tprint("%s = detail::fma(TCAST(2) * z, L_st.trace2(), %s);\n", fstr[index(1, 0)], fstr[index(1, 0)]);
		if (!nopot) {
			tprint("%s = detail::fma(x * x, L_st.trace2(), %s);\n", fstr[index(0, 0)], fstr[index(0, 0)]);
			tprint("%s = detail::fma(y * y, L_st.trace2(), %s);\n", fstr[index(0, 0)], fstr[index(0, 0)]);
			tprint("%s = detail::fma(z * z, L_st.trace2(), %s);\n", fstr[index(0, 0)], fstr[index(0, 0)]);
		}
	}
	tprint("if( scaling ) {\n");
	indent();
	tprint("rinv = TCAST(1) / L_st.scale();\n");
	tprint("r2inv = rinv * rinv;\n");
	if (!nopot) {
		tprint("f.potential *= rinv;\n");
	}
	tprint("f.force[0] *= r2inv;\n");
	tprint("f.force[1] *= r2inv;\n");
	tprint("f.force[2] *= r2inv;\n");
	deindent();
	tprint("}\n");
	if (!nopot) {
		tprint("f0.potential += f.potential;\n");
	}
	tprint("f0.force[0] += f.force[0];\n");
	tprint("f0.force[1] += f.force[1];\n");
	tprint("f0.force[2] += f.force[2];\n");
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");

	return fname;
}

void scaling(int P) {
}

std::string P2M(int P) {
	tprint("\n");
	auto fname = func_header("P2M", P + 1, true, false, true, true, "", "M", MUL, "m", LIT, "x", LIT, "y", LIT, "z", LIT);
	init_real("ax");
	init_real("ay");
	init_real("r2");
	init_real("tmp1");
	tprint("x = -x;");
	tprint("y = -y;");
	tprint("z = -z;");
	tprint("if( scaling ) {\n");
	indent();
	tprint("tmp1 = TCAST(1) / M_st.scale();\n");
	tprint("x *= tmp1;\n");
	tprint("y *= tmp1;\n");
	tprint("z *= tmp1;\n");
	deindent();
	tprint("}\n");
	tprint("r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	tprint("M[0] = m;\n");
	if (!nopot && periodic) {
		tprint("M_st.trace2() = m * r2;\n");
	}
	for (int m = 0; m <= P; m++) {
		if (m > 0) {
//	M[index(m, m)] = M[index(m - 1, m - 1)] * R / TCAST(2 * m);
			if (m - 1 > 0) {
				tprint("ax = M[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("ay = M[%i] * TCAST(%.20e);\n", index(m - 1, -(m - 1)), 1.0 / (2.0 * m));
				tprint("M[%i] = x * ax - y * ay;\n", index(m, m));
				tprint("M[%i] = detail::fma(y, ax, x * ay);\n", index(m, -m));
			} else {
				tprint("ax = M[%i] * TCAST(%.20e);\n", index(m - 1, m - 1), 1.0 / (2.0 * m));
				tprint("M[%i] = x * ax;\n", index(m, m));
				tprint("M[%i] = y * ax;\n", index(m, -m));
			}
		}
		if (m + 1 <= P) {
			if (m == 0) {
				tprint("M[%i] = z * M[%i];\n", index(m + 1, m), index(m, m));
			} else {
				tprint("M[%i] = z * M[%i];\n", index(m + 1, m), index(m, m));
				tprint("M[%i] = z * M[%i];\n", index(m + 1, -m), index(m, -m));
			}
		}
		for (int n = m + 2; n <= P; n++) {
			const double inv = double(1) / (double(n * n) - double(m * m));
			tprint("ax =  TCAST(%.20e) * z;\n", inv * double(2 * n - 1));
			tprint("ay =  TCAST(%.20e) * r2;\n", -(double) inv);
			tprint("M[%i] = detail::fma(ax, M[%i], ay * M[%i]);\n", index(n, m), index(n - 1, m), index(n - 2, m));
			if (m != 0) {
				tprint("M[%i] = detail::fma(ax, M[%i], ay * M[%i]);\n", index(n, -m), index(n - 1, -m), index(n - 2, -m));
			}
		}
	}
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

void math_float() {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	tprint("namespace detail {\n");
	tprint("SFMM_PREFIX inline float fma(float a, float b, float c) {\n");
	indent();
	tprint("return fmaf(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("SFMM_PREFIX float rsqrt(float);\n");
	tprint("SFMM_PREFIX float sqrt(float);\n");
	tprint("SFMM_PREFIX void sincos(float, float*, float*);\n");
	tprint("SFMM_PREFIX void erfcexp(float, float*, float*);\n");
	tprint("}\n");
	fclose(fp);
	for (int cuda = 0; cuda < 2; cuda++) {
		if (cuda) {
			fp = fopen("./generated_code/src/math/math_float.cu", "wt");
		} else {
			fp = fopen("./generated_code/src/math/math_float.cpp", "wt");
		}
		tprint("\n");
		tprint("#include \"typecast_float.hpp\"\n");
		tprint("#include <math.h>\n");
		tprint("\n");
		tprint("namespace sfmm {\n");
		tprint("namespace detail {\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
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
		tprint("y *= fmaf(x, y * y, TCAST(-1.5));\n");
		tprint("y *= fmaf(x, y * y, TCAST(-1.5));\n");
		tprint("y *= TCAST(1.5) - x * y * y;\n");
		tprint("return y;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T sqrt(float x) {\n");
		indent();
		tprint("return TCAST(1) / rsqrt(x);\n");
		deindent();
		tprint("}\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("void sincos(T x, T* s, T* c) {\n");
		indent();
		tprint("V ssgn, j, i, k;\n");
		tprint("T x2;\n");
		tprint("ssgn = VCAST(((*((U*) &x) & UCAST(0x80000000)) >> UCAST(30)) - UCAST(1));\n");
		tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFF)));\n");
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

		if (cuda) {
			tprint("__device__\n");
		}
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
			if (cuda) {
				tprint("__device__\n");
			}
			tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
			indent();
			tprint("T x2, nx2, q;\n");
			tprint("x2 = x * x;\n");
			tprint("nx2 = -x2;\n");
			tprint("*exp0 = exp(nx2);\n");

			constexpr double x0 = 2.75;
			tprint("if (x < TCAST(%.20e)) {\n", x0);
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
		tprint("\n");
		tprint("}\n");
		tprint("}\n");
		tprint("\n");
		fclose(fp);
	}

	fp = nullptr;
}

void math_vec_float() {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";

	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	tprint("#ifndef __CUDACC__\n");
	tprint("namespace detail {\n");
	tprint("inline v%isf fma(v%isf a, v%isf b, v%isf c) {\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("v%isf rsqrt(v%isf);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("v%isf sqrt(v%isf);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("void sincos(v%isf, v%isf*, v%isf*);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("void erfcexp(v%isf, v%isf*, v%isf*);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("}\n");
	tprint("#endif\n");
	fclose(fp);
	if (cuda) {
		fp = fopen("./generated_code/src/math/math_vec_float.cu", "wt");
	} else {
		fp = fopen("./generated_code/src/math/math_vec_float.cpp", "wt");
	}
	tprint("\n");
	tprint("#include \"%s\"\n", header.c_str());
	tprint("#include \"typecast_v%isf.hpp\"\n", VEC_FLOAT_SIZE);
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("namespace detail {\n");
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
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
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
	tprint("ssgn = V(((*((U*) &x) & UCAST(0x80000000)) >> UCAST(30)) - UCAST(1));\n");
	tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFF)));\n");
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
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
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
			tprint("y = detail::fma(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-fmaops);
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
				tprint("tmp0 = detail::fma(tmp0, q0, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n0 + 1));
			}
			if (n1 >= 1) {
				tprint("tmp1 = detail::fma(tmp1, q1, TCAST(%.20e));\n", dfactorial(2 * n1 - 1) * nonepow(n1));
			}
			n1--;
			n0--;
		}
		tprint("tmp0 *= TCAST(%.20e) * x * *exp0;\n", 2.0 / sqrt(M_PI));
		tprint("tmp1 = detail::fma(tmp1, q1, TCAST(1));\n");
		tprint("tmp0 = TCAST(1) - tmp0;\n");
		tprint("tmp1 *= *exp0 * TCAST(%.20e) / x;\n", 1.0 / sqrt(M_PI));
		tprint("*erfc0 = sw0 * tmp0 + sw1 * tmp1;\n");
		deindent();
		tprint("}\n");
		tprint("\n");
	}
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);

	fp = nullptr;
}

void math_double() {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
	constexpr double x0 = 1.0;
	constexpr double x1 = 4.7;

	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	tprint("namespace detail {\n");
	tprint("SFMM_PREFIX inline double fma(double a, double b, double c) {\n");
	indent();
	tprint("return std::fma(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("SFMM_PREFIX double rsqrt(double);\n");
	tprint("SFMM_PREFIX double sqrt(double);\n");
	tprint("SFMM_PREFIX void sincos(double, double*, double*);\n");
	tprint("SFMM_PREFIX void erfcexp(double, double*, double*);\n");
	tprint("}\n");
	fclose(fp);
	for (int cuda = 0; cuda < 2; cuda++) {
		if (cuda) {
			fp = fopen("./generated_code/src/math/math_double.cu", "wt");
		} else {
			fp = fopen("./generated_code/src/math/math_double.cpp", "wt");
		}
		tprint("\n");
		tprint("#include <math.h>\n");
		tprint("#include \"typecast_double.hpp\"\n");
		tprint("\n");
		tprint("namespace sfmm {\n");
		tprint("namespace detail {\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
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
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("T sqrt(T x) {\n");
		indent();
		tprint("return TCAST(1) / rsqrt(x);\n");
		deindent();
		tprint("}\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("void sincos(T x, T* s, T* c) {\n");
		indent();
		tprint("V ssgn, j, i, k;\n");
		tprint("T x2;\n");
		tprint("ssgn = VCAST(((*((U*) &x) & UCAST(0x8000000000000000LL)) >> UCAST(62LL)) - UCAST(1LL));\n");
		tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFFFFFFFFFFLL)));\n");
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
		tprint("%s *= TCAST(ssgn * k);\n", sout);
		tprint("%s *= TCAST(k);\n", cout);
		deindent();
		tprint("}\n");
		tprint("\n");
		if (cuda) {
			tprint("__device__\n");
		}
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
		if (cuda) {
			tprint("__device__\n");
		}
		tprint("void erfcexp(T x, T* erfc0, T* exp0) {\n");
		indent();
		tprint("T x2, nx2, q, x0;\n");
		tprint("x2 = x * x;\n");
		tprint("nx2 = -x2;\n");
		tprint("*exp0 = exp(nx2);\n");

		tprint("if (x < TCAST(%.20e)) {\n", x0);
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
		tprint("} else if (x < TCAST(%.20e)) {\n", x1);
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
		tprint("\n");
		tprint("}\n");
		tprint("}\n");
		tprint("\n");
		fclose(fp);
	}

	fp = nullptr;
}

void math_vec_double() {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
	constexpr double x0 = 1.0;
	constexpr double x1 = 4.7;

	fp = fopen("./generated_code/include/sfmmvd.hpp", "at");
	tprint("\n");
	tprint("#ifndef __CUDACC__\n");
	tprint("namespace detail {\n");
	tprint("inline v%idf fma(v%idf a, v%idf b, v%idf c) {\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("v%idf rsqrt(v%idf);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("v%idf sqrt(v%idf);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("void sincos(v%idf, v%idf*, v%idf*);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("void erfcexp(v%idf, v%idf*, v%idf*);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("}\n");
	tprint("#endif\n");
	fclose(fp);
	if (cuda) {
		fp = fopen("./generated_code/src/math/math_vec_double.cu", "wt");
	} else {
		fp = fopen("./generated_code/src/math/math_vec_double.cpp", "wt");
	}
	tprint("\n");
	tprint("#include \"%s\"\n", header.c_str());
	tprint("#include \"typecast_v%idf.hpp\"\n", VEC_DOUBLE_SIZE);
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("namespace detail {\n");
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
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= detail::fma(x, y * y, TCAST(-1.5));\n");
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
	tprint("ssgn = V(((*((U*) &x) & UCAST(0x8000000000000000LL)) >> UCAST(62LL)) - UCAST(1LL));\n");
	tprint("j = V((*((U*) &x) & UCAST(0x7FFFFFFFFFFFFFFFLL)));\n");
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
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", cout, cout, nonepow(n / 2) / factorial(n));
			tprint("%s = detail::fma(%s, x2, TCAST(%.20e));\n", sout, sout, nonepow((n - 1) / 2) / factorial(n - 1));
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
			tprint("y = detail::fma(y, xxx, TCAST(%.20e));\n", 1.0 / factorial(i)); //17*(2-detail::fmaops);
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
				tprint("res0 = detail::fma(res0, q0, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n0 + 1));
			}
			if (n1 >= 0) {
				tprint("res1 = detail::fma(res1, x1, TCAST(%.20e));\n", c0[n1]);
			}
			if (n2 >= 1) {
				tprint("res2 = detail::fma(res2, q2, TCAST(%.20e));\n", dfactorial(2 * n2 - 1) * nonepow(n2));
			}
			n0--;
			n1--;
			n2--;
		}
		tprint("res0 *= TCAST(%.20e) * x0 * *exp0;\n", 2.0 / sqrt(M_PI));
		tprint("res0 = TCAST(1) - res0;\n");
		tprint("res1 *= *exp0 / x0;\n");
		tprint("res2 = detail::fma(res2, q2, TCAST(1));\n");
		tprint("res2 *= *exp0 * TCAST(%.20e) / x2;\n", 1.0 / sqrt(M_PI));
	}
	tprint("*erfc0 = sw0 * res0 + sw1 * res1 + sw2 * res2;\n");

	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);
	fp = nullptr;
}

void typecast_functions() {
	if (fp) {
		fclose(fp);
	}
#if defined(FLOAT) || defined(CUDA_FLOAT)
	fp = fopen("./generated_code/include/typecast_float.hpp", "at");
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#define TCAST(a) ((float)(a))\n");
	tprint("#define UCAST(a) ((unsigned)(a))\n");
	tprint("#define VCAST(a) ((int)(a))\n");
	tprint("#define TCONVERT(a) T(a)\n");
	tprint("#define UCONVERT(a) U(a)\n");
	tprint("#define VCONVERT(a) V(a)\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("typedef float T;\n");
	tprint("typedef unsigned U;\n");
	tprint("typedef int V;\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);
#endif
#ifdef VEC_FLOAT
	fp = fopen((std::string("./generated_code/include/typecast_") + vf + ".hpp").c_str(), "at");
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#define TCAST(a) (v%isf(float(a)))\n", VEC_FLOAT_SIZE);
	tprint("#define UCAST(a) (v%iui32(unsigned(a)))\n", VEC_FLOAT_SIZE);
	tprint("#define VCAST(a) (v%isi32(int(a)))\n", VEC_FLOAT_SIZE);
	tprint("#define TCONVERT(a) T(a)\n");
	tprint("#define UCONVERT(a) U(a)\n");
	tprint("#define VCONVERT(a) V(a)\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("typedef v%isf T;\n", VEC_FLOAT_SIZE);
	tprint("typedef v%iui32 U;\n", VEC_FLOAT_SIZE);
	tprint("typedef v%isi32 V;\n", VEC_FLOAT_SIZE);
	tprint("}\n");
	tprint("\n");
	fclose(fp);
#endif
#if defined(DOUBLE) || defined(CUDA_DOUBLE)
	fp = fopen("./generated_code/include/typecast_double.hpp", "at");
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#define TCAST(a) ((double)(a))\n");
	tprint("#define UCAST(a) ((unsigned long long)(a))\n");
	tprint("#define VCAST(a) ((long long)(a))\n");
	tprint("#define TCONVERT(a) T(a)\n");
	tprint("#define UCONVERT(a) U(a)\n");
	tprint("#define VCONVERT(a) V(a)\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("typedef double T;\n");
	tprint("typedef unsigned long long U;\n");
	tprint("typedef long long V;\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);
#endif
#ifdef VEC_DOUBLE
	fp = fopen((std::string("./generated_code/include/typecast_") + vd + ".hpp").c_str(), "at");
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#define TCAST(a) (v%idf(double(a)))\n", VEC_DOUBLE_SIZE);
	tprint("#define UCAST(a) (v%iui64(uint64_t(a)))\n", VEC_DOUBLE_SIZE);
	tprint("#define VCAST(a) (v%isi64(int64_t(a)))\n", VEC_DOUBLE_SIZE);
	tprint("#define TCONVERT(a) T(a)\n");
	tprint("#define UCONVERT(a) U(a)\n");
	tprint("#define VCONVERT(a) V(a)\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("typedef v%idf T;\n", VEC_DOUBLE_SIZE);
	tprint("typedef v%iui64 U;\n", VEC_DOUBLE_SIZE);
	tprint("typedef v%isi64 V;\n", VEC_DOUBLE_SIZE);
	tprint("}\n");
	tprint("\n");
	fclose(fp);
#endif
	fp = nullptr;
}

int main() {
	SYSTEM("mkdir -p generated_code\n");
	SYSTEM("mkdir -p ./generated_code/include\n");
	SYSTEM("mkdir -p ./generated_code/src\n");
	SYSTEM("mkdir -p ./generated_code/src/math\n");
	tprint("\n");
	set_file(full_header.c_str());
	tprint("#pragma once\n");
	tprint("\n");
	tprint("#ifndef SFMM_FULL_HEADER42\n");
	tprint("#define SFMM_FULL_HEADER42\n");
	tprint("#ifdef __CUDA_ARCH__\n");
	tprint("#define SFMM_PREFIX __device__\n");
	tprint("#else\n");
	tprint("#define SFMM_PREFIX\n");
	tprint("#endif\n");
	tprint("\n");
	tprint("#include <math.h>\n");
	tprint("#include <cmath>\n");
	tprint("#include <cstdint>\n");
	tprint("\n");
	tprint("#define SFMM_IGNORE_SCALING (0x2)\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
	tprint("\n");
	tprint("template<class T>\n");
	tprint("struct force_type {\n");
	indent();
	tprint("T potential;\n");
	tprint("T force[3];\n");
	tprint("SFMM_PREFIX inline void init() {\n");
	indent();
	tprint("potential = force[0] = force[1] = force[2] = T(0);\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("};\n");
	for (periodic = 0; periodic <= 1; periodic++) {
		tprint("\n");
		tprint("template<class T, int P>\n");
		tprint("class expansion%s {\n", period_name());
		indent();
		tprint("T o[(P+1)*(P+1)];\n");
		tprint("T t;\n");
		tprint("T r;\n");
		deindent();
		tprint("public:\n");
		indent();
		tprint("SFMM_PREFIX expansion%s(T=T(1));\n", period_name());
		tprint("SFMM_PREFIX expansion%s(const expansion%s&);\n", period_name(), period_name());
		tprint("SFMM_PREFIX expansion%s& operator=(const expansion%s&);\n", period_name(), period_name());
		tprint("SFMM_PREFIX expansion%s& operator+=(expansion%s);\n", period_name(), period_name());
		tprint("SFMM_PREFIX void init(T r0 = T(1));\n");
		tprint("SFMM_PREFIX void rescale(T);\n");
		tprint("SFMM_PREFIX T* data();\n");
		tprint("SFMM_PREFIX const T* data() const;\n");
		tprint("SFMM_PREFIX T scale() const;\n");
		if (periodic) {
			tprint("SFMM_PREFIX T& trace2();\n");
			tprint("SFMM_PREFIX T trace2() const;\n");
		}
		deindent();
		tprint("};\n");
		tprint("\n");

		tprint("template<class T, int P>\n");
		tprint("class multipole%s {\n", period_name());
		indent();
		tprint("T o[P*P];\n");
		tprint("T t;\n");
		tprint("T r;\n");
		deindent();
		tprint("public:\n");
		indent();
		tprint("SFMM_PREFIX multipole%s(T=T(1));\n", period_name());
		tprint("SFMM_PREFIX multipole%s(const multipole%s&);\n", period_name(), period_name());
		tprint("SFMM_PREFIX multipole%s& operator=(const multipole%s&);\n", period_name(), period_name());
		tprint("SFMM_PREFIX multipole%s& operator+=(multipole%s);\n", period_name(), period_name());
		tprint("SFMM_PREFIX void init(T r0 = T(1));\n");
		tprint("SFMM_PREFIX void rescale(T);\n");
		tprint("SFMM_PREFIX T* data();\n");
		tprint("SFMM_PREFIX const T* data() const;\n");
		tprint("SFMM_PREFIX T scale() const;\n");
		if (periodic) {
			tprint("SFMM_PREFIX T& trace2();\n");
			tprint("SFMM_PREFIX T trace2() const;\n");
		}
		deindent();
		tprint("};\n");
		tprint("\n");

		for (int P = pmin - 1; P <= pmax; P++) {

			tprint("\n");
			tprint("template<class T>\n");
			tprint("class expansion%s<T,%i> {\n", period_name(), P);
			indent();
			tprint("T o[%i];\n", exp_sz(P));
			tprint("T t;\n");
			tprint("T r;\n");
			deindent();
			tprint("public:\n");
			indent();
			tprint("SFMM_PREFIX expansion%s(T r0 = T(1)) {\n", period_name());
			indent();
			tprint("r = r0;\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX void init(T r0 = T(1)) {\n");
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = T(0);\n");
			deindent();
			tprint("}\n");
			tprint("t = T(0);\n");
			tprint("r = r0;\n");
			deindent();
			tprint("}\n");
			tprint("SFMM_PREFIX expansion%s(const expansion%s& other) {\n", period_name(), period_name());
			indent();
			tprint("*this = other;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX expansion%s& operator=(const expansion%s& other) {\n", period_name(), period_name());
			indent();
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] = other.o[n];\n");
			deindent();
			tprint("}\n");
			tprint("t = other.t;\n");
			tprint("r = other.r;\n");
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX expansion%s& operator+=(expansion%s other) {\n", period_name(), period_name());
			indent();
			tprint("other.rescale(r);\n");
			tprint("for( int n = 0; n < %i; n++ ) {\n", exp_sz(P));
			indent();
			tprint("o[n] += other.o[n];\n");
			deindent();
			tprint("}\n");
			tprint("t += other.t;\n");
			tprint("r = other.r;\n");
			tprint("return *this;\n");
			deindent();
			tprint("}\n");

			tprint("SFMM_PREFIX void rescale(T r0) {\n");
			indent();
			tprint("const T a = r0 / r;\n");
			tprint("T b = a;\n");
			tprint("r = r0;\n");
			for (int n = 0; n <= P; n++) {
				for (int m = -n; m <= n; m++) {
					tprint("o[%i] *= b;\n", index(n, m));
				}
				if (n == 2) {
					tprint("t *= b;\n", exp_sz(P));
				}
				if (n != P) {
					tprint("b *= a;\n");
				}
			}
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
			tprint("SFMM_PREFIX T scale() const {\n");
			indent();
			tprint("return r;\n");
			deindent();
			tprint("}\n");
			if (periodic) {
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
			tprint("\n");
			if (P > pmin - 1) {
				tprint("\n");
				tprint("template<class T>\n");
				tprint("class multipole%s<T,%i> {\n", period_name(), P);
				indent();
				tprint("T o[%i];\n", mul_sz(P));
				tprint("T t;\n");
				tprint("T r;\n");
				deindent();
				tprint("public:\n");
				indent();
				tprint("SFMM_PREFIX multipole%s(T r0 = T(1)) {\n", period_name());
				indent();
				tprint("r = r0;\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX void init(T r0 = T(1)) {\n");
				indent();
				tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
				indent();
				tprint("o[n] = T(0);\n");
				deindent();
				tprint("}\n");
				tprint("t = T(0);\n");
				tprint("r = r0;\n");
				deindent();
				tprint("}\n");
				tprint("SFMM_PREFIX multipole%s(const multipole%s& other) {\n", period_name(), period_name());
				indent();
				tprint("*this = other;\n");
				deindent();
				tprint("}\n");

				tprint("SFMM_PREFIX multipole%s& operator=(const multipole%s& other) {\n", period_name(), period_name());
				indent();
				tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
				indent();
				tprint("o[n] = other.o[n];\n");
				deindent();
				tprint("}\n");
				tprint("t = other.t;\n");
				tprint("r = other.r;\n");
				tprint("return *this;\n");
				deindent();
				tprint("}\n");

				tprint("SFMM_PREFIX multipole%s& operator+=(multipole%s other) {\n", period_name(), period_name());
				indent();
				tprint("if( r != other.r ) {\n");
				indent();
				tprint("other.rescale(r);\n");
				deindent();
				tprint("}\n");
				tprint("for( int n = 0; n < %i; n++ ) {\n", mul_sz(P));
				indent();
				tprint("o[n] += other.o[n];\n");
				deindent();
				tprint("}\n");
				tprint("t += other.t;\n");
				tprint("r = other.r;\n");
				tprint("return *this;\n");
				deindent();
				tprint("}\n");

				tprint("SFMM_PREFIX void rescale(T r0) {\n");
				indent();
				tprint("const T a = r / r0;\n");
				tprint("T b = a;\n");
				tprint("r = r0;\n");
				for (int n = 1; n < P; n++) {
					for (int m = -n; m <= n; m++) {
						tprint("o[%i] *= b;\n", index(n, m));
					}
					if (n == 2) {
						tprint("t *= b;\n");
					}
					if (n != P - 1) {
						tprint("b *= a;\n");
					}
				}
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
				tprint("SFMM_PREFIX T scale() const {\n");
				indent();
				tprint("return r;\n");
				deindent();
				tprint("}\n");
				if (periodic && P > 1 ) {
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
				tprint("\n");
			}
		}
	}
	tprint("\n");
	set_file(full_header.c_str());
	tprint("#else\n");
	tprint("namespace sfmm {\n");
	tprint("#endif\n");
	typecast_functions();

	int ntypenames = 0;
	std::vector<std::string> rtypenames;
	std::vector<std::string> sitypenames;
	std::vector<std::string> uitypenames;
	std::vector<int> ucuda;
	int funcnum;
#if defined(CUDA_FLOAT)
	rtypenames.push_back("float");
	sitypenames.push_back("int32_t");
	uitypenames.push_back("uint32_t");
	ucuda.push_back(true);
	ntypenames++;
	funcnum = 0;
#endif
#if defined(CUDA_DOUBLE)
	rtypenames.push_back("double");
	sitypenames.push_back("int64_t");
	uitypenames.push_back("uint64_t");
	ucuda.push_back(true);
	ntypenames++;
	funcnum = 1;
#endif
#if defined(FLOAT)
	rtypenames.push_back("float");
	sitypenames.push_back("int32_t");
	uitypenames.push_back("uint32_t");
	ucuda.push_back(false);
	ntypenames++;
	funcnum = 0;
#endif
#if defined(DOUBLE)
	rtypenames.push_back("double");
	sitypenames.push_back("int64_t");
	uitypenames.push_back("uint64_t");
	ucuda.push_back(false);
	ntypenames++;
	funcnum = 1;
#endif
#ifdef VEC_FLOAT
	rtypenames.push_back(vf);
	sitypenames.push_back(vui32);
	uitypenames.push_back(vsi32);
	ucuda.push_back(false);
	ntypenames++;
	funcnum = 2;
#endif

#ifdef VEC_DOUBLE
	rtypenames.push_back(vd);
	sitypenames.push_back(vui64);
	uitypenames.push_back(vsi64);
	ucuda.push_back(false);
	ntypenames++;
	funcnum = 3;
#endif
	set_file(full_header.c_str());

#if defined(VEC_DOUBLE) || defined(VEC_FLOAT)
	tprint("#ifndef __CUDACC__\n");
	tprint("%s\n", vec_header().c_str());
	tprint("\n#endif");
	tprint("\n");
#endif

	for (int ti = 0; ti < ntypenames; ti++) {
		printf("%s cuda:%i\n", rtypenames[ti].c_str(), ucuda[ti]);
		cuda = ucuda[ti];
		prefix = ucuda[ti] ? "SFMM_PREFIX" : "";
		type = rtypenames[ti];
		sitype = sitypenames[ti];
		uitype = uitypenames[ti];
		tprint("\n#ifndef SFMM_FUNCS%i42\n", funcnum);
		tprint("#define SFMM_FUNCS%i42\n", funcnum);
#if defined(FLOAT) || defined(CUDA_FLOAT)
		math_float();
#endif

#if defined(DOUBLE) || defined(CUDA_DOUBLE)
		math_double();
#endif

#if defined(VEC_FLOAT)
		math_vec_float();
#endif

#if defined(VEC_DOUBLE)
		math_vec_double();
#endif
		if (is_vec(type)) {
			set_file(full_header.c_str());
			tprint("#ifndef __CUDACC__\n");
		}
		for (periodic = 0; periodic <= 1; periodic++) {
			for (nopot = 0; nopot <= 1; nopot++) {
				std::vector<std::unordered_map<std::string, flops_t>> flops_map(pmax + 1);
				std::vector<std::unordered_map<std::string, int>> rot_map(pmax + 1);
				for (int P = pmin - 1; P <= pmax; P++) {
					std::string fname;
					flops_t regular_harmonic_flops;
					flops_t regular_harmonic_xz_flops;
					flops_t fps;
					fname = regular_harmonic(P);
					fclose(fp);
					fp = nullptr;
					regular_harmonic_flops = count_flops(fname);

					fname = regular_harmonic_xz(P);
					fclose(fp);
					fp = nullptr;
					regular_harmonic_xz_flops = count_flops(fname);

					flops_t flops0, flops1, flops2;
					if (P < pmax && !nopot) {
						fname = M2M_norot(P);
						fclose(fp);
						fp = nullptr;
						flops0 += regular_harmonic_flops;
						flops0 += count_flops(fname);
						SYSTEM((std::string("rm -rf ") + fname).c_str());

						fname = M2M_rot1(P);
						fclose(fp);
						fp = nullptr;
						flops1 += regular_harmonic_xz_flops;
						flops1 += count_flops(fname);
						SYSTEM((std::string("rm -rf ") + fname).c_str());

						fname = M2M_rot2(P);
						fclose(fp);
						fp = nullptr;
						flops2 += count_flops(fname);
						SYSTEM((std::string("rm -rf ") + fname).c_str());

						if (flops2.load() > flops0.load() || flops2.load() > flops1.load()) {
							if (flops1.load() < flops0.load()) {
								M2M_rot1(P);
								flops_map[P]["M2M"] = flops1;
								rot_map[P]["M2M"] = 1;
							} else {
								M2M_norot(P);
								flops_map[P]["M2M"] = flops0;
								rot_map[P]["M2M"] = 0;
							}
						} else {
							M2M_rot2(P);
							flops_map[P]["M2M"] = flops2;
							rot_map[P]["M2M"] = 2;
						}

						flops0.reset();
						flops1.reset();
						flops2.reset();
					}
					if (P >= pmin) {
						fname = L2L_norot(P);
						fclose(fp);
						fp = nullptr;
						flops0 += regular_harmonic_flops;
						flops0 += count_flops(fname);
						SYSTEM((std::string("rm -rf ") + fname).c_str());

						fname = L2L_rot1(P);
						fclose(fp);
						fp = nullptr;
						flops1 += regular_harmonic_xz_flops;
						flops1 += count_flops(fname);
						SYSTEM((std::string("rm -rf ") + fname).c_str());

						fname = L2L_rot2(P);
						fclose(fp);
						fp = nullptr;
						flops2 += count_flops(fname);
						SYSTEM((std::string("rm -rf ") + fname).c_str());

						if (flops2.load() > flops0.load() || flops2.load() > flops1.load()) {
							if (flops1.load() < flops0.load()) {
								L2L_rot1(P);
								flops_map[P]["L2L"] = flops1;
								rot_map[P]["L2L"] = 1;
							} else {
								L2L_norot(P);
								flops_map[P]["L2L"] = flops0;
								rot_map[P]["L2L"] = 0;
							}
						} else {
							L2L_rot2(P);
							flops_map[P]["L2L"] = flops2;
							rot_map[P]["L2L"] = 2;
						}
					}
				}

				for (int P = pmin; P <= pmax; P++) {
					flops_t flops0, flops1, flops2;
					std::string fname;
					flops_t greens_flops;
					flops_t greens_xz_flops;
					flops_t fps;
					fname = greens(P);
					fclose(fp);
					fp = nullptr;
					greens_flops = count_flops(fname);

					flops0.reset();
					flops0.fma += 2 + (P + 1) + (P + 1) * (P + 1);
					flops0.r += 5 + (P + 1) * 6;
					flops0.asgn++;
					flops0 += sqrt_flops();
					flops0 += greens_flops;
					flops0 += erfcexp_flops();
					greens_ewald_real_flops = flops0;
					flops0.reset();

					fname = greens_xz(P);
					fclose(fp);
					fp = nullptr;
					greens_xz_flops = count_flops(fname);

					fname = M2L_norot(P, P);
					fclose(fp);
					fp = nullptr;
					flops0 += count_flops(fname);
					flops0 += greens_flops;
					flops0 += rescale_flops(P - 1);
					SYSTEM((std::string("rm -rf ") + fname).c_str());

					fname = M2L_rot1(P, P);
					fclose(fp);
					fp = nullptr;
					flops1 += greens_xz_flops;
					flops1 += count_flops(fname);
					flops1 += rescale_flops(P - 1);
					flops1 += copy_flops(P - 1);
					SYSTEM((std::string("rm -rf ") + fname).c_str());

					fname = M2L_rot2(P, P);
					fclose(fp);
					fp = nullptr;
					flops2 += count_flops(fname);
					flops2 += rescale_flops(P - 1);
					flops2 += copy_flops(P - 1);
					SYSTEM((std::string("rm -rf ") + fname).c_str());

					if (flops2.load() > flops0.load() || flops2.load() > flops1.load()) {
						if (flops1.load() < flops0.load()) {
							M2L_rot1(P, P);
							flops_map[P]["M2L"] = flops1;
							rot_map[P]["M2L"] = 1;
						} else {
							M2L_norot(P, P);
							flops_map[P]["M2L"] = flops0;
							rot_map[P]["M2L"] = 0;
						}
					} else {
						M2L_rot2(P, P);
						flops_map[P]["M2L"] = flops2;
						rot_map[P]["M2L"] = 2;
					}

					flops0.reset();
					flops1.reset();
					flops2.reset();
					fname = M2L_norot(P, 1);
					fclose(fp);
					fp = nullptr;
					flops0 += count_flops(fname);
					flops0 += greens_flops;
					flops0 += rescale_flops(P - 1);
					SYSTEM((std::string("rm -rf ") + fname).c_str());

					fname = M2L_rot1(P, 1);
					fclose(fp);
					fp = nullptr;
					flops1 += greens_xz_flops;
					flops1 += count_flops(fname);
					flops1 += rescale_flops(P - 1);
					SYSTEM((std::string("rm -rf ") + fname).c_str());

					fname = M2L_rot2(P, 1);
					fclose(fp);
					fp = nullptr;
					flops2 += count_flops(fname);
					flops2 += rescale_flops(P - 1);
					SYSTEM((std::string("rm -rf ") + fname).c_str());

					if (flops2.load() > flops0.load() || flops2.load() > flops1.load()) {
						if (flops1.load() < flops0.load()) {
							M2L_rot1(P, 1);
							flops_map[P]["M2P"] = flops1;
							rot_map[P]["M2P"] = 1;
						} else {
							M2L_norot(P, 1);
							flops_map[P]["M2P"] = flops0;
							rot_map[P]["M2P"] = 0;
						}
					} else {
						M2L_rot2(P, 1);
						flops_map[P]["M2P"] = flops2;
						rot_map[P]["M2P"] = 2;
					}

					fname = P2L(P);
					fclose(fp);
					fp = nullptr;
					flops0 = count_flops(fname);
					flops0 += accumulate_flops(P);
					flops_map[P]["P2L"] = flops0;

					fname = L2P(P);
					fclose(fp);
					fp = nullptr;
					flops0 = count_flops(fname);
					flops_map[P]["L2P"] = flops0;

					if (!nopot) {
						fname = P2M(P - 1);
						fclose(fp);
						fp = nullptr;
						flops0 = count_flops(fname);
						flops_map[P - 1]["P2M"] = flops0;
					}

					const double alpha = is_float(type) ? 2.4 : 2.25;
					fname = M2LG(P, P);
					fclose(fp);
					fp = nullptr;
					flops0 = count_flops(fname);
					auto M2LG_flops = flops0;

					if (periodic) {
						fname = greens_ewald(P, alpha);
						fclose(fp);
						fp = nullptr;
						flops0 = count_flops(fname);
						auto greens_ewald_flops = flops0;

						fname = M2L_ewald(P);
						fclose(fp);
						fp = nullptr;
						flops0 = count_flops(fname);
						flops0 += init_flops(P);
						flops0 += accumulate_flops(P);
						flops0 += M2LG_flops;
						flops0 += greens_ewald_flops;
						flops0 += rescale_flops(P - 1);
						flops_map[P]["M2L_ewald"] = flops0;
					}
				}

				printf("%s %s\n", periodic ? "periodic" : "nonperiodic", nopot ? "w/o potential" : "w/ potential");
				printf("P  | M2L      | P2L   | M2P     | P2M   | M2M     | L2L     | L2P   | M2L_ewald\n");
				for (int P = pmin; P <= pmax; P++) {
					printf("%2i | ", P);
					printf(" %5i %1i |", flops_map[P]["M2L"].load(), rot_map[P]["M2L"]);
					printf(" %5i |", flops_map[P]["P2L"].load());
					printf(" %5i %1i |", flops_map[P]["M2P"].load(), rot_map[P]["M2P"]);
					if (nopot) {
						printf("  n/a  |   n/a   |");
					} else {
						printf(" %5i |", flops_map[P - 1]["P2M"].load());
						printf(" %5i %1i |", flops_map[P - 1]["M2M"].load(), rot_map[P - 1]["M2M"]);
					}
					printf(" %5i %1i |", flops_map[P]["L2L"].load(), rot_map[P]["L2L"]);
					printf(" %5i |", flops_map[P]["L2P"].load());
					if (periodic) {
						printf(" %6i |", flops_map[P]["M2L_ewald"].load());
					} else {
						printf("   n/a  |");
					}
					printf("\n");
				}
			}
		}
		if (is_vec(type)) {
			set_file(full_header.c_str());
			tprint("#endif\n");
		}
	}
//	printf("./generated_code/include/sfmm.h");
	fflush(stdout);
	periodic = 1;
	set_file(full_header.c_str());
	tprint("namespace detail {\n");
	tprint("%s", detail_header.c_str());
	tprint("#ifndef __CUDACC__\n");
	tprint("%s", detail_header_vec.c_str());
	tprint("#endif\n");
	tprint("\n");
	tprint("#else\n", funcnum);
	tprint("namespace detail {\n");
	tprint("#endif\n", funcnum);
	tprint("#ifndef SFMM_GREEN_EWALD_REAL42\n");
	tprint("#define SFMM_GREEN_EWALD_REAL42\n");
	tprint("template<class T, int P, int ALPHA100>\n");
	tprint("SFMM_PREFIX void greens_ewald_real(expansion%s<T, P>& G_st, T x, T y, T z, int flags) {\n", period_name());
	indent();
	tprint("constexpr double ALPHA = ALPHA100 / 100.0;\n");
	tprint("expansion%s<T, P> Gr_st;\n", period_name());
	tprint("const T r2 = detail::fma(x, x, detail::fma(y, y, z * z));\n");
	tprint("const T r = detail::sqrt(r2);\n");
	tprint("greens(Gr_st, x, y, z, flags);\n");
	tprint("const T* Gr = Gr_st.data();\n");
	tprint("T* G = G_st.data();\n");
	tprint("const T xxx = T(ALPHA) * r;\n");
	tprint("T gam1, exp0;\n");
	tprint("detail::erfcexp(xxx, &gam1, &exp0);\n");
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
	tprint("G[i] = detail::fma(gam, Gr[i], G[i]);\n");
	deindent();
	tprint("}\n");
	tprint("gam0inv /= -(T(l) + T(0.5));\n");
	tprint("gam1 = detail::fma(T(l + 0.5), gam1, -xpow * exp0);\n");
	tprint("xpow *= xfac;\n");
	deindent();
	tprint("}\n");
	deindent();
	tprint("}\n");
	tprint("#endif\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return 0;
}
