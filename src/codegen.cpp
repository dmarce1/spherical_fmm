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
//		return r + 2 * fma + 4 * rdiv + con + rcmp;
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

#define DEBUGNAN

static int nopot = false;
static int nodip = false;
static bool fmaops = true;
static int periodic = 0;
static int scaled = 0;
static int pmin = PMIN;
static int pmax = PMAX;
static std::string basetype;
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
bool enable_scaled = true;
#endif
#ifdef CUDA_DOUBLE
static std::string header = "cusfmmd.hpp";
bool enable_scaled =false;
#endif
#ifdef CUDA_FLOAT
bool enable_scaled = true;
static std::string header = "cusfmmf.hpp";
#endif
#ifdef DOUBLE
bool enable_scaled = false;
static std::string header = "sfmmd.hpp";
#endif
#ifdef VEC_FLOAT
bool enable_scaled = true;
static std::string header = "sfmmvf.hpp";
#endif
#ifdef VEC_DOUBLE
bool enable_scaled = false;
static std::string header = "sfmmvd.hpp";
#endif
static std::string full_header = std::string("./generated_code/include/") + header;

static const char* pot_name() {
	return nopot ? "_wo_potential" : "";
}

static const char* period_name() {
	return periodic ? "_periodic" : "";
}

static const char* scaled_name() {
	return scaled ? "_scaled" : "";
}

static const char* dip_name() {
	return nodip ? "_wo_dipole" : "";
}

static std::string complex_header = "\n"
		"#ifndef SFMM_COMPLEX_DECL_HPP_42\n"
		"#define SFMM_COMPLEX_DECL_HPP_42\n"
		"\n"
		"template<class T>\n"
		"class complex {\n"
		"\tT x, y;\n"
		"public:\n"
		"\tSFMM_PREFIX complex() = default;\n"
		"\tSFMM_PREFIX complex(T a);\n"
		"\tSFMM_PREFIX complex(T a, T b);\n"
		"\tSFMM_PREFIX complex& operator+=(complex other);\n"
		"\tSFMM_PREFIX complex& operator-=(complex other);\n"
		"\tSFMM_PREFIX complex operator*(complex other) const;\n"
		"\tSFMM_PREFIX complex operator/(complex other) const;\n"
		"\tSFMM_PREFIX complex operator/=(complex other);\n"
		"\tSFMM_PREFIX complex operator/(T other) const;\n"
		"\tSFMM_PREFIX complex operator*(T other) const;\n"
		"\tSFMM_PREFIX complex& operator*=(T other);\n"
		"\tSFMM_PREFIX complex& operator*=(complex other);\n"
		"\tSFMM_PREFIX complex operator+(complex other) const;\n"
		"\tSFMM_PREFIX complex operator-(complex other) const;\n"
		"\tSFMM_PREFIX complex conj() const;\n"
		"\tSFMM_PREFIX T real() const;\n"
		"\tSFMM_PREFIX T imag() const;\n"
		"\tSFMM_PREFIX T& real();\n"
		"\tSFMM_PREFIX T& imag();\n"
		"\tSFMM_PREFIX T norm() const;\n"
		"\tSFMM_PREFIX T abs() const;\n"
		"\tSFMM_PREFIX complex operator-() const;\n"
		"};\n"
		"\n"
		"#endif\n"
		"";

static std::string complex_defs = "\n"
		"#ifndef SFMM_COMPLEX_DEF_HPP_42\n"
		"#define SFMM_COMPLEX_DEF_HPP_42\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T>::complex(T a) :\n"
		"\t\tx(a), y(0.0) {\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T>::complex(T a, T b) :\n"
		"\t\tx(a), y(b) {\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T>& complex<T>::operator+=(complex<T> other) {\n"
		"\tx += other.x;\n"
		"\ty += other.y;\n"
		"\treturn *this;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T>& complex<T>::operator-=(complex<T> other) {\n"
		"\tx -= other.x;\n"
		"\ty -= other.y;\n"
		"\treturn *this;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator*(complex<T> other) const {\n"
		"\tcomplex<T> a;\n"
		"\ta.x = x * other.x - y * other.y;\n"
		"\ta.y = fma(x, other.y, y * other.x);\n"
		"\treturn a;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator/(complex<T> other) const {\n"
		"\treturn *this * other.conj() / other.norm();\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator/=(complex<T> other) {\n"
		"\t*this = *this * other.conj() / other.norm();\n"
		"\treturn *this;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator/(T other) const {\n"
		"\tcomplex<T> b;\n"
		"\tb.x = x / other;\n"
		"\tb.y = y / other;\n"
		"\treturn b;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator*(T other) const {\n"
		"\tcomplex<T> b;\n"
		"\tb.x = x * other;\n"
		"\tb.y = y * other;\n"
		"\treturn b;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T>& complex<T>::operator*=(T other) {\n"
		"\tx *= other;\n"
		"\ty *= other;\n"
		"\treturn *this;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T>& complex<T>::operator*=(complex<T> other) {\n"
		"\t*this = *this * other;\n"
		"\treturn *this;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator+(complex<T> other) const {\n"
		"\tcomplex<T> a;\n"
		"\ta.x = x + other.x;\n"
		"\ta.y = y + other.y;\n"
		"\treturn a;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator-(complex<T> other) const {\n"
		"\tcomplex<T> a;\n"
		"\ta.x = x - other.x;\n"
		"\ta.y = y - other.y;\n"
		"\treturn a;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::conj() const {\n"
		"\tcomplex<T> a;\n"
		"\ta.x = x;\n"
		"\ta.y = -y;\n"
		"\treturn a;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX T complex<T>::real() const {\n"
		"\treturn x;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX T complex<T>::imag() const {\n"
		"\treturn y;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX T& complex<T>::real() {\n"
		"\treturn x;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX T& complex<T>::imag() {\n"
		"\treturn y;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX T complex<T>::norm() const {\n"
		"\treturn fma(x,x,y*y);\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX T complex<T>::abs() const {\n"
		"\treturn sqrt(norm());\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX complex<T> complex<T>::operator-() const {\n"
		"\tcomplex<T> a;\n"
		"\ta.x = -x;\n"
		"\ta.y = -y;\n"
		"\treturn a;\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"SFMM_PREFIX inline complex<T> operator*(T a, complex<T> b) {\n"
		"\treturn complex<T>(a*b.real(),a*b.imag());\n"
		"}\n"
		"\n"
		"template<class T>\n"
		"inline void swap(complex<T>& a, complex<T>& b) {\n"
		"\tstd::swap(a.real(), b.real());\n"
		"\tstd::swap(a.imag(), b.imag());\n"
		"}\n"
		"\n"
		"#endif\n"
		"";

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
	LIT, PTR, CPTR, EXP, HEXP, MUL, CEXP, CMUL, FORCE
};

void init_real(std::string var) {
	fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
	tprint("T %s(std::numeric_limits<%s>::signaling_NaN());\n", var.c_str(), basetype.c_str());
	fprintf(fp, "#else\n");
	tprint("T %s;\n", var.c_str(), type.c_str());
	fprintf(fp, "#endif\n");
	if (var == "tmp1") {
		for (int i = 2; i < nchains; i++) {
			init_real(std::string("tmp") + std::to_string(i));
		}
	}
}

void init_reals(std::string var, int cnt) {
	fprintf(fp, "#if !defined(NDEBUG) && !defined(__CUDA_ARCH__)\n");
	tprint("T %s [%i]={", var.c_str(), cnt);
	int ontab = ntab;
	ntab = 0;
	for (int n = 0; n < cnt; n++) {
		tprint("std::numeric_limits<%s>::signaling_NaN()", basetype.c_str());
		if (n != cnt - 1) {
			tprint(",");
		}
	}
	tprint("};\n");
	ntab = ontab;
	fprintf(fp, "#else\n");
	tprint("T %s [%i];\n", var.c_str(), cnt);
	fprintf(fp, "#endif\n");
//			" = {";
	/*	for (int n = 0; n < cnt - 1; n++) {
	 str += "std::numeric_limits<T>::signaling_NaN(), ";
	 }
	 str += "std::numeric_limits<T>::signaling_NaN()};";*/
}

template<class ...Args>
std::string func_args(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		str += std::string("expansion") + period_name() + scaled_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == HEXP) {
		str += std::string("detail::expansion_xz") + period_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == MUL) {
		str += std::string("multipole") + period_name() + scaled_name() + dip_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CEXP) {
		str += std::string("const expansion") + period_name() + scaled_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
	} else if (atype == CMUL) {
		str += std::string("const multipole") + period_name() + scaled_name() + dip_name() + "<" + type + ", " + std::to_string(P) + ">& " + arg + "_st";
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
std::string func_args_call(int P, const char* arg, arg_type atype, int term) {
	std::string str;
	if (atype == EXP) {
		str += std::string(arg) + "_st";
	} else if (atype == HEXP) {
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

template<class ... Args>
std::string func_header(const char* func, int P, bool pub, bool calcpot, bool scale, bool flags, std::string head, Args&& ...args) {
	static std::set<std::string> igen;
	std::string func_name = std::string(func);
	if (nopot && calcpot) {
		func_name += "_wo_potential";
	}
	std::string file_name = func_name + (std::string(func_name) == std::string("greens_ewald") ? "" : period_name()) + scaled_name() + dip_name()
			+ (cuda ? ".cu" : ".cpp");
	func_name = "void " + func_name;
	func_name += "(" + func_args(P, std::forward<Args>(args)..., 0);
	auto func_name2 = func_name;
	if (calcpot && !nopot) {
		func_name += ", int potopt = sfmmCalculateWithPotential)";
		func_name2 += ", int potopt)";

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
	if (calcpot && !nopot) {
		tprint("if( potopt == sfmmCalculateWithoutPotential ) {\n ");
		indent();
		std::string str = std::string(func) + std::string("_wo_potential(");
		str += func_args_call(P, std::forward<Args>(args)..., 0);
		str += ");\n";
		tprint("%s", str.c_str());
		tprint("return;\n");
		deindent();
		tprint("}\n");
	}
	func_args_cover(P, std::forward<Args>(args)..., 0);
	return file_name;
}

void set_tprint(bool c) {
	tprint_on = c;
}

int lindex(int l, int m) {
	return l * (l + 1) + m;
}

int mindex(int l, int m) {
	if (nodip) {
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

void z_rot(int P, const char* name, bool noevenhi, bool exclude, int noimaghi) {
	tprint("rx[0] = cosphi;\n");
	tprint("ry[0] = sinphi;\n");
	for (int m = 1; m <= P; m++) {
		tprint("rx[%i] = rx[%i] * cosphi - ry[%i] * sinphi;\n", m, m - 1, m - 1);
		tprint("ry[%i] = fma(rx[%i], sinphi, ry[%i] * cosphi);\n", m, m - 1, m - 1);
	}
	int mmin = 1;
	bool initR = true;
	std::function<int(int, int)> index;
	if (name[0] == 'M') {
		index = mindex;
	} else {
		index = lindex;
	}
	for (int m = 1; m <= P; m++) {
		tprint_new_chain();
		for (int l = m; l <= P; l++) {
			if (name == "M" && nodip && l == 1) {
				continue;
			}
			if (noevenhi) {
				if (l == P) {
					if (l % 2 != m % 2) {
						continue;
					}
				} else if (nodip && l == P - 1) {
					if (l % 2 == m % 2) {
						continue;
					}
				}
			}
			bool ionly = false;
			bool ronly = false;
			if (exclude) {
				if (l == P) {
					ionly = m % 2;
				} else if (nodip && l == P - 1) {
					ionly = m % 2;
				}
				if (l == P) {
					ronly = !(m % 2);
				} else if (nodip && l == P - 1) {
					ronly = !(m % 2);
				}
			}
			if (ionly) {
				tprint_chain("%s[%i] = -%s[%i] * ry[%i];\n", name, index(l, m), name, index(l, -m), m - 1);
				tprint_chain("%s[%i] *= rx[%i];\n", name, index(l, -m), m);
			} else if (ronly || (noimaghi && (l > (P - noimaghi)))) {
				tprint_chain("%s[%i] = %s[%i] * ry[%i];\n", name, index(l, -m), name, index(l, m), m - 1);
				tprint_chain("%s[%i] *= rx[%i];\n", name, index(l, m), m - 1);
			} else {
				bool sw = false;
				if (noevenhi && (l >= P - 1)) {
					if (nodip) {
						sw = true;
					} else {
						sw = m % 2 == P % 2;
					}
				}
				if (sw) {
					tprint_chain("%s[%i] = %s[%i] * ry[%i];\n", name, index(l, -m), name, index(l, m), m - 1);
					tprint_chain("%s[%i] *= rx[%i];\n", name, index(l, m), m - 1);
				} else {
					tprint_chain("tmp%i = %s[%i];\n", current_chain, name, index(l, m));
					tprint_chain("%s[%i] = %s[%i] * rx[%i] - %s[%i] * ry[%i];\n", name, index(l, m), name, index(l, m), m - 1, name, index(l, -m), m - 1);
					tprint_chain("%s[%i] = fma(tmp%i, ry[%i], %s[%i] * rx[%i]);\n", name, index(l, -m), current_chain, m - 1, name, index(l, -m), m - 1);
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
	std::function<int(int, int)> index;
	if (name[0] == 'M') {
		index = mindex;
	} else {
		index = lindex;
	}
	for (int n = 1; n <= P; n++) {
		if (name == "M" && nodip && n == 1) {
			continue;
		}
		int lmax = n;
		if (l_restrict) {
			lmax = std::min(n, P - n);
			if (nodip && n == P - 1) {
				lmax = 0;
			}
		}
		for (int m = -lmax; m <= lmax; m++) {
			bool flag = false;
			if (noevenhi) {
				if (P == n && n % 2 != abs(m) % 2) {
					continue;
				} else if (nodip && n == P - 1 && n % 2 == abs(m) % 2) {
					continue;
				}
			}
			tprint("A[%i] = %s[%i];\n", m + P, name, index(n, m));
		}
		std::vector<std::vector<std::pair<double, int>>>ops(2 * n + 1);
		int mmax = n;
		if (m_restrict && mmax > (P) - n) {
			mmax = (P + 1) - n;
		}
		for (int m = 0; m <= mmax; m++) {
			for (int l = 0; l <= lmax; l++) {
				bool flag = false;
				if (noevenhi) {
					if (P == n && n % 2 != abs(l) % 2) {
						continue;
					} else if (nodip && P - 1 == n && n % 2 == abs(l) % 2) {
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
	tprint("const T& zx1 = z;\n");
	for (int m = 1; m < P; m++) {
		tprint("const T zx%i = TCAST(%i) * z;\n", 2 * m + 1, 2 * m + 1);
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
	auto fname = func_header("greens_safe", P, true, false, false, true, "", "O", EXP, "x", LIT, "y", LIT, "z", LIT);
	const auto mul = [](std::string a, std::string b, std::string c, int l) {
		tprint( "flags[%i] *= safe_mul(%s, %s, %s);\n", l, a.c_str(), b.c_str(), c.c_str());
	};
	const auto mul2 = [](std::string a, std::string b, std::string c, int l) {
		tprint( "flags[%i] *= safe_mul(%s, %s, %s);\n", l, a.c_str(), b.c_str(), c.c_str());
	};
	const auto add = [](std::string a, std::string b, std::string c, int l) {
		tprint( "flags[%i] *= safe_add(%s, %s, %s);\n", l, a.c_str(), b.c_str(), c.c_str());
	};
	tprint("T flags[]={");
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
	tprint("O[0] *= flags[0];\n");
	for (int n = 1; n <= P; n++) {
		tprint("flags[%i] *= flags[%i];\n", n, n - 1);
		tprint("flag = flags[%i];\n", n);
		for (int m = -n; m <= n; m++) {
			tprint("O[%i] *= flag;\n", index(n, m), n);
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}");
	tprint("\n");
	return fname;
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
					char* gxstr = nullptr;
					char* gystr = nullptr;
					char* mxstr = nullptr;
					char* mystr = nullptr;
					if (m + l > 0) {
						ASPRINTF(&gxstr, "O[%i]", lindex(n + k, m + l));
						ASPRINTF(&gystr, "O[%i]", lindex(n + k, -m - l));
					} else if (m + l < 0) {
						if (abs(m + l) % 2 == 0) {
							ASPRINTF(&gxstr, "O[%i]", lindex(n + k, -m - l));
							ASPRINTF(&gystr, "O[%i]", lindex(n + k, m + l));
							gysgn = -1;
						} else {
							ASPRINTF(&gxstr, "O[%i]", lindex(n + k, -m - l));
							ASPRINTF(&gystr, "O[%i]", lindex(n + k, m + l));
							gxsgn = -1;
						}
					} else {
						greal = true;
						ASPRINTF(&gxstr, "O[%i]", lindex(n + k, 0));
					}
					if (l > 0) {
						ASPRINTF(&mxstr, "M[%i]", mindex(k, l));
						ASPRINTF(&mystr, "M[%i]", mindex(k, -l));
						mysgn = -1;
					} else if (l < 0) {
						if (l % 2 == 0) {
							ASPRINTF(&mxstr, "M[%i]", mindex(k, -l));
							ASPRINTF(&mystr, "M[%i]", mindex(k, l));
						} else {
							ASPRINTF(&mxstr, "M[%i]", mindex(k, -l));
							ASPRINTF(&mystr, "M[%i]", mindex(k, l));
							mxsgn = -1;
							mysgn = -1;
						}
					} else {
						mreal = true;
						ASPRINTF(&mxstr, "M[%i]", mindex(k, 0));
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
				tprint_chain("L[%i] = -L[%i];\n", lindex(n, m), lindex(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", lindex(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), lindex(n, m));
				}
				tprint_chain("L[%i] = -L[%i];\n", lindex(n, m), lindex(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", lindex(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", lindex(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), lindex(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("L[%i] = -L[%i];\n", lindex(n, -m), lindex(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", lindex(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), lindex(n, -m));
				}
				tprint_chain("L[%i] = -L[%i];\n", lindex(n, -m), lindex(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", lindex(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", lindex(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), lindex(n, -m));
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
	auto fname = func_header("P2L", P, true, true, true, true, "", "L", EXP, "M", LIT, "x", LIT, "y", LIT, "z", LIT);
	init_reals("O", exp_sz(P));
	init_real("tmp1");
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
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
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string greens(int P) {
	auto fname = func_header("greens", P, true, false, false, true, "", "O", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("r2");
	init_real("r2inv");
	init_real("ax0");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
	greens_body(P);
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}");
	tprint("\n");
	return fname;
}

std::string greens_xz(int P) {
	auto fname = func_header("greens_xz", P, false, false, false, true, "", "O", HEXP, "x", LIT, "z", LIT, "r2inv", LIT);
	init_real("ax0");
	init_real("ay0");
	init_real("ay1");
	init_real("ay2");
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
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2LG(int P, int Q) {
	auto fname = func_header("M2LG", P, true, true, false, true, "", "L", EXP, "M", CMUL, "O", EXP);
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
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
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
	auto fname = func_header("greens_ewald", P, true, true, false, true, "", "G", EXP, "x0", LIT, "y0", LIT, "z0", LIT);
	const auto c = tprint_on;
	tprint("expansion%s%s<%s, %i> Gr_st;\n", period_name(), scaled_name(), type.c_str(), P);
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
	tprint("r2 = fma(x0, x0, fma(y0, y0, z0 * z0));\n");
	tprint("rzero = TCONVERT(r2 < TCAST(%0.20e));\n", tiny());
	tprint("r = sqrt(r2) + rzero;\n");
	tprint("greens_safe(Gr_st, x0 + rzero, y0, z0);\n");
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
				tprint("detail::greens_ewald_real<%s, %i, %i>(G_st, %s, %s, %s);\n", type.c_str(), P, lround(alpha * 100), xstr.c_str(), ystr.c_str(),
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
					tprint("x2y2 = fma(TCAST(%i), y0, x2);\n", hy);
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
						tprint("hdotx = fma(TCAST(%i), z0, x2y2);\n", hz);
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
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;

}

std::string M2L_rot0(int P, int Q) {
	std::string fname;
	if (Q > 1) {
		fname = func_header("M2L", P, true, true, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	} else {
		fname = func_header("M2P", P, true, true, true, true, "", "f", FORCE, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	}
	tprint("/* algorithm= no rotation, full l^4 */\n");
	init_real("tmp1");
	if (Q == 1) {
		init_real("rinv");
		init_real("r2inv");
	}
	tprint("expansion%s%s<%s, %i> O_st;\n", period_name(), scaled_name(), type.c_str(), P);
	tprint("T* O(O_st.data());\n", type.c_str(), P);
	if (!(Q > 1 && scaled)) {
		tprint("const multipole%s%s%s<%s, %i>& M_st(M0_st);\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
		tprint("const T* M(M_st.data());\n", type.c_str(), P);
	} else {
		tprint("multipole%s%s%s<%s, %i> M_st;\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
		tprint("T* M(M_st.data());\n");
	}
	if (Q > 1) {
		init_real("a");
		init_real("b");
		tprint("expansion%s%s<%s, %i>& L_st(L0_st);\n", period_name(), scaled_name(), type.c_str(), P);
		tprint("T* L(L_st.data());\n", type.c_str(), P);
		if (scaled) {
			tprint("tmp1 = TCAST(1) / L_st.scale();\n");
			tprint("a = M0_st.r * tmp1;\n");
			tprint("b = a;\n");
			tprint("M_st.r = L_st.r;\n");
			tprint("M_st.o[0] = M0_st.o[0];\n");
			for (int n = 1; n < P; n++) {
				if (!(nodip && n == 1)) {
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
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
		}
	} else {
		init_reals("L", exp_sz(Q));
		tprint("L[0] = TCAST(0);\n");
		tprint("L[1] = TCAST(0);\n");
		tprint("L[2] = TCAST(0);\n");
		tprint("L[3] = TCAST(0);\n");
		if (scaled) {
			tprint("tmp1 = TCAST(1) / M0_st.scale();\n");
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
		}
	}
	tprint("greens(O_st, x, y, z);\n");
	m2lg_body(P, Q);
	if (Q == 1) {
		if (scaled) {
			tprint("rinv = TCAST(1) / M_st.scale();\n");
			tprint("r2inv = rinv * rinv;\n");
			if (!nopot) {
				tprint("f.potential = fma(L[0], rinv, f.potential);\n");
			}
			tprint("f.force[0] -= L[3] * r2inv;\n");
			tprint("f.force[1] -= L[1] * r2inv;\n");
			tprint("f.force[2] -= L[2] * r2inv;\n");
		} else {
			if (!nopot) {
				tprint("f.potential += L[0];\n");
			}
			tprint("f.force[0] -= L[3];\n");
			tprint("f.force[1] -= L[1];\n");
			tprint("f.force[2] -= L[2];\n");
		}
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
	tprint("/* algorithm= z rotation only, half l^4 */\n");
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
	init_reals("rx", std::max(P, Q+1));
	init_reals("ry", std::max(P, Q+1));
	init_real("tmp0");
	init_reals("L", exp_sz(Q));
	tprint("detail::expansion_xz%s<%s, %i> O_st;\n", period_name(), type.c_str(), P);
	tprint("multipole%s%s%s<%s, %i> M_st;\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
	tprint("T* O(O_st.data());\n", type.c_str(), P);
	tprint("T* M(M_st.data());\n");
	if (Q == 1) {
		init_real("rinv");
	}
	bool minit = false;
	if (scaled) {
		if (Q > 1) {
			init_real("a");
			init_real("b");
			tprint("tmp1 = TCAST(1) / L0_st.scale();\n");
			tprint("a = M0_st.r * tmp1;\n");
			tprint("b = a;\n");
			tprint("M_st.r = L0_st.r;\n");
			tprint("M_st.o[0] = M0_st.o[0];\n");
			for (int n = 1; n < P; n++) {
				if (!(nodip && n == 1)) {
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
			minit = true;
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
		}
	}
	if (!minit) {
		if (scaled) {
			tprint("tmp1 = TCAST(1) / M0_st.scale();\n");
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
		}
		for (int n = 0; n < mul_sz(P); n++) {
			tprint("M_st.o[%i] = M0_st.o[%i];\n", n, n);
		}
		if (periodic && P > 2) {
			tprint("M_st.t = M0_st.t;\n");
		}
		if (scaled) {
			tprint("M_st.r = M0_st.r;\n");
		}
	}

	tprint("R2 = fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("r2 = fma(z, z, R2);\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt(tmp1);\n");
	tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
	tprint("r2inv = TCAST(1) / (r2 + rzero);\n");
	tprint("cosphi = fma(x, Rinv, Rzero);\n");
	tprint("sinphi = -y * Rinv;\n");
	z_rot(P - 1, "M", false, false, false);
	tprint("detail::greens_xz(O_st, R, z, r2inv);\n");
	const auto oindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};
	bool first[(Q + 1)][(2 * Q + 1)];
	for (int n = 0; n <= Q; n++) {
		for (int m = -n; m <= n; m++) {
			first[n][n + m] = true;
		}
	}
	for (int n = nopot; n <= Q; n++) {
		for (int m = 0; m <= n; m++) {
			tprint_new_chain();
			const int kmax = std::min(P - n, P - 1);
			for (int sgn = -1; sgn <= 1; sgn += 2) {
				for (int k = 0; k <= kmax; k++) {
					if (nodip && k == 1) {
						continue;
					}
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
							ASPRINTF(&mxstr, "M[%i]", mindex(k, l));
							ASPRINTF(&mystr, "M[%i]", mindex(k, -l));
							mysgn = -1;
						} else if (l < 0) {
							if (l % 2 == 0) {
								ASPRINTF(&mxstr, "M[%i]", mindex(k, -l));
								ASPRINTF(&mystr, "M[%i]", mindex(k, l));
							} else {
								ASPRINTF(&mxstr, "M[%i]", mindex(k, -l));
								ASPRINTF(&mystr, "M[%i]", mindex(k, l));
								mxsgn = -1;
								mysgn = -1;
							}
						} else {
							mreal = true;
							ASPRINTF(&mxstr, "M[%i]", mindex(k, 0));
						}
						if (!mreal) {
							if (mxsgn * gxsgn == sgn) {
								if (first[n][n + m]) {
									tprint_chain("L[%i] = %s * %s;\n", lindex(n, m), mxstr, gxstr);
									first[n][n + m] = false;
								} else {
									tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", lindex(n, m), mxstr, gxstr, lindex(n, m));
								}
							}
							if (mysgn * gxsgn == sgn) {
								if (m > 0) {
									if (first[n][n - m]) {
										tprint_chain("L[%i] = %s * %s;\n", lindex(n, -m), mystr, gxstr);
										first[n][n - m] = false;
									} else {
										tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", lindex(n, -m), mystr, gxstr, lindex(n, -m));
									}
								}
							}
						} else {
							if (mxsgn * gxsgn == sgn) {
								if (first[n][n + m]) {
									tprint_chain("L[%i] = %s * %s;\n", lindex(n, m), mxstr, gxstr);
									first[n][n + m] = false;
								} else {
									tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", lindex(n, m), mxstr, gxstr, lindex(n, m));
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
				if (!first[n][n + m] && sgn == -1) {
					tprint_chain("L[%i] = -L[%i];\n", lindex(n, m), lindex(n, m));
				}
				if (!first[n][n - m] && sgn == -1) {
					tprint_chain("L[%i] = -L[%i];\n", lindex(n, -m), lindex(n, -m));
				}
			}
		}
	}
	tprint_flush_chains();
	tprint("sinphi = -sinphi;\n");
	if (nodip) {
		z_rot(Q, "L", false, false, 2 * ((Q == P) || (Q == 1 && P == 2)));
	} else {
		z_rot(Q, "L", false, false, Q == P);
	}
	if (Q > 1) {
		for (int n = 0; n < exp_sz(Q); n++) {
			tprint("L0[%i] += L[%i];\n", n, n);
		}
	} else {
		if (scaled) {
			tprint("rinv = TCAST(1) / M_st.scale();\n");
			tprint("r2inv = rinv * rinv;\n");
			if (!nopot) {
				tprint("f.potential = fma(L[0], rinv, f.potential);\n");
			}
			tprint("f.force[0] -= L[3] * r2inv;\n");
			tprint("f.force[1] -= L[1] * r2inv;\n");
			tprint("f.force[2] -= L[2] * r2inv;\n");
		} else {
			if (!nopot) {
				tprint("f.potential += L[0];\n");
			}
			tprint("f.force[0] -= L[3];\n");
			tprint("f.force[1] -= L[1];\n");
			tprint("f.force[2] -= L[2];\n");
		}
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
	tprint("/* algorithm= z rotation and x/z swap, l^3 */\n");
	tprint("multipole%s%s%s<%s, %i> M_st;\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
	tprint("T* M(M_st.data());\n");
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
	init_reals("rx", std::max(P, Q+1));
	init_reals("ry", std::max(P, Q+1));
	init_real("tmp0");
	init_real("r2przero");
	init_real("rinv");
	bool minit = false;
	if (scaled) {
		if (Q > 1) {
			init_real("a");
			init_real("b");
			tprint("tmp1 = TCAST(1) / L0_st.scale();\n");
			tprint("a = M0_st.r * tmp1;\n");
			tprint("b = a;\n");
			tprint("M_st.r = L0_st.r;\n");
			tprint("M_st.o[0] = M0_st.o[0];\n");
			for (int n = 1; n < P; n++) {
				if (!(nodip && n == 1)) {
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
			minit = true;
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
		}
	}
	if (!minit) {
		if (scaled) {
			tprint("tmp1 = TCAST(1) / M0_st.scale();\n");
			tprint("x *= tmp1;\n");
			tprint("y *= tmp1;\n");
			tprint("z *= tmp1;\n");
		}
		for (int n = 0; n < mul_sz(P); n++) {
			tprint("M_st.o[%i] = M0_st.o[%i];\n", n, n);
		}
		if (periodic && P > 2) {
			tprint("M_st.t = M0_st.t;\n");
		}
		if (scaled) {
			tprint("M_st.r = M0_st.r;\n");
		}
	}

	tprint("R2 = fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt(tmp1);\n");
	tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
	tprint("r2 = (fma(z, z, R2));\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("r2przero = (r2 + rzero);\n");
	tprint("rinv = rsqrt(r2przero);\n");
	tprint("cosphi = y * Rinv;\n");
	tprint("sinphi = fma(x, Rinv, Rzero);\n");
	z_rot(P - 1, "M", false, false, false);
	xz_swap(P - 1, "M", false, false, false, false);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = fma(z, rinv, rzero);\n");
	tprint("sinphi = -R * rinv;\n");
	z_rot(P - 1, "M", false, false, false);
	xz_swap(P - 1, "M", false, true, false, false);
	for( int i = 0; i < exp_sz(Q); i++) {
		tprint( "L[%i] = TCAST(0);\n", i);
	}
	m2l(P, Q, "M", "L");
	xz_swap(Q, "L", true, false, true, false);
	tprint("sinphi = -sinphi;\n");
	z_rot(Q, "L", true, false, false);
	xz_swap(Q, "L", true, false, false, true);
	tprint("cosphi = cosphi0;\n");
	tprint("sinphi = -sinphi0;\n");
	z_rot(Q, "L", false, true, false);
	if (Q > 1) {
		for (int n = nopot; n < exp_sz(Q); n++) {
			tprint("L0[%i] += L[%i];\n", n, n);
		}

	} else {
		if (scaled) {
			tprint("rinv = TCAST(1) / M_st.scale();\n");
			tprint("r2inv = rinv * rinv;\n");
			if (!nopot) {
				tprint("f.potential = fma(L[0], rinv, f.potential);\n");
			}
			tprint("f.force[0] -= L[3] * r2inv;\n");
			tprint("f.force[1] -= L[1] * r2inv;\n");
			tprint("f.force[2] -= L[2] * r2inv;\n");
		} else {
			if (!nopot) {
				tprint("f.potential += L[0];\n");
			}
			tprint("f.force[0] -= L[3];\n");
			tprint("f.force[1] -= L[1];\n");
			tprint("f.force[2] -= L[2];\n");
		}
	}
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2L_ewald(int P) {
	auto fname = func_header("M2L_ewald", P, true, true, true, true, "", "L0", EXP, "M0", CMUL, "x", LIT, "y", LIT, "z", LIT);
	tprint("expansion%s%s<%s, %i> G_st;\n", period_name(), scaled_name(), type.c_str(), P);
	tprint("expansion%s%s<%s,%i> L_st;\n", period_name(), scaled_name(), type.c_str(), P);
	tprint("multipole%s%s%s<%s,%i> M_st;\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
	tprint("T* G(G_st.data());\n", type.c_str(), P);
	tprint("T* M(M_st.data());\n");
	tprint("T* L(L_st.data());\n");
	if (scaled) {
		init_real("a");
		init_real("b");
	}
	for (int n = nopot; n < exp_sz(P); n++) {
		tprint("L[%i] = TCAST(0);\n", n);
	}
	if (scaled) {
		tprint("L_st.r = TCAST(1);\n");
	}
	if (periodic && P > 1) {
		tprint("L_st.trace2() = TCAST(0);\n");
	}
	if (scaled) {
		tprint("a = M0_st.r;\n");
		tprint("b = a;\n");
		tprint("M_st.r = T(1);\n");
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
	tprint("greens_ewald%s(G_st, x, y, z);\n", nopot ? "_wo_potential" : "");
	tprint("M2LG%s(L_st, M_st, G_st);\n", nopot ? "_wo_potential" : "");
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
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string regular_harmonic(int P) {
	auto fname = func_header("regular_harmonic", P, false, false, false, true, "", "Y", EXP, "x", LIT, "y", LIT, "z", LIT);
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
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
	auto fname = func_header("regular_harmonic_xz", P, false, false, false, true, "", "Y", HEXP, "x", LIT, "z", LIT);
	init_real("ax0");
	init_real("ay0");
	init_real("ax1");
	init_real("ay1");
	init_real("ax2");
	init_real("ay2");
	init_real("r2");
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
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2M_rot0(int P) {
	auto fname = func_header("M2M", P + 1, true, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	if (P < 2 && nodip || P < 1) {
		deindent();
		tprint("}\n");
		tprint("\n");
		tprint("}\n");
		tprint("\n");
		return fname;
	}
	tprint("/* algorithm= no rotation, full l^4 */\n");
	init_real("tmp1");
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("expansion%s%s<%s, %i> Y_st;\n", period_name(), scaled_name(), type.c_str(), P);
	tprint("T* Y(Y_st.data());\n", type.c_str(), P);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z);\n");
	if (P > 1 && !nopot && periodic) {
		tprint("M_st.trace2() = fma(TCAST(-4) * x, M[%i], M_st.trace2());\n", mindex(1, 1));
		tprint("M_st.trace2() = fma(TCAST(-4) * y, M[%i], M_st.trace2());\n", mindex(1, -1));
		tprint("M_st.trace2() = fma(TCAST(-2) * z, M[%i], M_st.trace2());\n", mindex(1, 0));
		tprint("M_st.trace2() = fma(x * x, M[%i], M_st.trace2());\n", mindex(0, 0));
		tprint("M_st.trace2() = fma(y * y, M[%i], M_st.trace2());\n", mindex(0, 0));
		tprint("M_st.trace2() = fma(z * z, M[%i], M_st.trace2());\n", mindex(0, 0));
	}
	for (int n = P; n >= 0; n--) {
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
				if (nodip && n - k == 1) {
					continue;
				}
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
					if (l > 0) {
						ASPRINTF(&gxstr, "Y[%i]", lindex(k, abs(l)));
						ASPRINTF(&gystr, "Y[%i]", lindex(k, -abs(l)));
					} else if (l < 0) {
						if (abs(l) % 2 == 0) {
							ASPRINTF(&gxstr, "Y[%i]", lindex(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", lindex(k, -abs(l)));
							gysgn = -1;
						} else {
							ASPRINTF(&gxstr, "Y[%i]", lindex(k, abs(l)));
							ASPRINTF(&gystr, "Y[%i]", lindex(k, -abs(l)));
							gxsgn = -1;
						}
					} else {
						ASPRINTF(&gxstr, "Y[%i]", lindex(k, 0));
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
	deindent();
	tprint("}\n");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string M2M_rot1(int P) {
	auto index = mindex;
	auto fname = func_header("M2M", P + 1, true, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	tprint("/* algorithm= z rotation only, half l^4 */\n");
	if (P < 2 && nodip || P < 1) {
		deindent();
		tprint("}\n");
		tprint("\n");
		tprint("}\n");
		tprint("\n");
		return fname;
	}
	tprint("detail::expansion_xz%s<%s, %i> Y_st;\n", period_name(), type.c_str(), P);
	tprint("T* Y(Y_st.data());\n", type.c_str(), P);
	init_reals("rx\n", P + 1);
	init_reals("ry\n", P + 1);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("tmp1");
	init_real("R2");
	init_real("Rzero");
	init_real("cosphi");
	init_real("sinphi");
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("R2 = fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt(tmp1);\n");
	tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
	tprint("cosphi = fma(x, Rinv, Rzero);\n");
	tprint("sinphi = -y * Rinv;\n");
	z_rot(P, "M", false, false, false);
	if (P > 1 && !nopot && periodic) {
		tprint("M_st.trace2() = fma(TCAST(-4) * R, M[%i], M_st.trace2());\n", index(1, 1));
		tprint("M_st.trace2() = fma(TCAST(-2) * z, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = fma(R * R, M[%i], M_st.trace2());\n", index(0, 0));
		tprint("M_st.trace2() = fma(z * z, M[%i], M_st.trace2());\n", index(0, 0));
	}
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("detail::regular_harmonic_xz(Y_st, -R, -z);\n");
	for (int n = P; n >= 0; n--) {
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
				if (nodip && n - k == 1) {
					continue;
				}
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
				tprint_chain("M[%i] = -M[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("M[%i] = -M[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("M[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("M[%i] = -M[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("M[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("M[%i] = fma(%s, %s, M[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
		tprint_flush_chains();
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
	auto index = mindex;
	auto fname = func_header("M2M", P + 1, true, true, true, true, "", "M", MUL, "x", LIT, "y", LIT, "z", LIT);
	tprint("/* algorithm= z rotation and x/z swap, l^3 */\n");
	if (P < 2 && nodip || P < 1) {
		deindent();
		tprint("}\n");
		tprint("\n");
		tprint("}\n");
		tprint("\n");
		return fname;
	}
	init_reals("A", 2 * P + 1);
	init_reals("rx\n", P + 1);
	init_reals("ry\n", P + 1);
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
	if (scaled) {
		tprint("tmp1 = TCAST(1) / M_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("R2 = fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt(tmp1);\n");
	tprint("r2 = fma(z, z, R2);\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = rsqrt(tmp1);\n");
	tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
	tprint("r = (TCAST(1) - rzero) / rinv;\n");
	tprint("cosphi = y * Rinv;\n");
	tprint("sinphi = fma(x, Rinv, Rzero);\n");
	z_rot(P, "M", false, false, false);
	xz_swap(P, "M", false, false, false, false);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = fma(z, rinv, rzero);\n");
	tprint("sinphi = -R * rinv;\n");
	z_rot(P, "M", false, false, false);
	xz_swap(P, "M", false, false, false, false);
	if (P > 1 && !nopot && periodic) {
		tprint("M_st.trace2() = fma(TCAST(-2) * r, M[%i], M_st.trace2());\n", index(1, 0));
		tprint("M_st.trace2() = fma(r * r, M[%i], M_st.trace2());\n", index(0, 0));
	}
	tprint("A[0] = TCAST(1);\n");
	for (int n = 1; n <= P; n++) {
		tprint("A[%i] = -r * A[%i];\n", n, n - 1);
	}
	for (int n = 2; n <= P; n++) {
		tprint("A[%i] *= TCAST(%.20e);\n", n, 1.0 / factorial(n));
	}
	for (int n = P; n >= 0; n--) {
		if (nodip && n == 1) {
			continue;
		}
		for (int m = 0; m <= n; m++) {
			tprint_new_chain();
			for (int k = 1; k <= n; k++) {
				if (nodip && n - k == 1) {
					continue;
				}
				if (abs(m) > n - k) {
					continue;
				}
				if (-abs(m) < k - n) {
					continue;
				}
				tprint_chain("M[%i] = fma(M[%i], A[%i], M[%i]);\n", index(n, m), index(n - k, m), k, index(n, m));
				if (m > 0) {
					tprint_chain("M[%i] = fma(M[%i], A[%i], M[%i]);\n", index(n, -m), index(n - k, -m), k, index(n, -m));
				}
			}
		}
		tprint_flush_chains();
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

std::string P2M(int P) {
	tprint("\n");
	auto fname = func_header("P2M", P + 1, true, true, true, true, "", "M", MUL, "m", LIT, "x", LIT, "y", LIT, "z", LIT);
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
	if (!nopot && periodic & P > 1) {
		tprint("M_st.trace2() = m * r2;\n");
	}
	for (int m = 0; m <= P; m++) {
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
	for (int m = 0; m <= P; m++) {
		tprint_new_chain();
		if (m + 1 <= P) {
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
		for (int n = m + 2; n <= P; n++) {
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
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	return fname;
}

std::string L2L_rot0(int P) {
	auto index = lindex;
	auto fname = func_header("L2L", P, true, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	tprint("/* algorithm= no rotation, full l^4 */\n");
	init_real("tmp1");
	if (scaled) {
		tprint("tmp1 = TCAST(1) / L_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("expansion%s%s<%s, %i> Y_st;\n", period_name(), scaled_name(), type.c_str(), P);
	tprint("T* Y(Y_st.data());\n", type.c_str(), P);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z);\n");

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
					tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
		tprint_flush_chains();
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = fma(TCAST(-2) * x, L_st.trace2(), L[%i]);\n", index(1, 1), index(1, 1));
		tprint("L[%i] = fma(TCAST(-2) * y, L_st.trace2(), L[%i]);\n", index(1, -1), index(1, -1));
		tprint("L[%i] = fma(TCAST(-2) * z, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L[%i] = fma(x * x, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = fma(y * y, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = fma(z * z, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
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
	auto index = lindex;

	auto fname = func_header("L2L", P, true, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	tprint("/* algorithm= z rotation only, half l^4 */\n");
	init_reals("rx", P+1);
	init_reals("ry", P+1);
	tprint("detail::expansion_xz%s<%s, %i> Y_st;\n", period_name(), type.c_str(), P);
	tprint("T* Y(Y_st.data());\n", type.c_str(), P);
	init_real("tmp0");
	init_real("R");
	init_real("Rinv");
	init_real("R2");
	init_real("Rzero");
	init_real("tmp1");
	init_real("cosphi");
	init_real("sinphi");
	if (scaled) {
		tprint("tmp1 = TCAST(1) / L_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("R2 = fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt(tmp1);\n");
	tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
	tprint("cosphi = fma(x, Rinv, Rzero);\n");
	tprint("sinphi = -y * Rinv;\n");
	z_rot(P, "L", false, false, false);
	const auto yindex = [](int l, int m) {
		return l*(l+1)/2+m;
	};

	tprint("detail::regular_harmonic_xz(Y_st, -R, -z);\n");
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
					tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("L[%i] = -L[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("L[%i] = fma(%s, %s, L[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
		tprint_flush_chains();
	}
	tprint("sinphi = -sinphi;\n");
	if (P > 1 && periodic) {
		tprint("L[%i] = fma(TCAST(-2) * R, L_st.trace2(), L[%i]);\n", index(1, 1), index(1, 1));
		tprint("L[%i] = fma(TCAST(-2) * z, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L[%i] = fma(R2, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
			tprint("L[%i] = fma(z * z, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
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
	auto index = lindex;
	auto fname = func_header("L2L", P, true, true, true, true, "", "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	tprint("/* algorithm= z rotation and x/z swap, l^3 */\n");
	init_reals("A", 2 * P + 1);
	init_reals("rx", P+1);
	init_reals("ry", P+1);
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
	if (scaled) {
		tprint("tmp1 = TCAST(1) / L_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("R2 = fma(x, x, y * y);\n");
	tprint("Rzero = TCONVERT( R2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = R2 + Rzero;\n");
	tprint("Rinv = rsqrt(tmp1);\n");
	tprint("r2 = fma(z, z, R2);\n");
	tprint("rzero = TCONVERT( r2 < TCAST(%.20e) );\n", tiny());
	tprint("tmp1 = r2 + rzero;\n");
	tprint("rinv = rsqrt(tmp1);\n");
	tprint("R = (TCAST(1) - Rzero) / Rinv;\n");
	tprint("r = (TCAST(1) - rzero) / rinv;\n");
	tprint("cosphi = y * Rinv;\n");
	tprint("sinphi = fma(x, Rinv, Rzero);\n");
	z_rot(P, "L", false, false, false);
	xz_swap(P, "L", true, false, false, false);
	tprint("cosphi0 = cosphi;\n");
	tprint("sinphi0 = sinphi;\n");
	tprint("cosphi = fma(z, rinv, rzero);\n");
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
				tprint_chain("L[%i] = fma(L[%i], A[%i], L[%i]);\n", index(n, m), index(n + k, m), k, index(n, m));
				if (m > 0) {
					tprint_chain("L[%i] = fma(L[%i], A[%i], L[%i]);\n", index(n, -m), index(n + k, -m), k, index(n, -m));
				}
			}
		}
		tprint_flush_chains();
	}
	if (P > 1 && periodic) {
		tprint("L[%i] = fma(TCAST(-2) * r, L_st.trace2(), L[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L[%i] = fma(r * r, L_st.trace2(), L[%i]);\n", index(0, 0), index(0, 0));
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
	auto index = lindex;
	auto fname = func_header("L2P", P, true, true, true, true, "", "f", FORCE, "L", EXP, "x", LIT, "y", LIT, "z", LIT);
	tprint("/* algorithm= no rotation, full l^4 */\n");
	init_real("tmp1");
	if (scaled) {
		tprint("tmp1 = TCAST(1) / L_st.scale();\n");
		tprint("x *= tmp1;\n");
		tprint("y *= tmp1;\n");
		tprint("z *= tmp1;\n");
	}
	tprint("expansion%s%s<%s, %i> Y_st;\n", period_name(), scaled_name(), type.c_str(), P);
	tprint("T* Y(Y_st.data());\n", type.c_str(), P);
	init_reals("L2", 4);
	tprint("detail::regular_harmonic(Y_st, -x, -y, -z);\n");
	if (!nopot) {
		tprint("L2[0] = L[0];\n");
	}
	tprint("L2[1] = L[1];\n");
	tprint("L2[2] = L[2];\n");
	tprint("L2[3] = L[3];\n");
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
				tprint_chain("L2[%i] = -L2[%i];\n", index(n, m), index(n, m));
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L2[%i] = fma(%s, %s, L2[%i]);\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str(), index(n, m));
				}
				tprint_chain("L2[%i] = -L2[%i];\n", index(n, m), index(n, m));
			} else {
				for (int i = 0; i < neg_real.size(); i++) {
					tprint_chain("L2[%i] -= %s * %s;\n", index(n, m), neg_real[i].first.c_str(), neg_real[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_real.size(); i++) {
				tprint_chain("L2[%i] = fma(%s, %s, L2[%i]);\n", index(n, m), pos_real[i].first.c_str(), pos_real[i].second.c_str(), index(n, m));
			}
			if (fmaops && neg_imag.size() >= 2) {
				tprint_chain("L2[%i] = -L2[%i];\n", index(n, -m), index(n, -m));
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L2[%i] = fma(%s, %s, L2[%i]);\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str(), index(n, -m));
				}
				tprint_chain("L2[%i] = -L2[%i];\n", index(n, -m), index(n, -m));
			} else {
				for (int i = 0; i < neg_imag.size(); i++) {
					tprint_chain("L2[%i] -= %s * %s;\n", index(n, -m), neg_imag[i].first.c_str(), neg_imag[i].second.c_str());
				}
			}
			for (int i = 0; i < pos_imag.size(); i++) {
				tprint_chain("L2[%i] = fma(%s, %s, L2[%i]);\n", index(n, -m), pos_imag[i].first.c_str(), pos_imag[i].second.c_str(), index(n, -m));
			}
		}
		tprint_flush_chains();
	}
	if (P > 1 && periodic) {
		tprint("L2[%i] = fma(TCAST(-2) * x, L_st.trace2(), L2[%i]);\n", index(1, 1), index(1, 1));
		tprint("L2[%i] = fma(TCAST(-2) * y, L_st.trace2(), L2[%i]);\n", index(1, -1), index(1, -1));
		tprint("L2[%i] = fma(TCAST(-2) * z, L_st.trace2(), L2[%i]);\n", index(1, 0), index(1, 0));
		if (!nopot) {
			tprint("L2[%i] = fma(x * x, L_st.trace2(), L2[%i]);\n", index(0, 0), index(0, 0));
			tprint("L2[%i] = fma(y * y, L_st.trace2(), L2[%i]);\n", index(0, 0), index(0, 0));
			tprint("L2[%i] = fma(z * z, L_st.trace2(), L2[%i]);\n", index(0, 0), index(0, 0));
		}
	}
	if (scaled) {
		if (!nopot) {
			tprint("L2[0] *= tmp1;");
		}
		tprint("tmp1 *= tmp1;\n");
		tprint("L2[1] *= tmp1;\n");
		tprint("L2[2] *= tmp1;\n");
		tprint("L2[3] *= tmp1;\n");
	}
	if (!nopot) {
		tprint("f.potential += L2[0];\n");
	}
	tprint("f.force[0] -= L2[3];\n");
	tprint("f.force[1] -= L2[1];\n");
	tprint("f.force[2] -= L2[2];\n");
	deindent();
	tprint("}");
	tprint("\n");
	tprint("}\n");
	tprint("\n");
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

void math_float(bool cu) {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	tprint("SFMM_PREFIX inline float fma(float a, float b, float c) {\n");
	indent();
	tprint("return fmaf(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("SFMM_PREFIX float rsqrt(float);\n");
	tprint("SFMM_PREFIX float sqrt(float);\n");
	tprint("SFMM_PREFIX void sincos(float, float*, float*);\n");
	tprint("SFMM_PREFIX void erfcexp(float, float*, float*);\n");
	tprint("SFMM_PREFIX float safe_mul(float&, float, float);\n");
	tprint("SFMM_PREFIX float safe_add(float&, float, float);\n");
	fclose(fp);
	if (cu) {
		fp = fopen("./generated_code/src/math/math_float.cu", "wt");
	} else {
		fp = fopen("./generated_code/src/math/math_float.cpp", "wt");
	}
	tprint("\n");
	tprint("#include \"%s\"\n", header.c_str());
	tprint("#include \"typecast_float.hpp\"\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
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
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
	tprint("y *= fma(x, y * y, TCAST(-1.5));\n");
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
				tprint("*erfc0 = fma(*erfc0, q, TCAST(%.20e));\n", 1.0 / dfactorial(2 * n + 1));
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
	}
	safe_math_float();
	tprint("}\n");
	tprint("\n");
	fclose(fp);

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
	tprint("inline v%isf fma(v%isf a, v%isf b, v%isf c) {\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("v%isf rsqrt(v%isf);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("v%isf sqrt(v%isf);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("void sincos(v%isf, v%isf*, v%isf*);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("void erfcexp(v%isf, v%isf*, v%isf*);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("v%isf safe_mul(v%isf&, v%isf, v%isf);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
	tprint("v%isf safe_add(v%isf&, v%isf, v%isf);\n", VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE, VEC_FLOAT_SIZE);
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

void math_double(bool cu) {
	if (fp) {
		fclose(fp);
	}
	const char* sout = "*s";
	const char* cout = "*c";
	constexpr double x0 = 1.0;
	constexpr double x1 = 4.7;

	fp = fopen(full_header.c_str(), "at");
	tprint("\n");
	tprint("SFMM_PREFIX inline double fma(double a, double b, double c) {\n");
	indent();
	tprint("return std::fma(a, b, c);\n");
	deindent();
	tprint("}\n");
	tprint("SFMM_PREFIX double rsqrt(double);\n");
	tprint("SFMM_PREFIX double sqrt(double);\n");
	tprint("SFMM_PREFIX void sincos(double, double*, double*);\n");
	tprint("SFMM_PREFIX void erfcexp(double, double*, double*);\n");
	tprint("SFMM_PREFIX double safe_mul(double&, double, double);\n");
	tprint("SFMM_PREFIX double safe_add(double&, double, double);\n");
	fclose(fp);
	if (cu) {
		fp = fopen("./generated_code/src/math/math_double.cu", "wt");
	} else {
		fp = fopen("./generated_code/src/math/math_double.cpp", "wt");
	}
	tprint("\n");
	tprint("#include \"%s\"\n", header.c_str());
	tprint("#include \"typecast_double.hpp\"\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
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
	safe_math_double();
	tprint("\n");
	tprint("}\n");
	tprint("\n");
	fclose(fp);

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
	tprint("\n");
	tprint("inline v%idf fma(v%idf a, v%idf b, v%idf c) {\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	indent();
	tprint("return a * b + c;\n");
	deindent();
	tprint("}\n");
	tprint("v%idf rsqrt(v%idf);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("v%idf sqrt(v%idf);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("void sincos(v%idf, v%idf*, v%idf*);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("void erfcexp(v%idf, v%idf*, v%idf*);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("v%idf safe_mul(v%idf&, v%idf, v%idf);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("v%idf safe_add(v%idf&, v%idf, v%idf);\n", VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE, VEC_DOUBLE_SIZE);
	tprint("#endif\n");
	fclose(fp);
	fp = fopen("./generated_code/src/math/math_vec_double.cpp", "wt");
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
	SYSTEM("mkdir -p ./generated_code/include/detail\n");
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
	tprint("#include <cstdio>\n");
	tprint("#include <cmath>\n");
	tprint("#include <cstdint>\n");
	tprint("#include <limits>\n");
	tprint("#include <utility>\n");
	tprint("\n");
	tprint("#define sfmmCalculateWithPotential    0\n");
	tprint("#define sfmmCalculateWithoutPotential 1\n");
	tprint("\n");
	tprint("namespace sfmm {\n");
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
	tprint("\n");
	tprint("#else\n");
	tprint("namespace sfmm {\n");
	tprint("#endif\n");
	tprint("\n");
	fprintf(fp, "%s\n", complex_header.c_str());
	nopot = 1;
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
	basetype = "float";
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
	basetype = "float";
	ucuda.push_back(false);
	ntypenames++;
	funcnum = 0;
#endif
#if defined(DOUBLE)
	rtypenames.push_back("double");
	sitypenames.push_back("int64_t");
	uitypenames.push_back("uint64_t");
	ucuda.push_back(false);
	basetype = "double";
	ntypenames++;
	funcnum = 1;
#endif
#ifdef VEC_FLOAT
	rtypenames.push_back(vf);
	sitypenames.push_back(vui32);
	uitypenames.push_back(vsi32);
	ucuda.push_back(false);
	basetype = "float";
	ntypenames++;
	funcnum = 2;
#endif

#ifdef VEC_DOUBLE
	rtypenames.push_back(vd);
	sitypenames.push_back(vui64);
	uitypenames.push_back(vsi64);
	basetype = "double";
	ucuda.push_back(false);
	ntypenames++;
	funcnum = 3;
#endif
#if defined(VEC_DOUBLE) || defined(VEC_FLOAT)
	tprint("#ifndef __CUDACC__\n");
	tprint("%s\n", vec_header().c_str());
	tprint("\n#endif");
	tprint("\n");
#endif
	tprint("#ifndef SFMM_EXPANSION_MEMBERS42\n");
	tprint("#define SFMM_EXPANSION_MEMBERS42\n");
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
			"\t\tconst int m0 = abs(m); \\\n"
			"\t\tconst int ip = n2n + m0; \\\n"
			"\t\tconst int im = n2n - m0; \\\n"
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
			"\t\tconst int m0 = abs(m); \\\n"
			"\t\tconst int ip = n2n + m0; \\\n"
			"\t\tconst int im = n2n - m0; \\\n"
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
	std::string str2 = "#define SFMM_EXPANSION_MEMBERS_WO_DIPOLE(classname, type, ppp) \\\n"
			"class reference { \\\n"
			"\tT* ax; \\\n"
			"\tT* ay; \\\n"
			"\tT rsgn; \\\n"
			"\tT isgn; \\\n"
			"public: \\\n"
			"\toperator complex<T>() const { \\\n"
			"\t\tif(ax != ay) { \\\n"
			"\t\t\treturn complex<T>(rsgn * *ax, isgn * *ay); \\\n"
			"\t\t} else { \\\n"
			"\t\t\treturn complex<T>(rsgn * *ax, T(0)); \\\n"
			"\t\t} \\\n"
			"\t} \\\n"
			"\treference& operator=(complex<T> other) { \\\n"
			"\t\t*ax = other.real() * rsgn; \\\n"
			"\t\tif(ax != ay) { \\\n"
			"\t\t\t*ay = other.imag() * isgn; \\\n"
			"\t\t} else { \\\n"
			"\t\t\t*ay = T(0); \\\n"
			"\t\t} \\\n"
			" \t\treturn *this; \\\n"
			"\t} \\\n"
			"\tfriend classname<type,ppp>; \\\n"
			"}; \\\n"
			"SFMM_PREFIX complex<T> operator()(int n, int m) const { \\\n"
			"\tcomplex<T> c; \\\n"
			"\tconst int n2n = n * n + n; \\\n"
			"\tconst int m0 = abs(m); \\\n"
			"\tconst int ip = n == 0 ? 0 : n2n + m0 - 3; \\\n"
			"\tconst int im = n == 0 ? 0 : n2n - m0 - 3; \\\n"
			"\tc.real() = o[ip]; \\\n"
			"\tc.imag() = o[im]; \\\n"
			"\tif( m < 0 ) { \\\n"
			"\t\tif( m % 2 == 0 ) { \\\n"
			"\t\t\tc.imag() = -c.imag(); \\\n"
			"\t\t} else { \\\n"
			"\t\t\tc.real() = -c.real(); \\\n"
			"\t\t} \\\n"
			"\t} else if( m == 0 ) { \\\n"
			"\t\tc.imag() = T(0); \\\n"
			"\t} \\\n"
			"\treturn c; \\\n"
			"} \\\n"
			"SFMM_PREFIX reference operator()(int n, int m) { \\\n"
			"\treference ref; \\\n"
			"\tconst int n2n = n * n + n; \\\n"
			"\tconst int m0 = abs(m); \\\n"
			"\tconst int ip = n == 0 ? 0 : n2n + m0 - 3; \\\n"
			"\tconst int im = n == 0 ? 0 : n2n - m0 - 3; \\\n"
			"\tref.ax = o + ip; \\\n"
			"\tref.ay = o + im; \\\n"
			"\tref.rsgn = ref.isgn = T(1); \\\n"
			"\tif( m < 0 ) { \\\n"
			"\t\tif( m % 2 == 0 ) { \\\n"
			"\t\t\tref.isgn = -ref.isgn; \\\n"
			"\t\t} else { \\\n"
			"\t\t\tref.rsgn = -ref.rsgn; \\\n"
			"\t\t} \\\n"
			"\t} \\\n"
			"\treturn ref; \\\n"
			"} \\\n"
			"SFMM_PREFIX classname(const classname& other) { \\\n"
			"\t*this = other; \\\n"
			"} \\\n"
			"SFMM_PREFIX T* data() { \\\n"
			"\treturn o; \\\n"
			"} \\\n"
			"SFMM_PREFIX const T* data() const { \\\n"
			"\treturn o; \\\n"
			"} \n";

	fprintf(fp, "%s", str1.c_str());
	fprintf(fp, "%s", str2.c_str());
	tprint("#endif\n\n");
	type = rtypenames.back();

	set_file(full_header.c_str());
	tprint("#ifndef SFMM_TYPES42\n");
	tprint("#define SFMM_TYPES42\n");
	for (scaled = 0; scaled <= enable_scaled; scaled++) {
		for (periodic = 0; periodic <= 1; periodic++) {
			for (nodip = 0; nodip <= 1; nodip++) {
				tprint("\n");
				if (!nodip) {
					tprint("namespace detail {\n");
					tprint("template<class T, int P>\n");
					tprint("class expansion_xz%s%s {\n", period_name(), scaled_name());
					tprint("};\n");
					tprint("}\n");
					tprint("template<class T, int P>\n");
					tprint("class expansion%s%s {\n", period_name(), scaled_name());
					tprint("};\n");
					tprint("\n");
				}
				tprint("template<class T, int P>\n");
				tprint("class multipole%s%s%s {\n", period_name(), scaled_name(), dip_name());
				tprint("};\n");
				tprint("\n");
			}
		}
	}
	tprint("#endif\n");

	for (scaled = 0; scaled <= enable_scaled; scaled++) {
		for (periodic = 0; periodic <= 1; periodic++) {
			for (nodip = 0; nodip <= 1; nodip++) {

				for (int P = pmin - 1; P <= pmax; P++) {
					if (!nodip) {
						tprint("\n");
						tprint("template<>\n");
						tprint("class expansion%s%s<%s,%i> {\n", period_name(), scaled_name(), type.c_str(), P);
						indent();
						tprint("typedef %s T;\n", type.c_str());
						tprint("T o[%i];\n", exp_sz(P));
						if (periodic && P > 1) {
							tprint("T t;\n");
						}
						if (scaled) {
							tprint("T r;\n");
						}

						deindent();
						tprint("public:\n");
						indent();
						tprint("SFMM_EXPANSION_MEMBERS(expansion%s%s, %s, %i);\n", period_name(), scaled_name(), type.c_str(), P);
						tprint("SFMM_PREFIX expansion%s%s& operator=(const expansion%s%s& other) {\n", period_name(), scaled_name(), period_name(), scaled_name());
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

						tprint("SFMM_PREFIX expansion%s%s& operator+=(expansion%s%s other) {\n", period_name(), scaled_name(), period_name(), scaled_name());
						indent();
						if (scaled) {
							tprint("if( r != other.r ) {\n");
							indent();
							tprint("other.rescale(r);\n");
							deindent();
							tprint("}\n");
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

						tprint("SFMM_PREFIX expansion%s%s(T r0 = T(1)) {\n", period_name(), scaled_name());
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
						tprint("SFMM_PREFIX void init(T r0 = T(1)) {\n");
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

						tprint("SFMM_PREFIX void rescale(T r0) {\n");
						indent();
						if (scaled) {
							tprint("const T a = r0 / r;\n");
							tprint("T b = a;\n");
							tprint("r = r0;\n");
							for (int n = 0; n <= P; n++) {
								for (int m = -n; m <= n; m++) {
									tprint("o[%i] *= b;\n", lindex(n, m));
								}
								if (periodic && P > 1 && n == 2) {
									tprint("t *= b;\n", exp_sz(P));
								}
								if (n != P) {
									tprint("b *= a;\n");
								}
							}
						}
						deindent();
						tprint("}\n");
						tprint("SFMM_PREFIX T scale() const {\n");
						indent();
						if (scaled) {
							tprint("return r;\n");
						} else {
							tprint("return T(1);\n");
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

						if (scaled && periodic && P >= pmin) {
							tprint("friend void M2L_ewald(expansion_periodic_scaled<T, %i>&, const multipole_periodic_scaled<T, %i>&, T, T, T, int);\n", P, P);
							tprint("friend void M2L_ewald%s(expansion_periodic_scaled<T, %i>&, const multipole_periodic_scaled<T, %i>&, T, T, T);\n", pot_name(), P,
									P);
							tprint("friend void M2L_ewald(expansion_periodic_scaled<T, %i>&, const multipole_periodic_scaled_wo_dipole<T, %i>&, T, T, T, int);\n", P,
									P);
							tprint("friend void M2L_ewald%s(expansion_periodic_scaled<T, %i>&, const multipole_periodic_scaled_wo_dipole<T, %i>&, T, T, T);\n",
									pot_name(), P, P);
						}
						if (scaled && P >= pmin) {
							tprint("friend void M2L(expansion%s_scaled<T, %i>&, const multipole%s_scaled<T, %i>&, T, T, T, int);\n", period_name(), P, period_name(),
									P);
							tprint("friend void M2L%s(expansion%s_scaled<T, %i>&, const multipole%s_scaled<T, %i>&, T, T, T);\n", pot_name(), period_name(), P,
									period_name(), P);
							tprint("friend void M2L(expansion%s_scaled<T, %i>&, const multipole%s_scaled_wo_dipole<T, %i>&, T, T, T, int);\n", period_name(), P,
									period_name(), P);
							tprint("friend void M2L%s(expansion%s_scaled<T, %i>&, const multipole%s_scaled_wo_dipole<T, %i>&, T, T, T);\n", pot_name(), period_name(),
									P, period_name(), P);
						}
						deindent();
						tprint("};\n");
					}
					tprint("\n");

					if (P > pmin - 1) {
						tprint("\n");
						tprint("template<>\n");
						tprint("class multipole%s%s%s<%s,%i> {\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
						indent();
						tprint("typedef %s T;\n", type.c_str());
						tprint("T o[%i];\n", mul_sz(P));
						if (periodic && P > 2) {
							tprint("T t;\n");
						}
						if (scaled) {
							tprint("T r;\n");
						}
						deindent();
						tprint("public:\n");
						indent();
						if (nodip && P > 1) {
							tprint("SFMM_EXPANSION_MEMBERS_WO_DIPOLE(multipole%s%s%s, %s, %i);\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
						} else {
							tprint("SFMM_EXPANSION_MEMBERS(multipole%s%s%s, %s, %i);\n", period_name(), scaled_name(), dip_name(), type.c_str(), P);
						}
						tprint("SFMM_PREFIX multipole%s%s%s& operator=(const multipole%s%s%s& other) {\n", period_name(), scaled_name(), dip_name(), period_name(),
								scaled_name(), dip_name());
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

						tprint("SFMM_PREFIX multipole%s%s%s& operator+=(multipole%s%s%s other) {\n", period_name(), scaled_name(), dip_name(), period_name(),
								scaled_name(), dip_name());
						indent();
						if (scaled) {
							tprint("if( r != other.r ) {\n");
							indent();
							tprint("other.rescale(r);\n");
							deindent();
							tprint("}\n");
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

						tprint("SFMM_PREFIX multipole%s%s%s(T r0 = T(1)) {\n", period_name(), scaled_name(), dip_name());
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
						if( scaled ) {
							tprint("r = r0;\n");
						}
						deindent();
						tprint("}\n");
						tprint("SFMM_PREFIX void init(T r0 = T(1)) {\n");
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
						tprint("SFMM_PREFIX void rescale(T r0) {\n");
						indent();
						if (scaled) {
							tprint("const T a = r / r0;\n");
							tprint("T b = a;\n");
							tprint("r = r0;\n");
							for (int n = 1; n < P; n++) {
								if (!(nodip && n == 1)) {
									for (int m = -n; m <= n; m++) {
										tprint("o[%i] *= b;\n", mindex(n, m));
									}
									if (periodic && P > 2 && n == 2) {
										tprint("t *= b;\n");
									}
								}
								if (n != P - 1) {
									tprint("b *= a;\n");
								}
							}
						}
						deindent();
						tprint("}\n");
						tprint("SFMM_PREFIX T scale() const {\n");
						indent();
						if (scaled) {
							tprint("return r;\n");
						} else {
							tprint("return T(1);\n");
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
						if (periodic) {
							tprint("friend void M2L_ewald(expansion_periodic%s<T, %i>&, const multipole_periodic%s%s<T, %i>&, T, T, T, int);\n", scaled_name(), P,
									scaled_name(), dip_name(), P);
							tprint("friend void M2L_ewald%s(expansion_periodic%s<T, %i>&, const multipole_periodic%s%s<T, %i>&, T, T, T);\n", pot_name(),
									scaled_name(), P, scaled_name(), dip_name(), P);
						}
						tprint("friend void M2L(expansion%s%s<T, %i>&, const multipole%s%s%s<T, %i>&, T, T, T, int);\n", period_name(), scaled_name(), P,
								period_name(), scaled_name(), dip_name(), P);
						tprint("friend void M2L%s(expansion%s%s<T, %i>&, const multipole%s%s%s<T, %i>&, T, T, T);\n", pot_name(), period_name(), scaled_name(), P,
								period_name(), scaled_name(), dip_name(), P);
						tprint("friend void M2P(force_type<T>&, const multipole%s%s%s<T, %i>&, T, T, T, int);\n", period_name(), scaled_name(), dip_name(), P);
						tprint("friend void M2P%s(force_type<T>&, const multipole%s%s%s<T, %i>&, T, T, T);\n", pot_name(), period_name(), scaled_name(), dip_name(),
								P);
						deindent();
						tprint("};\n");
						tprint("\n");
					}
					if (!scaled && !nodip) {
						tprint("\n");
						tprint("namespace detail {\n");
						tprint("template<>\n");
						tprint("class expansion_xz%s%s<%s,%i> {\n", period_name(), scaled_name(), type.c_str(), P);
						indent();
						tprint("typedef %s T;\n", type.c_str());
						tprint("T o[%i];\n", half_exp_sz(P));
						if (periodic && P > 1) {
							tprint("T t;\n");
						}
						deindent();
						tprint("public:\n");
						indent();
						tprint("SFMM_PREFIX expansion_xz%s%s() {\n", period_name(), scaled_name());
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
				}
			}
		}
	}
	nopot = 0;
	typecast_functions();

	set_file(full_header.c_str());

	for (int ti = 0; ti < ntypenames; ti++) {
		printf("%s cuda:%i\n", rtypenames[ti].c_str(), ucuda[ti]);
		cuda = ucuda[ti];
		prefix = ucuda[ti] ? "SFMM_PREFIX" : "";
		type = rtypenames[ti];
		sitype = sitypenames[ti];
		uitype = uitypenames[ti];
		tprint("\n#ifndef SFMM_FUNCS%i42\n", funcnum);
		tprint("#define SFMM_FUNCS%i42\n", funcnum);
#if defined(FLOAT)
		math_float(false);
#endif

#if defined(DOUBLE)
		math_double(false);
#endif

#if  defined(CUDA_FLOAT)
		math_float(true);
#endif

#if defined(CUDA_DOUBLE)
		math_double(true);
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
				flops_t regular_harmonic_xz_flops[pmax + 1];
				flops_t greens_xz_flops[pmax + 1];
				for (scaled = 0; scaled <= enable_scaled; scaled++) {
					flops_t regular_harmonic_flops[pmax + 1];
					flops_t greens_flops[pmax + 1];
					flops_t greens_safe_flops[pmax + 1];
					flops_t greens_ewald_flops[pmax + 1];
					flops_t greens_ewald_real_flops0[pmax + 1];
					for (nodip = 0; nodip <= 1; nodip++) {
						std::vector<std::unordered_map<std::string, flops_t>> flops_map(pmax + 1);
						std::vector<std::unordered_map<std::string, int>> rot_map(pmax + 1);
						for (int P = pmin - 1; P <= pmax; P++) {
							std::string fname;
							flops_t fps;
							if (!nodip) {
								fname = regular_harmonic(P);
								fclose(fp);
								fp = nullptr;
								regular_harmonic_flops[P] = count_flops(fname);
							}
							if (!nodip && !scaled) {
								fname = regular_harmonic_xz(P);
								fclose(fp);
								fp = nullptr;
								regular_harmonic_xz_flops[P] = count_flops(fname);
							}
							flops_t flops0, flops1, flops2;
							if (P < pmax) {
								fname = M2M_rot0(P);
								fclose(fp);
								fp = nullptr;
								flops0 += regular_harmonic_flops[P];
								flops0 += count_flops(fname);
								SYSTEM((std::string("rm -rf ") + fname).c_str());

								fname = M2M_rot1(P);
								fclose(fp);
								fp = nullptr;
								flops1 += regular_harmonic_xz_flops[P];
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
										M2M_rot0(P);
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
							if (P >= pmin && !nodip) {
								fname = L2L_rot0(P);
								fclose(fp);
								fp = nullptr;
								flops0 += regular_harmonic_flops[P];
								flops0 += count_flops(fname);
								SYSTEM((std::string("rm -rf ") + fname).c_str());

								fname = L2L_rot1(P);
								fclose(fp);
								fp = nullptr;
								flops1 += regular_harmonic_xz_flops[P];
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
										L2L_rot0(P);
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
							flops_t fps;
							if (!nodip) {

								fname = greens(P);
								fclose(fp);
								fp = nullptr;
								greens_flops[P] = count_flops(fname);

								fname = greens_safe(P);
								fclose(fp);
								fp = nullptr;
								greens_safe_flops[P] = count_flops(fname);

								flops0.reset();
								flops0 += sqrt_flops();
								flops0 += erfcexp_flops();
								//fma+2+(P+1)*(1+(2*P+1)) r+5+(P+1)*(5)  rdiv+(P+1)
								flops0.fma += 2 + (P + 1) * (2 + 2 * (P + 1));
								flops0.r += 5 + 5 * (P + 1);
								flops0.rdiv += P + 1;
								greens_ewald_real_flops0[P] = flops0;
								flops0.reset();
								if (!scaled) {
									fname = greens_xz(P);
									fclose(fp);
									fp = nullptr;
									greens_xz_flops[P] = count_flops(fname);
								}
							}
							greens_ewald_real_flops = greens_ewald_real_flops0[P];
							flops0.reset();
							flops1.reset();
							flops2.reset();

							fname = M2L_rot0(P, P);
							fclose(fp);
							fp = nullptr;
							flops0 += count_flops(fname);
							flops0 += greens_flops[P];
							SYSTEM((std::string("rm -rf ") + fname).c_str());

							fname = M2L_rot1(P, P);
							fclose(fp);
							fp = nullptr;
							flops1 += greens_xz_flops[P];
							flops1 += count_flops(fname);
							SYSTEM((std::string("rm -rf ") + fname).c_str());

							fname = M2L_rot2(P, P);
							fclose(fp);
							fp = nullptr;
							flops2 += count_flops(fname);
							SYSTEM((std::string("rm -rf ") + fname).c_str());
							//		printf( "%i %i %i\n", flops0.load(), flops1.load(), flops2.load());
							if (flops2.load() > flops0.load() || flops2.load() > flops1.load()) {
								if (flops1.load() < flops0.load()) {
									M2L_rot2(P, P);
									flops_map[P]["M2L"] = flops1;
									rot_map[P]["M2L"] = 1;
								} else {
									M2L_rot2(P, P);
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
							fname = M2L_rot0(P, 1);
							fclose(fp);
							fp = nullptr;
							flops0 += count_flops(fname);
							flops0 += greens_flops[P];
							SYSTEM((std::string("rm -rf ") + fname).c_str());

							fname = M2L_rot1(P, 1);
							fclose(fp);
							fp = nullptr;
							flops1 += greens_xz_flops[P];
							flops1 += count_flops(fname);
							SYSTEM((std::string("rm -rf ") + fname).c_str());

							fname = M2L_rot2(P, 1);
							fclose(fp);
							fp = nullptr;
							flops2 += count_flops(fname);
							SYSTEM((std::string("rm -rf ") + fname).c_str());

							if (flops2.load() > flops0.load() || flops2.load() > flops1.load()) {
								if (flops1.load() < flops0.load()) {
									M2L_rot1(P, 1);
									flops_map[P]["M2P"] = flops1;
									rot_map[P]["M2P"] = 1;
								} else {
									M2L_rot0(P, 1);
									flops_map[P]["M2P"] = flops0;
									rot_map[P]["M2P"] = 0;
								}
							} else {
								M2L_rot2(P, 1);
								flops_map[P]["M2P"] = flops2;
								rot_map[P]["M2P"] = 2;
							}
							if (!nodip) {
								fname = P2L(P);
								fclose(fp);
								fp = nullptr;
								flops0 = count_flops(fname);
								flops_map[P]["P2L"] = flops0;

								fname = L2P(P);
								fclose(fp);
								fp = nullptr;
								flops0 = count_flops(fname);
								flops_map[P]["L2P"] = flops0;
							}

							fname = P2M(P - 1);
							fclose(fp);
							fp = nullptr;
							flops0 = count_flops(fname);
							flops_map[P - 1]["P2M"] = flops0;

							const double alpha = is_float(type) ? 2.4 : 2.25;
							fname = M2LG(P, P);
							fclose(fp);
							fp = nullptr;
							flops0 = count_flops(fname);
							auto M2LG_flops = flops0;

							if (periodic) {
								if (!nodip) {
									fname = greens_ewald(P, alpha);
									fclose(fp);
									fp = nullptr;
									flops0 = count_flops(fname);
									flops0 += greens_safe_flops[P];
									greens_ewald_flops[P] = flops0;
								}
								fname = M2L_ewald(P);
								fclose(fp);
								fp = nullptr;
								flops0 = count_flops(fname);
								flops0 += M2LG_flops;
								flops0 += greens_ewald_flops[P];
								flops_map[P]["M2L_ewald"] = flops0;
							}
						}

						printf("\ntype: %s  target: %s\nperiodic [%s]  potential[%s]  scaled [%s]  dipole [%s]   \n", type.c_str(), cuda ? "GPU" : "CPU",
								periodic ? "X" : " ", nopot ? " " : "X", scaled ? "X" : " ", nodip ? " " : "X");
						printf("P  | M2L      | P2L   | M2P     | P2M   | M2M     | L2L     | L2P   | M2L_ewald\n");
						for (int P = pmin; P <= pmax; P++) {
							printf("%2i | ", P);
							printf(" %5i %1i |", flops_map[P]["M2L"].load(), rot_map[P]["M2L"]);
							if (!nodip) {
								printf(" %5i |", flops_map[P]["P2L"].load());
							} else {
								printf("  n/a  |");
							}
							printf(" %5i %1i |", flops_map[P]["M2P"].load(), rot_map[P]["M2P"]);
							printf(" %5i |", flops_map[P - 1]["P2M"].load());
							printf(" %5i %1i |", flops_map[P - 1]["M2M"].load(), rot_map[P - 1]["M2M"]);
							if (!nodip) {
								printf(" %5i %1i |", flops_map[P]["L2L"].load(), rot_map[P]["L2L"]);
								printf(" %5i |", flops_map[P]["L2P"].load());
							} else {

								printf("   n/a   |");
								printf("  n/a  |");
							}
							if (periodic) {
								printf(" %6i |", flops_map[P]["M2L_ewald"].load());
							} else {
								printf("   n/a  |");
							}
							printf("\n");
						}
						fflush(stdout);
					}
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
	tprint("#endif\n\n", funcnum);
	tprint("#ifndef SFMM_GREEN_EWALD_REAL42\n");
	tprint("#define SFMM_GREEN_EWALD_REAL42\n");
	for (scaled = 0; scaled <= enable_scaled; scaled++) {
		tprint("\n");
		tprint("template<class T, int P, int ALPHA100>\n");
		tprint("SFMM_PREFIX void greens_ewald_real(expansion%s%s<T, P>& G_st, T x, T y, T z) {\n", period_name(), scaled_name());
		indent();
		tprint("constexpr double ALPHA = ALPHA100 / 100.0;\n");
		tprint("expansion%s%s<T, P> Gr_st;\n", period_name(), scaled_name());
		tprint("const T r2 = fma(x, x, fma(y, y, z * z));\n");
		tprint("const T r = sqrt(r2);\n");
		tprint("greens(Gr_st, x, y, z);\n");
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
	}
	tprint("\n");
	tprint("#endif\n");
	tprint("}\n");
	fprintf(fp, "%s\n", complex_defs.c_str());
	tprint("}\n");
	return 0;
}
