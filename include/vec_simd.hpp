#define SFMM_SIMD_BINARY_OP(vtype, type, op) \
   inline vtype operator op (const vtype& u ) const { \
      vtype w; \
      w.v = v op u.v; \
		  return w; \
   } \
	  inline vtype& operator op##= (const vtype& u ) { \
       *this = *this op u; \
	      return *this; \
   }

#define SFMM_SIMD_UNARY_OP(vtype, type, op) \
   inline vtype operator op () const { \
      vtype w; \
      w.v = op v; \
      return w; \
   }

#define SFMM_SIMD_CVT_OP_PROT(vtype,type,ovtype,otype) \
   inline vtype(const ovtype&); \
   inline vtype& operator=(const ovtype&); \
   inline vtype& operator=(const otype&)

#define SFMM_SIMD_CVT_OP_DEF(vtype,type,ovtype,otype) \
   inline vtype::vtype(const ovtype& other) { \
	     v = __builtin_convertvector(other.v, simd_t); \
   } \
   inline vtype& vtype::operator=(const ovtype& other) { \
	     v = __builtin_convertvector(other.v, simd_t); \
	     return *this; \
   }

#define SFMM_SIMD_MEMBERS(vtype,type) \
   inline vtype(const type& other) { \
	     v = other - simd_t{}; \
   } \
   inline vtype& operator=(const type& other) { \
	     v = other - simd_t{}; \
	     return *this; \
   } \
   inline vtype& pad(int n) { \
		const int& e = size(); \
		for(int i = n; i < e; i++) { \
			v[i] = v[0]; \
		} \
		return *this; \
   } \
   static inline vtype mask(int n) { \
   	vtype mk; \
   	for( int i = 0; i < n; i++) { \
   		mk[i] = type(1); \
   	} \
   	for( int i = n; i < size(); i++) { \
   		mk[i] = type(0); \
   	} \
   	return mk; \
   } \
	inline void set_NaN() { \
		for( int i = 0; i < size(); i++) { \
			v[i] = std::numeric_limits<type>::signaling_NaN(); \
		} \
	}

#define SFMM_SIMD_CMP_OP_PROT(vtype,vstype,stype,op) \
   inline vstype operator op (const vtype&) const

#define SFMM_SIMD_CMP_OP_DEF(vtype,vstype,sitype,op) \
   inline vstype vtype::operator op (const vtype& other) const { \
	     vstype w; \
      w.v = -(v op other.v); \
      return w; \
   }

#define SFMM_SIMD_FWD(vtype)              \
class vtype

#define SFMM_SIMD_SIZE(sz) \
   inline static constexpr size_t size() { \
      return sz; \
   }

#define SFMM_SIMD_REAL_TYPE(vtype, type, vstype, stype, vutype, utype, size)              \
   class vtype {                                           \
      typedef type simd_t __attribute__ ((vector_size(size*sizeof(type))));  \
      simd_t v;  \
   public: \
      inline vtype() : v() {} \
      inline type operator[](int i) const {  \
         return v[i]; \
      }\
	     inline type& operator[](int i) {  \
		     return v[i]; \
	     }\
      SFMM_SIMD_BINARY_OP(vtype, type, +); \
      SFMM_SIMD_BINARY_OP(vtype, type, -); \
      SFMM_SIMD_BINARY_OP(vtype, type, *); \
      SFMM_SIMD_BINARY_OP(vtype, type, /); \
      SFMM_SIMD_UNARY_OP(vtype, type, +); \
      SFMM_SIMD_UNARY_OP(vtype, type, -); \
      SFMM_SIMD_CVT_OP_PROT(vtype, type, vstype, stype); \
      SFMM_SIMD_CVT_OP_PROT(vtype, type, vutype, utype); \
      SFMM_SIMD_MEMBERS(vtype, type); \
      SFMM_SIMD_CMP_OP_PROT(vtype,  vstype, stype, <); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, <=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, ==); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, !=); \
      SFMM_SIMD_SIZE(size);\
      friend class vstype; \
      friend class vutype; \
   }

#define SFMM_SIMD_INTEGER_TYPE(vtype, type, votype, otype, vrtype, rtype, vstype, stype, size)              \
   class vtype {                                           \
      typedef type simd_t __attribute__ ((vector_size(size*sizeof(type))));  \
      simd_t v;  \
   public: \
      inline vtype() : v() {} \
      inline type operator[](int i) const {  \
         return v[i]; \
      }\
      inline type& operator[](int i) {  \
         return v[i]; \
      }\
      SFMM_SIMD_BINARY_OP(vtype, type, +); \
      SFMM_SIMD_BINARY_OP(vtype, type, -); \
      SFMM_SIMD_BINARY_OP(vtype, type, *); \
      SFMM_SIMD_BINARY_OP(vtype, type, /); \
      SFMM_SIMD_BINARY_OP(vtype, type, &); \
      SFMM_SIMD_BINARY_OP(vtype, type, ^); \
      SFMM_SIMD_BINARY_OP(vtype, type, |); \
      SFMM_SIMD_BINARY_OP(vtype, type, >>); \
      SFMM_SIMD_BINARY_OP(vtype, type, <<); \
      SFMM_SIMD_UNARY_OP(vtype, type, +); \
      SFMM_SIMD_UNARY_OP(vtype, type, -); \
      SFMM_SIMD_UNARY_OP(vtype, type, ~); \
      SFMM_SIMD_MEMBERS(vtype,type); \
      SFMM_SIMD_CVT_OP_PROT(vtype, type, vrtype, rtype); \
      SFMM_SIMD_CVT_OP_PROT(vtype, type, votype, otype); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, <); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, <=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, ==); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, !=); \
      SFMM_SIMD_SIZE(size);\
      friend class vrtype; \
      friend class votype; \
   }

#define SFMM_SIMD_REAL_TYPE_DEF(vtype, type, vstype, stype, vutype, utype, size)\
   SFMM_SIMD_CVT_OP_DEF(vtype, type, vstype, stype); \
   SFMM_SIMD_CVT_OP_DEF(vtype, type, vutype, utype)

#define SFMM_SIMD_INTEGER_TYPE_DEF(vtype, type, votype, otype, vrtype, rtype, size)              \
   SFMM_SIMD_CVT_OP_DEF(vtype,type,vrtype,rtype); \
   SFMM_SIMD_CVT_OP_DEF(vtype,type,votype,otype)

#define SFMM_SIMD_FACTORY(vrtype,rtype,vstype,stype,vutype,utype,size) \
   SFMM_SIMD_FWD(vrtype); \
   SFMM_SIMD_FWD(vutype); \
   SFMM_SIMD_FWD(vstype); \
   SFMM_SIMD_REAL_TYPE(vrtype, rtype, vstype, stype, vutype, utype,  size); \
   SFMM_SIMD_INTEGER_TYPE(vutype, utype, vstype, stype, vrtype, rtype, vstype, stype, size); \
   SFMM_SIMD_INTEGER_TYPE(vstype, stype, vutype, utype, vrtype, rtype, vstype, stype, size); \
   SFMM_SIMD_REAL_TYPE_DEF(vrtype, rtype, vstype, stype, vutype, utype,  size); \
   SFMM_SIMD_INTEGER_TYPE_DEF(vutype, utype, vstype, stype, vrtype, rtype,  size); \
   SFMM_SIMD_INTEGER_TYPE_DEF(vstype, stype, vutype, utype, vrtype, rtype,  size); \
   SFMM_SIMD_CMP_OP_DEF(vrtype, vstype, stype, <); \
   SFMM_SIMD_CMP_OP_DEF(vrtype, vstype, stype, >); \
   SFMM_SIMD_CMP_OP_DEF(vrtype, vstype, stype, <=); \
   SFMM_SIMD_CMP_OP_DEF(vrtype, vstype, stype, >=); \
   SFMM_SIMD_CMP_OP_DEF(vrtype, vstype, stype, ==); \
   SFMM_SIMD_CMP_OP_DEF(vrtype, vstype, stype, !=); \
   SFMM_SIMD_CMP_OP_DEF(vutype, vstype, stype, <); \
   SFMM_SIMD_CMP_OP_DEF(vutype, vstype, stype, >); \
   SFMM_SIMD_CMP_OP_DEF(vutype, vstype, stype, <=); \
   SFMM_SIMD_CMP_OP_DEF(vutype, vstype, stype, >=); \
   SFMM_SIMD_CMP_OP_DEF(vutype, vstype, stype, ==); \
   SFMM_SIMD_CMP_OP_DEF(vutype, vstype, stype, !=); \
   SFMM_SIMD_CMP_OP_DEF(vstype, vstype, stype, <); \
   SFMM_SIMD_CMP_OP_DEF(vstype, vstype, stype, >); \
   SFMM_SIMD_CMP_OP_DEF(vstype, vstype, stype, <=); \
   SFMM_SIMD_CMP_OP_DEF(vstype, vstype, stype, >=); \
   SFMM_SIMD_CMP_OP_DEF(vstype, vstype, stype, ==); \
   SFMM_SIMD_CMP_OP_DEF(vstype, vstype, stype, !=)
