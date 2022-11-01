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

#define SFMM_SIMD_BROADCAST_OP(vtype,type) \
   inline vtype(const type& other) { \
	     v = other - simd_t{}; \
   } \
   inline vtype& operator=(const type& other) { \
	     v = other - simd_t{}; \
	     return *this; \
   }

#define SFMM_SIMD_CMP_OP_PROT(vtype,vstype,stype,op) \
   inline vstype operator op (const vtype&) const

#define SFMM_SIMD_CMP_OP_DEF(vtype,vstype,sitype,op) \
   inline vstype vtype::operator op (const vtype& other) const { \
	     vstype w; \
      w.v = (-(v op other.v)); \
      return w; \
   }

#define SFMM_SIMD_FWD(vtype)              \
class vtype;

#define SFMM_SIMD_SIZE(sz) \
   inline static size_t size() { \
      return sz; \
   } \

#ifdef NDEBUG
#define SFMM_SIMD_PADDING(type, sz)  \
   inline void pad(int, int=type(1)) { \
   }
#else
#define SFMM_SIMD_PADDING(type, sz)  \
   inline void pad(int n, type value = type(1)) { \
      for( int i = sz - n; i < sz; i++ ) { \
         v[i] = value; \
      } \
   }
#endif

#define SFMM_SIMD_MASK_PROT(vtype, type, sz) \
	inline static vtype mask(int)

#define SFMM_SIMD_MASK_DEF(vtype, type, sz) \
	inline vtype vtype::mask(int tailcnt) { \
		vtype result; \
		const int mid = sz - tailcnt; \
		for( int i = 0; i < mid; i++) { \
			result[i] = type(1); \
		} \
		for( int i = mid; i < sz; i++) { \
			result[i] = type(0); \
		} \
		return result; \
	}

#define SFMM_SIMD_LOAD_PROT(vtype, type, size) \
	inline vtype load(type* ptr, type* end); \
   inline vtype load_with_padding(type* ptr, type* end, type pad_val = type(1))

#define SFMM_SIMD_LOAD_DEF(vtype, type, size) \
	inline vtype vtype::load(type* ptr, type* end) { \
		const int iend = end - ptr > size ? end - ptr : size; \
		const int tailcnt = size - iend; \
		for( int i = 0; i < iend; i++) { \
			v[i] = ptr[i]; \
		} \
		return mask(tailcnt); \
   } \
   inline vtype vtype::load_with_padding(type* ptr, type* end, type pad_val) { \
   	const vtype m = load(ptr, end); \
   	pad(end - ptr > size ? 0 : size - (end - ptr), pad_val); \
   	return m; \
   }

#define SFMM_SIMD_REAL_TYPE(vtype, type, vstype, stype, vutype, utype, size)              \
   class vtype {                                           \
      typedef type simd_t __attribute__ ((vector_size(size*sizeof(type))));  \
      simd_t v;  \
   public: \
      inline constexpr vtype() : v() {} \
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
      SFMM_SIMD_BROADCAST_OP(vtype, type); \
      SFMM_SIMD_CMP_OP_PROT(vtype,  vstype, stype, <); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, <=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, ==); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, !=); \
      SFMM_SIMD_PADDING(type, size);\
      SFMM_SIMD_SIZE(size);\
      SFMM_SIMD_MASK_PROT(vtype, type, size); \
      SFMM_SIMD_LOAD_PROT(vtype, type, size); \
      friend class vstype; \
      friend class vutype; \
   }

#define SFMM_SIMD_INTEGER_TYPE(vtype, type, votype, otype, vrtype, rtype, vstype, stype, size)              \
   class vtype {                                           \
      typedef type simd_t __attribute__ ((vector_size(size*sizeof(type))));  \
      simd_t v;  \
   public: \
      inline constexpr vtype() : v() {} \
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
      SFMM_SIMD_BROADCAST_OP(vtype,type); \
      SFMM_SIMD_CVT_OP_PROT(vtype, type, vrtype, rtype); \
      SFMM_SIMD_CVT_OP_PROT(vtype, type, votype, otype); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, <); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, <=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, >=); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, ==); \
      SFMM_SIMD_CMP_OP_PROT(vtype, vstype, stype, !=); \
      SFMM_SIMD_PADDING(type, size);\
      SFMM_SIMD_SIZE(size);\
      SFMM_SIMD_MASK_PROT(vtype, type, size); \
      SFMM_SIMD_LOAD_PROT(vtype, type, size); \
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
   SFMM_SIMD_CMP_OP_DEF(vstype, vstype, stype, !=); \
   SFMM_SIMD_MASK_DEF(vrtype, rtype, size); \
   SFMM_SIMD_MASK_DEF(vstype, stype, size); \
   SFMM_SIMD_MASK_DEF(vutype, utype, size); \
   SFMM_SIMD_LOAD_DEF(vrtype, rtype, size); \
   SFMM_SIMD_LOAD_DEF(vstype, stype, size); \
   SFMM_SIMD_LOAD_DEF(vutype, utype, size)

