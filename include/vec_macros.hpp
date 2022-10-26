#pragma once

#define create_binary_op(vtype, type, op) \
		inline vtype operator op (const vtype& u ) const { \
			vtype w; \
			w.v = v op u.v; \
			return w; \
		} \
		inline vtype& operator op##= (const vtype& u ) { \
			*this = *this op u; \
			return *this; \
		}

#define create_unary_op(vtype, type, op) \
		inline vtype operator op () const { \
			vtype w; \
			w.v = op v; \
			return w; \
		}

#define create_convert_op_prot(vtype,type,ovtype,otype) \
		inline vtype(const ovtype&); \
		inline vtype& operator=(const ovtype&); \
		inline vtype& operator=(const otype&)

#define create_convert_op_def(vtype,type,ovtype,otype) \
	inline vtype::vtype(const ovtype& other) { \
		v = __builtin_convertvector(other.v, simd_t); \
	} \
	inline vtype& vtype::operator=(const ovtype& other) { \
		v = __builtin_convertvector(other.v, simd_t); \
		return *this; \
	}


#define create_broadcast_op(vtype,type) \
	inline vtype(const type& other) { \
		v = other - simd_t{}; \
	} \
	inline vtype& operator=(const type& other) { \
		v = other - simd_t{}; \
		return *this; \
	}

#define create_compare_op_prot(vtype,vstype,stype,op) \
	inline vstype operator op (const vtype&) const

#define create_compare_op_def(vtype,vstype,sitype,op) \
	inline vstype vtype::operator op (const vtype& other) const { \
		vstype w; \
		w.v = (-(v op other.v)); \
		return w; \
	}

#define create_vec_types_fwd(vtype)              \
	class vtype

#define create_rvec_types(vtype, type, vstype, stype, vutype, utype, size)              \
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
	create_binary_op(vtype, type, +); \
	create_binary_op(vtype, type, -); \
	create_binary_op(vtype, type, *); \
	create_binary_op(vtype, type, /); \
	create_unary_op(vtype, type, +); \
	create_unary_op(vtype, type, -); \
	create_convert_op_prot(vtype,type,vstype,stype); \
	create_convert_op_prot(vtype,type,vutype,utype); \
	create_broadcast_op(vtype,type); \
	create_compare_op_prot(vtype, vstype, stype, <); \
	create_compare_op_prot(vtype, vstype, stype, >); \
	create_compare_op_prot(vtype, vstype, stype, <=); \
	create_compare_op_prot(vtype, vstype, stype, >=); \
	create_compare_op_prot(vtype, vstype, stype, ==); \
	create_compare_op_prot(vtype, vstype, stype, !=); \
	friend class vstype; \
	friend class vutype; \
}

#define create_ivec_types(vtype, type, votype, otype, vrtype, rtype, vstype, stype, size)              \
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
	create_binary_op(vtype, type, +); \
	create_binary_op(vtype, type, -); \
	create_binary_op(vtype, type, *); \
	create_binary_op(vtype, type, /); \
	create_binary_op(vtype, type, &); \
	create_binary_op(vtype, type, ^); \
	create_binary_op(vtype, type, |); \
	create_binary_op(vtype, type, >>); \
	create_binary_op(vtype, type, <<); \
	create_unary_op(vtype, type, +); \
	create_unary_op(vtype, type, -); \
	create_unary_op(vtype, type, ~); \
	create_broadcast_op(vtype,type); \
	create_convert_op_prot(vtype,type,vrtype,rtype); \
	create_convert_op_prot(vtype,type,votype,otype); \
	create_compare_op_prot(vtype,vstype,stype,<); \
	create_compare_op_prot(vtype,vstype,stype,>); \
	create_compare_op_prot(vtype,vstype,stype,<=); \
	create_compare_op_prot(vtype,vstype,stype,>=); \
	create_compare_op_prot(vtype,vstype,stype,==); \
	create_compare_op_prot(vtype,vstype,stype,!=); \
	friend class vrtype; \
	friend class votype; \
}

#define create_rvec_types_def(vtype, type, vstype, stype, vutype, utype, size)\
		create_convert_op_def(vtype, type, vstype, stype); \
		create_convert_op_def(vtype, type, vutype, utype)

#define create_ivec_types_def(vtype, type, votype, otype, vrtype, rtype, size)              \
		create_convert_op_def(vtype,type,vrtype,rtype); \
		create_convert_op_def(vtype,type,votype,otype)

#define create_vec_types(vrtype,rtype,vstype,stype,vutype,utype,size) \
		create_vec_types_fwd(vrtype); \
		create_vec_types_fwd(vutype); \
		create_vec_types_fwd(vstype); \
		create_rvec_types(vrtype,rtype,vstype,stype,vutype,utype, size); \
		create_ivec_types(vutype,utype,vstype,stype,vrtype,rtype,vstype,stype,size); \
		create_ivec_types(vstype,stype,vutype,utype,vrtype,rtype,vstype,stype,size); \
		create_rvec_types_def(vrtype,rtype,vstype,stype,vutype,utype, size); \
		create_ivec_types_def(vutype,utype,vstype,stype,vrtype,rtype, size); \
		create_ivec_types_def(vstype,stype,vutype,utype,vrtype,rtype, size); \
		create_compare_op_def(vrtype,vstype,stype,<); \
		create_compare_op_def(vrtype,vstype,stype,>); \
		create_compare_op_def(vrtype,vstype,stype,<=); \
		create_compare_op_def(vrtype,vstype,stype,>=); \
		create_compare_op_def(vrtype,vstype,stype,==); \
		create_compare_op_def(vrtype,vstype,stype,!=); \
		create_compare_op_def(vutype,vstype,stype,<); \
		create_compare_op_def(vutype,vstype,stype,>); \
		create_compare_op_def(vutype,vstype,stype,<=); \
		create_compare_op_def(vutype,vstype,stype,>=); \
		create_compare_op_def(vutype,vstype,stype,==); \
		create_compare_op_def(vutype,vstype,stype,!=); \
		create_compare_op_def(vstype,vstype,stype,<); \
		create_compare_op_def(vstype,vstype,stype,>); \
		create_compare_op_def(vstype,vstype,stype,<=); \
		create_compare_op_def(vstype,vstype,stype,>=); \
		create_compare_op_def(vstype,vstype,stype,==); \
		create_compare_op_def(vstype,vstype,stype,!=)


