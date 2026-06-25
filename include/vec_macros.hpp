#ifndef VEC_MACROS12_HPP_
#define VEC_MACROS12_HPP_


#define create_binary_op(type, op) \
		inline type operator op (const type& u ) const { \
			type w; \
			w.v = v op u.v; \
			return w; \
		} \
		inline type& operator op##= (const type& u ) { \
			*this = *this op u; \
			return *this; \
		}

#define create_unary_op(type, op) \
		inline type operator op () const { \
			type w; \
			w.v = op v; \
			return w; \
		}

#define create_convert_op_prot(type,otype) \
		inline vec_##type(const vec_##otype&); \
		inline vec_##type& operator=(const vec_##otype&); \
		inline vec_##type& operator=(const otype&)

#define create_convert_op_def(type,otype) \
	inline vec_##type::vec_##type(const vec_##otype& other) { \
		v = __builtin_convertvector(other.v, vtype); \
	} \
	inline vec_##type& vec_##type::operator=(const vec_##otype& other) { \
		v = __builtin_convertvector(other.v, vtype); \
		return *this; \
	}


#define create_broadcast_op(type) \
	inline vec_##type(const type& other) { \
		v = other - vtype{}; \
	} \
	inline vec_##type& operator=(const type& other) { \
		v = other - vtype{}; \
		return *this; \
	}

#define create_compare_op_prot(type,sitype,  op) \
	inline vec_##sitype operator op (const vec_##type&) const

#define create_compare_op_def(type,sitype,  op) \
	inline vec_##sitype vec_##type::operator op (const vec_##type& other) const { \
		vec_##sitype w; \
		w.v = (-(v op other.v)); \
		return w; \
	}

#define create_vec_types_fwd(type)              \
	class vec_##type

#define create_rvec_types(type, sitype, uitype, size)              \
	class vec_##type {                                           \
		typedef type vtype __attribute__ ((vector_size(size*sizeof(type))));  \
		vtype v;  \
	public: \
	inline constexpr vec_##type() : v() {} \
	inline type operator[](int i) const {  \
		return v[i]; \
	}\
	inline type& operator[](int i) {  \
		return v[i]; \
	}\
	create_binary_op(vec_##type, +); \
	create_binary_op(vec_##type, -); \
	create_binary_op(vec_##type, *); \
	create_binary_op(vec_##type, /); \
	create_unary_op(vec_##type, +); \
	create_unary_op(vec_##type, -); \
	create_convert_op_prot(type, sitype); \
	create_convert_op_prot(type, uitype); \
	create_broadcast_op(type); \
	create_compare_op_prot(type, sitype, <); \
	create_compare_op_prot(type, sitype, >); \
	create_compare_op_prot(type, sitype, <=); \
	create_compare_op_prot(type, sitype, >=); \
	create_compare_op_prot(type, sitype, ==); \
	create_compare_op_prot(type, sitype, !=); \
	friend class vec_##sitype; \
	friend class vec_##uitype; \
}

#define create_ivec_types(type, otype, rtype, sitype, size)              \
	class vec_##type {                                           \
		typedef type vtype __attribute__ ((vector_size(size*sizeof(type))));  \
		vtype v;  \
	public: \
	inline constexpr vec_##type() : v() {} \
	inline type operator[](int i) const {  \
		return v[i]; \
	}\
	inline type& operator[](int i) {  \
		return v[i]; \
	}\
	create_binary_op(vec_##type, +); \
	create_binary_op(vec_##type, -); \
	create_binary_op(vec_##type, *); \
	create_binary_op(vec_##type, /); \
	create_binary_op(vec_##type, &); \
	create_binary_op(vec_##type, ^); \
	create_binary_op(vec_##type, |); \
	create_binary_op(vec_##type, >>); \
	create_binary_op(vec_##type, <<); \
	create_unary_op(vec_##type, +); \
	create_unary_op(vec_##type, -); \
	create_unary_op(vec_##type, ~); \
	create_broadcast_op(type); \
	create_convert_op_prot(type, rtype); \
	create_convert_op_prot(type, otype); \
	create_compare_op_prot(type,sitype,  <); \
	create_compare_op_prot(type,sitype,  >); \
	create_compare_op_prot(type, sitype, <=); \
	create_compare_op_prot(type,sitype,  >=); \
	create_compare_op_prot(type, sitype,  ==); \
	create_compare_op_prot(type, sitype,  !=); \
	friend class vec_##rtype; \
	friend class vec_##otype; \
}

#define create_rvec_types_def(type, sitype, uitype, size)\
		create_convert_op_def(type, sitype); \
		create_convert_op_def(type, uitype)

#define create_ivec_types_def(type, otype, rtype, size)              \
		create_convert_op_def(type, rtype); \
		create_convert_op_def(type, otype)

#define create_vec_types(rtype, sitype, uitype, size) \
		create_vec_types_fwd(rtype); \
		create_vec_types_fwd(uitype); \
		create_vec_types_fwd(sitype); \
		create_rvec_types(rtype, sitype, uitype, size); \
		create_ivec_types(uitype, sitype, rtype, sitype, size); \
		create_ivec_types(sitype, uitype, rtype, sitype, size); \
		create_rvec_types_def(rtype, sitype, uitype, size); \
		create_ivec_types_def(uitype, sitype, rtype, size); \
		create_ivec_types_def(sitype, uitype, rtype, size); \
		create_compare_op_def(rtype, sitype, <); \
		create_compare_op_def(rtype,sitype,  >); \
		create_compare_op_def(rtype, sitype, <=); \
		create_compare_op_def(rtype, sitype, >=); \
		create_compare_op_def(rtype,sitype,  ==); \
		create_compare_op_def(rtype, sitype, !=); \
		create_compare_op_def(uitype,sitype,  <); \
		create_compare_op_def(uitype, sitype, >); \
		create_compare_op_def(uitype, sitype, <=); \
		create_compare_op_def(uitype, sitype, >=); \
		create_compare_op_def(uitype, sitype, ==); \
		create_compare_op_def(uitype, sitype, !=); \
		create_compare_op_def(sitype, sitype, <); \
		create_compare_op_def(sitype,sitype,  >); \
		create_compare_op_def(sitype,sitype,  <=); \
		create_compare_op_def(sitype, sitype, >=); \
		create_compare_op_def(sitype, sitype, ==); \
		create_compare_op_def(sitype, sitype, !=)


#endif /* VEC_MACROS_HPP_ */
