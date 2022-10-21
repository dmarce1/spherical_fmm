#ifndef VEC_MACROS_HPP_
#define VEC_MACROS_HPP_


#define create_binary_op(type, op) \
		inline type operator op (const type& u ) const { \
			type w; \
			w.v = v + u.v; \
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
		inline type##_vec(const otype##_vec&); \
		inline type##_vec(const otype&); \
		inline type##_vec& operator=(const otype##_vec&); \
		inline type##_vec& operator=(const otype&)

#define create_convert_op_def(type,otype) \
	inline type##_vec::type##_vec(const otype##_vec& other) { \
		v = __builtin_convertvector(other.v, vtype); \
	} \
	inline type##_vec& type##_vec::operator=(const otype##_vec& other) { \
		v = __builtin_convertvector(other.v, vtype); \
		return *this; \
	}

#define create_broadcast_op(type) \
	inline type##_vec(const type& other) { \
		v = other - vtype{}; \
	} \
	inline type##_vec& operator=(const type& other) { \
		v = other - vtype{}; \
		return *this; \
	}

#define create_compare_op_prot(type, op) \
	inline type##_vec operator op (const type##_vec&) const

#define create_compare_op_def(type, op) \
	inline type##_vec type##_vec::operator op (const type##_vec& other) const { \
		type##_vec w; \
		w.v = (-(v op other.v)); \
		return w; \
	}

#define create_vec_types_fwd(type)              \
	class type##_vec

#define create_rvec_types(type, sitype, uitype, size)              \
	class type##_vec {                                           \
		typedef type vtype __attribute__ ((vector_size(size*sizeof(type))));  \
		vtype v;  \
	public: \
	inline constexpr type##_vec() : v() {} \
	create_binary_op(type##_vec, +); \
	create_binary_op(type##_vec, -); \
	create_binary_op(type##_vec, *); \
	create_binary_op(type##_vec, /); \
	create_unary_op(type##_vec, +); \
	create_unary_op(type##_vec, -); \
	create_convert_op_prot(type, sitype); \
	create_convert_op_prot(type, uitype); \
	create_broadcast_op(type); \
	create_compare_op_prot(type, <); \
	create_compare_op_prot(type, >); \
	create_compare_op_prot(type, <=); \
	create_compare_op_prot(type, >=); \
	create_compare_op_prot(type, ==); \
	create_compare_op_prot(type, !=); \
	friend class sitype##_vec; \
	friend class uitype##_vec; \
}

#define create_ivec_types(type, otype, rtype, size)              \
	class type##_vec {                                           \
		typedef type vtype __attribute__ ((vector_size(size*sizeof(type))));  \
		vtype v;  \
	public: \
	inline constexpr type##_vec() : v() {} \
	create_binary_op(type##_vec, +); \
	create_binary_op(type##_vec, -); \
	create_binary_op(type##_vec, *); \
	create_binary_op(type##_vec, /); \
	create_binary_op(type##_vec, &); \
	create_binary_op(type##_vec, ^); \
	create_binary_op(type##_vec, |); \
	create_binary_op(type##_vec, >>); \
	create_binary_op(type##_vec, <<); \
	create_unary_op(type##_vec, +); \
	create_unary_op(type##_vec, -); \
	create_unary_op(type##_vec, ~); \
	create_broadcast_op(type); \
	create_convert_op_prot(type, rtype); \
	create_convert_op_prot(type, otype); \
	create_compare_op_prot(type, <); \
	create_compare_op_prot(type, >); \
	create_compare_op_prot(type, <=); \
	create_compare_op_prot(type, >=); \
	create_compare_op_prot(type, ==); \
	create_compare_op_prot(type, !=); \
	friend class rtype##_vec; \
	friend class otype##_vec; \
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
		create_ivec_types(uitype, sitype, rtype, size); \
		create_ivec_types(sitype, uitype, rtype, size); \
		create_rvec_types_def(rtype, sitype, uitype, size); \
		create_ivec_types_def(uitype, sitype, rtype, size); \
		create_ivec_types_def(sitype, uitype, rtype, size); \
		create_compare_op_def(rtype, <); \
		create_compare_op_def(rtype, >); \
		create_compare_op_def(rtype, <=); \
		create_compare_op_def(rtype, >=); \
		create_compare_op_def(rtype, ==); \
		create_compare_op_def(rtype, !=); \
		create_compare_op_def(uitype, <); \
		create_compare_op_def(uitype, >); \
		create_compare_op_def(uitype, <=); \
		create_compare_op_def(uitype, >=); \
		create_compare_op_def(uitype, ==); \
		create_compare_op_def(uitype, !=); \
		create_compare_op_def(sitype, <); \
		create_compare_op_def(sitype, >); \
		create_compare_op_def(sitype, <=); \
		create_compare_op_def(sitype, >=); \
		create_compare_op_def(sitype, ==); \
		create_compare_op_def(sitype, !=)


#endif /* VEC_MACROS_HPP_ */
