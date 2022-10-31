#include <stdio.h>
#include <sfmmf.hpp>
#include <sfmmd.hpp>

#include <array>
#include <vector>
#include <fenv.h>

constexpr double theta_max = 0.6;
constexpr double hsoft = 0.01;

#define NDIM 3
#define BUCKET_SIZE 16
#define MIN_CLOUD 4
#define LEFT 0
#define RIGHT 1
#define NCHILD 2
#define TEST_SIZE 10000

using rtype = double;

#define VEC3_BINARY_OP( op ) \
		vec3 operator op (const vec3& other) const { \
			vec3 result; \
			for( int dim = 0; dim < NDIM; dim++ ) { \
				result[dim] = (*this)[dim] op other[dim]; \
			} \
			return result; \
		} \
		vec3& operator op##= (const vec3& other) { \
			for( int dim = 0; dim < NDIM; dim++ ) { \
				(*this)[dim] op##= other[dim]; \
			} \
			return *this; \
		}

#define VEC3_SINGLE_BINARY_OP( op ) \
		vec3 operator op (const rtype other) const { \
			vec3 result; \
			for( int dim = 0; dim < NDIM; dim++ ) { \
				result[dim] = (*this)[dim] op other; \
			} \
			return result; \
		} \
		vec3& operator op##= (const rtype other) { \
			for( int dim = 0; dim < NDIM; dim++ ) { \
				(*this)[dim] op##= other; \
			} \
			return *this; \
		}

struct vec3: public std::array<rtype, NDIM> {
	VEC3_BINARY_OP( + )
	VEC3_BINARY_OP( - )
	VEC3_BINARY_OP( * )
	VEC3_BINARY_OP( / )
	VEC3_SINGLE_BINARY_OP( * )
	VEC3_SINGLE_BINARY_OP( / )
	vec3 operator-() const {
		vec3 result;
		for (int dim = 0; dim < NDIM; dim++) {
			result[dim] = -(*this)[dim];
		}
		return result;
	}
	vec3& operator=(rtype a) {
		for (int dim = 0; dim < NDIM; dim++) {
			(*this)[dim] = a;
		}
		return *this;
	}
};

template<class T>
inline T sqr(T a) {
	return a * a;
}

inline rtype abs(vec3 vec) {
	return sqrt(sqr(vec[0]) + sqr(vec[1]) + sqr(vec[2]));
}

namespace sfmm {

}

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

template<class T, int ORDER>
class tree {

	using multipole_type = sfmm::multipole_wo_dipole<T,ORDER>;
	using expansion_type = sfmm::expansion<T,ORDER>;
	using force_type = sfmm::force_type<T>;
	struct particle {
		vec3 x;
		force_type f;
	};

	multipole_type multipole;
	std::vector<tree> children;
	std::vector<particle> parts;
	vec3 begin;
	vec3 end;
	vec3 center;
	tree* parent;
	double radius;

	static std::vector<tree*> nodes;

	struct check_type {
		tree* ptr;
		bool opened;
	};

	void list_iterate(std::vector<check_type>& checklist, std::vector<tree*>& Plist, std::vector<tree*>& Clist, bool leaf) {
		std::vector<check_type> nextlist;
		for (auto& check : checklist) {
			bool far = leaf && check.opened;
			if (!far) {
				const vec3 dx = center - check.ptr->center;
				const double d = abs(dx);
				far = radius + check.ptr->radius < theta_max * d;
			}
			if (far) {
				if (check.opened) {
					Plist.push_back(check.ptr);
				} else {
					Clist.push_back(check.ptr);
				}
			} else {
				if (check.ptr->children.size()) {
					for (int ci = 0; ci < NCHILD; ci++) {
						check_type chk;
						chk.ptr = &(check.ptr->children[ci]);
						chk.opened = false;
						nextlist.push_back(chk);
					}
				} else {
					check.opened = true;
					nextlist.push_back(check);
				}
			}
		}
		checklist = nextlist;
	}

	void P2P(force_type& f, T m, T x, T y, T z) {
		const T r2 = sqr(x) + sqr(y) + sqr(z);
		static const T h2 = hsoft * hsoft;
		static const T hinv = T(1) / hsoft;
		static const T hinv3 = sqr(hinv) * hinv;
		if (r2 > h2) {
			const T rinv = sfmm::rsqrt(r2);
			T rinv3 = sqr(rinv) * rinv;
			f.potential += m * rinv;
			f.force[0] -= m * x * rinv3;
			f.force[1] -= m * y * rinv3;
			f.force[2] -= m * z * rinv3;
		} else if (r2 > 0.0) {
			f.potential += m * (T(1.5) * hinv - T(0.5) * r2 * hinv3);
			f.force[0] -= m * x * hinv3;
			f.force[1] -= m * y * hinv3;
			f.force[2] -= m * z * hinv3;
		}
	}
public:

	tree() {
		nodes.push_back(this);
	}

	void set_root() {
		begin = 0;
		end = 1;
		parent = nullptr;
		children = decltype(children)();
	}

	void add_particle(vec3 p, int depth = 0) {
		const int xdim = depth % NDIM;
		if (children.size()) {
			if (p[xdim] < children[LEFT].end[xdim]) {
				children[LEFT].add_particle(p, depth + 1);
			} else {
				children[RIGHT].add_particle(p, depth + 1);
			}
		} else if (parts.size() >= BUCKET_SIZE) {
			children.resize(NCHILD);
			children[LEFT].begin = children[RIGHT].begin = begin;
			children[LEFT].end = children[RIGHT].end = end;
			children[LEFT].end[xdim] = children[RIGHT].begin[xdim] = 0.5 * (begin[xdim] + end[xdim]);
			for (auto& par : parts) {
				add_particle(par.x, depth);
			}
			parts = decltype(parts)();
			add_particle(p, depth);
		} else {
			particle part;
			part.x = p;
			parts.push_back(part);
		}
	}

	void compute_multipoles(tree* par = nullptr, int depth = 0) {
		parent = par;
		if (children.size()) {
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci].compute_multipoles(this, depth + 1);
			}
		}
		double scale = end[0] - begin[0];
		multipole.init(scale);
		if (parts.size()) {
			center = T(0);
			for (const auto& part : parts) {
				center += part.x;
			}
			center /= parts.size();
			radius = 0.0;
			for (const auto& part : parts) {
				multipole_type M;
				vec3 dx = part.x - center;
				radius = std::max(radius, (double) abs(dx));
				sfmm::P2M(M, T(1), dx[0], dx[1], dx[2]);
				multipole += M;
			}
		} else if (children.size()) {
			const auto& left = children[LEFT];
			const auto& right = children[RIGHT];
			center = left.center * left.multipole(0, 0).real() + right.center * right.multipole(0, 0).real();
			const double total = left.multipole(0, 0).real() + right.multipole(0, 0).real();
			center /= total;
			radius = 0.0;
			for (int ci = 0; ci < NCHILD; ci++) {
				vec3 dx = children[ci].center - center;
				radius = std::max(radius, ((double) abs(dx) + children[ci].radius));
				auto M = children[ci].multipole;
				sfmm::M2M(M, dx[0], dx[1], dx[2]);
				multipole += M;
			}
		} else {
			center = (begin + end) * 0.5;
		}
		multipole.rescale(radius);
	}

	void compute_gravity_field(expansion_type expansion = expansion_type(), std::vector<check_type> checklist = std::vector<check_type>()) {
		std::vector<tree*> Clist, Plist;
		vec3 dx;
		if (parent) {
			expansion.rescale(radius);
			dx = parent->center - center;
			sfmm::L2L(expansion, dx[0], dx[1], dx[2]);
		} else {
			check_type ck;
			ck.ptr = this;
			ck.opened = false;
			checklist.push_back(ck);
			expansion.init(radius);
		}
		list_iterate(checklist, Plist, Clist, false);
		for (auto src : Clist) {
			if (src->children.size() == 0 && (src->parts.size() < MIN_CLOUD)) {
				for (const auto& part : src->parts) {
					dx = part.x - center;
					sfmm::P2L(expansion, T(1), dx[0], dx[1], dx[2]);
				}
			} else {
				dx = src->center - center;
				sfmm::M2L(expansion, src->multipole, dx[0], dx[1], dx[2]);
			}
		}
		for (auto src : Plist) {
			for (const auto& part : src->parts) {
				dx = part.x - center;
				sfmm::P2L(expansion, T(1), dx[0], dx[1], dx[2]);
			}
		}
		if (children.size()) {
			if (checklist.size()) {
				for (int ci = 0; ci < NCHILD; ci++) {
					children[ci].compute_gravity_field(expansion, checklist);
				}
			}
		} else {
			Plist.resize(0);
			Clist.resize(0);
			while (checklist.size()) {
				list_iterate(checklist, Plist, Clist, true);
			}
			for (auto& part : parts) {
				part.f.init();
				dx = center - part.x;
				sfmm::L2P(part.f, expansion, dx[0], dx[1], dx[2]);
			}
			for (auto src : Clist) {
				if (parts.size() < MIN_CLOUD) {
					for (auto& snk_part : parts) {
						for (const auto& src_part : src->parts) {
							dx = src_part.x - snk_part.x;
							P2P(snk_part.f, T(1), dx[0], dx[1], dx[2]);
						}
					}
				} else {
					for (auto& part : parts) {
						dx = src->center - part.x;
						sfmm::M2P(part.f, src->multipole, dx[0], dx[1], dx[2]);
					}
				}
			}
			for (auto src : Plist) {
				for (auto& snk_part : parts) {
					for (const auto& src_part : src->parts) {
						dx = src_part.x - snk_part.x;
						P2P(snk_part.f, T(1), dx[0], dx[1], dx[2]);
					}
				}
			}
		}
	}

	double compare_analytic(double sample_odds) {
		double err = 0.0;
		double norm = 0.0;
		for (const auto& snk_node : nodes) {
			for (const auto& snk_part : snk_node->parts) {
				if (rand1() > sample_odds) {
					continue;
				}
				force_type fa;
				fa.init();
				for (const auto& src_node : nodes) {
					for (const auto& src_part : src_node->parts) {
						const vec3 dx = src_part.x - snk_part.x;
						P2P(fa, T(1), dx[0], dx[1], dx[2]);
					}
				}
				double famag = 0.0;
				double fnmag = 0.0;
				for (int dim = 0; dim < NDIM; dim++) {
					famag += sqr(fa.force[0]) + sqr(fa.force[1]) + sqr(fa.force[2]);
					fnmag += sqr(snk_part.f.force[0]) + sqr(snk_part.f.force[1]) + sqr(snk_part.f.force[2]);
				}
				//	printf("%e %e %e\n", famag, fnmag, (famag - fnmag) / famag);
				famag = sqrt(famag);
				fnmag = sqrt(fnmag);
				norm += sqr(famag);
				err += sqr(famag - fnmag);
			}
		}
		err = sqrt(err / norm);
		return err;
	}
	void initialize() {
		for (int i = 0; i < TEST_SIZE; i++) {
			vec3 x;
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = rand1();
			}
			add_particle(x);
		}
	}

};

template<class T, int ORDER>
std::vector<tree<T, ORDER>*> tree<T, ORDER>::nodes;

template<class T, int ORDER = PMIN>
struct run_tests {
	void operator()() const {
		tree<T, ORDER> root;
		root.set_root();
		root.initialize();
		root.compute_multipoles();
		root.compute_gravity_field();
		printf("%i %e\n", ORDER, root.compare_analytic(0.1));
		run_tests<T, ORDER + 1> run;
		run();
	}
};

template<class T>
struct run_tests<T, PMAX + 1> {
	void operator()() const {
	}
};

int main(int argc, char **argv) {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_OVERFLOW);
	feenableexcept(FE_INVALID);
	printf("\ndouble\n");
	run_tests<double> run2;
	run2();
	printf("float\n");
	run_tests<float> run1;
	run1();
	return 0;
}
