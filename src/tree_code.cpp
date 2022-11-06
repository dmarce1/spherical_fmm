#include <stdio.h>
#include <sfmm.hpp>

#include <array>
#include <vector>
#include <fenv.h>
#include <future>
#include <atomic>

#define NDIM 3
#define BUCKET_SIZE 32
#define MIN_CLOUD 4
#define LEFT 0
#define RIGHT 1
#define NCHILD 2
#define MIN_THREAD 1024
#define TEST_SIZE 1000000
//#define FLAGS (sfmmWithRandomOptimization | sfmmProfilingOn)

using rtype = double;

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

template<class T, class V, class MV, int ORDER, int FLAGS>
class tree {

	template<class W>
	using multipole_type = sfmm::multipole<W,ORDER>;
	template<class W>
	using expansion_type = sfmm::expansion<W,ORDER>;
	template<class W>
	using force_type = sfmm::force_type<W>;
	struct particle {
		sfmm::vec3<T> x;
		force_type<T> f;
	};

	multipole_type<T> multipole;
	std::vector<tree> children;
	std::pair<int, int> part_range;
	sfmm::vec3<T> begin;
	sfmm::vec3<T> end;
	sfmm::vec3<T> center;
	tree* parent;
	T radius;

	static const T theta_max;
	static const T hsoft;

	static std::vector<particle> parts;
	static std::atomic<int> nthread_avail;

	struct check_type {
		tree* ptr;
		bool opened;
	};

	template<class W>
	size_t P2P(sfmm::force_type<W>& f, W m, sfmm::vec3<W> dx) {
		static const W h2(hsoft * hsoft);
		static const W hinv(W(1) / hsoft);
		static const W hinv3(sfmm::sqr(hinv) * hinv);
		const W r2 = sfmm::sqr(dx[0]) + sfmm::sqr(dx[1]) + sfmm::sqr(dx[2]);
		const W wn(m * (r2 < h2));
		const W wf(m * (r2 >= h2));
		sfmm::vec3<W> fn, ff;
		W rzero(r2 < W(1e-30));
		W pn, pf;
		const W rinv = sfmm::rsqrt(r2 + rzero);
		W rinv3 = sfmm::sqr(rinv) * rinv;
		pf = rinv;
		ff = dx * rinv3;
		pn = (W(1.5) * hinv - W(0.5) * r2 * hinv3);
		fn = dx * hinv3;
		f.potential += pn * wn + pf * wf;
		f.force -= fn * wn + ff * wf;
		return 8 * 41;
	}

	void list_iterate(std::vector<check_type>& checklist, std::vector<tree*>& Plist, std::vector<tree*>& Clist, bool leaf) {
		static thread_local std::vector<check_type> nextlist;
		nextlist.resize(0);
		auto c0 = V(theta_max*theta_max);
		for (int i = 0; i < checklist.size(); i += sfmm::simd_size<V>()) {
			sfmm::vec3<V> dx;
			V rsum;
			const int end = std::min(sfmm::simd_size<V>(), checklist.size() - i);
			for (int j = 0; j < end; j++) {
				const auto& check = checklist[i + j];
				sfmm::load(dx, center - check.ptr->center, j);
				sfmm::load(rsum, radius + check.ptr->radius, j);
			}
			sfmm::apply_padding(dx, end);
			sfmm::apply_padding(rsum, end);
			V far(sqr(rsum) < c0 * sqr(dx));
			for (int j = 0; j < end; j++) {
				auto& check = checklist[i + j];
				if (sfmm::access(far, j) || (leaf && check.opened)) {
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
		}
		checklist = nextlist;
	}

public:

	tree() {
	}

	void set_root() {
		begin = 0;
		end = 1;
		parent = nullptr;
		children = decltype(children)();
	}

	size_t form_tree(tree* par = nullptr, int depth = 0) {
		size_t flops = 0;
		const int xdim = depth % SFMM_NDIM;
		if (par == nullptr) {
			part_range.first = 0;
			part_range.second = parts.size();
		}
		parent = par;
		const T xmid = T(0.5) * (begin[xdim] + end[xdim]);
		const int nparts = part_range.second - part_range.first;
		T scale = end[0] - begin[0];
		multipole.init(scale);
		if (nparts > BUCKET_SIZE) {
			sfmm::vec3<MV> dx;
			multipole_type<MV> M;
			MV cr;
			children.resize(NCHILD);
			int lo = part_range.first;
			int hi = part_range.second;
			while (lo < hi) {
				if (parts[lo].x[xdim] >= xmid) {
					while (lo != hi) {
						hi--;
						if (parts[hi].x[xdim] < xmid) {
							std::swap(parts[lo], parts[hi]);
							break;
						}
					}
				}
				lo++;
			}
			auto& left = children[LEFT];
			auto& right = children[RIGHT];
			left.part_range = right.part_range = part_range;
			left.part_range.second = right.part_range.first = hi;
			right.begin = left.begin = begin;
			left.end = right.end = end;
			left.end[xdim] = right.begin[xdim] = 0.5 * (begin[xdim] + end[xdim]);
			if (nthread_avail-- >= 0 && part_range.second - part_range.first > MIN_THREAD) {
				auto fut = std::async([this,depth]() {
					const auto flops = children[LEFT].form_tree(this, depth + 1);
					nthread_avail++;
					return flops;
				});
				flops += children[RIGHT].form_tree(this, depth + 1);
				flops += fut.get();
			} else {
				nthread_avail++;
				for (int ci = 0; ci < NCHILD; ci++) {
					flops += children[ci].form_tree(this, depth + 1);
				}
			}
			const auto& cleft = children[LEFT];
			const auto& cright = children[RIGHT];
			center = left.center * cleft.multipole(0, 0).real() + right.center * cright.multipole(0, 0).real();
			const T total = cleft.multipole(0, 0).real() + cright.multipole(0, 0).real();
			center /= total;
			radius = 0.0;
			for (int i = 0; i < NCHILD; i += sfmm::simd_size<MV>()) {
				const int end = std::min((int) sfmm::simd_size<MV>(), NCHILD - i);
				for (int j = 0; j < end; j++) {
					const int ci = i + j;
					sfmm::load(dx, children[ci].center - center, j);
					sfmm::load(cr, children[ci].radius, j);
					sfmm::load(M, children[ci].multipole, j);
				}
				radius = std::max(radius, sfmm::reduce_max(abs(dx) + cr));
				flops += sfmm::M2M(M, dx, FLAGS);
				multipole += sfmm::reduce_sum(M);
			}
		} else {
			if (nparts) {
				center = T(0);
				for (int i = part_range.first; i < part_range.second; i++) {
					center += parts[i].x;
				}
				center /= nparts;
				radius = 0.0;
				for (int i = part_range.first; i < part_range.second; i += sfmm::simd_size<V>()) {
					sfmm::vec3<V> dx;
					const int end = std::min((int) sfmm::simd_size<V>(), part_range.second - i);
					for (int j = 0; j < end; j++) {
						const auto& part = parts[i + j];
						load(dx, part.x - center, j);
					}
					multipole_type<V> M;
					radius = std::max(radius, sfmm::reduce_max(abs(dx)));
					flops += sfmm::P2M(M, sfmm::create_mask<V>(end), dx);
					multipole += sfmm::reduce_sum(M);
				}
				radius = std::max(hsoft, radius);
			} else {
				center = (begin + end) * 0.5;
				radius = hsoft;
			}
		}
		multipole.rescale(radius);
		return flops;
	}

	size_t compute_gravity_field(expansion_type<T> expansion = expansion_type<T>(), std::vector<check_type> checklist = std::vector<check_type>()) {
		size_t flops = 0;
		static thread_local std::vector<tree*> Clist;
		static thread_local std::vector<tree*> Plist;
		Clist.resize(0);
		Plist.resize(0);
		expansion_type<V> L;
		multipole_type<V> M;
		force_type<V> F;
		sfmm::vec3<V> dx;

		if (parent) {
			expansion.rescale(radius);
			flops += sfmm::L2L(expansion, parent->center - center, FLAGS);
		} else {
			check_type ck;
			ck.ptr = this;
			ck.opened = false;
			checklist.push_back(ck);
			expansion.init(radius);
		}
		list_iterate(checklist, Plist, Clist, false);
		for (int i = 0; i < Clist.size(); i += sfmm::simd_size<V>()) {
			L.init();
			const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
			for (int j = 0; j < end; j++) {
				const auto& src = Clist[i + j];
				load(dx, src->center - center, j);
				load(M, src->multipole, j);
			}
			apply_padding(dx, end);
			apply_padding(M, end);
			flops += sfmm::M2L(L, M, dx, FLAGS);
			apply_mask(L, end);
			expansion += reduce_sum(L);
		}
		for (int i = 0; i < Plist.size(); i++) {
			for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
				const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
				L.init();
				for (int k = 0; k < end; k++) {
					load(dx, parts[j + k].x - center, k);
				}
				apply_padding(dx, end);
				flops += sfmm::P2L(L, sfmm::create_mask<V>(end), dx);
				expansion += reduce_sum(L);
			}
		}
		if (children.size()) {
			if (checklist.size()) {
				if (nthread_avail-- >= 0 && part_range.second - part_range.first > MIN_THREAD) {
					auto fut = std::async([this,&expansion](decltype(checklist) checklist) {
						const auto flops = children[LEFT].compute_gravity_field(expansion, std::move(checklist));
						nthread_avail++;
						return flops;
					}, checklist);
					flops += children[RIGHT].compute_gravity_field(expansion, std::move(checklist));
					flops += fut.get();
				} else {
					nthread_avail++;
					flops += children[LEFT].compute_gravity_field(expansion, checklist);
					flops += children[RIGHT].compute_gravity_field(expansion, std::move(checklist));
				}
			}
		} else {
			Plist.resize(0);
			Clist.resize(0);
			while (checklist.size()) {
				list_iterate(checklist, Plist, Clist, true);
			}
			load(L, expansion);
			for (int i = part_range.first; i < part_range.second; i += sfmm::simd_size<V>()) {
				sfmm::vec3<V> part_x;
				force_type<V> F;
				F.init();
				const int end = std::min((int) sfmm::simd_size<V>(), part_range.second - i);
				for (int j = 0; j < end; j++) {
					load(dx, center - parts[i + j].x, j);
				}
				flops += sfmm::L2P(F, L, dx, FLAGS);
				for (int j = 0; j < end; j++) {
					store(parts[i + j].f, F, j);
				}
			}
			for (int i = 0; i < Clist.size(); i += sfmm::simd_size<V>()) {
				for (int k = part_range.first; k < part_range.second; k++) {
					auto& part = parts[k];
					F.init();
					const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
					for (int j = 0; j < end; j++) {
						const auto& src = Clist[i + j];
						load(dx, src->center - part.x, j);
						load(M, src->multipole, j);
					}
					apply_padding(dx, end);
					apply_padding(M, end);
					sfmm::M2P(F, M, dx, FLAGS);
					for (int j = 0; j < end; j++) {
						accumulate(part.f, F, j);
					}
				}
			}
			for (int i = 0; i < Plist.size(); i++) {
				for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
					const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
					for (int l = part_range.first; l < part_range.second; l++) {
						auto& part = parts[l];
						for (int k = 0; k < end; k++) {
							const auto& src_part = parts[j + k];
							load(dx, parts[j + k].x - part.x, k);
						}
						apply_padding(dx, end);
						F.init();
						flops += P2P(F, sfmm::create_mask<V>(end), dx);
						part.f += sfmm::reduce_sum(F);
					}
				}
			}
		}
		return flops;
	}

	T compare_analytic(T sample_odds) {
		double err = 0.0;
		double norm = 0.0;
		for (int i = 0; i < parts.size(); i++) {
			if (rand1() > sample_odds) {
				continue;
			}
			const auto& snk_part = parts[i];
			force_type<double> fa;
			fa.init();
			for (int j = 0; j < parts.size(); j++) {
				if (i == j) {
					continue;
				}
				const auto& src_part = parts[j];
				sfmm::vec3<double> dx;
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = src_part.x[dim] - snk_part.x[dim];
				}
				P2P<double>(fa, double(1), dx);
			}
			double famag = 0.0;
			double fnmag = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				famag += sfmm::sqr(fa.force[0]) + sfmm::sqr(fa.force[1]) + sfmm::sqr(fa.force[2]);
				fnmag += sfmm::sqr(snk_part.f.force[0]) + sfmm::sqr(snk_part.f.force[1]) + sfmm::sqr(snk_part.f.force[2]);
			}
			//	printf("%e %e %e\n", famag, fnmag, (famag - fnmag) / famag);
			famag = sqrt(famag);
			fnmag = sqrt(fnmag);
			norm += sfmm::sqr(famag);
			err += sfmm::sqr(famag - fnmag);
		}
		err = sqrt(err / norm);
		return err;
	}
	void initialize() {
		parts.resize(0);
		for (int i = 0; i < TEST_SIZE; i++) {
			sfmm::vec3<T> x;
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = rand1();
			}
			particle p;
			p.x = x;
			parts.push_back(p);
		}
	}

};

template<class T, class V, class M, int ORDER, int FLAGS>
std::vector<typename tree<T, V, M, ORDER, FLAGS>::particle> tree<T, V, M, ORDER, FLAGS>::parts;

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<int> tree<T, V, M, ORDER, FLAGS>::nthread_avail(std::thread::hardware_concurrency() - 1);

template<class T, class V, class M, int ORDER, int FLAGS>
const T tree<T, V, M, ORDER, FLAGS>::theta_max = 0.6;

template<class T, class V, class M, int ORDER, int FLAGS>
const T tree<T, V, M, ORDER, FLAGS>::hsoft = 0.01;

template<class T, class V, class M, int ORDER, int FLAGS>
struct run_tests {
	void operator()() const {
		sfmm::timer tm, ftm;
		double tree_time, force_time;
		tree<T, V, M, ORDER, FLAGS> root;
		root.set_root();
		root.initialize();
		tm.start();
		size_t flops = root.form_tree();
		tm.stop();
		tree_time = tm.read();
		tm.reset();
		tm.start();
		ftm.start();
		flops += root.compute_gravity_field();
		tm.stop();
		ftm.stop();
		force_time = tm.read();
		tm.reset();
		tm.start();
		const auto error = root.compare_analytic(100.0 / TEST_SIZE);
		tm.stop();
		printf("%i %e %e %e %e %e Gflops\n", ORDER, tree_time, force_time, tm.read(), error, flops / ftm.read() / (1024.0 * 1024.0 * 1024.0));
		run_tests<T, V, M, ORDER + 1, FLAGS> run;
		run();
	}
};

template<class T, class V, class M, int FLAGS>
struct run_tests<T, V, M, PMAX + 1, FLAGS> {
	void operator()() const {
	}
};

int main(int argc, char **argv) {
//	feenableexcept(FE_DIVBYZERO);
	//feenableexcept(FE_OVERFLOW);
//	feenableexcept(FE_INVALID);
	run_tests<float, sfmm::simd_f32, sfmm::m2m_simd_f32, PMIN, sfmmWithDoubleRotationOptimization /*| sfmmProfilingOn*/> run1;
	run_tests<float, sfmm::simd_f32, sfmm::m2m_simd_f32, PMIN, sfmmWithDoubleRotationOptimization /*| sfmmProfilingOn*/ | sfmmCalculateWithoutPotential> run2;
	auto prof = sfmm::detail::operator_best_rotations();
//	printf("%s\n", prof.c_str());
	run1();
	run2();
//	prof = sfmm::operator_profiling_results();
	//printf("%s\n", prof.c_str());
	//prof = sfmm::detail::operator_best_rotations();
//	printf("%s\n", prof.c_str());
//	sfmm::detail::operator_write_new_bestops_source();
	return 0;
}
