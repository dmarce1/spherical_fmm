#include <stdio.h>
#include <sfmm.hpp>

#include <array>
#include <vector>
#include <fenv.h>
#include <future>
#include <atomic>

#define DEBUG

#define NDIM 3
#define BUCKET_SIZE 32
#define MIN_CLOUD 4
#define LEFT 0
#define RIGHT 1
#define NCHILD 2
#define MIN_THREAD 1024
#define TEST_SIZE 10000
//#define FLAGS (sfmmWithRandomOptimization | sfmmProfilingOn)

using rtype = double;

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

int P2P_ewald(sfmm::force_type<double>& f, double m, sfmm::vec3<double> dx) {
	//dx = -dx;
	double& pot = f.potential;
	double& fx = f.force[0];
	double& fy = f.force[1];
	double& fz = f.force[2];
	double& dx0 = dx[0];
	double& dx1 = dx[1];
	double& dx2 = dx[2];
	const double cons1 = (double) (4.0 / sqrt(4.0 * atan(1)));
	const auto r2 = sfmm::sqr(dx0) + sfmm::sqr(dx1) + sfmm::sqr(dx2);  // 5

	if (r2 > 0.) {
		const double dx = dx0;
		const double dy = dx1;
		const double dz = dx2;
		const double r2 = sfmm::sqr(dx) + sfmm::sqr(dy) + sfmm::sqr(dz);
		const double r = sqrt(r2);
		const double rinv = 1. / r;
		const double r2inv = rinv * rinv;
		const double r3inv = r2inv * rinv;
		double exp0 = exp(-4.0 * r2);
		double erf0 = erf(2.0 * r);
		const double expfactor = cons1 * r * exp0;
		const double d0 = erf0 * rinv;
		const double d1 = (expfactor - erf0) * r3inv;
		pot += m * d0;
		fx -= m * dx * d1;
		fy -= m * dy * d1;
		fz -= m * dz * d1;
		for (int xi = -4; xi <= +4; xi++) {
			for (int yi = -4; yi <= +4; yi++) {
				for (int zi = -4; zi <= +4; zi++) {
					const bool center = xi * xi + yi * yi + zi * zi == 0;
					if (center) {
						continue;
					}
					const double dx = dx0 - xi;
					const double dy = dx1 - yi;
					const double dz = dx2 - zi;
					const double r2 = dx * dx + dy * dy + dz * dz;
					const double r = sqrt(r2);
					if (r > 3.6) {
						continue;
					}
					const double rinv = 1. / r;
					const double r2inv = rinv * rinv;
					const double r3inv = r2inv * rinv;
					double exp0 = exp(-4.0 * r2);
					double erfc0 = erfc(2.0 * r);
					const double expfactor = cons1 * r * exp0;
					const double d0 = -erfc0 * rinv;
					const double d1 = (expfactor + erfc0) * r3inv;
					pot += m * d0;
					fx -= m * dx * d1;
					fy -= m * dy * d1;
					fz -= m * dz * d1;
				}
			}
		}
		pot += (double) m * (M_PI / 4.0);
		for (int xi = -3; xi <= +3; xi++) {
			//	printf( "%i\n", xi);
			for (int yi = -3; yi <= +3; yi++) {
				for (int zi = -3; zi <= +3; zi++) {
					const double hx = xi;
					const double hy = yi;
					const double hz = zi;
					const double h2 = hx * hx + hy * hy + hz * hz;
					if (h2 > 0.0 && h2 <= 10) {
						const double hdotx = dx0 * hx + dx1 * hy + dx2 * hz;
						const double omega = (double) (2.0 * M_PI) * hdotx;
						double c, s;
						sincos(omega, &s, &c);
						const double c0 = -1. / h2 * exp((double) (-sfmm::sqr(M_PI) * 0.25) * h2) * (double) (1. / (M_PI));
						const double c1 = -s * 2.0 * M_PI * c0;
						pot += m * c0 * c;
						fx -= m * c1 * hx;
						fy -= m * c1 * hy;
						fz -= m * c1 * hz;
					}
				}
			}
		}
	} else {
		pot += m * 2.8372975;
	}
	return 0;
}

template<class W>
static size_t P2P(sfmm::force_type<W>& f, W m, sfmm::vec3<W> dx) {
	const static double hsoft = 0.01;
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
	f.potential -= pn * wn + pf * wf;
	f.force -= fn * wn + ff * wf;
	return 41;
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
	static const T mass;
	static const int Ngrid;

	static std::vector<particle> parts;
	static std::vector<tree> forest;

	struct check_type {
		tree* ptr;
		bool opened;
	};

	void list_iterate(std::vector<check_type>& checklist, std::vector<tree*>& Plist, std::vector<tree*>& Clist, bool leaf, bool ewald) {
		static thread_local std::vector<check_type> nextlist;
		nextlist.resize(0);
		auto c0 = V(theta_max);
		for (int i = 0; i < checklist.size(); i += sfmm::simd_size<V>()) {
			sfmm::vec3<V> dx;
			V rsum;
			const int end = std::min(sfmm::simd_size<V>(), checklist.size() - i);
			for (int j = 0; j < end; j++) {
				const auto& check = checklist[i + j];
				sfmm::load(dx, sfmm::distance(center, check.ptr->center), j);
				sfmm::load(rsum, radius + check.ptr->radius, j);
			}
			sfmm::apply_padding(dx, end);
			sfmm::apply_padding(rsum, end);
			V far;
			if (ewald) {
				far = (rsum < c0 * sfmm::max(abs(dx), V(0.5) - rsum));
			} else {
				far = sfmm::sqr(rsum) < sfmm::sqr(c0) * sfmm::sqr(dx);
			}
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

	static int partition_parts(std::pair<int, int> part_range, int xdim, T xmid) {
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
		return hi;
	}

	static void sort_grid(sfmm::vec3<int> cell_begin = { 0, 0, 0 }, sfmm::vec3<int> cell_end = { Ngrid, Ngrid, Ngrid }, std::pair<int, int> part_range =
			std::make_pair(0, parts.size())) {
#ifdef DEBUG
		bool flag = false;
		for (int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				T begin = (T) cell_begin[dim] / Ngrid;
				T end = (T) cell_end[dim] / Ngrid;
				if (parts[i].x[dim] < begin || parts[i].x[dim] > end) {
					flag = true;
				}
			}
		}
		for (int i = part_range.first; i < part_range.second; i++) {
			if (flag) {
				for (int dim = 0; dim < NDIM; dim++) {
					T begin = (T) cell_begin[dim] / Ngrid;
					T end = (T) cell_end[dim] / Ngrid;
					printf("particle out of range! %e %e %e \n", begin, parts[i].x[dim], end);
				}
			}
		}
		if (flag) {
			abort();
		}
#endif
		sfmm::vec3<int> cell_end_left;
		sfmm::vec3<int> cell_begin_right;
		std::pair<int, int> range_left;
		std::pair<int, int> range_right;
		int cmid;
		int xdim;
		T xmid;
		int largest = 0;
		for (int dim = 0; dim < SFMM_NDIM; dim++) {
			const int span = cell_end[dim] - cell_begin[dim];
			if (span > largest) {
				largest = span;
				xdim = dim;
			}
		}
		if (largest == 1) {
			//printf( "%i %i %i : %i %i\n", cell_begin[0], cell_begin[1], cell_begin[2], part_range.first, part_range.second);
			tree root;
			root.part_range = part_range;
			for (int dim = 0; dim < SFMM_NDIM; dim++) {
				root.begin[dim] = (T) cell_begin[dim] / Ngrid;
				root.end[dim] = (T) cell_end[dim] / Ngrid;
			}
			forest.push_back(root);
		} else {
			cmid = (cell_end[xdim] + cell_begin[xdim]) / 2;
			xmid = (T) cmid / Ngrid;
			int pmid = partition_parts(part_range, xdim, xmid);
			range_left = range_right = part_range;
			range_left.second = range_right.first = pmid;
			cell_end_left = cell_end;
			cell_begin_right = cell_begin;
			cell_end_left[xdim] = cell_begin_right[xdim] = cmid;
			sort_grid(cell_begin, cell_end_left, range_left);
			sort_grid(cell_begin_right, cell_end, range_right);
		}
	}

	static size_t form_trees() {
		size_t flops = 0;
		std::vector<std::future<size_t>> futs;
		for (auto& tr : forest) {
			auto* ptr = &tr;
			futs.push_back(std::async([ptr]() {return ptr->form_tree();}));
		}
		for (auto& f : futs) {
			flops += f.get();
		}
		return flops;
	}

	size_t form_tree(tree* par = nullptr, int depth = 0) {
		size_t flops = 0;
		const int xdim = depth % SFMM_NDIM;
		parent = par;
		const T xmid = T(0.5) * (begin[xdim] + end[xdim]);
		const int nparts = part_range.second - part_range.first;
		T scale = end[0] - begin[0];
		multipole.init(scale);
#ifdef DEBUG
		for (int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				if (parts[i].x[dim] < begin[dim] || parts[i].x[dim] > end[dim]) {
					printf("particle out of range! %e %e %e %i\n", begin[dim], parts[i].x[dim], end[dim], depth);
				}
			}
		}
#endif
		if (nparts > BUCKET_SIZE) {
			sfmm::vec3<MV> dx;
			multipole_type<MV> M;
			MV cr;
			children.resize(NCHILD);
			int imid = partition_parts(part_range, xdim, xmid);
			auto& left = children[LEFT];
			auto& right = children[RIGHT];
			left.part_range = right.part_range = part_range;
			left.part_range.second = right.part_range.first = imid;
			right.begin = left.begin = begin;
			left.end = right.end = end;
			left.end[xdim] = right.begin[xdim] = 0.5 * (begin[xdim] + end[xdim]);
			for (int ci = 0; ci < NCHILD; ci++) {
				flops += children[ci].form_tree(this, depth + 1);
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
				flops += sfmm::simd_size<MV>() * sfmm::M2M(M, dx, FLAGS);
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
					flops += sfmm::simd_size<V>() * sfmm::P2M(M, sfmm::create_mask<V>(end) * mass, dx);
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

	static size_t compute_gravity() {
		std::vector<check_type> checklist;
		size_t flops = 0;
		for (auto& root : forest) {
			check_type entry;
			entry.ptr = &root;
			entry.opened = false;
			checklist.push_back(entry);
		}
		std::vector<std::future<size_t>> futs;
		for (auto& tr : forest) {
			auto* ptr = &tr;
			futs.push_back(std::async([ptr,checklist]() {
				size_t flops = 0;
				expansion_type<T> expansion;
				expansion.init();
				return flops + ptr->compute_cell_gravity(expansion, checklist, checklist);
			}));
		}
		for (auto& f : futs) {
			flops += f.get();
		}
		return flops;
	}

	size_t compute_cell_gravity(expansion_type<T> expansion, std::vector<check_type> checklist, std::vector<check_type> echecklist) {
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
		}
		list_iterate(echecklist, Plist, Clist, false, true);
		for (int i = 0; i < Clist.size(); i += sfmm::simd_size<V>()) {
			L.init();
			const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
			for (int j = 0; j < end; j++) {
				const auto& src = Clist[i + j];
				load(dx, sfmm::distance(center, src->center), j);
				load(M, src->multipole, j);
			}
			apply_padding(dx, end);
			apply_padding(M, end);
			flops += sfmm::simd_size<V>() * sfmm::M2L_ewald(L, M, dx, FLAGS);
			apply_mask(L, end);
			expansion += reduce_sum(L);
		}
		for (int i = 0; i < Plist.size(); i++) {
			for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
				const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
				L.init();
				for (int k = 0; k < end; k++) {
					load(dx, sfmm::distance(center, parts[j + k].x), k);
				}
				apply_padding(dx, end);
				flops += sfmm::simd_size<V>() * sfmm::P2L_ewald(L, sfmm::create_mask<V>(end) * mass, dx);
				expansion += reduce_sum(L);
			}
		}
		Clist.resize(0);
		Plist.resize(0);
		list_iterate(checklist, Plist, Clist, false, false);
		for (int i = 0; i < Clist.size(); i += sfmm::simd_size<V>()) {
			L.init();
			const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
			for (int j = 0; j < end; j++) {
				const auto& src = Clist[i + j];
				load(dx, sfmm::distance(center, src->center), j);
				load(M, src->multipole, j);
			}
			apply_padding(dx, end);
			apply_padding(M, end);
			flops += sfmm::simd_size<V>() * sfmm::M2L(L, M, dx, FLAGS);
			apply_mask(L, end);
			expansion += reduce_sum(L);
		}
		for (int i = 0; i < Plist.size(); i++) {
			for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
				const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
				L.init();
				for (int k = 0; k < end; k++) {
					load(dx, sfmm::distance(center, parts[j + k].x), k);
				}
				apply_padding(dx, end);
				flops += sfmm::simd_size<V>() * sfmm::P2L(L, sfmm::create_mask<V>(end) * mass, dx);
				expansion += reduce_sum(L);
			}
		}
		if (children.size()) {
			if (checklist.size()) {
				flops += children[LEFT].compute_cell_gravity(expansion, checklist, echecklist);
				flops += children[RIGHT].compute_cell_gravity(expansion, std::move(checklist), std::move(echecklist));
			}
		} else {
			load(L, expansion);
			for (int i = part_range.first; i < part_range.second; i += sfmm::simd_size<V>()) {
				force_type<V> F;
				F.init();
				const int end = std::min((int) sfmm::simd_size<V>(), part_range.second - i);
				for (int j = 0; j < end; j++) {
					load(dx, center - parts[i + j].x, j);
				}
				flops += sfmm::simd_size<V>() * sfmm::L2P(F, L, dx, FLAGS);
				for (int j = 0; j < end; j++) {
					store(parts[i + j].f, F, j);
				}
			}
			Plist.resize(0);
			Clist.resize(0);
			while (echecklist.size()) {
				list_iterate(echecklist, Plist, Clist, true, true);
			}
			for (int i = 0; i < Clist.size(); i += sfmm::simd_size<V>()) {
				for (int k = part_range.first; k < part_range.second; k++) {
					auto& part = parts[k];
					F.init();
					const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
					for (int j = 0; j < end; j++) {
						const auto& src = Clist[i + j];
						load(dx, sfmm::distance(part.x, src->center), j);
						load(M, src->multipole, j);
					}
					apply_padding(dx, end);
					apply_padding(M, end);
					flops += sfmm::simd_size<V>() * sfmm::M2P_ewald(F, M, dx, FLAGS);
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
							load(dx, sfmm::distance(part.x, parts[j + k].x), k);
						}
						apply_padding(dx, end);
						F.init();
						flops += sfmm::simd_size<V>() * P2P_ewald(F, sfmm::create_mask<V>(end) * mass, dx);
						part.f += sfmm::reduce_sum(F);
					}
				}
			}
			Plist.resize(0);
			Clist.resize(0);
			while (checklist.size()) {
				list_iterate(checklist, Plist, Clist, true, false);
			}
			for (int i = 0; i < Clist.size(); i += sfmm::simd_size<V>()) {
				for (int k = part_range.first; k < part_range.second; k++) {
					auto& part = parts[k];
					F.init();
					const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
					for (int j = 0; j < end; j++) {
						const auto& src = Clist[i + j];
						load(dx, sfmm::distance(part.x, src->center), j);
						load(M, src->multipole, j);
					}
					apply_padding(dx, end);
					apply_padding(M, end);
					flops += sfmm::simd_size<V>() * sfmm::M2P(F, M, dx, FLAGS);
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
							load(dx, sfmm::distance(part.x, parts[j + k].x), k);
						}
						apply_padding(dx, end);
						F.init();
						flops += sfmm::simd_size<V>() * P2P(F, sfmm::create_mask<V>(end) * mass, dx);
						part.f += sfmm::reduce_sum(F);
					}
				}
			}
		}
		return flops;
	}

	static std::pair<T, T> compare_analytic(T sample_odds) {
		double err = 0.0;
		double norm = 0.0;
		double perr = 0.0;
		double pnorm = 0.0;
		for (int i = 0; i < parts.size(); i++) {
			if (rand1() > sample_odds) {
				continue;
			}
			const auto& snk_part = parts[i];
			force_type<double> fa;
			fa.init();
			const int nthreads = std::thread::hardware_concurrency();
			std::vector<std::future<void>> futs;
			std::mutex mutex;
			for (int proc = 0; proc < nthreads; proc++) {
				const int b = (size_t) proc * parts.size() / nthreads;
				const int e = (size_t) (proc + 1) * parts.size() / nthreads;
				futs.push_back(std::async([b,e,&fa,&mutex,snk_part]() {
					for (int j = b; j < e; j++) {
						force_type<double> fc;
						const auto& src_part = parts[j];
						sfmm::vec3<double> dx;
						for (int dim = 0; dim < NDIM; dim++) {
							dx[dim] = sfmm::distance(snk_part.x[dim], src_part.x[dim]);
						}
						fc.init();
						P2P_ewald(fc, mass, dx);
						force_type<double> fd;
						fd.init();
						P2P<double>(fd, mass, dx);
						//	printf( "%e\n", fc.force[0]*fd.force[0]+fc.force[1]*fd.force[1]+fc.force[2]*fd.force[2]);
						std::lock_guard<std::mutex> lock(mutex);
						fa.force += fd.force + fc.force;
						fa.potential += fd.potential + fc.potential;
//				fa += P2P_ewald(mass, dx);
					}
				}));
			}
			for (auto& f : futs) {
				f.get();
			}
			double famag = 0.0;
			double fnmag = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				famag += sfmm::sqr(fa.force[0]) + sfmm::sqr(fa.force[1]) + sfmm::sqr(fa.force[2]);
				fnmag += sfmm::sqr(snk_part.f.force[0]) + sfmm::sqr(snk_part.f.force[1]) + sfmm::sqr(snk_part.f.force[2]);
			}
			//printf("%i %e %e %e \n", i, fa.potential, snk_part.f.potential, (fa.potential - snk_part.f.potential)/fa.potential);
			famag = sqrt(famag);
			fnmag = sqrt(fnmag);
			perr += sfmm::sqr(snk_part.f.potential - fa.potential);
			pnorm += sfmm::sqr(fa.potential);
			norm += sfmm::sqr(famag);
			err += sfmm::sqr(famag - fnmag);
		}
		err = sqrt(err / norm);
		perr = sqrt(perr / pnorm);
		return std::make_pair(perr, err);
	}

	static void initialize() {
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
const T tree<T, V, M, ORDER, FLAGS>::theta_max = 0.55;

template<class T, class V, class M, int ORDER, int FLAGS>
const T tree<T, V, M, ORDER, FLAGS>::mass = 1.0 / TEST_SIZE;

template<class T, class V, class M, int ORDER, int FLAGS>
const T tree<T, V, M, ORDER, FLAGS>::hsoft = 0.01;

template<class T, class V, class M, int ORDER, int FLAGS>
const int tree<T, V, M, ORDER, FLAGS>::Ngrid = 12;

template<class T, class V, class M, int ORDER, int FLAGS>
std::vector<tree<T, V, M, ORDER, FLAGS>> tree<T, V, M, ORDER, FLAGS>::forest;

template<class T, class V, class M, int ORDER, int FLAGS>
struct run_tests {
	void operator()() const {
		sfmm::timer tm, ftm;
		using tree_type = tree<T, V, M, ORDER, FLAGS>;
		double tree_time, force_time;
		tree_type::initialize();
		tree_time = tm.read();
		tm.reset();
		tm.start();
		ftm.start();
		fflush(stdout);
		tree_type::sort_grid();
		size_t flops = tree_type::form_trees();
		flops += tree_type::compute_gravity();
		tm.stop();
		ftm.stop();
		force_time = tm.read();
		tm.reset();
		tm.start();
		const auto error = tree_type::compare_analytic(50.0 / TEST_SIZE);
		tm.stop();
		printf("%i %e %e %e %e %e %e Gflops\n", ORDER, tree_time, force_time, tm.read(), error.first, error.second,
				flops / ftm.read() / (1024.0 * 1024.0 * 1024.0));
		run_tests<T, V, M, ORDER + 1, FLAGS> run;
		run();
	}
};

template<class T, class V, class M, int FLAGS>
struct run_tests<T, V, M, PMAX + 1, FLAGS> {
	void operator()() const {
	}
};

void ewald() {
	constexpr int N = 8;
	sfmm::multipole<double, N> M;
	sfmm::expansion<double, N> L;
	M.init();
	M[0] = 1.0;
	sfmm::vec3<double> dx;
	for (double x = 0.0; x < 0.5; x += 0.001) {
		dx[2] = dx[1] = double(0.0);
		dx[0] = double(x);
		sfmm::M2L_ewald(L, M, dx);
		printf("%e ", x);
		for (int i = 0; i < L.size(); i++) {
			printf("%e ", L[i]);
		}
		printf("\n");
	}
}

void random_unit(double& x, double& y, double& z) {
	const double theta = acos(2 * rand1() - 1.0);
	const double phi = rand1() * 2.0 * M_PI;
	x = cos(phi) * sin(theta);
	y = sin(phi) * sin(theta);
	z = cos(theta);
}

namespace sfmm {
constexpr int P = 5;
void M2L_ewald2(sfmm::expansion<double, P>, sfmm::multipole<double, P>, sfmm::vec3<double> x) {
	constexpr double alpha = 2.0;
	expansion<double, P> G;
	expansion<double, P> Gr;
	const auto index = [](int l, int m) {
		return l*l + l + m;
	};
	G.init();
	for (int xi = -3; xi <= 3; xi++) {
		for (int yi = -3; yi <= 3; yi++) {
			for (int zi = -3; zi <= 3; zi++) {
				vec3<double> x0 = { (double) xi, (double) yi, (double) zi };
				vec3<double> dx = x - x0;
				greens(Gr, dx);
				const double r = abs(dx);
				const double r2 = r * r;
				double igamma[P + 1];
				double cgamma[P + 1];
				igamma[0] = sqrt(M_PI) * erfc(alpha * r);
				cgamma[0] = sqrt(M_PI);
				for (int l = 0; l < P; l++) {
					const double s = l + 0.5;
					igamma[l + 1] = s * igamma[l] + pow(r2, s) * exp(-r2);
					cgamma[l + 1] = s * cgamma[l];
				}
				for (int l = 0; l <= P; l++) {
					for (int m = -l; m <= l; m++) {
						G[index(l, m)] += igamma[l] / cgamma[l] * Gr[index(l, m)];
					}
				}
			}
		}

	}
	for (int xi = -2; xi <= 2; xi++) {
		for (int yi = -2; yi <= 2; yi++) {
			for (int zi = -2; zi <= 2; zi++) {
				vec3<double> h = { (double) xi, (double) yi, (double) zi };
				if (abs(h) > 0.0) {
					const double hdotx = h[0] * x[0] + h[1] * x[1] + h[2] * x[2];
					double cgamma[P + 1];
					cgamma[0] = sqrt(M_PI);
					for (int l = 0; l < P; l++) {
						const double s = l + 0.5;
						cgamma[l + 1] = s * cgamma[l];
					}

				}
			}
		}
	}
}
}

template<int P>
void test2() {
	sfmm::vec3<double> dx0;
	sfmm::vec3<double> dx1;
	sfmm::vec3<double> dx2;
	sfmm::vec3<double> dx3;
	sfmm::vec3<double> dx4;
	sfmm::vec3<double> x0;
	sfmm::vec3<double> x1;
	sfmm::vec3<double> x2;
	sfmm::vec3<double> x3;
	sfmm::vec3<double> x4;
	sfmm::vec3<double> x5;
	constexpr int n = 1000;
	double poterr = 0, potnorm = 0;
	double ferr = 0, fnorm = 0;
	for (int i = 0; i < n; i++) {
		rand1();
		random_unit(dx0[0], dx0[1], dx0[2]);
		random_unit(dx1[0], dx1[1], dx1[2]);
		random_unit(dx2[0], dx2[1], dx2[2]);
		random_unit(dx3[0], dx3[1], dx3[2]);
		random_unit(dx4[0], dx4[1], dx4[2]);
		const auto alpha = 0.25;
		dx0 *= alpha * 0.125;
		dx1 *= alpha * 0.125;
		dx2 *= alpha;
		dx3 *= alpha * 0.125;
		dx4 *= alpha * 0.125;
		x0 = sfmm::vec3<double>( { 0, 0, 0 });
		x1 = x0 + dx0;
		x2 = x1 + dx1;
		x3 = x2 + dx2;
		x4 = x3 + dx3;
		x5 = x4 + dx4;

		sfmm::multipole<double, P> M, M1;
		sfmm::expansion<double, P> L1;
		sfmm::expansion<double, P> L;
		sfmm::force_type<double> f1;
		M.init();
		L.init();
		M1.init();
//		x0 *= 0.0;
		//	x1 *= 0.0;
		//x3 *= 0.0;
		//	x4 *= 0.0;
		sfmm::P2M(M, double(0.5), x0 - x1);
		sfmm::P2M(M1, double(0.5), { 0, 0, 0 });
		sfmm::M2M(M, x1 - x2);
		sfmm::M2L_ewald(L1, M1, { 0, 0, 0 });
		L += L1;
		sfmm::M2L_ewald(L1, M, x3 - x2);
		L += L1;
		sfmm::M2L(L1, M, x3 - x2, sfmmWithSingleRotationOptimization);
		L += L1;
		f1.init();
		sfmm::L2L(L, x3 - x4);
		sfmm::L2P(f1, L, x4 - x5);
		sfmm::force_type<double> f2;
		f2.init();
		//	f1.init();
		P2P(f2, 0.5, x5 - x0);
		P2P_ewald(f2, 0.5, { 0, 0, 0 });
		P2P_ewald(f2, 0.5, x5 - x0);
		poterr += sfmm::sqr(f1.potential - f2.potential);
		potnorm += sfmm::sqr(f1.potential);
		const double fabs1 = sfmm::sqr(f1.force[0]) + sfmm::sqr(f1.force[1]) + sfmm::sqr(f1.force[2]);
		const double fabs2 = sfmm::sqr(f2.force[0]) + sfmm::sqr(f2.force[1]) + sfmm::sqr(f2.force[2]);
		ferr += sfmm::sqr(fabs1 - fabs2);
		fnorm += sfmm::sqr(fabs2);
		//	printf( "%e\n", f1.force[0]*f2.force[0]+f1.force[1]*f2.force[1]+f1.force[2]*f2.force[2]);
	}
	printf("%i %e %e\n", P, poterr / potnorm, ferr / fnorm);

}

int main(int argc, char **argv) {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_OVERFLOW);
	feenableexcept(FE_INVALID);
	ewald();
	return 0;
	//return 0;
	//test2<5>();
	//test2<6>();
	//test2<7>();
	//test2<8>();
//	 return 0;
	//ewald();
	//return 0;
	run_tests<double, double, double, PMIN, sfmmWithoutOptimization | sfmmProfilingOn> run1;
	//run_tests<double, sfmm::simd_f32, sfmm::m2m_simd_f32, PMIN, sfmmWithSingleRotationOptimization | sfmmProfilingOn> run2;
//	run_tests<double, sfmm::simd_f32, sfmm::m2m_simd_f32, PMIN, sfmmWithDoubleRotationOptimization | sfmmProfilingOn> run3;
	run1();
//	run2();
//	run3();
//	sfmm::detail::operator_write_new_bestops_source();
	return 0;
}
