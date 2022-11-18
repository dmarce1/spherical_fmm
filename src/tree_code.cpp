#include <stdio.h>
#include <sfmm.hpp>

#include <array>
#include <vector>
#include <fenv.h>
#include <future>
#include <atomic>

#define DEBUG

#define NDIM 3
#define BUCKET_SIZE 8
#define MIN_CLOUD 4
#define LEFT 0
#define RIGHT 1
#define NCHILD 2
#define MIN_THREAD 1024
#define TEST_SIZE (1024*1024)
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
	static std::atomic<long long> p2p;
	static std::atomic<long long> m2p;
	static std::atomic<long long> p2l;
	static std::atomic<long long> m2l;
	static std::atomic<long long> p2p_ewald;
	static std::atomic<long long> m2p_ewald;
	static std::atomic<long long> p2l_ewald;
	static std::atomic<long long> m2l_ewald;
	static std::atomic<long long> node_count;

	static std::vector<sfmm::vec3<T>> parts;
	static std::vector<force_type<T>> forces;
	static std::vector<tree> forest;

	struct check_type {
		tree* ptr;
		bool opened;
	};

	void list_iterate(std::vector<check_type>& checklist, std::vector<tree*>& Plist, std::vector<tree*>& Clist, bool leaf, bool ewald) {
		static thread_local std::vector<check_type> nextlist;
		nextlist.resize(0);
		auto c0 = T(1.0 / theta_max);
		for (int i = 0; i < checklist.size(); i += sfmm::simd_size<V>()) {
			sfmm::vec3<V> dx;
			V rsum;
			V rsum0;
			V rsum1;
			V rsum2;
			const int end = std::min(sfmm::simd_size<V>(), checklist.size() - i);
			for (int j = 0; j < end; j++) {
				const auto& check = checklist[i + j];
				sfmm::load(dx, sfmm::distance(center, check.ptr->center), j);
				sfmm::load(rsum, radius + check.ptr->radius, j);
				sfmm::load(rsum0, c0 * radius + c0 * check.ptr->radius, j);
				sfmm::load(rsum1, radius + c0 * check.ptr->radius, j);
				sfmm::load(rsum2, c0 * radius + check.ptr->radius, j);
			}
			sfmm::apply_padding(dx, end);
			sfmm::apply_padding(rsum, end);
			sfmm::apply_padding(rsum0, end);
			sfmm::apply_padding(rsum1, end);
			sfmm::apply_padding(rsum2, end);
			V far0, far1, far2;
			if (ewald) {
				const auto max = sfmm::max(abs(dx), V(0.5) - rsum);
				far0 = (rsum0 < max);
				far1 = (rsum1 < max);
				far2 = (rsum2 < max);
			} else {
				far0 = sfmm::sqr(rsum0) < sfmm::sqr(dx);
				far1 = sfmm::sqr(rsum1) < sfmm::sqr(dx);
				far2 = sfmm::sqr(rsum2) < sfmm::sqr(dx);
			}
			for (int j = 0; j < end; j++) {
				auto& check = checklist[i + j];
				bool used = false;
				if (leaf) {
					if (check.opened) {
						Plist.push_back(check.ptr);
						used = true;
					} else {
						if (sfmm::access(far1, j)) {
							Clist.push_back(check.ptr);
							used = true;
						}
					}
				} else {
					if (check.opened) {
						if (sfmm::access(far2, j)) {
							Plist.push_back(check.ptr);
							used = true;
						}
					} else {
						if (sfmm::access(far0, j)) {
							Clist.push_back(check.ptr);
							used = true;
						}
					}
				}
				if (!used) {
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
			if (parts[lo][xdim] >= xmid) {
				while (lo != hi) {
					hi--;
					if (parts[hi][xdim] < xmid) {
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
				if (parts[i][dim] < begin || parts[i][dim] > end) {
					flag = true;
				}
			}
		}
		for (int i = part_range.first; i < part_range.second; i++) {
			if (flag) {
				for (int dim = 0; dim < NDIM; dim++) {
					T begin = (T) cell_begin[dim] / Ngrid;
					T end = (T) cell_end[dim] / Ngrid;
					printf("particle out of range! %e %e %e \n", begin, parts[i][dim], end);
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
		node_count++;
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
				if (parts[i][dim] < begin[dim] || parts[i][dim] > end[dim]) {
					printf("particle out of range! %e %e %e %i\n", begin[dim], parts[i][dim], end[dim], depth);
				}
			}
		}
#endif
//		center = (begin + end) * 0.5;
		if (nparts > BUCKET_SIZE) {
			sfmm::vec3<T> dx;
			multipole_type<T> M;
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
			for (int ci = 0; ci < NCHILD; ci ++) {
				dx = children[ci].center - center;
				M = children[ci].multipole;
				radius = std::max(radius, sfmm::reduce_max(abs(dx) + children[ci].radius));
				flops += sfmm::M2M(M, dx, FLAGS);
				multipole += sfmm::reduce_sum(M);
			}
		} else {
			if (nparts) {
				center = T(0);
				for (int i = part_range.first; i < part_range.second; i++) {
					center += parts[i];
				}
				center /= nparts;
				radius = 0.0;
				for (int i = part_range.first; i < part_range.second; i += sfmm::simd_size<V>()) {
					sfmm::vec3<V> dx;
					const int end = std::min((int) sfmm::simd_size<V>(), part_range.second - i);
					for (int j = 0; j < end; j++) {
						const auto& part = parts[i + j];
						load(dx, part - center, j);
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

	size_t compute_cell_gravity(expansion_type<float> expansion, std::vector<check_type> checklist, std::vector<check_type> echecklist) {
		size_t flops = 0;
		static thread_local std::vector<tree*> Clist;
		static thread_local std::vector<tree*> Plist;
		Clist.resize(0);
		Plist.resize(0);
		expansion_type<V> L;
		multipole_type<V> M;
		force_type<V> F;
		sfmm::vec3<V> x0;
		sfmm::vec3<V> x1;

		if (parent) {
			expansion.rescale(radius);
			flops += sfmm::L2L(expansion, parent->center, center, FLAGS);
		}
		list_iterate(echecklist, Plist, Clist, false, true);
		x1 = sfmm::vec3<V>(center);
		for (int i = 0; i < Clist.size(); i += sfmm::simd_size<V>()) {
			L.init();
			M.init();
			const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
			for (int j = 0; j < end; j++) {
				const auto& src = Clist[i + j];
				load(x0, src->center, j);
				load(M, src->multipole, j);
			}
			m2l_ewald += end;
			apply_padding(x0, end);
			apply_padding(M, end);
			flops += sfmm::simd_size<V>() * sfmm::M2L_ewald(L, M, x0, x1, FLAGS);
			apply_mask(L, end);
			expansion += (reduce_sum(L));
		}
		for (int i = 0; i < Plist.size(); i++) {
			for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
				const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
				L.init();
				for (int k = 0; k < end; k++) {
					load(x0, parts[j + k], k);
				}
				p2l_ewald += end;
				apply_padding(x0, end);
				flops += sfmm::simd_size<V>() * sfmm::P2L_ewald(L, sfmm::create_mask<V>(end) * mass, x0, x1, FLAGS);
				expansion += (reduce_sum(L));
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
				load(x0, src->center, j);
				load(M, src->multipole, j);
			}
			m2l += end;
			apply_padding(x0, end);
			apply_padding(M, end);
			flops += sfmm::simd_size<V>() * sfmm::M2L(L, M, x0, x1, FLAGS);
			apply_mask(L, end);
			expansion += (reduce_sum(L));
		}
		for (int i = 0; i < Plist.size(); i++) {
			for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
				const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
				L.init();
				for (int k = 0; k < end; k++) {
					load(x0, parts[j + k], k);
				}
				p2l += end;
				apply_padding(x0, end);
				flops += sfmm::simd_size<V>() * sfmm::P2L(L, sfmm::create_mask<V>(end) * mass, x0, x1, FLAGS);
				expansion += (reduce_sum(L));
			}
		}
		if (children.size()) {
			if (checklist.size()) {
				flops += children[LEFT].compute_cell_gravity(expansion, checklist, echecklist);
				flops += children[RIGHT].compute_cell_gravity(expansion, std::move(checklist), std::move(echecklist));
			}
		} else {
			load(L, sfmm::expansion<float,ORDER>(expansion));
			for (int i = part_range.first; i < part_range.second; i += sfmm::simd_size<V>()) {
				force_type<V> F;
				F.init();
				const int end = std::min((int) sfmm::simd_size<V>(), part_range.second - i);
				for (int j = 0; j < end; j++) {
					load(x0, parts[i + j], j);
				}
				flops += sfmm::simd_size<V>() * sfmm::L2P(F, L, x1, x0, FLAGS);
				for (int j = 0; j < end; j++) {
					store(forces[i + j], F, j);
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
					x1 = sfmm::vec3<sfmm::simd_f32>(part);
					F.init();
					const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
					for (int j = 0; j < end; j++) {
						const auto& src = Clist[i + j];
						load(x0, src->center, j);
						load(M, src->multipole, j);
					}
					m2p_ewald += end;
					apply_padding(x0, end);
					apply_padding(M, end);
					flops += sfmm::simd_size<V>() * sfmm::M2P_ewald(F, M, x0, x1, FLAGS);
					for (int j = 0; j < end; j++) {
						accumulate(forces[k], F, j);
					}
				}
			}
			for (int i = 0; i < Plist.size(); i++) {
				for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
					const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
					for (int l = part_range.first; l < part_range.second; l++) {
						auto& part = parts[l];
						x1 = sfmm::vec3<sfmm::simd_f32>(part);
						for (int k = 0; k < end; k++) {
							load(x0, parts[j + k], k);
						}
						p2p_ewald += end;
						apply_padding(x0, end);
						F.init();
						flops += sfmm::simd_size<V>() * P2P_ewald(F, sfmm::create_mask<V>(end) * mass, x0, x1, FLAGS);
						forces[l] += sfmm::reduce_sum(F);
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
					x1 = sfmm::vec3<sfmm::simd_f32>(part);
					F.init();
					const int end = std::min(sfmm::simd_size<V>(), Clist.size() - i);
					for (int j = 0; j < end; j++) {
						const auto& src = Clist[i + j];
						load(x0, src->center, j);
						load(M, src->multipole, j);
					}
					m2p += end;
					apply_padding(x0, end);
					apply_padding(M, end);
					flops += sfmm::simd_size<V>() * sfmm::M2P(F, M, x0, x1, FLAGS);
					for (int j = 0; j < end; j++) {
						accumulate(forces[k], F, j);
					}
				}
			}
			for (int i = 0; i < Plist.size(); i++) {
				for (int j = Plist[i]->part_range.first; j < Plist[i]->part_range.second; j += sfmm::simd_size<V>()) {
					const int end = std::min((int) sfmm::simd_size<V>(), Plist[i]->part_range.second - j);
					for (int l = part_range.first; l < part_range.second; l++) {
						auto& part = parts[l];
						x1 = sfmm::vec3<sfmm::simd_f32>(part);
						for (int k = 0; k < end; k++) {
							load(x0, parts[j + k], k);
						}
						p2p += end;
						apply_padding(x0, end);
						F.init();
						flops += sfmm::simd_size<V>() * P2P(F, sfmm::create_mask<V>(end) * mass, x0, x1, FLAGS);
						forces[l] += sfmm::reduce_sum(F);
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
				futs.push_back(std::async([i,b,e,&fa,&mutex,snk_part]() {
					for (int j = b; j < e; j++) {
						force_type<double> fe2;
						force_type<double> fd;
						const auto& src_part = parts[j];
						sfmm::vec3<double> dx;
						for (int dim = 0; dim < NDIM; dim++) {
							dx[dim] = sfmm::distance(snk_part[dim], src_part[dim]);
						}
						fd.init();
						fe2.init();
						P2P_ewald(fe2, mass, dx);
						P2P(fd, (double) mass, dx);
						std::lock_guard<std::mutex> lock(mutex);
						fa.force += fd.force + fe2.force;
						fa.potential += fd.potential + fe2.potential;
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
				fnmag += sfmm::sqr(forces[i].force[0]) + sfmm::sqr(forces[i].force[1]) + sfmm::sqr(forces[i].force[2]);
			}
			famag = sqrt(famag);
			fnmag = sqrt(fnmag);
			//printf( "%e\n", (famag - fnmag) / famag);
			std::lock_guard<std::mutex> lock(mutex);
			perr += sfmm::sqr(forces[i].potential - fa.potential);
			pnorm += sfmm::sqr(fa.potential);
			norm += sfmm::sqr(famag);
			err += sfmm::sqr(famag - fnmag);
		}
		err = sqrt(err / norm);
		perr = sqrt(perr / pnorm);
		return std::make_pair(perr, err);
	}

	static void initialize() {
		forest.resize(0);
		parts.resize(0);
		forces.resize(0);
		for (int i = 0; i < TEST_SIZE; i++) {
			sfmm::vec3<T> x;
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = rand1();
			}
			parts.push_back(x);
		}
		forces.resize(parts.size());
	}

	static void show_counters() {
		printf("Node count = %lli\n", (long long) node_count);
		printf("P2P = %lli (%e)\n", (long long) p2p, (long long) p2p / (double) node_count);
		printf("P2L = %lli (%e)\n", (long long) p2l, (long long) p2l / (double) node_count);
		printf("M2P = %lli (%e)\n", (long long) m2p, (long long) m2p / (double) node_count);
		printf("M2L = %lli (%e)\n", (long long) m2l, (long long) m2l / (double) node_count);
		printf("P2P_ewald = %lli (%e)\n", (long long) p2p_ewald, (long long) p2p_ewald / (double) node_count);
		printf("P2L_ewald = %lli (%e)\n", (long long) p2l_ewald, (long long) p2l_ewald / (double) node_count);
		printf("M2P_ewald = %lli (%e)\n", (long long) m2p_ewald, (long long) m2p_ewald / (double) node_count);
		printf("M2L_ewald = %lli (%e)\n", (long long) m2l_ewald, (long long) m2l_ewald / (double) node_count);
	}

	static void reset_counters() {
		node_count = p2p = m2p = p2l = m2l = p2p_ewald = m2p_ewald = p2l_ewald = p2p_ewald = 0;
	}

};

template<class T, class V, class M, int ORDER, int FLAGS>
std::vector<sfmm::vec3<T>> tree<T, V, M, ORDER, FLAGS>::parts;

template<class T, class V, class M, int ORDER, int FLAGS>
std::vector<sfmm::force_type<T>> tree<T, V, M, ORDER, FLAGS>::forces;

template<class T, class V, class M, int ORDER, int FLAGS>
const T tree<T, V, M, ORDER, FLAGS>::theta_max = 0.55;

template<class T, class V, class M, int ORDER, int FLAGS>
const T tree<T, V, M, ORDER, FLAGS>::mass = 1.0 / TEST_SIZE;

template<class T, class V, class M, int ORDER, int FLAGS>
const T tree<T, V, M, ORDER, FLAGS>::hsoft = 0.01;

template<class T, class V, class M, int ORDER, int FLAGS>
const int tree<T, V, M, ORDER, FLAGS>::Ngrid = 4;

template<class T, class V, class M, int ORDER, int FLAGS>
std::vector<tree<T, V, M, ORDER, FLAGS>> tree<T, V, M, ORDER, FLAGS>::forest;

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::p2p(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::m2p(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::p2l(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::m2l(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::p2p_ewald(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::m2p_ewald(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::p2l_ewald(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::m2l_ewald(0);

template<class T, class V, class M, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, M, ORDER, FLAGS>::node_count(0);

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
		const auto error = tree_type::compare_analytic(16.0 / TEST_SIZE);
		printf("%i %e %e %e %e %e %e Gflops\n", ORDER, tree_time, force_time, tm.read(), error.first, error.second,
				flops / ftm.read() / (1024.0 * 1024.0 * 1024.0));
	//	tree_type::show_counters();
		tree_type::reset_counters();
		run_tests<T, V, M, ORDER + 1, FLAGS> run;
	//	run();
	}
};

template<class T, class V, class M, int FLAGS>
struct run_tests<T, V, M, PMAX + 1, FLAGS> {
	void operator()() const {
	}
};

void ewald() {
	/*	constexpr int N = 8;
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
	 }*/
}

void random_unit(double& x, double& y, double& z) {
	const double theta = acos(2 * rand1() - 1.0);
	const double phi = rand1() * 2.0 * M_PI;
	x = cos(phi) * sin(theta);
	y = sin(phi) * sin(theta);
	z = cos(theta);
}

template<int P>
void test2() {
	/*sfmm::vec3<double> dx0;
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
*/
}

int main(int argc, char **argv) {
	feenableexcept(FE_DIVBYZERO);
	feenableexcept(FE_OVERFLOW);
	feenableexcept(FE_INVALID);
	sfmm::vec3<double> t;
	sfmm::vec3<float> t1(t);
//	ewald();
//	return 0;
	//return 0;
	//test2<5>();
	//test2<6>();
	//test2<7>();
	//test2<8>();
//	 return 0;
	//ewald();
	//return 0;
//	run_tests<double, sfmm::simd_f64, sfmm::m2m_simd_f64, 5, sfmmWithBestOptimization> run1;
//	run_tests<double, sfmm::simd_f64, sfmm::m2m_simd_f64, 6, sfmmWithBestOptimization> run2;
//	run_tests<double, sfmm::simd_f64, sfmm::m2m_simd_f64, 7, sfmmWithBestOptimization> run3;
//	run_tests<double, sfmm::simd_f64, sfmm::m2m_simd_f64, 8, sfmmWithBestOptimization> run4;
	run_tests<float, sfmm::simd_f32, sfmm::m2m_simd_f32, 5, sfmmWithBestOptimization> run5;
//	run_tests<float, sfmm::simd_f32, sfmm::m2m_simd_f32, 6, sfmmWithBestOptimization> run6;
//	run_tests<float, sfmm::simd_f32, sfmm::m2m_simd_f32, 7, sfmmWithBestOptimization> run7;
//	run_tests<float, sfmm::simd_f32, sfmm::m2m_simd_f32, 8, sfmmWithBestOptimization> run8;
	//run_tests<double, sfmm::simd_f32, sfmm::m2m_simd_f32, PMIN, sfmmWithSingleRotationOptimization | sfmmProfilingOn> run2;
//	run_tests<double, sfmm::simd_f32, sfmm::m2m_simd_f32, PMIN, sfmmWithDoubleRotationOptimization | sfmmProfilingOn> run3;
	run5();
//	run6();
//	run7();
//	run8();
//	run1();
//	run2();
//	run3();
//	run2();
//	run3();
//	sfmm::detail::operator_write_new_bestops_source();
	return 0;
}
