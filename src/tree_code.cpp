#include <stdio.h>
#include <sfmm.hpp>

#include <array>
#include <vector>
#include <fenv.h>
#include <future>
#include <atomic>

#define DEBUG

#define NDIM 3
#define BUCKET_SIZE 40
#define MIN_CLOUD 4
#define LEFT 0
#define RIGHT 1
#define NCHILD 2
#define MIN_THREAD 1024
#define TEST_SIZE (2*1024*1024)
//#define FLAGS (sfmmWithRandomOptimization | sfmmProfilingOn)

using rtype = double;

double rand1() {
	return (rand() + 0.5) / RAND_MAX;
}

void prefetch(void* ptr, int cnt) {
	for (int i = 0; i < cnt; i += 32) {
		__builtin_prefetch((char*) ptr + i);
	}
}

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
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
	sfmm::vec3<FT> center;
	tree* parent;
	T radius;
	T scale;

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
	static std::atomic<int> threads_avail;

	static std::vector<sfmm::vec3<FT>> parts;
	static std::vector<force_type<T>> forces;
	static std::vector<tree> forest;

	struct check_type {
		tree* ptr;
		bool opened;
	};

public:

	tree() {
	}

	static int partition_parts(std::pair<int, int> part_range, int xdim, T xmid) {
		int lo = part_range.first;
		int hi = part_range.second;
		while (lo < hi) {
			if (parts[lo][xdim].to_double() >= xmid) {
				while (lo != hi) {
					hi--;
					if (parts[hi][xdim].to_double() < xmid) {
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
				if (parts[i][dim].to_double() < begin || parts[i][dim].to_double() > end) {
					flag = true;
				}
			}
		}
		for (int i = part_range.first; i < part_range.second; i++) {
			if (flag) {
				for (int dim = 0; dim < NDIM; dim++) {
					T begin = (T) cell_begin[dim] / Ngrid;
					T end = (T) cell_end[dim] / Ngrid;
					printf("particle out of range! %e %e %e \n", begin, parts[i][dim].to_double(), end);
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
		scale = end[0] - begin[0];
		multipole.init(scale);
#ifdef DEBUG
		for (int i = part_range.first; i < part_range.second; i++) {
			for (int dim = 0; dim < NDIM; dim++) {
				if (parts[i][dim].to_double() < begin[dim] || parts[i][dim].to_double() > end[dim]) {
					printf("particle out of range! %e %e %e %i\n", begin[dim], parts[i][dim].to_double(), end[dim], depth);
				}
			}
		}
#endif
//		center = (begin + end) * 0.5;
		sfmm::vec3<double> center0;
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
			if (threads_avail-- >= 0) {
				auto fut = std::async([this, depth]() {
					size_t flops = children[LEFT].form_tree(this, depth + 1);
					threads_avail++;
					return flops;
				});
				flops += children[RIGHT].form_tree(this, depth + 1);
				flops += fut.get();
			} else {
				threads_avail++;
				flops += children[LEFT].form_tree(this, depth + 1);
				flops += children[RIGHT].form_tree(this, depth + 1);
			}
			const auto& cleft = children[LEFT];
			const auto& cright = children[RIGHT];
			for (int dim = 0; dim < NDIM; dim++) {
				center0[dim] = left.center[dim].to_double() * cleft.multipole(0, 0).real() + right.center[dim].to_double() * cright.multipole(0, 0).real();
			}
			const T total = cleft.multipole(0, 0).real() + cright.multipole(0, 0).real();
			center0 /= total;
			radius = 0.0;
			for (int ci = 0; ci < NCHILD; ci++) {
				for (int dim = 0; dim < NDIM; dim++) {
					dx[dim] = children[ci].center[dim].to_double() - center0[dim];
				}
				M = children[ci].multipole;
				radius = std::max(radius, sfmm::reduce_max(abs(dx) + children[ci].radius));
				flops += sfmm::M2M(M, dx, FLAGS);
				multipole += sfmm::reduce_sum(M);
			}
			center = center0;
		} else {
			if (nparts) {
				node_count++;
				center0 = T(0);
				for (int i = part_range.first; i < part_range.second; i++) {
					for (int dim = 0; dim < SFMM_NDIM; dim++) {
						center0[dim] += parts[i][dim].to_double();
					}
				}
				center0 /= nparts;
				radius = 0.0;
				multipole_type<V> M;
				M.init(scale);
				for (int i = part_range.first; i < part_range.second; i += V::size()) {
					sfmm::vec3<V> dx;
					const int cnt = std::min((int) V::size(), part_range.second - i);
					for (int j = 0; j < cnt; j++) {
						const auto& part = parts[i + j];
						for (int dim = 0; dim < SFMM_NDIM; dim++) {
							sfmm::load(dx[dim], T(part[dim].to_double() - center0[dim]), j);
						}
					}
					apply_padding(dx, cnt);
					radius = std::max(radius, sfmm::reduce_max(abs(dx)));
					flops += V::size() * sfmm::P2M(M, sfmm::create_mask<V>(cnt) * mass, dx);
				}
				multipole = sfmm::reduce_sum(M);
				center = center0;
				radius = std::max(hsoft, radius);
			} else {
				center0 = (begin + end) * 0.5;
				center = center0;
				radius = hsoft;
			}
		}
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

	size_t compute_cell_gravity(expansion_type<T> expansion, std::vector<check_type> dchecklist, std::vector<check_type> echecklist) {
		size_t flops = 0;
		expansion_type<V> L(scale);
		multipole_type<V> M(scale);
		force_type<V> F;
		sfmm::vec3<FV> xsnk;
		sfmm::vec3<FV> xsrc;
		if (parent) {
			expansion.rescale(scale);
			flops += sfmm::L2L(expansion, parent->center, center, FLAGS);
		}
		if (!children.size()) {
			for (int i = part_range.first; i < part_range.second; i++) {
				force_type<T> F;
				F.init();
				flops += sfmm::P2P(F, mass, sfmm::vec3<T>( { 0, 0, 0 }), FLAGS);
				forces[i].potential = -F.potential;
				for (int dim = 0; dim < NDIM; dim++) {
					forces[i].force[dim] = T(0);
				}
			}
		}
		static thread_local std::vector<tree*> M2Llist;
		static thread_local std::vector<tree*> P2Llist;
		static thread_local std::vector<tree*> M2Plist;
		static thread_local std::vector<tree*> P2Plist;
		static thread_local std::vector<check_type> nextlist;
		for (int ewald = 0; ewald <= 1; ewald++) {
			const T c0(1.0 / theta_max);
			auto& checklist = ewald ? echecklist : dchecklist;
			M2Llist.resize(0);
			P2Llist.resize(0);
			M2Plist.resize(0);
			P2Plist.resize(0);
			do {
				for (int i = 0; i < checklist.size(); i += V::size()) {
					sfmm::vec3<V> dx;
					V rsum;
					V rm2l;
					V rp2l;
					V rm2p;
					const int end = std::min(V::size(), checklist.size() - i);
					for (int j = 0; j < end; j++) {
						const auto& check = checklist[i + j];
						sfmm::load(dx, sfmm::distance(center, check.ptr->center), j);
						sfmm::load(rsum, radius + check.ptr->radius, j);
						sfmm::load(rm2l, c0 * radius + c0 * check.ptr->radius, j);
						sfmm::load(rm2p, radius + c0 * check.ptr->radius, j);
						sfmm::load(rp2l, c0 * radius + check.ptr->radius, j);
					}
					sfmm::apply_padding(dx, end);
					sfmm::apply_padding(rsum, end);
					sfmm::apply_padding(rm2l, end);
					sfmm::apply_padding(rp2l, end);
					sfmm::apply_padding(rm2p, end);
					V Cm2l, Cl2p, Cm2p;
					auto D = abs(dx);
					if (ewald) {
						D = max(D, V(0.5) - rsum);
					}
					Cm2l = D - rm2l;
					Cl2p = D - rp2l;
					Cm2p = D - rm2p;
					for (int j = 0; j < end; j++) {
						const auto check = checklist[j + i];
						bool used = false;
						if (Cm2l[j] > 0.0) {
							M2Llist.push_back(check.ptr);
							used = true;
						} else if (!children.size() && check.ptr->children.size()) {
							if (Cm2p[j] > T(0)) {
								M2Plist.push_back(check.ptr);
								used = true;
							}
						} else if (children.size() && !check.ptr->children.size()) {
							if (Cl2p[j] > T(0)) {
								P2Llist.push_back(check.ptr);
								used = true;
							}
						} else if (!children.size() && !check.ptr->children.size()) {
							if (Cl2p[j] > T(0) && Cl2p[j] > Cm2p[j]) {
								P2Llist.push_back(check.ptr);
							} else if (Cm2p[j] > T(0) && Cl2p[j] < Cm2p[j]) {
								M2Plist.push_back(check.ptr);
							} else {
								P2Plist.push_back(check.ptr);
							}
							used = true;
						}
						if (!used) {
							if (check.ptr->children.size()) {
								for (int ci = 0; ci < NCHILD; ci++) {
									check_type chk;
									chk.ptr = &(check.ptr->children[ci]);
									nextlist.push_back(chk);
								}
							} else {
								nextlist.push_back(check);
							}
						}
					}
				}
				checklist = std::move(nextlist);
				nextlist.resize(0);
			} while (!children.size() && checklist.size());
			xsnk = center;
			int inters;
			inters = 0;
			L.init(scale);
			for (int i = 0; i < M2Llist.size(); i += V::size()) {
				const int cnt = std::min(V::size(), M2Llist.size() - i);
				inters += cnt;
				for (int j = 0; j < cnt; j++) {
					const auto& src = M2Llist[i + j];
					load(xsrc, src->center, j);
					load(M, src->multipole, j);
				}
				apply_padding(xsrc, cnt);
				apply_mask(M, cnt);
				if (ewald) {
					flops += cnt * sfmm::M2L_ewald(L, M, xsrc, xsnk, FLAGS);
				} else {
					flops += cnt * sfmm::M2L(L, M, xsrc, xsnk, FLAGS);
				}
			}
			expansion += reduce_sum(L);
			(ewald ? m2l_ewald : m2l) += inters;
			inters = 0;
			L.init(scale);
			for (int i = 0; i < P2Llist.size(); i++) {
				for (int j = P2Llist[i]->part_range.first; j < P2Llist[i]->part_range.second; j += V::size()) {
					const int cnt = std::min((int) V::size(), P2Llist[i]->part_range.second - j);
					inters += cnt;
					for (int k = 0; k < cnt; k++) {
						load(xsrc, parts[j + k], k);
					}
					apply_padding(xsrc, cnt);
					if (ewald) {
						flops += cnt * sfmm::P2L_ewald(L, sfmm::create_mask<V>(cnt) * mass, xsrc, xsnk, FLAGS);
					} else {
						flops += cnt * sfmm::P2L(L, sfmm::create_mask<V>(cnt) * mass, xsrc, xsnk, FLAGS);
					}
				}
			}
			expansion += (reduce_sum(L));
			(ewald ? p2l_ewald : p2l) += inters;
			if (!children.size()) {
				inters = 0;
				for (int i = 0; i < M2Plist.size(); i += V::size()) {
					const int cnt = std::min(V::size(), M2Plist.size() - i);
					for (int j = 0; j < cnt; j++) {
						const auto& src = M2Plist[i + j];
						load(xsrc, src->center, j);
						load(M, src->multipole, j);
					}
					apply_padding(xsrc, cnt);
					apply_mask(M, cnt);
					for (int k = part_range.first; k < part_range.second; k++) {
						auto& part = parts[k];
						for (int dim = 0; dim < SFMM_NDIM; dim++) {
							xsnk[dim] = FV(part[dim]);
						}
						inters += cnt;
						F.init();
						if (ewald) {
							flops += cnt * sfmm::M2P_ewald(F, M, xsrc, xsnk, FLAGS);
						} else {
							flops += cnt * sfmm::M2P(F, M, xsrc, xsnk, FLAGS);
						}
						forces[k] += reduce_sum(F);
					}
				}
				(ewald ? m2p_ewald : m2p) += inters;
				inters = 0;
				const auto nparts = part_range.second - part_range.first;
				std::array<force_type<V>, BUCKET_SIZE> Forces;
				std::array<sfmm::vec3<FV>, BUCKET_SIZE> xsnk;
				for (int l = part_range.first; l < part_range.second; l++) {
					for (int dim = 0; dim < SFMM_NDIM; dim++) {
						xsnk[l - part_range.first][dim] = FV(parts[l][dim]);
					}
				}
				for (int i = 0; i < nparts; i++) {
					Forces[i].init();
				}
				for (int i = 0; i < P2Plist.size(); i++) {
					for (int j = P2Plist[i]->part_range.first; j < P2Plist[i]->part_range.second; j += V::size()) {
						const int cnt = std::min((int) V::size(), P2Plist[i]->part_range.second - j);
						inters += cnt;
						for (int k = 0; k < cnt; k++) {
							load(xsrc, parts[j + k], k);
						}
						sfmm::apply_padding(xsrc, cnt);
						if (ewald) {
							for (int l = 0; l < nparts; l++) {
								flops += cnt * sfmm::P2P_ewald(Forces[l], sfmm::create_mask<V>(cnt) * mass, xsrc, xsnk[l], FLAGS);
							}
						} else {
							for (int l = 0; l < nparts; l++) {
								flops += cnt * sfmm::P2P(Forces[l], sfmm::create_mask<V>(cnt) * mass, xsrc, xsnk[l], FLAGS);
							}
						}
					}
				}
				for (int i = 0; i < nparts; i++) {
					forces[part_range.first + i] += reduce_sum(Forces[i]);
				}
				(ewald ? p2p_ewald : p2p) += inters;
				inters = 0;
			}
		}
		if (children.size()) {
			if (dchecklist.size() || echecklist.size()) {
				if (threads_avail-- >= 0) {
					auto fut = std::async([this, expansion](std::vector<check_type> dchecklist, std::vector<check_type> echecklist ) {
						size_t flops = children[LEFT].compute_cell_gravity(expansion, std::move(dchecklist), std::move(echecklist));
						threads_avail++;
						return flops;
					}, dchecklist, echecklist);
					flops += children[RIGHT].compute_cell_gravity(expansion, std::move(dchecklist), std::move(echecklist));
					flops += fut.get();
				} else {
					threads_avail++;
					flops += children[LEFT].compute_cell_gravity(expansion, dchecklist, echecklist);
					flops += children[RIGHT].compute_cell_gravity(expansion, std::move(dchecklist), std::move(echecklist));
				}
			}
		} else {
			load(L, (expansion));
			xsnk = center;
			for (int i = part_range.first; i < part_range.second; i += V::size()) {
				force_type<V> F;
				F.init();
				const int cnt = std::min((int) V::size(), part_range.second - i);
				for (int j = 0; j < cnt; j++) {
					load(xsrc, parts[j + i], j);
				}
				apply_padding(xsrc, cnt);
				flops += cnt * sfmm::L2P(F, L, xsnk, xsrc, FLAGS);
				for (int j = 0; j < cnt; j++) {
					accumulate(forces[i + j], F, j);
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
		force_type<T> f;
		f.init();
		P2P(f, mass, { 0, 0, 0 });
		double pot0 = f.potential;
		for (int i = 0; i < parts.size(); i++) {
			if (rand1() > sample_odds) {
				continue;
			}
			const auto& snk_part = parts[i];
			force_type<T> fa;
			fa.init();
			fa.potential = -pot0;
			const int nthreads = 2 * std::thread::hardware_concurrency();
			std::vector<std::future<void>> futs;
			std::mutex mutex;
			for (int proc = 0; proc < nthreads; proc++) {
				const int b = (size_t) proc * parts.size() / nthreads;
				const int e = (size_t) (proc + 1) * parts.size() / nthreads;
				futs.push_back(std::async([i,b,e,&fa,&mutex,snk_part]() {
					sfmm::vec3<FV> xsrc, xsnk;
					force_type<V> fe;
					force_type<V> f;
					xsnk = sfmm::vec3<FV>(parts[i]);
					for (int j = b; j < e; j += sfmm::simd_size<FV>()) {
						int cnt = std::min((int) sfmm::simd_size<FV>(), e - j);
						for( int k = j; k < j + cnt; k++) {
							load(xsrc, parts[k], k - j);
						}
						f.init();
						apply_padding(xsrc, cnt);
						P2P(f, sfmm::create_mask<V>(cnt) * mass, xsrc, xsnk);
						P2P_ewald(f, sfmm::create_mask<V>(cnt) * mass, xsrc, xsnk);
						std::lock_guard<std::mutex> lock(mutex);
						fa += reduce_sum(f);
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
			printf("%e %e %e\n", (famag - fnmag) / famag, fa.potential, forces[i].potential);
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
		int N = 100;
		sfmm::vec3<double> center[N];
		for (int j = 0; j < N; j++) {
			for (int dim = 0; dim < NDIM; dim++) {
				center[j][dim] = rand1();
			}
		}
		for (int i = 0; i < TEST_SIZE; i++) {
			sfmm::vec3<FT> x;
			int j = rand() % N;
			for (int dim = 0; dim < NDIM; dim++) {
				x[dim] = fmod((2.0 * rand1() - 1.0) * 0.01 + center[j][dim], 1.0);
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

}
;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<sfmm::vec3<FT>> tree<T, V, FT, FV, ORDER, FLAGS>::parts;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<sfmm::force_type<T>> tree<T, V, FT, FV, ORDER, FLAGS>::forces;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
const T tree<T, V, FT, FV, ORDER, FLAGS>::theta_max = 0.55;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<int> tree<T, V, FT, FV, ORDER, FLAGS>::threads_avail(2 * std::thread::hardware_concurrency() - 1);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
const T tree<T, V, FT, FV, ORDER, FLAGS>::mass = 1.0 / TEST_SIZE;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
const T tree<T, V, FT, FV, ORDER, FLAGS>::hsoft = 1e-6;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
const int tree<T, V, FT, FV, ORDER, FLAGS>::Ngrid = 1;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::vector<tree<T, V, FT, FV, ORDER, FLAGS>> tree<T, V, FT, FV, ORDER, FLAGS>::forest;

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2p(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2p(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2l(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2l(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2p_ewald(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2p_ewald(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::p2l_ewald(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::m2l_ewald(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
std::atomic<long long> tree<T, V, FT, FV, ORDER, FLAGS>::node_count(0);

template<class T, class V, class FT, class FV, int ORDER, int FLAGS>
struct run_tests {
	void operator()() const {
		sfmm::timer tm, ftm;
		using tree_type = tree<T, V, FT, FV, ORDER, FLAGS>;
		double tree_time, force_time;
		srand(1);
		tree_type::initialize();
		tree_time = tm.read();
		fflush(stdout);
		tree_type::sort_grid();
		tm.reset();
		tm.start();
		size_t flops = tree_type::form_trees();
		tm.stop();
		tree_time = tm.read();
		tm.reset();
		tm.start();
		flops += tree_type::compute_gravity();
		tm.stop();
		force_time = tm.read();
		const auto error = tree_type::compare_analytic(25.0 / TEST_SIZE);
		printf("%i %e %e %e %e %e Gflops\n", ORDER, tree_time, force_time, error.first, error.second, flops / force_time / (1024.0 * 1024.0 * 1024.0));
		tree_type::show_counters();
		tree_type::reset_counters();
		run_tests<T, V, FT, FV, ORDER + 1, FLAGS> run;
		run();
	}
};

template<class T, class V, class FT, class FV, int FLAGS>
struct run_tests<T, V, FT, FV, PMAX + 1, FLAGS> {
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
	for (double a = 0.0; a < 1.0; a += 0.1) {
		for (double b = 0.0; b < 1.0; b += 0.1) {
			sfmm::fixed32 A(a);
			sfmm::fixed32 B(b);
			sfmm::simd_fixed32 aa, bb;
			aa[0] = A;
			bb[0] = B;
			auto d = sfmm::distance(aa, bb)[0];
		}
	}

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
	run_tests<double, simd::simd_f64, sfmm::fixed64, sfmm::simd_fixed64, PMIN, sfmmWithDoubleRotationOptimization> run4;
	run_tests<float, simd::simd_f32, sfmm::fixed32, sfmm::simd_fixed32, PMIN, sfmmWithBestOptimization> run5;
	run5();
	run4();
	return 0;
}
