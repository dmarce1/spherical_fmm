
template<class W>
static int P2P(force_type<W>& f, W m, vec3<W> dx, int flags = sfmmDefaultFlags) {
	const static double hsoft = 0.01;
	static const W h2(hsoft * hsoft);
	static const W hinv(W(1) / hsoft);
	static const W hinv3(sqr(hinv) * hinv);
	const W r2 = fma(dx[0], dx[0], fma(dx[1], dx[1], dx[2] * dx[2])); // 5
	const W wn(r2 < h2);                                              // 1
	const W wf(r2 >= h2);                                             // 1
	vec3<W> fn, ff;
	W rzero(r2 < W(1e-30));                                           // 1
	W pn, pf;
	const W rinv = rsqrt(r2 + rzero);                                 // 11
	W rinv3 = sqr(rinv) * rinv;                                       // 3
	pf = rinv;
	ff = dx * rinv3;                                                  // 1
	pn = (W(1.5) * hinv - W(0.5) * r2 * hinv3);                       // 4
	fn = dx * hinv3;                                                  // 3
	m = -m;                                                           // 1
	f.potential = m * fma(pn, wn, pf * wf);                           // 4
	f.force[0] = m * fma(fn[0], wn, ff[0] * wf);                      // 4
	f.force[1] = m * fma(fn[1], wn, ff[1] * wf);                      // 4
	f.force[2] = m * fma(fn[2], wn, ff[2] * wf);                      // 4
	return 47;
}


