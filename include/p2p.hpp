
template<class W>
static int P2P(force_type<W>& f, W m, vec3<W> dx) {
	const static double hsoft = 0.01;
	static const W h2(hsoft * hsoft);
	static const W hinv(W(1) / hsoft);
	static const W hinv3(sqr(hinv) * hinv);
	const W r2 = fma(dx[0], dx[0], fma(dx[1], dx[1], dx[2] * dx[2]));
	const W wn(m * (r2 < h2));
	const W wf(m * (r2 >= h2));
	vec3<W> fn, ff;
	W rzero(r2 < W(1e-30));
	W pn, pf;
	const W rinv = rsqrt(r2 + rzero);
	W rinv3 = sqr(rinv) * rinv;
	pf = rinv;
	ff = dx * rinv3;
	pn = (W(1.5) * hinv - W(0.5) * r2 * hinv3);
	fn = dx * hinv3;
	f.potential = -fma(pn, wn, pf * wf);
	f.force[0] = -fma(fn[0], wn, ff[0] * wf);
	f.force[1] = -fma(fn[1], wn, ff[1] * wf);
	f.force[2] = -fma(fn[2], wn, ff[2] * wf);
	return 39;
}


