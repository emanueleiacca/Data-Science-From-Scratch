# **Extracting the Multiscale Backbone of Complex Weighted Networks — a story about keeping what matters**

> When every road looks important, the map becomes useless. The art is to keep **all the scales that matter** without inventing an arbitrary cutoff.

These notes explain the **disparity filter** (Serrano–Boguñá–Vespignani, PNAS 2009): a principled way to extract the **multiscale backbone** of a weighted network. We’ll walk it like a lab demo: intuition → null model → formula → algorithm → examples → pitfalls.

---

## **1) Problem: filtering without breaking the scales**

Weighted networks (air traffic, trade, metabolism, brain, social activity) are **heterogeneous**: both degrees and weights vary over many orders of magnitude. If you set a **global threshold** (keep edges with weight $\omega \ge \omega_c$), two bad things happen:

1. You **punish small nodes** (low total strength $s_i$): even their most relevant edges fall below $\omega_c$.
2. You inject an **artificial scale** and erase important fine structure.

Minimum spanning trees (MST) go to the other extreme: they **delete cycles** by design, flatten clustering, and over-simplify.

**Goal.** Keep edges that are **statistically significant for each node** — so that *small* nodes can still keep their **locally dominant** edges. That is the multiscale spirit.

---

## **2) Local view: normalize weights and look for dominance**

For node $i$ with strength $s_i = \sum_j \omega_{ij}$, normalize its incident weights:

$$p_{ij} = \frac{\omega_{ij}}{s_i}, \qquad \sum_{j \in N(i)} p_{ij} = 1.$$

Think of ${p_{ij}}$ as **how node $i$ splits its attention** across neighbors. If one edge carries a disproportionately large share, it should be a candidate for the backbone — **for that node**.

A classic summary of local heterogeneity is the **disparity** (a concentration index):

$$\Upsilon_i(k) = k \sum_{j\in N(i)} p_{ij}^2,$$

where $k$ is the degree of $i$. If all $p_{ij} = 1/k$ (perfect homogeneity), then $\Upsilon_i = 1$; if a single edge dominates, $\Upsilon_i = k$. Real networks typically show $\Upsilon_i(k)$ growing with $k$, signaling **peaked allocations**.

But $\Upsilon_i$ alone doesn’t tell us **which** edges are significant. For that we need a **null model** and **p‑values** at the edge level.

---

## **3) Null model: what “random” looks like locally**

Fix a node $i$ with degree $k$. Under the null hypothesis, $i$ splits its unit budget **uniformly at random** among its $k$ edges. Geometrically: throw $k-1$ uniform cut points on $[0,1]$; the $k$ subinterval lengths are the $p_{ij}$’s.

This is the **stick‑breaking** (Dirichlet) picture. For any single edge, the marginal distribution is

$$p \sim \mathrm{Beta}(1, k-1) \quad \Rightarrow \quad f(p) = (k-1)(1-p)^{k-2}, ; p\in(0,1).$$

The right‑tail probability (one‑sided p‑value) of observing $p_{ij}$ or larger under the null is

$$\alpha_{ij} = \Pr{P \ge p_{ij}} = (1 - p_{ij})^{,k-1}.$$

So an edge $(i,j)$ is **locally significant for node $i$** at level $\alpha$ if

$$\boxed{\alpha_{ij} = (1 - p_{ij})^{k-1} < \alpha ;; \Longleftrightarrow ;; p_{ij} > 1 - \alpha^{1/(k-1)}.}$$

Interpretation: as $k$ grows, even moderate $p_{ij}$ can be significant because they beat a harsher uniform baseline.

---

## **4) The disparity filter (algorithm)**

**Input.** Undirected weighted network with weights $\omega_{ij} > 0$.

**Step 1 — Normalize at nodes.** For each node $i$, compute $s_i$ and $p_{ij} = \omega_{ij}/s_i$.

**Step 2 — Test edges locally.** For each incident edge $(i,j)$, compute $\alpha_{ij} = (1 - p_{ij})^{k_i - 1}$.

**Step 3 — Keep if significant at either end (OR‑rule).** Preserve edge $(i,j)$ in the backbone if

$$\alpha_{ij} < \alpha \quad \textbf{or} \quad \alpha_{ji} < \alpha.$$

This **protects small nodes**: an edge can be crucial for a low‑strength node even if it is not for a hub. The OR‑rule preserves **percolation** and avoids shattering the graph.

**Parameter.** The significance level $\alpha$ tunes sparsity. Empirically, values in roughly $[10^{-3}, 0.5]$ give stable backbones; a practical “sweet spot” in many datasets is around $\alpha \in [0.01, 0.1]$.

---

## **5) Why this is multiscale (and why thresholds miss it)**

* **No global cutoff.** Significance depends on **relative share $p_{ij}$** and **local degree $k_i$**, not on absolute $\omega_{ij}$. Small yet **locally dominant** edges survive.
* **Scales preserved.** The heavy tail of $P(\omega)$ is **not chopped**; only statistically negligible crumbs vanish.
* **Cycles and clustering survive.** Unlike MSTs, the backbone can keep triangles and motifs, preserving local geometry.

---

## **6) Worked intuition on examples**

### U.S. airport network (2006)

* $\sim 1{,}078$ airports, $\sim 11{,}890$ links, weights = annual passengers per route.
* With $\alpha \approx 0.05$: keep **>80% of total weight**, **~66% of nodes**, and only **~17% of edges**.
* As $\alpha$ decreases, the **degree distribution** of the backbone settles into a clear heavy tail (exponent about $\gamma \approx 2.3$), while clustering stays roughly constant until very small $\alpha$.
* Geography pops out: **star‑like basins** around regional hubs (e.g., Alaska, Midwest). You see **both** the top traffic corridors **and** statistically relevant local feeders.

### Florida Bay food web (dry season)

* Directed, weighted carbon flows between species.
* With a stringent $\alpha \approx 8\times 10^{-4}$, the backbone still includes the **top 40% heaviest** links and **about half of the total weight**.
* Reveals **subsystems** and **keystone‑like species** with few links but **locally dominant** fluxes — nodes that a global threshold might wrongly discard.

---

## **7) Practical guide (how to use it well)**

* **Pick $\alpha$ by stability.** Sweep $\alpha$ and watch: fraction of nodes/weight kept, clustering, and the backbone’s degree tail. Choose the region where these stabilize but edges are sparse.
* **Undirected vs. directed.** For directed networks, normalize and test **incoming** and **outgoing** sets separately (apply the same Beta‑tail logic per side).
* **Visuals with meaning.** Plot the backbone over spatial coordinates (airports, mobility). The filter often exposes **latent geography** and **catchment areas**.
* **Compare to nulls.** Always benchmark against a **degree‑preserving weight reshuffle** to confirm that detected heterogeneity exceeds random fluctuations.

---

## **8) Limitations (and when not to use it)**

* Works best when the network shows **strong disorder** (heavy‑tailed $P(\omega)$, local correlations). If weights are close to i.i.d. and homogeneous, disparity filtering can look like a fancy threshold.
* The uniform Dirichlet null is **local** and **agnostic**. If your domain suggests a different baseline (e.g., gravity model for mobility, distance‑decay, mass constraints), adapt the null accordingly.

---

## **9) One‑page recap (math you actually need)**

* Normalize: $p_{ij} = \omega_{ij}/s_i$, $; s_i = \sum_j \omega_{ij}$.
* Null (per edge given degree $k_i$): $p \sim \mathrm{Beta}(1,k_i-1)$.
* p‑value: $\alpha_{ij} = (1 - p_{ij})^{k_i-1}$.
* Significance at level $\alpha$: keep $(i,j)$ if $\alpha_{ij} < \alpha$ or $\alpha_{ji} < \alpha$.
* Disparity index (optional diagnostic): $\Upsilon_i = k_i \sum_j p_{ij}^2$.

<>
This is how you **thin** a dense weighted network into a **meaningful backbone** that respects the system’s **multiscale nature** — keeping dominant edges for **each node**, not just the globally heavy ones.
