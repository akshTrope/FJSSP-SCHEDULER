# FJSP Solver: CP-SAT + WOA + LNS Hybrid

This project solves the **Flexible Job Shop Scheduling Problem (FJSP)**: given a set of jobs (each made of several sequential operations) and a set of machines (where each operation can run on one of several eligible machines, often at different speeds), find an assignment of operations to machines and an ordering on each machine that **minimizes the makespan** — the time at which the last operation finishes.

FJSP is NP-hard. No algorithm can guarantee both an exact answer and fast runtime on every instance. This project combines an **exact solver** (CP-SAT) with an **approximate search** (Whale Optimization Algorithm + Large Neighborhood Search) so that easy instances get a *proven* optimal answer quickly, and hard instances still get a *good* answer in reasonable time.

---

## 1. The core idea, in one paragraph

Always try the exact solver first. If it proves it found the best possible schedule, stop there — nothing can beat it. If it only finds a decent-but-unproven schedule (or nothing at all) within a time budget, hand that result to a metaheuristic search as a head start, let the search try to improve on it, and keep whichever answer — the exact solver's or the search's — turns out better.

---

## 2. The three building blocks

### 2.1 CP-SAT (exact solver)

CP-SAT is a constraint-programming solver (via Google OR-Tools). It models the whole scheduling problem as a set of variables and logical constraints — which machine each operation uses, when it starts and ends, that no two operations overlap on the same machine, that a job's operations run in the right order — and searches for an assignment that minimizes the makespan, using logical deduction to prune huge parts of the search space rather than guessing.

It can return one of three outcomes, given a time limit:

| Status | Meaning |
| --- | --- |
| `OPTIMAL` | Found a schedule **and proved** no better one exists. |
| `FEASIBLE` | Found a valid schedule, but ran out of time before proving it's the best. |
| `UNKNOWN` | Ran out of time before finding *any* valid schedule. |

CP-SAT is excellent on small-to-medium instances, where it typically returns `OPTIMAL` in well under a second. On larger or more structurally difficult instances, it can get stuck in `FEASIBLE` or `UNKNOWN` within a practical time budget — not because no good schedule exists, but because *proving* optimality (or even finding a good incumbent) gets combinatorially harder.

### 2.2 WOA (Whale Optimization Algorithm)

WOA is a metaheuristic inspired by how humpback whales hunt in groups. A population of "whales" search in parallel; each whale is one candidate solution. Every iteration, each whale moves using one of three rules:

- **Encircle prey** — move toward the current best whale.
- **Spiral bubble-net attack** — approach the best whale along a spiral path, so nearby regions get probed too, not just a straight line.
- **Search for prey** — move toward a randomly chosen whale instead of the best one, to keep exploring new territory.

A shrinking parameter gradually shifts the population from broad exploration (early iterations) to focused refinement around the best-known solution (later iterations). Two additional decay-weight functions (`w(t)`, cosine decay, and `v(t)`, exponential decay) further tune this exploration-to-exploitation balance over the run, scaling the "chase the best whale" and "chase a random whale" moves respectively.

WOA never looks at jobs or machines directly — it only ever moves numbers around in a fixed numeric space. To judge how good a whale is, its numbers get **decoded** into an actual schedule, and the resulting makespan becomes its fitness score (lower is better).

### 2.3 LNS (Large Neighborhood Search)

LNS is the "surgical fix" step. Instead of trying to improve a whole schedule at once:

1. Take the current best schedule.
2. **Freeze** almost all of it — leave most operations exactly where they are.
3. **Unfreeze** a small window of operations (by default \~10): the ones on the **critical path** (the specific chain of operations actually responsible for the current makespan) plus their immediate neighbors.
4. Hand just that small window to CP-SAT and ask it to solve it **exactly**.
5. Splice the result back into the full schedule.

Because the window is small and fixed in size, CP-SAT can solve it in a fraction of a second — even on instances where solving the *whole* schedule exactly would be too slow. This lets the search borrow CP-SAT's exactness for the part of the problem that matters most (the bottleneck), without paying for full-instance exactness.

---

## 3. The full pipeline, step by step

```
                    ┌─────────────────────┐
                    │   Run CP-SAT first   │
                    │  (fixed time budget) │
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
        OPTIMAL            FEASIBLE          UNKNOWN / no
      (proven best)    (valid, unproven)   feasible schedule
            │                  │                  │
            │                  ▼                  ▼
            │        Seed WOA's population   Run WOA + LNS
            │        with CP-SAT's schedule  from scratch
            │        (one whale, not all)    (heuristic + random
            │                  │              starting whales)
            │                  ▼                  │
            │           Run WOA + LNS             │
            │                  │                  │
            │                  ▼                  ▼
            │         Compare hybrid's      Hybrid's result
            │         result to CP-SAT's     is the final
            │         feasible answer,          answer
            │         keep the better one
            │                  │
            ▼                  ▼
     ┌─────────────────────────────┐
     │      Return final schedule   │
     └─────────────────────────────┘
```

**Why seed only one whale, not the whole population?** CP-SAT's `FEASIBLE` answer isn't always close to optimal — it can be weak, particularly on instances with a lot of tied/symmetric machine choices (see Section 5). Replacing the *entire* population with copies of a possibly-weak seed would anchor the whole search on a bad starting point. Replacing just one whale gives the seed a chance to help if it's good, while the rest of the population still explores freely and can find something better if the seed was misleading.

**Why does LNS fire only periodically, not every iteration?** Each LNS call is a full CP-SAT solve (of a small window), which costs real time. Firing it every few iterations (a tunable `lns_frequency`) balances search quality against wall-clock cost.

---

## 4. Key parameters

| Parameter | Where | What it controls |
| --- | --- | --- |
| `cpsat_time_budget` | CP-SAT (initial pass) | How long CP-SAT gets to try to prove optimal before falling back to the hybrid. |
| `hybrid_time_budget` | WOA+LNS | Total wall-clock time the metaheuristic search is allowed. |
| `woa_population_size` | WOA | Number of whales searching in parallel. More whales explore more of the space per iteration, at the cost of more schedule evaluations. |
| `woa_max_iterations` | WOA | How many rounds of whale movement to run. |
| `w_min`, `w_max` | WOA (adaptive weight) | Bounds of the decay curves controlling how bold/cautious whale movement is over the run. |
| `lns_frequency` | LNS | How often (in WOA iterations) to run one LNS "deep fix" step. |
| `lns_window_size` | LNS | How many operations get unfrozen per LNS call. Bigger windows are more powerful but slower to solve exactly. |
| `seed_chromosome` | WOA | Optional external starting schedule (e.g. CP-SAT's incumbent) to seed one whale with. |

---

---

## 6. Known strengths

- **Never worse than CP-SAT alone.** The final comparison step guarantees the hybrid only replaces CP-SAT's answer if it's genuinely better — there's no risk of regressing below what CP-SAT already achieved.
- **No wasted effort on easy instances.** Small, low-symmetry instances resolve to `OPTIMAL` almost immediately and never touch the metaheuristic path at all.
- **Exact correction where it matters most.** LNS targets the critical path specifically, so the limited time spent on exact sub-solving is concentrated on the operations actually responsible for the makespan.
- **Seeding compounds with search.** When CP-SAT's incumbent is strong, the hybrid starts close to a good answer instead of from random noise, which should reduce how many iterations are needed to make further progress.