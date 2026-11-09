"""
Nouvelle DP avec Delay Sharing T=1
Implémentation fidèle aux équations du PDF (slides 14-23).

STRUCTURE D'INFORMATION T=1 (slide 22)
  Δ_t    = {Y^(K)_{1,t-1}, U^(K)_{1,t-1}}   (public, partagé à t)
  Λ^k_t  = {Y^k_t}                              (privé, seulement agent k)
  I^k_t  = Δ_t ∪ Λ^k_t

  • t=1 : Δ_1 = ∅            → I^k_1 = {Y^k_1}   (identique à T=∞)
  • t=2 : Δ_2 = {Y^(K)_1, U^(K)_1} → agent k CONNAÎT Y^{-k}_1
  • t≥2 : la stratégie γ^k_t peut exploiter Y^{-k}_{t-1} (public)

DIFFÉRENCE CLEF AVEC T=∞ (pas de partage)
  T=∞ : futur = Σ_{y^k_{t+1}} P(y^k_{t+1}|ξ,u^k) · V_{t+1}(ξ_{t+1})
  T=1 : futur = Σ_{y^k_{t+1}} Σ_{y^{-k}_t_réalisé}
                  P(y^k_{t+1}, y^{-k}_t | ξ, u^k) · V_{t+1}(ξ_{t+1})

  À t+1, y^{-k}_t entre dans Δ_{t+1} (devient PUBLIC) : on l'obtient
  comme variable d'intégration supplémentaire, et le belief est mis à
  jour en conditionnant sur la VALEUR RÉALISÉE de y^{-k}_t.

COÛT PAR AGENT (eq. 19-20 du PDF)
  V^k_t(ξ^k_t) = inf_{u^k} [
      Σ_{x,y^{-k}} ℓ(x, u^k, γ^{-k}(y^{-k})) · ξ^k_t(x, y^{-k})
    + Σ_{y^k_{t+1}, y^{-k}_t_réal.}
          P(y^k_{t+1}, y^{-k}_t_réal. | ξ^k_t, u^k, γ^{-k})
        · V^k_{t+1}(T^k_{t+1}[y^k_{t+1}, y^{-k}_t_réal., u^k, γ^{-k}](ξ^k_t))
  ]
"""

import time
import itertools
from collections import defaultdict
import signal
import os

# 1. PARAMÈTRES SYSTÈME
#  PARAMÈTRE PRINCIPAL À MODIFIER 
# Augmenter HORIZON ralentit exponentiellement le multi-start DP (T=1) :
#   K=2 : H≤8 raisonnable  (~quelques secondes)
#   K=3 : H≤5 raisonnable  (~quelques minutes)
#   K=4 : H≤4 raisonnable  (~10-30 min avec cache partagé)
# Le timeout adaptatif coupe automatiquement les runs trop longs.
HORIZON = 3   # ← MODIFIABLE LIBREMENT
DELAY    = 1
OBS_ACCURACY = [0.9, 0.8, 0.85, 0.75]


def obs_prob(agent_k, y, x):
    """P(Y^k = y | X = x)"""
    acc = OBS_ACCURACY[agent_k]
    return acc if y == x else (1.0 - acc)


def transition(x, x_next, u_joint):
    """P(X_{t+1}=x' | X_t=x, u_joint) — vote majoritaire."""
    vote   = sum(u_joint) / len(u_joint)
    target = 1 if vote >= 0.5 else 0
    if target == x:
        return 0.8 if x_next == x else 0.2
    else:
        return 0.3 if x_next == x else 0.7


def stage_cost(x, u_joint):
    """ℓ(x,u) = Σ_k (x−u^k)² + λ·Σ_{k<j} (u^k−u^j)²"""
    K    = len(u_joint)
    lam  = 2.0
    ind  = sum((x - u)**2 for u in u_joint)
    coup = sum((u_joint[i] - u_joint[j])**2
               for i in range(K) for j in range(i+1, K))
    return ind + lam * coup


# 2. CROYANCES ET OPÉRATEUR T^k_{t+1} POUR T=1
def belief_init(agent_k, y_k_1, K, p_x=None):
    """
    Croyance initiale à t=1 : Δ_1=∅, I^k_1={Y^k_1}.
    ξ^k_1(x, y^{-k}) = P(X_1=x, Y^{-k}_1=y^{-k} | Y^k_1=y_k_1)
    """
    if p_x is None:
        p_x = {0: 0.6, 1: 0.4}
    other_agents   = [j for j in range(K) if j != agent_k]
    y_others_space = list(itertools.product([0, 1], repeat=len(other_agents)))
    belief = {}
    for x in [0, 1]:
        for y_oth in y_others_space:
            p = obs_prob(agent_k, y_k_1, x) * p_x[x]
            for idx, j in enumerate(other_agents):
                p *= obs_prob(j, y_oth[idx], x)
            belief[(x, tuple(y_oth))] = p
    total = sum(belief.values())
    return {k: v / total for k, v in belief.items()} if total > 0 else belief


def update_belief_T1(agent_k, belief, y_k_next, y_others_realized,
                     u_k, gamma_others_fn, K):
    """
    Opérateur T^k_{t+1} (éq. 4-6, slides 15-16) pour T=1.

    À t+1, y_others_realized = y^{-k}_t est entré dans Δ_{t+1} (PUBLIC).
    On conditionne donc sur cette valeur RÉALISÉE (et non on l'intègre).

    ξ^k_{t+1}(x', y'^{-k}) ∝
      Σ_x  ξ^k_t(x, y^{-k}_t_réal.)          ← sélectionne la ligne réalisée
           · S_{t+1}(x'|x, u^k, u^{-k}_t)
           · Q^k_{t+1}(y^k_{t+1}|x', u^k, u^{-k}_t)
           · Π_{j≠k} Q^j_{t+1}(y'^j_{t+1}|x', u^k, u^{-k}_t)
    """
    other_agents   = [j for j in range(K) if j != agent_k]
    y_others_space = list(itertools.product([0, 1], repeat=len(other_agents)))
    new_belief     = defaultdict(float)

    # Actions des autres déduites de leurs obs RÉALISÉES (maintenant publiques)
    u_others = tuple(gamma_others_fn[j](y_others_realized[idx])
                     for idx, j in enumerate(other_agents))
    u_joint  = list(u_others)
    u_joint.insert(agent_k, u_k)
    u_joint  = tuple(u_joint)

    for (x, y_others_curr), p in belief.items():
        if p < 1e-14:
            continue
        # Condition sur y^{-k}_t réalisé (T=1 : maintenant public dans Δ_{t+1})
        if y_others_curr != tuple(y_others_realized):
            continue

        for x_next in [0, 1]:
            p_trans = transition(x, x_next, u_joint)
            p_obs_k = obs_prob(agent_k, y_k_next, x_next)

            # Intégration sur Y^{-k}_{t+1} (encore privé des autres à t+1)
            for y_others_next in y_others_space:
                p_obs_oth = 1.0
                for idx, j in enumerate(other_agents):
                    p_obs_oth *= obs_prob(j, y_others_next[idx], x_next)
                new_belief[(x_next, tuple(y_others_next))] += (
                    p * p_trans * p_obs_k * p_obs_oth
                )

    total = sum(new_belief.values())
    if total > 1e-14:
        return {k: v / total for k, v in new_belief.items()}
    # Belief dégénéré : retourner uniforme
    n = 2 * len(y_others_space)
    return {(x, tuple(yo)): 1.0 / n
            for x in [0, 1] for yo in y_others_space}


def joint_prob_T1(agent_k, belief, y_k_next, y_others_realized,
                  u_k, gamma_others_fn, K):
    """
    P(Y^k_{t+1}=y_k_next, Y^{-k}_t=y_others_realized | ξ^k_t, u^k_t, γ^{-k})

    Terme de pondération de la somme future pour T=1 (éq. 21-23, slide 21).
    """
    other_agents = [j for j in range(K) if j != agent_k]
    u_others = tuple(gamma_others_fn[j](y_others_realized[idx])
                     for idx, j in enumerate(other_agents))
    u_joint  = list(u_others)
    u_joint.insert(agent_k, u_k)
    u_joint  = tuple(u_joint)

    prob = 0.0
    for (x, y_others_curr), p in belief.items():
        if y_others_curr != tuple(y_others_realized):
            continue
        for x_next in [0, 1]:
            prob += (p
                     * transition(x, x_next, u_joint)
                     * obs_prob(agent_k, y_k_next, x_next))
    return prob


def encode_belief(belief):
    return tuple(sorted((k, round(v, 6)) for k, v in belief.items()))


# 3. SOLVEUR DP — AGENT k — NOUVELLE DP T=1

class AgentSolverT1:
    """
    Résout l'équation DP (slides 19-23) pour un agent k avec T=1.

    Équation de Bellman :
      V^k_t(ξ^k_t) = inf_{u^k} [
          Σ_{x,y^{-k}} ℓ(x,u^k,γ^{-k}(y^{-k})) · ξ^k_t(x,y^{-k})
        + Σ_{y^k_{t+1}} Σ_{y^{-k}_t_réal.}
              P(y^k_{t+1}, y^{-k}_t_réal. | ξ, u^k, γ^{-k})
            · V^k_{t+1}(T^k_{t+1}[y^k_{t+1}, y^{-k}_t_réal., u^k, γ^{-k}](ξ))
      ]
    """

    def __init__(self, agent_k, K, horizon, shared_memo=None):
        self.k            = agent_k
        self.K            = K
        self.horizon      = horizon
        # shared_memo : dict externe partagé entre runs du multi-start.
        # Quand deux initialisations convergent vers le même belief sous
        # le même gamma^{-k}, le résultat est réutilisé sans recalcul.
        self.memo         = shared_memo if shared_memo is not None else {}
        self.other_agents = [j for j in range(K) if j != agent_k]
        self.yo_space     = list(itertools.product([0, 1],
                                 repeat=len(self.other_agents)))

    def clear(self):
        self.memo.clear()

    def _encode_gamma(self, gamma_fn):
        return tuple((j, gamma_fn[j](0), gamma_fn[j](1))
                     for j in sorted(gamma_fn.keys()))

    def solve(self, t, belief, gamma_others_fn):
        key = (t, encode_belief(belief), self._encode_gamma(gamma_others_fn))
        if key in self.memo:
            return self.memo[key]
        if t == self.horizon:
            return 0.0, None

        best_u, min_cost = None, float('inf')

        for u_k in [0, 1]:
            # ── Coût immédiat ──────────────────────────────────────────
            imm = 0.0
            for (x, y_oth), p in belief.items():
                u_oth   = tuple(gamma_others_fn[j](y_oth[idx])
                                for idx, j in enumerate(self.other_agents))
                u_joint = list(u_oth)
                u_joint.insert(self.k, u_k)
                imm += p * stage_cost(x, tuple(u_joint))

            # ── Coût futur — double somme T=1 (éq. 20) ────────────────
            # Σ_{y^{-k}_t_réal.} Σ_{y^k_{t+1}}
            #   P(y^k_{t+1}, y^{-k}_t_réal. | ξ, u^k, γ^{-k})
            #   · V^k_{t+1}(T^k_{t+1}[y^k_{t+1}, y^{-k}_t_réal., u^k, γ^{-k}](ξ))
            fut = 0.0
            for y_oth_real in self.yo_space:   # y^{-k}_t qui devient public
                for y_k_next in [0, 1]:        # nouvelle obs privée de k
                    pj = joint_prob_T1(
                        self.k, belief, y_k_next, y_oth_real,
                        u_k, gamma_others_fn, self.K)
                    if pj < 1e-12:
                        continue
                    b_next = update_belief_T1(
                        self.k, belief, y_k_next, y_oth_real,
                        u_k, gamma_others_fn, self.K)
                    v_next, _ = self.solve(t + 1, b_next, gamma_others_fn)
                    fut += pj * v_next

            total = imm + fut
            if total < min_cost:
                min_cost, best_u = total, u_k

        self.memo[key] = (min_cost, best_u)
        return min_cost, best_u


# 4. ALGORITHME PbP AVEC T=1 — COÛTS PAR AGENT

def make_gamma_fn(strategy_dict):
    """Convertit {y: u} en callable y → u."""
    return lambda y: strategy_dict[y]


def evaluate_agent_cost(k, K, horizon, strategies, p_x,
                        shared_memos=None):
    """
    Cout individuel V^k = E_{Y^k_1}[V^k_0(xi^k_1(Y^k_1))].

    shared_memos : dict optionnel {k: memo_dict} fourni par le multi-start
                   pour reutiliser les calculs entre les 4^K runs.
    """
    gamma_fn = {j: make_gamma_fn(strategies[j])
                for j in range(K) if j != k}
    memo     = shared_memos[k] if shared_memos is not None else None
    solver   = AgentSolverT1(k, K, horizon, shared_memo=memo)
    v_k      = 0.0
    for y_k in [0, 1]:
        p_y_k  = sum(p_x[x] * obs_prob(k, y_k, x) for x in [0, 1])
        belief = belief_init(k, y_k, K, p_x)
        v, _   = solver.solve(0, belief, gamma_fn)
        v_k   += p_y_k * v
    return v_k


def _pbp_one_run(K, horizon, init_strategies, p_x, max_iter=20, tol=1e-6,
                 verbose=True, shared_memos=None):
    """
    Un seul run PbP depuis une initialisation donnee.

    Retourne (strategies, agent_costs, v_joint, n_iters).
    Si verbose=False, n'imprime rien.
    shared_memos : {k: dict} transmis a AgentSolverT1 et evaluate_agent_cost
                   pour partager le cache entre tous les runs du multi-start.
    """
    strategies  = [dict(s) for s in init_strategies]
    agent_costs = [0.0] * K
    prev_cost   = float('inf')
    n_iters     = 0

    for iteration in range(max_iter):
        n_iters = iteration + 1
        changed = False

        for k in range(K):
            gamma_fn = {j: make_gamma_fn(strategies[j])
                        for j in range(K) if j != k}
            memo     = shared_memos[k] if shared_memos is not None else None
            solver   = AgentSolverT1(k, K, horizon, shared_memo=memo)

            new_gamma_k = {}
            for y_k in [0, 1]:
                belief   = belief_init(k, y_k, K, p_x)
                _, u_opt = solver.solve(0, belief, gamma_fn)
                new_gamma_k[y_k] = u_opt if u_opt is not None else y_k

            if new_gamma_k != strategies[k]:
                strategies[k] = new_gamma_k
                changed        = True

        for k in range(K):
            agent_costs[k] = evaluate_agent_cost(
                k, K, horizon, strategies, p_x, shared_memos=shared_memos)

        v_joint = sum(agent_costs) / K

        if verbose:
            strats = " | ".join(f"γ{k+1}={strategies[k]}" for k in range(K))
            costs  = " | ".join(f"V{k+1}={agent_costs[k]:.4f}" for k in range(K))
            print(f"  Iter {iteration+1:2d} | {strats}")
            print(f"         | {costs} | V_moy={v_joint:.4f}")

        if not changed or abs(prev_cost - v_joint) < tol:
            if verbose:
                print(f"\n  ✅ Nash atteint à l'itération {iteration+1} "
                      f"(init = {[dict(s) for s in init_strategies]})")
            break
        prev_cost = v_joint

    return strategies, agent_costs, v_joint, n_iters


def run_pbp_T1(K, horizon, max_iter=20, tol=1e-6, p_x=None):
    """
    DP PbP (T=1) — mono-start depuis l'initialisation identité {0:0, 1:1}.

    C'est le point de référence « naïf » : une seule exécution depuis
    le point de départ standard. Rapide, mais peut rester coincé dans
    un équilibre de Nash local sous-optimal si l'identité est elle-même
    une meilleure réponse à elle-même (puits d'attraction immédiat).
    """
    if p_x is None:
        p_x = {0: 0.6, 1: 0.4}

    print(f"\n{'='*65}")
    print(f"  DP PbP Mono-Start (T=1) — K={K} | H={horizon}")
    print(f"  Init : identité {{0:0, 1:1}} pour tous les agents")
    print(f"{'='*65}")

    init = [{0: 0, 1: 1}] * K   # identité
    strats, agent_costs, v_joint, _ = _pbp_one_run(
        K, horizon, init, p_x, max_iter=max_iter, tol=tol, verbose=True)

    print(f"{'─'*65}")
    for k in range(K):
        print(f"    γ{k+1}·ᵒ = {strats[k]}   V{k+1} = {agent_costs[k]:.4f}")

    return strats, agent_costs, v_joint


def run_pbp_T1_multistart(K, horizon, max_iter=20, tol=1e-6, p_x=None):
    """
    DP PbP (T=1) — Multi-Start exhaustif sur les 4^K initialisations
    stationnaires pures.

    Principe (inspiré des remarques de l'analyse) :
      Pour chaque profil d'initialisation (γ¹_init,...,γ^K_init) ∈ (S_pur)^K :
        1. Lancer _pbp_one_run depuis cette initialisation
        2. Collecter le Nash local obtenu et son coût
      Retenir le Nash de coût minimal sur toutes les exécutions.

    Avantage sur la Brute Force :
      • La BF est exhaustive mais STATIONNAIRE : elle ne cherche que parmi
        les 4 fonctions γ^k : {0,1}→{0,1} constantes dans le temps.
      • La DP multi-start part des mêmes 4^K points mais laisse les agents
        explorer les stratégies NON-STATIONNAIRES via le belief ξ^k_t(T=1).
      • Résultat attendu : le meilleur Nash DP multi-start ≤ meilleur Nash BF
        (espace strictement plus riche, exhaustivement initialisé).

    Complexité : 4^K runs DP × max_iter × K × O(2^K · H · memo)
    """
    if p_x is None:
        p_x = {0: 0.6, 1: 0.4}

    n_inits = 4 ** K

    #  Calibration : mesure du temps d'un run de référence 
    # On lance un run rapide (init identité) pour estimer le temps réel
    # d'un run DP avec les paramètres (K, H) courants, AVANT de lancer
    # les 4^K runs. Le timeout est ensuite fixé à max(10s, 10 × t_calib).
    # Cela rend le multistart robuste à TOUT (K, H) sans paramètre hardcodé.
    print(f"\n{'='*65}")
    print(f"  DP PbP Multi-Start (T=1) — K={K} | H={horizon}")
    print(f"  {n_inits} initialisations | memo partage entre runs")
    print(f"  Calibration en cours...", end=" ", flush=True)

    calib_init   = [{0: 0, 1: 1}] * K   # identité, run le plus rapide
    shared_memos = {k: {} for k in range(K)}   # memo initialise ici, reutilise ensuite

    t_calib = time.perf_counter()
    _pbp_one_run(K, horizon, calib_init, p_x,
                 max_iter=1, tol=1e-6, verbose=False,
                 shared_memos=shared_memos)
    t_calib = time.perf_counter() - t_calib   # secondes

    # Timeout = max(10s, 15 × temps calibration)
    # Suffisant pour laisser converger un run plus lent que la calib,
    # mais évite les blocages sur des configurations très lourdes.
    run_timeout = max(10, int(t_calib * 15)) if hasattr(signal, 'SIGALRM') else None

    t_total_est = t_calib * n_inits   # estimation grossière (avec cache : beaucoup moins)
    print(f"OK ({t_calib*1000:.0f}ms/run)")
    print(f"  Timeout par run   : {run_timeout}s  "
          f"(= max(10s, 15 × {t_calib*1000:.0f}ms))")
    print(f"  Durée max estimée : {t_total_est/60:.1f} min "
          f"(avec memo partage : beaucoup moins)")
    print(f"{'='*65}")

    all_init_profiles = list(itertools.product(range(4), repeat=K))
    runs      = []
    skipped   = 0
    t_ms_start = time.perf_counter()

    for run_idx, init_profile in enumerate(all_init_profiles):
        init_strats = [_PURE_STRATEGIES[i] for i in init_profile]

        # ── Timeout via SIGALRM (Unix / macOS) 
        timed_out = False
        if run_timeout and hasattr(signal, 'SIGALRM'):
            def _timeout_handler(signum, frame):
                raise TimeoutError()
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(run_timeout)

        try:
            t_run = time.perf_counter()
            strats, agent_costs, v_joint, n_it = _pbp_one_run(
                K, horizon, init_strats, p_x,
                max_iter=max_iter, tol=tol, verbose=False,
                shared_memos=shared_memos)
            dt_run = (time.perf_counter() - t_run) * 1000
        except TimeoutError:
            timed_out = True
            skipped  += 1
            strats, agent_costs, v_joint, n_it = (
                list(init_strats), [float('inf')]*K, float('inf'), 0)
            dt_run = run_timeout * 1000
        finally:
            if run_timeout and hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        if not timed_out:
            runs.append((v_joint, strats, agent_costs, init_profile, n_it))

        # Barre de progression 
        elapsed  = time.perf_counter() - t_ms_start
        pct      = (run_idx + 1) / n_inits * 100
        filled   = int(20 * (run_idx + 1) / n_inits)
        bar      = '█' * filled + '░' * (20 - filled)
        memo_sz  = sum(len(shared_memos[k]) for k in range(K))
        eta      = (elapsed / (run_idx + 1)) * (n_inits - run_idx - 1)
        tag      = " TIMEOUT" if timed_out else ""
        print(f"  [{bar}] {run_idx+1:>3}/{n_inits} ({pct:4.0f}%) "
              f"{dt_run:6.0f}ms  V={v_joint:7.4f}  "
              f"cache={memo_sz}  ETA={eta:.0f}s{tag}", flush=True)

    if skipped:
        print(f"\n  ⚠  {skipped}/{n_inits} runs annulés (timeout {run_timeout}s)")
        print(f"     → Pour un multistart complet, augmenter run_timeout")
        print(f"       ou réduire H dans la variable HORIZON.")

    #  Meilleur Nash trouvé toutes initialisations confondues 
    runs.sort(key=lambda r: r[0])
    best_cost, best_strats, best_agent_costs, best_init, best_niters = runs[0]

    # Résumé des Nash distincts 
    seen = {}
    for v, strats, ac, init_p, n_it in runs:
        key = tuple(tuple(sorted(s.items())) for s in strats)
        if key not in seen or v < seen[key][0]:
            seen[key] = (v, strats, ac)

    distinct_nash = sorted(seen.values(), key=lambda x: x[0])

    total_memo = sum(len(m) for m in shared_memos.values())
    print(f"\n{'─'*65}")
    print(f"  Cache partage : {total_memo} entrees totales "
          f"({[len(shared_memos[k]) for k in range(K)]} par agent)")
    print(f"  Nash distincts trouves ({len(distinct_nash)}) :")
    for v, strats, ac in distinct_nash:
        marker = "  ← MEILLEUR ★" if abs(v - best_cost) < 1e-9 else ""
        print(f"    {[dict(s) for s in strats]}  V_moy={v:.4f}{marker}")

    print(f"\n  ★  Meilleur Nash Multi-Start : V_moy = {best_cost:.4f}")
    print(f"     Init gagnante : {[_PURE_STRATEGIES[i] for i in best_init]}")
    print(f"     Stratégies    : {[dict(s) for s in best_strats]}")
    for k in range(K):
        print(f"     V{k+1} = {best_agent_costs[k]:.4f}")

    return best_strats, best_agent_costs, best_cost, runs


# 5. BRUTE FORCE DÉCENTRALISÉE — RÉFÉRENCE SANS DP
# Principe : énumérer EXHAUSTIVEMENT les 4^K profils de stratégies
# STATIONNAIRES pures γ^k : {0,1} → {0,1} (constantes dans le temps),
# évaluer le coût exact de chaque profil par récurrence forward,
# puis identifier les équilibres de Nash et retenir le meilleur.
# Espace des stratégies stationnaires par agent (4 fonctions binaires) :
#   s0 = {0:0, 1:0}   (toujours 0)
#   s1 = {0:0, 1:1}   (identité, i.e. u = y^k)
#   s2 = {0:1, 1:0}   (inverse)
#   s3 = {0:1, 1:1}   (toujours 1)
#
# Complexité : 4^K évaluations × O(2^K · H) par évaluation → tractable.
# Pour K=4 : 256 profils × ~2ms = ~0.5 s.
#
# Comparaison juste avec la DP PbP :
#   BF exhaustive   → meilleur Nash dans l'espace STATIONNAIRE  (4^K profils)
#   DP PbP (T=1)    → Nash dans l'espace NON-STATIONNAIRE via belief
#                     (espace strictement plus large ; convergence locale)
#   DP Multi-start  → meilleur Nash non-stationnaire sur 4^K initialisations
#
# Le gain de la DP apparaît à travers :
#   (a) la richesse de l'espace non-stationnaire exploré
#   (b) la garantie de convergence vers un point fixe de Bellman

# Les 4 stratégies pures stationnaires possibles par agent
_PURE_STRATEGIES = [
    {0: 0, 1: 0},   # s0 : toujours 0
    {0: 0, 1: 1},   # s1 : identité (u = y)
    {0: 1, 1: 0},   # s2 : inverse
    {0: 1, 1: 1},   # s3 : toujours 1
]


def eval_cost_forward(K, H, strategies, p_x=None):
    """
    Évaluation EXACTE du coût social pour un profil de stratégies fixes.

    Par récurrence forward sur la distribution jointe P(X_t, Y^1_t,...,Y^K_t).
    Retourne le coût social total (= V^k pour tous k dans ce modèle).

    Complexité : O(H · 2^{K+1}) — très rapide.
    """
    if p_x is None:
        p_x = {0: 0.6, 1: 0.4}

    obs_space = list(itertools.product([0, 1], repeat=K))

    # Distribution initiale P(X_1=x, Y^(K)_1=y_all)
    dist = {}
    for x in [0, 1]:
        for y_all in obs_space:
            p = p_x[x]
            for k, y_k in enumerate(y_all):
                p *= obs_prob(k, y_k, x)
            dist[(x, tuple(y_all))] = p

    total_cost = 0.0

    for _ in range(H):
        new_dist = defaultdict(float)
        for (x, y_all), p in dist.items():
            if p < 1e-15:
                continue
            u_joint = tuple(strategies[k][y_all[k]] for k in range(K))
            total_cost += p * stage_cost(x, u_joint)

            for x_next in [0, 1]:
                p_trans = transition(x, x_next, u_joint)
                for y_next in obs_space:
                    p_obs = 1.0
                    for k, y_k_next in enumerate(y_next):
                        p_obs *= obs_prob(k, y_k_next, x_next)
                    new_dist[(x_next, tuple(y_next))] += p * p_trans * p_obs

        dist = dict(new_dist)

    return total_cost


def find_best_nash_bf(K, H, p_x=None):
    """
    Brute Force décentralisée : trouve TOUS les équilibres de Nash parmi
    les stratégies stationnaires pures, retourne le meilleur.

    Algorithme :
      1. Calculer le coût pour chaque profil joint (4^K évaluations)
      2. Vérifier la condition Nash pour chaque profil :
         profil est Nash ssi aucun agent k ne peut réduire le coût
         en changeant sa stratégie seul (meilleure réponse)
      3. Retenir le Nash de coût minimal (meilleur équilibre)

    Retourne :
      best_strategies  : liste des stratégies du meilleur Nash
      best_cost        : coût social du meilleur Nash
      all_nash         : tous les équilibres Nash trouvés (profil, coût)
      cost_table       : dict {profil_idx: coût} pour analyse
    """
    if p_x is None:
        p_x = {0: 0.6, 1: 0.4}

    all_profiles = list(itertools.product(range(4), repeat=K))

    # ── Étape 1 : précalcul des coûts 
    cost_table = {}
    for profile_idx in all_profiles:
        strats = [_PURE_STRATEGIES[i] for i in profile_idx]
        cost_table[profile_idx] = eval_cost_forward(K, H, strats, p_x)

    # ── Étape 2 : vérification Nash 
    all_nash = []
    for profile_idx in all_profiles:
        c_curr = cost_table[profile_idx]
        is_nash = True
        for k in range(K):
            # L'agent k peut-il réduire le coût en déviant ?
            for alt_idx in range(4):
                if alt_idx == profile_idx[k]:
                    continue
                dev = list(profile_idx)
                dev[k] = alt_idx
                if cost_table[tuple(dev)] < c_curr - 1e-9:
                    is_nash = False
                    break
            if not is_nash:
                break
        if is_nash:
            strats = [_PURE_STRATEGIES[i] for i in profile_idx]
            all_nash.append((profile_idx, strats, c_curr))

    if not all_nash:
        # Fallback : retourner le profil de coût minimal (ne devrait pas arriver)
        best_idx = min(cost_table, key=cost_table.get)
        best_strats = [_PURE_STRATEGIES[i] for i in best_idx]
        return best_strats, cost_table[best_idx], [], cost_table

    # ── Étape 3 : meilleur Nash 
    all_nash.sort(key=lambda x: x[2])   # tri par coût croissant
    best_profile_idx, best_strategies, best_cost = all_nash[0]

    return best_strategies, best_cost, all_nash, cost_table


def run_brute_force(K, H, p_x=None):
    """Interface haut-niveau pour le benchmark."""
    print(f"\n{'='*65}")
    print(f"  Brute Force Décentralisée — K={K} agents | H={H}")
    print(f"{'='*65}")

    if p_x is None:
        p_x = {0: 0.6, 1: 0.4}

    t0 = time.perf_counter()
    best_strats, best_cost, all_nash, cost_table = find_best_nash_bf(K, H, p_x)
    t_bf = (time.perf_counter() - t0) * 1000

    n_profiles = 4 ** K
    print(f"  {n_profiles} profils évalués | {len(all_nash)} équilibres Nash trouvés")
    print(f"  Équilibres Nash (triés par coût) :")
    for _, strats, c in all_nash:
        marker = " ← MEILLEUR" if abs(c - best_cost) < 1e-9 else ""
        print(f"    {[dict(s) for s in strats]}  coût={c:.4f}{marker}")
    print(f"{'─'*65}")
    print(f"  Meilleur Nash BF : coût = {best_cost:.4f}")
    print(f"  Stratégies       : {[dict(s) for s in best_strats]}")
    print(f"  Temps total      : {t_bf:.1f} ms")

    # Coût par agent = coût social (identique pour tous dans ce modèle)
    agent_costs_bf = [best_cost] * K

    return best_strats, agent_costs_bf, best_cost, t_bf


# 6. ARCHITECTURE CENTRALISÉE (BORNE INFÉRIEURE)

class CentralizedSolver:
    """Contrôleur omniscient — borne inférieure du coût social."""

    def __init__(self, K, horizon):
        self.K       = K
        self.horizon = horizon
        self.memo    = {}

    def _encode(self, belief):
        return tuple(sorted((k, round(v, 6)) for k, v in belief.items()))

    def _update(self, belief, y_all, u_joint):
        new_b = {}
        for x_next in [0, 1]:
            p = 0.0
            for x, px in belief.items():
                p_obs = 1.0
                for k, y_k in enumerate(y_all):
                    p_obs *= obs_prob(k, y_k, x_next)
                p += px * transition(x, x_next, u_joint) * p_obs
            new_b[x_next] = p
        total = sum(new_b.values())
        return ({k: v / total for k, v in new_b.items()}
                if total > 0 else new_b)

    def _obs_prob(self, belief, y_all, u_joint):
        prob = 0.0
        for x, px in belief.items():
            for x_next in [0, 1]:
                p_obs = 1.0
                for k, y_k in enumerate(y_all):
                    p_obs *= obs_prob(k, y_k, x_next)
                prob += px * transition(x, x_next, u_joint) * p_obs
        return prob

    def solve(self, t, belief):
        key = (t, self._encode(belief))
        if key in self.memo:
            return self.memo[key]
        if t == self.horizon:
            return 0.0, None
        best_cost, best_u = float('inf'), None
        for u_joint in itertools.product([0, 1], repeat=self.K):
            imm = sum(px * stage_cost(x, u_joint) for x, px in belief.items())
            fut = 0.0
            for y_all in itertools.product([0, 1], repeat=self.K):
                prob_y = self._obs_prob(belief, y_all, u_joint)
                if prob_y < 1e-12:
                    continue
                b_next    = self._update(belief, y_all, u_joint)
                v_next, _ = self.solve(t + 1, b_next)
                fut      += prob_y * v_next
            total = imm + fut
            if total < best_cost:
                best_cost, best_u = total, u_joint
        self.memo[key] = (best_cost, best_u)
        return best_cost, best_u


# 7. BENCHMARK : BRUTE FORCE vs. NOUVELLE DP (T=1) vs. CENTRALISÉ

if __name__ == "__main__":
    W = 74
    print("\n" + "=" * W)
    print("  BENCHMARK — BF vs. DP Mono-Start vs. DP Multi-Start vs. Centralisé")
    print("=" * W)

    p_x     = {0: 0.6, 1: 0.4}
    results = {}

    for K in [2, 3, 4]:
        h = HORIZON
        print(f"\n{'═'*W}")
        print(f"  ▶  K={K} agents | Horizon={h} | Délai T={DELAY}")
        print(f"{'═'*W}")

        #  1. Brute Force 
        t0     = time.perf_counter()
        bf_strats, bf_ac, cost_bf, _ = run_brute_force(K, h, p_x)
        t_bf   = (time.perf_counter() - t0) * 1000

        #  2. DP Mono-Start (init identité) 
        t0     = time.perf_counter()
        ms_strats, ms_ac, cost_ms = run_pbp_T1(K=K, horizon=h, p_x=p_x)
        t_ms   = (time.perf_counter() - t0) * 1000

        #  3. DP Multi-Start (4^K initialisations) 
        t0     = time.perf_counter()
        best_strats, best_ac, cost_dp_ms, all_runs = run_pbp_T1_multistart(
            K=K, horizon=h, p_x=p_x)
        t_dp_ms = (time.perf_counter() - t0) * 1000

        #  4. Centralisé (borne inférieure) 
        print(f"\n  Calcul architecture centralisée (K={K}, H={h})...")
        solver_c  = CentralizedSolver(K, h)
        t0        = time.perf_counter()
        cost_c, _ = solver_c.solve(0, p_x)
        t_c       = (time.perf_counter() - t0) * 1000
        print(f"  ✓ Centralisé terminé. Coût = {cost_c:.4f}  |  Temps = {t_c:.1f} ms")

        def gap(x):
            return (x - cost_c) / cost_c * 100

        results[K] = dict(
            horizon=h,
            cost_c=cost_c,       t_c=t_c,
            cost_bf=cost_bf,     t_bf=t_bf,      bf_ac=list(bf_ac),
            cost_ms=cost_ms,     t_ms=t_ms,      ms_ac=list(ms_ac),
            cost_dp_ms=cost_dp_ms, t_dp_ms=t_dp_ms, dp_ms_ac=list(best_ac),
            gap_bf=gap(cost_bf),
            gap_ms=gap(cost_ms),
            gap_dp_ms=gap(cost_dp_ms),
            gain_multi_over_mono=(cost_ms - cost_dp_ms) / cost_ms * 100
                                  if cost_ms > 0 else 0,
        )

    # ══════════════════════════════════════════════════════════════════
    # TABLEAU 1 — Coûts individuels par agent (3 méthodes décentralisées)
    # ══════════════════════════════════════════════════════════════════
    max_K = max(results.keys())
    W2    = 93

    print("\n\n" + "=" * W2)
    print("  COÛTS INDIVIDUELS PAR AGENT À L'ÉQUILIBRE NASH")
    print("=" * W2)

    agent_cols = " | ".join(f"{'V'+str(k+1):>9}" for k in range(max_K))
    hdr_line   = (f"  {'K':>2} | {'H':>2} | {'Méthode':<24} | "
                  f"{agent_cols} | {'V_moy':>9}")
    sep_line   = ("  ──-+-──-+-────────────────────────-+-"
                  + "-+-".join("─"*9 for _ in range(max_K))
                  + "-+-─────────")
    print(hdr_line)
    print(sep_line)

    LABELS = [
        ("BF   (stationnaire)",    "bf_ac"),
        ("DP   (mono-start)",      "ms_ac"),
        ("DP ★ (multi-start)",     "dp_ms_ac"),
    ]

    for K in [2, 3, 4]:
        r = results[K]
        for label, key in LABELS:
            ac   = r[key]
            vmoy = sum(ac[:K]) / K
            cs   = " | ".join(
                f"{ac[k]:>9.4f}" if k < K else f"{'—':>9}"
                for k in range(max_K)
            )
            print(f"  {K:>2} | {r['horizon']:>2} | {label:<24} | {cs} | {vmoy:>9.4f}")
        print(sep_line)

    # ══════════════════════════════════════════════════════════════════
    # TABLEAU 2 — Comparaison globale + temps
    # ══════════════════════════════════════════════════════════════════
    W3 = 110
    print("\n\n" + "=" * W3)
    print("  COMPARAISON GLOBALE : COÛTS & TEMPS DE CALCUL")
    print("=" * W3)
    print(f"  {'K':>2} | {'H':>2} │ "
          f"{'BF':>9} {'Gap':>7} │ "
          f"{'DP mono':>9} {'Gap':>7} │ "
          f"{'DP multi★':>9} {'Gap':>7} │ "
          f"{'Centr.':>9} │ "
          f"{'T.BF':>7} {'T.mono':>8} {'T.multi':>8} {'T.centr':>8}")
    sep3 = ("  ──-+-──-┼" + "─"*18 + "┼" + "─"*18 + "┼"
            + "─"*18 + "┼" + "─"*10 + "┼" + "─"*34)
    print(sep3)

    for K in [2, 3, 4]:
        r = results[K]
        costs3 = sorted([r['cost_bf'], r['cost_ms'], r['cost_dp_ms']])

        def icon(c):
            if abs(c - costs3[0])  < 1e-9: return "★"
            if abs(c - costs3[-1]) < 1e-9: return "▼"
            return " "

        print(
            f"  {K:>2} | {r['horizon']:>2} │ "
            f"{icon(r['cost_bf'])}{r['cost_bf']:>8.4f} +{r['gap_bf']:>5.1f}% │ "
            f"{icon(r['cost_ms'])}{r['cost_ms']:>8.4f} +{r['gap_ms']:>5.1f}% │ "
            f"{icon(r['cost_dp_ms'])}{r['cost_dp_ms']:>8.4f} +{r['gap_dp_ms']:>5.1f}% │ "
            f" {r['cost_c']:>8.4f} │ "
            f"{r['t_bf']:>6.0f}ms"
            f"{r['t_ms']:>7.0f}ms"
            f"{r['t_dp_ms']:>7.0f}ms"
            f"{r['t_c']:>7.0f}ms"
        )

    # ══════════════════════════════════════════════════════════════════
    # TABLEAU 3 — Gain du Multi-Start sur le Mono-Start
    # ══════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 72)
    print("  GAIN DU MULTI-START : APPORT RÉEL DE LA NOUVELLE DP (T=1)")
    print("=" * 72)
    print(f"  {'K':>2} | {'H':>2} | {'DP Mono':>10} | {'DP Multi★':>10} | "
          f"{'Gain Mono→Multi':>16} | {'Multi vs BF':>13}")
    print(f"  ──-+-──-+-──────────-+-──────────-+-────────────────-+-─────────────")
    for K in [2, 3, 4]:
        r      = results[K]
        g_mm   = r['gain_multi_over_mono']
        diff_bf = r['cost_dp_ms'] - r['cost_bf']
        if   diff_bf < -1e-9: verdict = "★ MIEUX que BF"
        elif abs(diff_bf) < 1e-9: verdict = "= ÉGAL à BF"
        else:                  verdict = "▼ MOINS BON que BF"
        print(
            f"  {K:>2} | {r['horizon']:>2} | "
            f"{r['cost_ms']:>10.4f} | "
            f"{r['cost_dp_ms']:>10.4f} | "
            f"{'−'+f'{g_mm:.2f}%':>16} | "
            f"{verdict:<13}  (Δ={diff_bf:+.4f})"
        )

    # ══════════════════════════════════════════════════════════════════
    # LÉGENDE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─'*W3}")
    print("  LÉGENDE")
    print("  ─────────────────────────────────────────────────────────────────────")
    print("  BF (stationnaire)  : 4^K profils de stratégies STATIONNAIRES pures")
    print("                       γ^k : {0,1}→{0,1} constante dans le temps.")
    print("                       Énumération exhaustive → meilleur Nash garanti.")
    print()
    print("  DP mono-start      : PbP (Nouvelle DP, T=1) depuis l'init identité.")
    print("                       Risque de puits local si l'identité est déjà")
    print("                       une meilleure réponse à elle-même (convergence 1 iter).")
    print()
    print("  DP multi-start ★   : PbP (T=1) lancée depuis les 4^K inits stationnaires.")
    print("                       Chaque run explore les stratégies NON-STATIONNAIRES")
    print("                       via le belief ξ^k_t — espace strictement plus riche.")
    print("                       Si ★ MIEUX que BF : preuve que T=1 apporte un gain réel.")
    print()
    print("  Centralisé         : Borne inférieure absolue (oracle omniscient).")
    print("  Gap +X%            : Surcoût par rapport à la borne centralisée.")
    print("  ★ meilleur décentralisé  |  ▼ pire parmi les 3 méthodes")
    print()
    print("  Structure T=1 (slide 22 du PDF) :")
    print("    Δ_t   = {Y^(K)_{1,t-1}, U^(K)_{1,t-1}}  (info publique à t)")
    print("    Λ^k_t = {Y^k_t}                            (info privée de k à t)")
    print("    Futur DP : Σ_{y^k_{t+1}, y^{-k}_t_réal.} P(·) · V_{t+1}(ξ_{t+1})")