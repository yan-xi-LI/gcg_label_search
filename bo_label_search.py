import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import main as gcg


# ============================================================
# BO Config
# ============================================================
# This script reuses the fixed project layout from main.py:
#   ./train/pos, ./train/neg, ./dev/pos, ./dev/neg, ./test/pos, ./test/neg
# and reuses the exact same evaluation objective implemented in main.py.

SEARCHABLE_SHORTLIST_SIZE = 256   # shortlist labels first, then BO searches pairs on this shortlist
PAIR_POOL_LIMIT = None            # set e.g. 50000 to randomly subsample pair universe if needed
N_INIT_RANDOM = 20
INIT_STRATEGY = "max_separation"   # "random" or "max_separation"
N_BO_ITERS = 40
PCA_DIM = 32
RF_NUM_TREES = 200
UCB_BETA = 1.5
OUTPUT_JSONL = "bo_search_history.jsonl"
OUTPUT_BEST_JSON = "bo_best_result.json"


# ============================================================
# Small utilities
# ============================================================


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleRandomForestRegressor:
    """
    A small dependency-free random forest regressor.
    We only need two things for BO:
      1. predictive mean
      2. predictive uncertainty proxy via ensemble std
    """

    def __init__(
        self,
        n_trees: int = 200,
        max_depth: int = 8,
        min_leaf: int = 2,
        feature_subsample_ratio: float = 0.5,
        bootstrap_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.feature_subsample_ratio = feature_subsample_ratio
        self.bootstrap_ratio = bootstrap_ratio
        self.seed = seed
        self.trees: List[Dict] = []
        self.rng = np.random.RandomState(seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.trees = []
        n, d = x.shape
        if n == 0:
            raise ValueError("Cannot fit random forest with no data.")

        for tree_idx in range(self.n_trees):
            sample_size = max(2, int(math.ceil(self.bootstrap_ratio * n)))
            row_idx = self.rng.choice(n, size=sample_size, replace=True)
            feat_count = max(1, int(math.ceil(self.feature_subsample_ratio * d)))
            feat_idx = np.sort(self.rng.choice(d, size=feat_count, replace=False))
            tree = self._build_tree(x[row_idx][:, feat_idx], y[row_idx], depth=0)
            self.trees.append({"feat_idx": feat_idx, "tree": tree})

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trees:
            raise ValueError("Random forest has not been fit yet.")
        preds = []
        for item in self.trees:
            cur_x = x[:, item["feat_idx"]]
            preds.append(self._predict_tree_batch(item["tree"], cur_x))
        pred_mat = np.stack(preds, axis=0)
        return pred_mat.mean(axis=0), pred_mat.std(axis=0)

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> Dict:
        if (
            depth >= self.max_depth
            or len(y) <= self.min_leaf * 2
            or np.allclose(y, y[0])
        ):
            return {"leaf": True, "value": float(np.mean(y))}

        best_feature = None
        best_threshold = None
        best_loss = float("inf")

        n, d = x.shape
        for feature in range(d):
            vals = x[:, feature]
            if np.allclose(vals, vals[0]):
                continue

            unique_vals = np.unique(vals)
            if unique_vals.shape[0] > 16:
                quantiles = np.linspace(0.1, 0.9, 9)
                thresholds = np.quantile(unique_vals, quantiles)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thr in thresholds:
                left = vals <= thr
                right = ~left
                if left.sum() < self.min_leaf or right.sum() < self.min_leaf:
                    continue
                left_y = y[left]
                right_y = y[right]
                loss = left_y.var() * len(left_y) + right_y.var() * len(right_y)
                if loss < best_loss:
                    best_loss = float(loss)
                    best_feature = feature
                    best_threshold = float(thr)

        if best_feature is None:
            return {"leaf": True, "value": float(np.mean(y))}

        vals = x[:, best_feature]
        left = vals <= best_threshold
        right = ~left
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(x[left], y[left], depth + 1),
            "right": self._build_tree(x[right], y[right], depth + 1),
        }

    def _predict_tree_batch(self, tree: Dict, x: np.ndarray) -> np.ndarray:
        out = np.empty(x.shape[0], dtype=np.float32)
        for i in range(x.shape[0]):
            out[i] = self._predict_tree_one(tree, x[i])
        return out

    def _predict_tree_one(self, tree: Dict, row: np.ndarray) -> float:
        cur = tree
        while not cur["leaf"]:
            if row[cur["feature"]] <= cur["threshold"]:
                cur = cur["left"]
            else:
                cur = cur["right"]
        return float(cur["value"])


# ============================================================
# BO-specific feature construction
# ============================================================


def get_token_embedding_matrix(model, token_ids: Sequence[int]) -> torch.Tensor:
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    return emb[torch.tensor(list(token_ids), dtype=torch.long)]



def pca_reduce(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    rank = min(out_dim, x.shape[0] - 1, x.shape[1])
    if rank <= 0:
        return x[:, :1]
    u, s, _ = torch.pca_lowrank(x, q=rank, center=False)
    return u[:, :rank] * s[:rank]



def build_pair_feature(pos_vec: np.ndarray, neg_vec: np.ndarray) -> np.ndarray:
    diff = pos_vec - neg_vec
    abs_diff = np.abs(diff)
    prod = pos_vec * neg_vec
    cosine_num = float(np.dot(pos_vec, neg_vec))
    cosine_den = float(np.linalg.norm(pos_vec) * np.linalg.norm(neg_vec)) + 1e-8
    cosine = np.array([cosine_num / cosine_den], dtype=np.float32)
    return np.concatenate([pos_vec, neg_vec, diff, abs_diff, prod, cosine], axis=0).astype(np.float32)



def shortlist_candidate_ids(
    tokenizer,
    searchable_ids: List[int],
    shortlist_size: int,
) -> List[int]:
    # Prefer diverse random shortlist over scanning full pair universe.
    if shortlist_size >= len(searchable_ids):
        return searchable_ids[:]

    # Try to mix in both plain and leading-space word pieces.
    with_space = []
    without_space = []
    for tid in searchable_ids:
        piece = gcg.token_piece_str(tokenizer, tid)
        if piece.startswith(" "):
            with_space.append(tid)
        else:
            without_space.append(tid)

    half = shortlist_size // 2
    chosen = []
    if with_space:
        chosen.extend(random.sample(with_space, k=min(len(with_space), half)))
    remaining = shortlist_size - len(chosen)
    if without_space:
        chosen.extend(random.sample(without_space, k=min(len(without_space), remaining)))
    remaining = shortlist_size - len(chosen)
    if remaining > 0:
        leftovers = [tid for tid in searchable_ids if tid not in set(chosen)]
        chosen.extend(random.sample(leftovers, k=min(len(leftovers), remaining)))

    return list(dict.fromkeys(chosen))



def build_pair_universe(candidate_ids: List[int], pair_limit: int = None) -> List[Tuple[int, int]]:
    pairs = [(p, n) for p in candidate_ids for n in candidate_ids if p != n]
    if pair_limit is not None and len(pairs) > pair_limit:
        pairs = random.sample(pairs, k=pair_limit)
    return pairs




def select_init_indices_max_separation(pair_features: np.ndarray, k: int, seed: int) -> List[int]:
    """
    Greedy farthest-point sampling in pair-feature space.
    This makes BO warm-start points semantically diverse instead of purely random.
    """
    n = pair_features.shape[0]
    if k >= n:
        return list(range(n))
    rng = np.random.RandomState(seed)

    # Start from one random point to avoid a deterministic bias toward any fixed region.
    first = int(rng.randint(0, n))
    selected = [first]

    # Track each point's distance to the currently selected set.
    diff = pair_features - pair_features[first:first + 1]
    min_sq_dist = np.sum(diff * diff, axis=1)
    min_sq_dist[first] = -1.0

    while len(selected) < k:
        next_idx = int(np.argmax(min_sq_dist))
        selected.append(next_idx)
        diff = pair_features - pair_features[next_idx:next_idx + 1]
        sq_dist = np.sum(diff * diff, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, sq_dist)
        min_sq_dist[selected] = -1.0

    return selected


def choose_initial_indices(pair_features: np.ndarray, k: int, strategy: str, seed: int) -> List[int]:
    if k <= 0:
        return []
    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(range(pair_features.shape[0]), k=min(k, pair_features.shape[0]))
    if strategy == "max_separation":
        return select_init_indices_max_separation(pair_features, k=min(k, pair_features.shape[0]), seed=seed)
    raise ValueError(f"Unknown INIT_STRATEGY: {strategy}")

# ============================================================
# Evaluation + logging
# ============================================================


def evaluate_pair_with_splits(
    model,
    tokenizer,
    train_data,
    dev_data,
    test_data,
    pos_tid: int,
    neg_tid: int,
) -> Dict:
    pos_tids = [pos_tid]
    neg_tids = [neg_tid]
    train_stats = gcg.evaluate_candidate_pair(model, tokenizer, train_data, pos_tids, neg_tids)
    dev_stats = gcg.evaluate_candidate_pair(model, tokenizer, dev_data, pos_tids, neg_tids)
    test_stats = gcg.evaluate_candidate_pair(model, tokenizer, test_data, pos_tids, neg_tids)
    return {
        "pos_tid": pos_tid,
        "neg_tid": neg_tid,
        "pos_label": gcg.label_str(tokenizer, pos_tids),
        "neg_label": gcg.label_str(tokenizer, neg_tids),
        "train": asdict(train_stats),
        "dev": asdict(dev_stats),
        "test": asdict(test_stats),
    }



def write_jsonl(path: str, row: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Main BO loop
# ============================================================


def main() -> None:
    set_all_seeds(gcg.SEED)

    print("[INFO] Loading datasets from fixed folders...")
    train_data = gcg.load_train_folder(gcg.TRAIN_DIR)
    dev_data = gcg.load_train_folder(gcg.DEV_DIR)
    test_data = gcg.load_train_folder(gcg.TEST_DIR)
    print(f"[INFO] Loaded train={len(train_data)} samples from {gcg.TRAIN_DIR}")
    print(f"[INFO] Loaded dev={len(dev_data)} samples from {gcg.DEV_DIR}")
    print(f"[INFO] Loaded test={len(test_data)} samples from {gcg.TEST_DIR}")

    print("[INFO] Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(gcg.MODEL_NAME_OR_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        gcg.MODEL_NAME_OR_PATH,
        torch_dtype=gcg.DTYPE,
        device_map=None,
    ).to(gcg.DEVICE)
    model.eval()

    vocab_size = model.get_output_embeddings().weight.shape[0]
    searchable_ids = gcg.get_searchable_token_ids(tokenizer, vocab_size)
    candidate_ids = shortlist_candidate_ids(tokenizer, searchable_ids, SEARCHABLE_SHORTLIST_SIZE)
    pair_universe = build_pair_universe(candidate_ids, PAIR_POOL_LIMIT)

    print(f"[INFO] Searchable token count: {len(searchable_ids)} / {vocab_size}")
    print(f"[INFO] BO shortlist size: {len(candidate_ids)}")
    print(f"[INFO] BO pair universe size: {len(pair_universe)}")

    token_emb = get_token_embedding_matrix(model, candidate_ids)
    token_emb = pca_reduce(token_emb, PCA_DIM)
    token_emb_np = token_emb.numpy().astype(np.float32)
    id_to_idx = {tid: i for i, tid in enumerate(candidate_ids)}

    pair_features = []
    for pos_tid, neg_tid in pair_universe:
        pair_features.append(
            build_pair_feature(
                token_emb_np[id_to_idx[pos_tid]],
                token_emb_np[id_to_idx[neg_tid]],
            )
        )
    pair_features = np.stack(pair_features, axis=0)

    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    init_count = min(N_INIT_RANDOM, len(pair_universe))
    init_indices = choose_initial_indices(pair_features, k=init_count, strategy=INIT_STRATEGY, seed=gcg.SEED)
    evaluated = set()
    history = []
    best_row = None

    print("=" * 120)
    print(f"[INIT EVALUATIONS | strategy={INIT_STRATEGY}]")
    print("=" * 120)
    for idx in init_indices:
        pos_tid, neg_tid = pair_universe[idx]
        row = evaluate_pair_with_splits(model, tokenizer, train_data, dev_data, test_data, pos_tid, neg_tid)
        row["phase"] = "init"
        row["init_strategy"] = INIT_STRATEGY
        row["iter"] = len(history)
        row["pair_index"] = idx
        history.append(row)
        evaluated.add(idx)
        write_jsonl(OUTPUT_JSONL, row)

        if best_row is None or row["dev"]["objective"] > best_row["dev"]["objective"]:
            best_row = row

        print(
            f"[INIT {len(history):02d}] pos={row['pos_label']!r} neg={row['neg_label']!r}  "
            f"train_obj={row['train']['objective']:.4f}  dev_obj={row['dev']['objective']:.4f}  "
            f"dev_inj_acc={row['dev']['inj_acc']:.4f}  dev_asr={row['dev']['asr_flip_to_pos']:.4f}"
        )

    print("\n" + "=" * 120)
    print("[BAYESIAN OPTIMIZATION LOOP]")
    print("=" * 120)
    for bo_iter in range(1, N_BO_ITERS + 1):
        remaining = [i for i in range(len(pair_universe)) if i not in evaluated]
        if not remaining:
            print("[INFO] Pair universe exhausted. Stopping early.")
            break

        x_train = np.stack([pair_features[i] for i in evaluated], axis=0)
        y_train = np.array([history_item["dev"]["objective"] for history_item in history], dtype=np.float32)

        rf = SimpleRandomForestRegressor(
            n_trees=RF_NUM_TREES,
            max_depth=8,
            min_leaf=2,
            feature_subsample_ratio=0.5,
            bootstrap_ratio=0.8,
            seed=gcg.SEED + bo_iter,
        )
        rf.fit(x_train, y_train)

        x_cand = pair_features[remaining]
        mu, sigma = rf.predict(x_cand)
        acquisition = mu + UCB_BETA * sigma
        best_local = int(np.argmax(acquisition))
        next_idx = remaining[best_local]

        pos_tid, neg_tid = pair_universe[next_idx]
        row = evaluate_pair_with_splits(model, tokenizer, train_data, dev_data, test_data, pos_tid, neg_tid)
        row["phase"] = "bo"
        row["iter"] = len(history)
        row["pair_index"] = next_idx
        row["pred_mean"] = float(mu[best_local])
        row["pred_std"] = float(sigma[best_local])
        row["acquisition"] = float(acquisition[best_local])

        history.append(row)
        evaluated.add(next_idx)
        write_jsonl(OUTPUT_JSONL, row)

        improved = False
        if best_row is None or row["dev"]["objective"] > best_row["dev"]["objective"]:
            best_row = row
            improved = True

        print(
            f"[BO {bo_iter:02d}/{N_BO_ITERS}] pos={row['pos_label']!r} neg={row['neg_label']!r}  "
            f"pred_mu={row['pred_mean']:.4f} pred_std={row['pred_std']:.4f} acq={row['acquisition']:.4f}  "
            f"train_obj={row['train']['objective']:.4f} dev_obj={row['dev']['objective']:.4f}  "
            f"dev_inj_acc={row['dev']['inj_acc']:.4f} dev_asr={row['dev']['asr_flip_to_pos']:.4f}"
            + ("  <-- NEW BEST" if improved else "")
        )

    print("\n" + "#" * 120)
    print("[BEST ON DEV VIA BO]")
    print(f"pos: {best_row['pos_label']!r} (tid={best_row['pos_tid']})")
    print(f"neg: {best_row['neg_label']!r} (tid={best_row['neg_tid']})")
    print(f"train objective: {best_row['train']['objective']:.6f}")
    print(f"dev objective  : {best_row['dev']['objective']:.6f}")
    print(f"test objective : {best_row['test']['objective']:.6f}")
    print(f"dev clean_acc  : {best_row['dev']['clean_acc']:.6f}")
    print(f"dev inj_acc    : {best_row['dev']['inj_acc']:.6f}")
    print(f"dev ASR        : {best_row['dev']['asr_flip_to_pos']:.6f}")
    print("#" * 120)

    with open(OUTPUT_BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote BO history to {OUTPUT_JSONL}")
    print(f"[INFO] Wrote best result to {OUTPUT_BEST_JSON}")


if __name__ == "__main__":
    main()
