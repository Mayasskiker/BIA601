
def fitness_vectorized(chrom_array, mi_all, corr_abs, alpha=0.01, beta=0.08, eps=1e-9):

    c = chrom_array.astype(float)
    n = c.sum()
    if n < 1:
        return 0.0

    sum_mi = float((mi_all * c).sum())

    tot = float(c @ (corr_abs @ c))
    diag = (c * np.diag(corr_abs)).sum()
    pair_sum = tot - diag
    pair_count = max(1.0, n * (n - 1.0))
    avg_abs_corr = pair_sum / pair_count

    penalty_count = alpha * n
    penalty_corr = beta * avg_abs_corr * n

    fitness = sum_mi - penalty_count - penalty_corr
    return float(fitness)

def evaluate_population(pop_matrix, mi_all, corr_abs, alpha=0.01, beta=0.08):

    pop_size = pop_matrix.shape[0]
    fitnesses = np.zeros(pop_size, dtype=float)
    for i in range(pop_size):
        fitnesses[i] = fitness_vectorized(pop_matrix[i], mi_all, corr_abs, alpha, beta)
    return fitnesses

def model_based_fitness(df, features, target_col, chrom, sample_n=5000, cv=3, random_state=42):

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np

    idx = np.where(chrom == 1)[0]
    if len(idx) == 0:
        return 0.0
    sel_features = [features[i] for i in idx]
    X = df[sel_features]
    y = df[target_col]
    if len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=random_state)
        X = df_sample[sel_features]
        y = df_sample[target_col]
    try:
        clf = LogisticRegression(max_iter=300, solver='liblinear')
        scoring = 'roc_auc' if y.nunique() == 2 else 'accuracy'
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        return float(np.mean(scores))
    except Exception:
        return 0.0

def genetic_algorithm_feature_selection_fast(df, features, target_col,
                                             pop_size=40, generations=40,
                                             min_features=3, max_features=None,
                                             mutation_rate=0.03, elite_frac=0.12,
                                             alpha=0.01, beta=0.08,
                                             early_stopping=True, patience=8,
                                             prefilter_k=None, sample_n_for_stats=20000,
                                             random_state=42, verbose=False,
                                             use_model_fitness=False):

    random.seed(random_state)
    np.random.seed(random_state)

    num_features = len(features)
    if max_features is None:
        max_features = max(1, int(num_features * 0.6))

    max_features = min(max_features, num_features)
    min_features = min(min_features, max_features)


    if prefilter_k is not None and prefilter_k < num_features:
        mi_full, _ = compute_precomputed_metrics(df, features, target_col, sample_n=sample_n_for_stats)
        top_idx = np.argsort(mi_full)[-prefilter_k:]
        selected_feat_list = [features[i] for i in top_idx]
        df = df[[*selected_feat_list, target_col]].copy()
        features = selected_feat_list
        num_features = len(features)
        if verbose:
            print(f"Prefilter: reduced features to top {num_features}")

    mi_all, corr_abs = compute_precomputed_metrics(df, features, target_col, sample_n=sample_n_for_stats)

    def init_pop_matrix(pop_size, num_features):
        mat = np.zeros((pop_size, num_features), dtype=np.int8)
        for i in range(pop_size):
            n = random.randint(min_features, min(max_features, num_features))
            idx = np.random.choice(num_features, size=n, replace=False)
            mat[i, idx] = 1
        return mat

    pop = init_pop_matrix(pop_size, num_features)
    best_fit = -1e9
    best_chrom = None
    no_improve = 0
    start_time = time.time()
    gen_ran = 0

    for gen in range(generations):
        gen_ran = gen + 1
        fitnesses = evaluate_population(pop, mi_all, corr_abs, alpha, beta)

        if use_model_fitness:
            top_tmp = np.argsort(fitnesses)[-10:]
            for idx in top_tmp:
                chrom = pop[idx]
                model_score = model_based_fitness(df, features, target_col, chrom)
                fitnesses[idx] = 0.7 * fitnesses[idx] + 0.3 * model_score

        idx_best = int(np.argmax(fitnesses))
        if fitnesses[idx_best] > best_fit + 1e-12:
            best_fit = float(fitnesses[idx_best])
            best_chrom = pop[idx_best].copy()
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            print(f"Gen {gen+1}/{generations} - best_fit={best_fit:.6f} - no_improve={no_improve}")

        if early_stopping and no_improve >= patience:
            if verbose:
                print("Early stopping triggered.")
            break

        elite_n = max(1, int(pop_size * elite_frac))
        sorted_idx = np.argsort(fitnesses)[::-1]
        new_pop_list = [pop[i].copy() for i in sorted_idx[:elite_n]]

        while len(new_pop_list) < pop_size:
            aspirants = np.random.randint(0, pop_size, size=3)
            p1_idx = aspirants[int(np.argmax(fitnesses[aspirants]))]
            aspirants = np.random.randint(0, pop_size, size=3)
            p2_idx = aspirants[int(np.argmax(fitnesses[aspirants]))]
            p1 = pop[p1_idx].copy()
            p2 = pop[p2_idx].copy()

            if num_features > 1:
                pt = random.randint(1, num_features - 1)
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1 = p1.copy()
                c2 = p2.copy()

            mask1 = (np.random.rand(num_features) < mutation_rate)
            mask2 = (np.random.rand(num_features) < mutation_rate)
            c1 = np.bitwise_xor(c1, mask1.astype(np.int8))
            c2 = np.bitwise_xor(c2, mask2.astype(np.int8))

            for c in (c1, c2):
                s = c.sum()
                if s < min_features:
                    zeros = np.where(c == 0)[0]
                    need = int(min_features - s)
                    if len(zeros) > 0:
                        choose = np.random.choice(zeros, size=min(need, len(zeros)), replace=False)
                        c[choose] = 1
                elif s > max_features:
                    ones = np.where(c == 1)[0]
                    remove = int(s - max_features)
                    if remove > 0 and len(ones) > 0:
                        choose = np.random.choice(ones, size=min(remove, len(ones)), replace=False)
                        c[choose] = 0

            new_pop_list.append(c1)
            if len(new_pop_list) < pop_size:
                new_pop_list.append(c2)

        pop = np.array(new_pop_list, dtype=np.int8)

    elapsed = time.time() - start_time
    chosen_idx = np.where(best_chrom == 1)[0].tolist() if best_chrom is not None else []
    selected_features = [features[i] for i in chosen_idx]
    details = {
        "best_fitness": best_fit,
        "generations_ran": gen_ran,
        "time_seconds": elapsed,
        "num_features_initial": len(features),
        "num_selected": len(selected_features)
    }
    return selected_features, details
def fitness_vectorized(chrom_array, mi_all, corr_abs, alpha=0.01, beta=0.08, eps=1e-9):

    c = chrom_array.astype(float)
    n = c.sum()
    if n < 1:
        return 0.0

    sum_mi = float((mi_all * c).sum())

    tot = float(c @ (corr_abs @ c))
    diag = (c * np.diag(corr_abs)).sum()
    pair_sum = tot - diag
    pair_count = max(1.0, n * (n - 1.0))
    avg_abs_corr = pair_sum / pair_count

    penalty_count = alpha * n
    penalty_corr = beta * avg_abs_corr * n

    fitness = sum_mi - penalty_count - penalty_corr
    return float(fitness)

def evaluate_population(pop_matrix, mi_all, corr_abs, alpha=0.01, beta=0.08):

    pop_size = pop_matrix.shape[0]
    fitnesses = np.zeros(pop_size, dtype=float)
    for i in range(pop_size):
        fitnesses[i] = fitness_vectorized(pop_matrix[i], mi_all, corr_abs, alpha, beta)
    return fitnesses

def model_based_fitness(df, features, target_col, chrom, sample_n=5000, cv=3, random_state=42):

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np

    idx = np.where(chrom == 1)[0]
    if len(idx) == 0:
        return 0.0
    sel_features = [features[i] for i in idx]
    X = df[sel_features]
    y = df[target_col]
    if len(df) > sample_n:
        df_sample = df.sample(n=sample_n, random_state=random_state)
        X = df_sample[sel_features]
        y = df_sample[target_col]
    try:
        clf = LogisticRegression(max_iter=300, solver='liblinear')
        scoring = 'roc_auc' if y.nunique() == 2 else 'accuracy'
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        return float(np.mean(scores))
    except Exception:
        return 0.0

def genetic_algorithm_feature_selection_fast(df, features, target_col,
                                             pop_size=40, generations=40,
                                             min_features=3, max_features=None,
                                             mutation_rate=0.03, elite_frac=0.12,
                                             alpha=0.01, beta=0.08,
                                             early_stopping=True, patience=8,
                                             prefilter_k=None, sample_n_for_stats=20000,
                                             random_state=42, verbose=False,
                                             use_model_fitness=False):

    random.seed(random_state)
    np.random.seed(random_state)

    num_features = len(features)
    if max_features is None:
        max_features = max(1, int(num_features * 0.6))

    max_features = min(max_features, num_features)
    min_features = min(min_features, max_features)


    if prefilter_k is not None and prefilter_k < num_features:
        mi_full, _ = compute_precomputed_metrics(df, features, target_col, sample_n=sample_n_for_stats)
        top_idx = np.argsort(mi_full)[-prefilter_k:]
        selected_feat_list = [features[i] for i in top_idx]
        df = df[[*selected_feat_list, target_col]].copy()
        features = selected_feat_list
        num_features = len(features)
        if verbose:
            print(f"Prefilter: reduced features to top {num_features}")

    mi_all, corr_abs = compute_precomputed_metrics(df, features, target_col, sample_n=sample_n_for_stats)

    def init_pop_matrix(pop_size, num_features):
        mat = np.zeros((pop_size, num_features), dtype=np.int8)
        for i in range(pop_size):
            n = random.randint(min_features, min(max_features, num_features))
            idx = np.random.choice(num_features, size=n, replace=False)
            mat[i, idx] = 1
        return mat

    pop = init_pop_matrix(pop_size, num_features)
    best_fit = -1e9
    best_chrom = None
    no_improve = 0
    start_time = time.time()
    gen_ran = 0

    for gen in range(generations):
        gen_ran = gen + 1
        fitnesses = evaluate_population(pop, mi_all, corr_abs, alpha, beta)

        if use_model_fitness:
            top_tmp = np.argsort(fitnesses)[-10:]
            for idx in top_tmp:
                chrom = pop[idx]
                model_score = model_based_fitness(df, features, target_col, chrom)
                fitnesses[idx] = 0.7 * fitnesses[idx] + 0.3 * model_score

        idx_best = int(np.argmax(fitnesses))
        if fitnesses[idx_best] > best_fit + 1e-12:
            best_fit = float(fitnesses[idx_best])
            best_chrom = pop[idx_best].copy()
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            print(f"Gen {gen+1}/{generations} - best_fit={best_fit:.6f} - no_improve={no_improve}")

        if early_stopping and no_improve >= patience:
            if verbose:
                print("Early stopping triggered.")
            break

        elite_n = max(1, int(pop_size * elite_frac))
        sorted_idx = np.argsort(fitnesses)[::-1]
        new_pop_list = [pop[i].copy() for i in sorted_idx[:elite_n]]

        while len(new_pop_list) < pop_size:
            aspirants = np.random.randint(0, pop_size, size=3)
            p1_idx = aspirants[int(np.argmax(fitnesses[aspirants]))]
            aspirants = np.random.randint(0, pop_size, size=3)
            p2_idx = aspirants[int(np.argmax(fitnesses[aspirants]))]
            p1 = pop[p1_idx].copy()
            p2 = pop[p2_idx].copy()

            if num_features > 1:
                pt = random.randint(1, num_features - 1)
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1 = p1.copy()
                c2 = p2.copy()

            mask1 = (np.random.rand(num_features) < mutation_rate)
            mask2 = (np.random.rand(num_features) < mutation_rate)
            c1 = np.bitwise_xor(c1, mask1.astype(np.int8))
            c2 = np.bitwise_xor(c2, mask2.astype(np.int8))

            for c in (c1, c2):
                s = c.sum()
                if s < min_features:
                    zeros = np.where(c == 0)[0]
                    need = int(min_features - s)
                    if len(zeros) > 0:
                        choose = np.random.choice(zeros, size=min(need, len(zeros)), replace=False)
                        c[choose] = 1
                elif s > max_features:
                    ones = np.where(c == 1)[0]
                    remove = int(s - max_features)
                    if remove > 0 and len(ones) > 0:
                        choose = np.random.choice(ones, size=min(remove, len(ones)), replace=False)
                        c[choose] = 0

            new_pop_list.append(c1)
            if len(new_pop_list) < pop_size:
                new_pop_list.append(c2)

        pop = np.array(new_pop_list, dtype=np.int8)

    elapsed = time.time() - start_time
    chosen_idx = np.where(best_chrom == 1)[0].tolist() if best_chrom is not None else []
    selected_features = [features[i] for i in chosen_idx]
    details = {
        "best_fitness": best_fit,
        "generations_ran": gen_ran,
        "time_seconds": elapsed,
        "num_features_initial": len(features),
        "num_selected": len(selected_features)
    }
    return selected_features, details