from app import app
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split

from app.utils import preprocess_data, get_target_column
from app.ga import genetic_algorithm_feature_selection_fast, compute_feature_importance_table
from app.traditional import lasso_feature_selection, pca_feature_selection
from app.stats_methods import chi2_feature_selection
from app.evaluate import get_final_evaluation, _is_classification



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "dataset" not in request.files:
            flash("لم يتم إرسال ملف dataset")
            return redirect(request.url)
        f = request.files["dataset"]
        if f.filename == "":
            flash("لم تختَر ملفًا")
            return redirect(request.url)

        filename = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        f.save(filename)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
            else:
                df = pd.read_excel(filename)
        except Exception as e:
            flash("فشل في قراءة الملف: " + str(e))
            return redirect(request.url)

        try:
            df_proc, features, target = preprocess_data(df, missing_threshold=0.6, corr_threshold=0.0, verbose=False)
        except Exception as e:
            flash("خطأ في تجهيز البيانات: " + str(e))
            return redirect(request.url)

        try:
            is_clf = _is_classification(df_proc[target])
            stratify_opt = df_proc[target] if is_clf else None
            df_train, df_test = train_test_split(
                df_proc,
                test_size=0.25,
                random_state=42,
                stratify=stratify_opt
            )
        except Exception as e:
            flash(f"خطأ أثناء تقسيم البيانات (قد يكون بسبب قلة البيانات): {e}")
            return redirect(request.url)

        try:
            importance_table = compute_feature_importance_table(df_train, features, target, sample_n=20000)
            importance_top10 = importance_table.head(15).to_dict(orient='records')
        except Exception:
            importance_top10 = []

        num_features = len(features)
        if num_features > 400:
            prefilter_k = 300
        elif num_features > 200:
            prefilter_k = 200
        elif num_features > 100:
            prefilter_k = 120
        else:
            prefilter_k = None

        ga_params = {
            "pop_size": 40,
            "generations": 40,
            "min_features": 3,
            "max_features": min(30, max(1, int(num_features * 0.4))),
            "mutation_rate": 0.03,
            "elite_frac": 0.12,
            "alpha": 0.01,
            "beta": 0.08,
            "early_stopping": True,
            "patience": 8,
            "prefilter_k": prefilter_k,
            "sample_n_for_stats": 20000,
            "random_state": 42,
            "verbose": False,
            "use_model_fitness": False
        }

        t0 = time.time()
        try:
            ga_features, ga_details = genetic_algorithm_feature_selection_fast(df_train, features, target, **ga_params)
            ga_time = time.time() - t0
        except Exception as e:
            flash("خطأ أثناء تشغيل الخوارزمية الجينية: " + str(e))
            return redirect(request.url)

        try:
            lasso_features = lasso_feature_selection(df_train, features, target, alpha=0.01)
        except Exception:
            lasso_features = []
        try:
            pca_features = pca_feature_selection(df_train, features, n_components=5)
        except Exception:
            pca_features = []
        try:
            chi2_features = chi2_feature_selection(df_train, features, target, k=5)
        except Exception:
            chi2_features = []

        comparisons = []
        all_features = features.copy()

        methods = [
            ("الخوارزمية الجينية", ga_features),
            ("Lasso", lasso_features),
            ("PCA", pca_features),
            ("Chi2", chi2_features),
            ("All Features (Baseline)", all_features)
        ]

        for name, flist in methods:
            try:
                res = get_final_evaluation(df_train, df_test, flist, target)
                res.update({"method": name})
            except Exception as e:
                res = {"method": name, "num_features": len(flist) if flist else 0,
                       "metric": None, "score_mean": None, "score_std": None,
                       "time_seconds": 0.0, "error": str(e)}
            comparisons.append(res)

        chart_data = {
            'labels': [c['method'] for c in comparisons],
            'scores': [c['score_mean'] if c['score_mean'] is not None else 0 for c in comparisons],
            'feature_counts': [c['num_features'] for c in comparisons]
        }

        return render_template("results.html",
                               filename=os.path.basename(filename),
                               ga_features=ga_features,
                               ga_details=ga_details,
                               ga_time=ga_time,
                               lasso_features=lasso_features,
                               pca_features=pca_features,
                               chi2_features=chi2_features,
                               target=target,
                               num_features=num_features,
                               importance_top10=importance_top10,
                               comparisons=comparisons,
                               chart_data=chart_data)

    return render_template("index.html")
