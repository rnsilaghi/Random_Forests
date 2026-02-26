# Import packages
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.5f}'.format

# Define a path for import and export
path = r'C:\Users\rsila\OneDrive\Desktop\UMich\FIN 427\Project 1\Aggregate Dataset\Random Forrest\\'

# Import and view data
returns01 = pd.read_csv(path + 'Aggregate data 20260216_2204.csv')
returns01['month'] = pd.to_datetime(returns01['month'], format='%d-%b-%y')
print(returns01.dtypes)
print(returns01.head(10))
print(returns01.columns)

# -------------------------
# Define y, x, fn (same as your original)
# -------------------------
y = returns01['indadjret']
x = returns01[['lag1mcreal','finbm','finchgroa','finilliq','finsdage',
               'fingender','finnpm','finmom12','fincr','finos','finbeta','finnwca','finlnshchg',
               'finmom11','finroic','bmmiss','chgroamiss','illiqmiss','sdagemiss','gendermiss',
               'npmmiss','mom12miss','crmiss','osmiss','betamiss','nwcamiss','lnshchgmiss','mom11miss','roicmiss']]

fn = list(x.columns)

# =========================
# RANDOM FOREST: IN-SAMPLE SPECS + ALL-MONTH QUINTILES
# Base spec = your in-class parameters
# =========================

# ---- Baseline (class) settings you provided ----
base_params = dict(
    max_depth=3,
    min_weight_fraction_leaf=0.10,
    n_estimators=20,
    random_state=11610,
    max_features=10,
    bootstrap=True,
    max_samples=0.60,
    n_jobs=-1
)

# ---- "Edit them a little" via 4 nested loops (your choices) ----
# Loop dimensions: number of trees, max_features, max_samples, random_state
trees_list        = [20, 50, 100]           # tweak #trees
max_features_list = [8, 10, "sqrt"]         # tweak feature selection
max_samples_list  = [0.50, 0.60, 0.80]      # tweak sample selection per tree
seeds_list        = [11610, 2024]           # tweak randomness (optional but useful)
total_specs = len(trees_list) * len(max_features_list) * len(max_samples_list) * len(seeds_list)

# Storage
spec_rows = []
importances_all = []
quintile_overall_all = []
quintile_panel_all = []

spec_id = 0

# Fully in-sample: fit + score on full dataset
X = returns01[fn]
Y = returns01["indadjret"]

for n_trees in trees_list:
    for mf in max_features_list:
        for ms in max_samples_list:
            for seed in seeds_list:
                spec_id += 1
                print(f"[{spec_id}/{total_specs}] Running RF: trees={n_trees}, max_features={mf}, max_samples={ms}, seed={seed}")
                params = base_params.copy()
                params.update({
                    "n_estimators": n_trees,
                    "max_features": mf,
                    "max_samples": ms,
                    "random_state": seed
                })

                rf = RandomForestRegressor(**params)
                rfmodel = rf.fit(X, Y)

                # In-sample R^2 score
                insample_r2 = rfmodel.score(X, Y)

                # Predictions (keep the same returns02 naming style)
                returns02 = returns01.copy()
                pred_col = f"pred_indadjret_spec{spec_id}"
                returns02[pred_col] = rfmodel.predict(returns02[fn])

                # Feature importance (sorted)
                fi = pd.DataFrame({
                    "fn": fn,
                    "importance": rfmodel.feature_importances_
                }).sort_values("importance", ascending=False)

                fi["spec_id"] = spec_id
                fi["n_estimators"] = n_trees
                fi["max_features"] = str(mf)
                fi["max_samples"] = ms
                fi["random_state"] = seed
                fi["max_depth"] = params["max_depth"]
                fi["min_weight_fraction_leaf"] = params["min_weight_fraction_leaf"]

                importances_all.append(fi)

                # Save spec summary row
                spec_rows.append({
                    "spec_id": spec_id,
                    "insample_r2": insample_r2,
                    "n_estimators": n_trees,
                    "max_features": str(mf),
                    "max_samples": ms,
                    "random_state": seed,
                    "max_depth": params["max_depth"],
                    "min_weight_fraction_leaf": params["min_weight_fraction_leaf"]
                })

                # ---- Quintiles across ALL months (within-month quintiles on predicted returns) ----
                # Handle ties by ranking first to avoid qcut duplicate-edge errors
                def _qcut_ranked(s, q=5):
                    r = s.rank(method="first")
                    return pd.qcut(r, q, labels=False) + 1  # 1..5

                returns02["pred_q"] = returns02.groupby("month")[pred_col].transform(_qcut_ranked).astype(int)

                # Choose characteristics to summarize by quintile (edit as you like)
                char_cols = [
                    "indadjret",      # realized
                    pred_col,         # predicted
                    "lag1mcreal",
                    "finbm",
                    "finmom12",
                    "finbeta",
                    "finroic",
                    "finilliq",
                    "finlnshchg",
                    "fincr"
                ]
                # Keep only those that exist
                char_cols = [c for c in char_cols if c in returns02.columns]

                quintile_panel = (
                    returns02
                    .groupby(["month", "pred_q"], as_index=False)[char_cols]
                    .mean(numeric_only=True)
                )
                quintile_panel["spec_id"] = spec_id
                quintile_panel["n_estimators"] = n_trees
                quintile_panel["max_features"] = str(mf)
                quintile_panel["max_samples"] = ms
                quintile_panel["random_state"] = seed

                # Overall quintile characteristics: equal-weight months (mean of month-level means)
                quintile_overall = (
                    quintile_panel
                    .groupby("pred_q", as_index=False)[char_cols]
                    .mean(numeric_only=True)
                )
                quintile_overall["spec_id"] = spec_id
                quintile_overall["n_estimators"] = n_trees
                quintile_overall["max_features"] = str(mf)
                quintile_overall["max_samples"] = ms
                quintile_overall["random_state"] = seed

                quintile_panel_all.append(quintile_panel)
                quintile_overall_all.append(quintile_overall)

# Combine outputs
specs_df = pd.DataFrame(spec_rows).sort_values("insample_r2", ascending=False)
importances_df = pd.concat(importances_all, ignore_index=True)
quintile_panel_df = pd.concat(quintile_panel_all, ignore_index=True)
quintile_overall_df = pd.concat(quintile_overall_all, ignore_index=True)

print(specs_df.head(10))
print(importances_df.head(20))
print(quintile_overall_df.head(10))

# ---- Export ----
out_file = path + "RF_InSample_SpecSweep_And_AllMonthQuintiles.xlsx"
with pd.ExcelWriter(out_file) as writer:
    specs_df.to_excel(writer, sheet_name="Specs (In-sample R2)", index=False)
    importances_df.to_excel(writer, sheet_name="Feature Importances", index=False)
    quintile_overall_df.to_excel(writer, sheet_name="Quintiles Overall", index=False)
    quintile_panel_df.to_excel(writer, sheet_name="Quintiles MonthxQ", index=False)

print("Wrote:", out_file)
