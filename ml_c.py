import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers


# =========================
# CONFIG
# =========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_CSV  = "data/backtest_data/binanceGlobalAccounts_BTC_binance_5m_with_factors.csv"
OUTPUT_CSV = "data/backtest_data/strategy_return_with_preds.csv"
TIME_COL   = "open_time"

# label selection / creation
PREFERRED_LABELS = ["label", "label_SM_minmax", "future_ret", "ret_fwd_1bar"]

# ---- CHANGED: make horizon a LIST for grid search ----
LABEL_HORIZON_LIST = [288, 576, 1440, 2880, 4000]  # example spans, edit as you like

# train/val/test split ratios (chronological)
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15  # test is remainder

# AutoEncoder
LATENT_DIM = 16
AE_HIDDEN = (128, 64)
AE_DROPOUT = 0.10
AE_NOISE   = 0.05
AE_LR      = 1e-3
AE_EPOCHS  = 200
AE_BATCH   = 256

# MLP regressor
MLP_HIDDEN = (64, 32)
MLP_DROPOUT = 0.20
MLP_LR      = 1e-3
MLP_EPOCHS  = 200
MLP_BATCH   = 256

# XGB regressor (only if xgboost installed)
XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist",
)
XGB_EARLY_STOP = 50
XGB_EVAL_METRIC = "rmse"

# stacking stability
STACK_RIDGE_ALPHA = 1.0
# =========================


def build_autoencoder(input_dim, latent_dim=16, hidden=(128, 64), dropout=0.1, noise=0.05, lr=1e-3):
    inp = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inp)
    x = layers.GaussianNoise(noise)(x)

    # Encoder
    for h in hidden:
        x = layers.Dense(h)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.Dropout(dropout)(x)

    z = layers.Dense(latent_dim, name="latent")(x)

    # Decoder (mirror)
    x = z
    for h in reversed(hidden):
        x = layers.Dense(h)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.Dropout(dropout)(x)

    out = layers.Dense(input_dim, name="recon")(x)

    ae = keras.Model(inp, out, name="autoencoder")
    enc = keras.Model(inp, z, name="encoder")

    ae.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return ae, enc


def build_mlp(input_dim, hidden=(64, 32), dropout=0.2, lr=1e-3):
    inp = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inp)
    for h in hidden:
        x = layers.Dense(h)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out, name="mlp_reg")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model


# ---- MIN CHANGE: add optional force_create so grid search always uses horizon label ----
def pick_or_create_label(df: pd.DataFrame, label_horizon_bars: int, force_create: bool = True):
    label_col = None

    if not force_create:
        for c in PREFERRED_LABELS:
            if c in df.columns:
                label_col = c
                break

    if label_col is None:
        h = int(label_horizon_bars)
        col_name = f"ret_fwd_{h}"
        df[col_name] = np.log(df["close"].shift(-h) / df["close"])
        label_col = col_name
        print(f"[INFO] Created label '{label_col}' with horizon {h} bars.")
    else:
        print(f"[INFO] Using existing label '{label_col}'.")

    # drop last H rows where label is NaN after shift(-h)
    h = int(label_horizon_bars) if label_col.startswith("ret_fwd_") else 0
    if h > 0:
        df = df.iloc[:-h].copy().reset_index(drop=True)

    return df, label_col


def get_feature_cols(df: pd.DataFrame, label_col: str):
    exclude = {
        TIME_COL, "close_time", "timestamp", "openDate",
        label_col
    }
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude]
    return feature_cols


def chronological_split(X, y, train_frac=0.7, val_frac=0.15):
    N = len(X)
    train_end = int(N * train_frac)
    val_end   = int(N * (train_frac + val_frac))

    X_train_raw = X[:train_end]
    y_train = y[:train_end]

    X_val_raw = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test_raw = X[val_end:]
    y_test = y[val_end:]

    return (X_train_raw, y_train,
            X_val_raw, y_val,
            X_test_raw, y_test,
            train_end, val_end)


def eval_regression(y_true, y_pred, name="Model"):
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)

    if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        corr = np.corrcoef(y_pred, y_true)[0, 1]
    else:
        corr = np.nan

    print(f"[{name}] RMSE={rmse:.6f}, MAE={mae:.6f}, Corr={corr:.6f}")
    return rmse, mae, corr


def learn_simplex_weights(pred_mlp_val, pred_xgb_val, y_val, alpha=1.0):
    A = np.vstack([pred_mlp_val, pred_xgb_val]).T
    stacker = Ridge(alpha=alpha, fit_intercept=True, random_state=SEED)
    stacker.fit(A, y_val)
    w = stacker.coef_.astype(np.float64)

    w = np.clip(w, 0.0, None)
    s = w.sum()
    if not np.isfinite(s) or s <= 1e-12:
        w = np.array([0.5, 0.5], dtype=np.float64)
    else:
        w = w / s
    return w


def overall_ic(df, label_col, pred_col):
    d = df[[label_col, pred_col]].dropna()
    ic_p = d[pred_col].corr(d[label_col], method="pearson")
    ic_s = d[pred_col].corr(d[label_col], method="spearman")
    return ic_p, ic_s


def rolling_ic(df, label_col, pred_col, window=288):
    d = df[[label_col, pred_col]].dropna()
    roll_p = d[pred_col].rolling(window).corr(d[label_col])
    pred_rank  = d[pred_col].rank()
    label_rank = d[label_col].rank()
    roll_s = pred_rank.rolling(window).corr(label_rank)
    return pd.DataFrame({"ic_pearson_roll": roll_p, "ic_spearman_roll": roll_s})


# ---- ADDED: daily IC series + stats (std, skew, t-test) ----
def ic_series_by_day(test_df, label_col, pred_col):
    d = test_df[[TIME_COL, label_col, pred_col]].dropna().copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL])
    d["date"] = d[TIME_COL].dt.floor("1D")

    ics = []
    for dt, g in d.groupby("date"):
        if len(g) < 3:
            ics.append((dt, np.nan))
        else:
            ics.append((dt, g[pred_col].corr(g[label_col])))
    return pd.Series({k: v for k, v in ics}).sort_index()


def ic_stats(ic_s: pd.Series):
    x = ic_s.dropna().values
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, skew=np.nan, t=np.nan)

    mean = x.mean()
    std  = x.std(ddof=1) if n > 1 else np.nan
    skew = pd.Series(x).skew()

    if n > 1 and std > 1e-12:
        t_stat = mean / (std / np.sqrt(n))
    else:
        t_stat = np.nan

    return dict(n=n, mean=mean, std=std, skew=skew, t=t_stat)


# ---- ADDED: time-series stratification on TEST ----
def time_series_stratification(test_df, label_col, pred_col, n_bins=5):
    d = test_df[[label_col, pred_col]].dropna().copy()
    if len(d) == 0:
        print(f"[Stratify][{pred_col}] no data")
        return None

    d["layer"] = pd.qcut(d[pred_col], n_bins, labels=False, duplicates="drop")
    layer_mean = d.groupby("layer")[label_col].mean()

    top = layer_mean.iloc[-1]
    bot = layer_mean.iloc[0]
    spread = top - bot

    print(f"[Stratify][{pred_col}] mean fwd ret per layer:")
    print(layer_mean)
    print(f"[Stratify][{pred_col}] top-bottom spread = {spread:.6f}")
    return layer_mean


# ---- CHANGED: remove all plot code, add IC stats + stratification ----
def ic_by_day(df, test_idx, preds, label_col, pred_name="pred", horizon_bars=288):
    test_df = df.iloc[test_idx].copy()
    test_df[pred_name] = preds

    ic_p, ic_s = overall_ic(test_df, label_col, pred_name)
    print(f"[IC][{pred_name}] Overall Pearson={ic_p:.6f}, Spearman={ic_s:.6f}")

    ic_roll = rolling_ic(test_df, label_col, pred_name, window=horizon_bars)
    mean_p = ic_roll["ic_pearson_roll"].mean()
    mean_s = ic_roll["ic_spearman_roll"].mean()
    last_p = ic_roll["ic_pearson_roll"].iloc[-1]
    last_s = ic_roll["ic_spearman_roll"].iloc[-1]
    print(f"[IC][{pred_name}] Rolling({horizon_bars}) mean Pearson={mean_p:.6f}, Spearman={mean_s:.6f}")
    print(f"[IC][{pred_name}] Rolling({horizon_bars}) last Pearson={last_p:.6f}, Spearman={last_s:.6f}")

    # daily IC stats
    ic_day = ic_series_by_day(test_df, label_col, pred_name)
    st = ic_stats(ic_day)
    print(f"[IC][{pred_name}] Daily IC stats: n={st['n']}, mean={st['mean']:.6f}, "
          f"std={st['std']:.6f}, skew={st['skew']:.6f}, t={st['t']:.3f}")

    # time-series stratification
    time_series_stratification(test_df, label_col, pred_name, n_bins=5)

    return ic_roll, st


def run_one_horizon(label_horizon_bars: int):
    # -------- Load --------
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    print("\n==============================")
    print(f"[GRID] LABEL_HORIZON_BARS = {label_horizon_bars}")
    print("[INFO] Loaded:", df.shape)

    # -------- Label --------
    df, label_col = pick_or_create_label(df, label_horizon_bars, force_create=True)

    # -------- Features --------
    feature_cols = get_feature_cols(df, label_col)
    print("[INFO] Num features:", len(feature_cols))

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[label_col].values.astype(np.float32)

    # -------- Split --------
    (X_train_raw, y_train,
     X_val_raw, y_val,
     X_test_raw, y_test,
     train_end, val_end) = chronological_split(X_raw, y_raw, TRAIN_FRAC, VAL_FRAC)

    # -------- Scale on train only --------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    print("[INFO] Split shapes:",
          X_train.shape, X_val.shape, X_test.shape)

    # -------- AE --------
    ae, encoder = build_autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=LATENT_DIM,
        hidden=AE_HIDDEN,
        dropout=AE_DROPOUT,
        noise=AE_NOISE,
        lr=AE_LR
    )

    ae_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    print("[INFO] Training AutoEncoder...")
    ae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH,
        callbacks=ae_callbacks,
        verbose=1
    )

    Z_train = encoder.predict(X_train, verbose=0)
    Z_val   = encoder.predict(X_val, verbose=0)
    Z_test  = encoder.predict(X_test, verbose=0)
    print("[INFO] Latent shapes:",
          Z_train.shape, Z_val.shape, Z_test.shape)

    # -------- MLP --------
    mlp = build_mlp(
        input_dim=Z_train.shape[1],
        hidden=MLP_HIDDEN,
        dropout=MLP_DROPOUT,
        lr=MLP_LR
    )

    mlp_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    print("[INFO] Training MLP regressor...")
    mlp.fit(
        Z_train, y_train,
        validation_data=(Z_val, y_val),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH,
        callbacks=mlp_callbacks,
        verbose=1
    )

    pred_mlp_val  = mlp.predict(Z_val).reshape(-1)
    pred_mlp_test = mlp.predict(Z_test).reshape(-1)
    rmse_mlp, mae_mlp, corr_mlp = eval_regression(y_test, pred_mlp_test, "MLP")

    # -------- XGB (optional) --------
    pred_xgb_val = pred_xgb_test = None
    rmse_xgb = mae_xgb = corr_xgb = np.nan
    if xgb is not None:
        print("[INFO] Training XGB regressor...")
        try:
            xgb_model = xgb.XGBRegressor(
                **XGB_PARAMS,
                eval_metric=XGB_EVAL_METRIC,
                early_stopping_rounds=XGB_EARLY_STOP,
            )
            xgb_model.fit(
                Z_train, y_train,
                eval_set=[(Z_val, y_val)],
                verbose=200
            )
        except TypeError:
            xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
            xgb_model.fit(
                Z_train, y_train,
                eval_set=[(Z_val, y_val)],
                eval_metric=XGB_EVAL_METRIC,
                early_stopping_rounds=XGB_EARLY_STOP,
                verbose=200
            )

        best_iter = getattr(xgb_model, "best_iteration", None)
        if best_iter is not None:
            pred_xgb_val  = xgb_model.predict(Z_val,  iteration_range=(0, best_iter + 1))
            pred_xgb_test = xgb_model.predict(Z_test, iteration_range=(0, best_iter + 1))
        else:
            pred_xgb_val  = xgb_model.predict(Z_val)
            pred_xgb_test = xgb_model.predict(Z_test)

        rmse_xgb, mae_xgb, corr_xgb = eval_regression(y_test, pred_xgb_test, "XGB")

    # -------- Blends --------
    pred_blend_avg_test = None
    pred_blend_stack_test = None
    w = None
    rmse_blend_avg = mae_blend_avg = corr_blend_avg = np.nan
    rmse_blend_stack = mae_blend_stack = corr_blend_stack = np.nan

    if pred_xgb_test is not None:
        pred_blend_avg_test = 0.5 * pred_mlp_test + 0.5 * pred_xgb_test
        rmse_blend_avg, mae_blend_avg, corr_blend_avg = eval_regression(y_test, pred_blend_avg_test, "Blend-Avg")

        w = learn_simplex_weights(pred_mlp_val, pred_xgb_val, y_val, alpha=STACK_RIDGE_ALPHA)
        print("[INFO] Stacking weights (simplex, Ridge):", w)

        pred_blend_stack_test = w[0]*pred_mlp_test + w[1]*pred_xgb_test
        rmse_blend_stack, mae_blend_stack, corr_blend_stack = eval_regression(y_test, pred_blend_stack_test, "Blend-Stack")

    # -------- IC tracking on TEST --------
    test_idx = np.arange(val_end, len(df))

    _, st_mlp = ic_by_day(df, test_idx, pred_mlp_test, label_col,
                         pred_name="pred_mlp", horizon_bars=label_horizon_bars)
    st_xgb = st_avg = st_stack = dict(n=0, mean=np.nan, std=np.nan, skew=np.nan, t=np.nan)

    if pred_xgb_test is not None:
        _, st_xgb = ic_by_day(df, test_idx, pred_xgb_test, label_col,
                              pred_name="pred_xgb", horizon_bars=label_horizon_bars)
        _, st_avg = ic_by_day(df, test_idx, pred_blend_avg_test, label_col,
                              pred_name="pred_blend_avg", horizon_bars=label_horizon_bars)
        _, st_stack = ic_by_day(df, test_idx, pred_blend_stack_test, label_col,
                                pred_name="pred_blend_stack", horizon_bars=label_horizon_bars)

    # -------- Save predictions --------
    pred_mlp_full = np.full(len(df), np.nan, dtype=np.float32)
    pred_mlp_full[train_end:val_end] = pred_mlp_val
    pred_mlp_full[val_end:] = pred_mlp_test

    pred_xgb_full = np.full(len(df), np.nan, dtype=np.float32)
    if pred_xgb_test is not None:
        pred_xgb_full[train_end:val_end] = pred_xgb_val
        pred_xgb_full[val_end:] = pred_xgb_test

    pred_blend_avg_full = np.full(len(df), np.nan, dtype=np.float32)
    pred_blend_stack_full = np.full(len(df), np.nan, dtype=np.float32)
    if pred_blend_avg_test is not None:
        pred_blend_avg_full[val_end:] = pred_blend_avg_test
    if pred_blend_stack_test is not None:
        pred_blend_stack_full[val_end:] = pred_blend_stack_test

    preds_df = pd.DataFrame({
        "pred_mlp": pred_mlp_full,
        "pred_xgb": pred_xgb_full,
        "pred_blend_avg": pred_blend_avg_full,
        "pred_blend_stack": pred_blend_stack_full,
    }, index=df.index)

    out = pd.concat([df, preds_df], axis=1)

    out_csv = OUTPUT_CSV.replace(".csv", f"_h{label_horizon_bars}.csv")
    out.to_csv(out_csv, index=False)
    print("[INFO] Saved:", out_csv)

    # -------- Return summary for grid --------
    summary = {
        "horizon_bars": label_horizon_bars,

        "rmse_mlp": rmse_mlp, "mae_mlp": mae_mlp, "corr_mlp": corr_mlp,
        "ic_mean_mlp": st_mlp["mean"], "ic_std_mlp": st_mlp["std"],
        "ic_skew_mlp": st_mlp["skew"], "ic_t_mlp": st_mlp["t"], "ic_n_mlp": st_mlp["n"],

        "rmse_xgb": rmse_xgb, "mae_xgb": mae_xgb, "corr_xgb": corr_xgb,
        "ic_mean_xgb": st_xgb["mean"], "ic_std_xgb": st_xgb["std"],
        "ic_skew_xgb": st_xgb["skew"], "ic_t_xgb": st_xgb["t"], "ic_n_xgb": st_xgb["n"],

        "rmse_blend_avg": rmse_blend_avg, "mae_blend_avg": mae_blend_avg, "corr_blend_avg": corr_blend_avg,
        "ic_mean_blend_avg": st_avg["mean"], "ic_std_blend_avg": st_avg["std"],
        "ic_skew_blend_avg": st_avg["skew"], "ic_t_blend_avg": st_avg["t"], "ic_n_blend_avg": st_avg["n"],

        "rmse_blend_stack": rmse_blend_stack, "mae_blend_stack": mae_blend_stack, "corr_blend_stack": corr_blend_stack,
        "ic_mean_blend_stack": st_stack["mean"], "ic_std_blend_stack": st_stack["std"],
        "ic_skew_blend_stack": st_stack["skew"], "ic_t_blend_stack": st_stack["t"], "ic_n_blend_stack": st_stack["n"],
    }
    return summary


def main():
    results = []
    for h in LABEL_HORIZON_LIST:
        results.append(run_one_horizon(h))

    res_df = pd.DataFrame(results).sort_values("horizon_bars")
    grid_csv = OUTPUT_CSV.replace(".csv", "_grid_results.csv")
    res_df.to_csv(grid_csv, index=False)
    print("\n==============================")
    print("[GRID] Finished. Results saved ->", grid_csv)
    print(res_df)


if __name__ == "__main__":
    main()



