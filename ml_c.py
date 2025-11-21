import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
try:
    # sklearn>=1.4
    from sklearn.metrics import root_mean_squared_error
except Exception:
    root_mean_squared_error = None
    from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Optional for XGB
try:
    import xgboost as xgb
except Exception:
    xgb = None
    print("[WARN] xgboost not installed. XGB stage will be skipped. pip install xgboost")


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
LABEL_HORIZON_BARS = 12   # if label not found, create log return over next N bars

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


def pick_or_create_label(df: pd.DataFrame):
    label_col = None
    for c in PREFERRED_LABELS:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        # forward log return over next LABEL_HORIZON_BARS bars
        # y_t = log(close_{t+H}/close_t)
        h = int(LABEL_HORIZON_BARS)
        df["ret_fwd"] = np.log(df["close"].shift(-h) / df["close"])
        label_col = "ret_fwd"
        print(f"[INFO] Created label '{label_col}' with horizon {h} bars.")
    else:
        print(f"[INFO] Using existing label '{label_col}'.")

    # drop last H rows where label is NaN after shift(-h)
    h = int(LABEL_HORIZON_BARS) if label_col == "ret_fwd" else 0
    if h > 0:
        df = df.iloc[:-h].copy()

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
    corr = np.corrcoef(y_pred, y_true)[0, 1] if len(y_true) > 2 else np.nan
    print(f"[{name}] RMSE={rmse:.6f}, MAE={mae:.6f}, Corr={corr:.6f}")
    return rmse, mae, corr


def main():
    # -------- Load --------
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    print("[INFO] Loaded:", df.shape)

    # -------- Label --------
    df, label_col = pick_or_create_label(df)

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

    eval_regression(y_test, pred_mlp_test, "MLP")

    # -------- XGB (optional) --------
    pred_xgb_val = pred_xgb_test = None
    if xgb is not None:
        print("[INFO] Training XGB regressor...")

        # New xgboost (>=2.1): early_stopping_rounds + eval_metric must be in constructor
        # (fit() no longer accepts them). :contentReference[oaicite:2]{index=2}
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
            # Fallback for older xgboost if needed
            xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
            xgb_model.fit(
                Z_train, y_train,
                eval_set=[(Z_val, y_val)],
                eval_metric=XGB_EVAL_METRIC,
                early_stopping_rounds=XGB_EARLY_STOP,
                verbose=200
            )

        # Predict using best_iteration if present
        best_iter = getattr(xgb_model, "best_iteration", None)
        if best_iter is not None:
            pred_xgb_val  = xgb_model.predict(Z_val,  iteration_range=(0, best_iter + 1))
            pred_xgb_test = xgb_model.predict(Z_test, iteration_range=(0, best_iter + 1))
        else:
            pred_xgb_val  = xgb_model.predict(Z_val)
            pred_xgb_test = xgb_model.predict(Z_test)

        eval_regression(y_test, pred_xgb_test, "XGB")

    # -------- Blends --------
    pred_blend_avg_test = None
    pred_blend_stack_test = None

    if pred_xgb_test is not None:
        # (A) simple avg
        pred_blend_avg_test = 0.5 * pred_mlp_test + 0.5 * pred_xgb_test
        eval_regression(y_test, pred_blend_avg_test, "Blend-Avg")

        # (B) stacking weights learned on val
        A = np.vstack([pred_mlp_val, pred_xgb_val]).T
        w, *_ = np.linalg.lstsq(A, y_val, rcond=None)
        print("[INFO] Stacking weights:", w)

        pred_blend_stack_test = w[0]*pred_mlp_test + w[1]*pred_xgb_test
        eval_regression(y_test, pred_blend_stack_test, "Blend-Stack")

    # -------- Save predictions --------
    # Create full-length prediction arrays aligned to df index
    pred_mlp_full = np.full(len(df), np.nan, dtype=np.float32)
    pred_mlp_full[val_end:] = pred_mlp_test

    pred_xgb_full = np.full(len(df), np.nan, dtype=np.float32)
    if pred_xgb_test is not None:
        pred_xgb_full[val_end:] = pred_xgb_test

    pred_blend_avg_full = np.full(len(df), np.nan, dtype=np.float32)
    pred_blend_stack_full = np.full(len(df), np.nan, dtype=np.float32)

    if pred_blend_avg_test is not None:
        pred_blend_avg_full[val_end:] = pred_blend_avg_test
    if pred_blend_stack_test is not None:
        pred_blend_stack_full[val_end:] = pred_blend_stack_test

    # Use concat to avoid fragmentation
    preds_df = pd.DataFrame({
        "pred_mlp": pred_mlp_full,
        "pred_xgb": pred_xgb_full,
        "pred_blend_avg": pred_blend_avg_full,
        "pred_blend_stack": pred_blend_stack_full,
    }, index=df.index)

    out = pd.concat([df, preds_df], axis=1)
    out.to_csv(OUTPUT_CSV, index=False)
    print("[INFO] Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()


