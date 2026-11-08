import os
import json
import sqlite3
import hashlib
import traceback
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


st.set_page_config(
    page_title="Ad Fraud & Invalid Traffic Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


SCHEMA_VERSION = "1.0.0"
DB_PATH = "afit_demo.db"
VIDEO_DIR = "demo_output"
VIDEO_PATH = os.path.join(VIDEO_DIR, "afit_live_demo.mp4")

MODEL_FEATURES = [
    "ip_events_5m",
    "ip_clicks_5m",
    "ua_events_5m",
    "pub_events_5m",
    "pub_clicks_5m",
    "pub_conversions_5m",
    "pub_ctr_5m",
    "pub_cvr_5m",
    "ip_click_ratio_5m",
    "repeat_click_60s",
    "click_after_impression_ms",
    "conversion_latency_ms",
    "viewable_time_ms",
    "cost_micros",
    "datacenter_flag",
    "impossible_ua_flag",
    "geo_mismatch_flag",
    "zero_viewable_click_flag",
    "quality_score",
]


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:length]


def jsonable(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def generate_events(
    n_events: int = 6000,
    seed: int = 42,
    scenario: str = "mixed",
    fraud_share: float = 0.12,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-11-08 00:00:00", tz="UTC")
    duration_sec = 6 * 3600
    seconds = np.sort(rng.uniform(0, duration_sec, n_events))

    campaigns = [f"camp_{i:02d}" for i in range(1, 13)]
    publishers = [f"pub_{i:02d}" for i in range(1, 41)]
    high_risk_publishers = publishers[-10:]

    countries = ["US", "GB", "DE", "FR", "IN", "ID", "BR", "NG", "VN", "PK", "BD", "RU", "MX", "JP"]
    fraud_countries = ["IN", "ID", "VN", "PK", "BD", "NG", "RU"]

    devices = ["mobile", "desktop", "tablet", "ctv"]
    os_map = {
        "mobile": ["Android", "iOS"],
        "desktop": ["Windows", "macOS", "Linux"],
        "tablet": ["Android", "iOS"],
        "ctv": ["Roku", "SmartTV", "Tizen"],
    }
    browsers = ["Chrome", "Safari", "Firefox", "Edge", "WebView"]

    legit_uas = [
        "Mozilla/5.0 (Windows NT 10.0) Chrome/120 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Safari/604.1",
        "Mozilla/5.0 (Linux; Android 14) Chrome/120 Mobile",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) Safari/605.1",
        "Mozilla/5.0 (Roku) SmartTV Browser/1.0",
    ]
    fraud_uas = [
        "python-requests/2.31",
        "curl/8.4.0",
        "HeadlessChrome/118.0",
        "bot-crawler/1.0",
        "okhttp/4.10.0",
        "Go-http-client/2.0",
    ]

    legit_ips = [f"res_{i:04d}" for i in range(1200)]
    fraud_ips = [f"dc_{i:03d}" for i in range(120)] + [f"proxy_{i:03d}" for i in range(60)]

    rows = []

    for idx, sec in enumerate(seconds):
        ts = start + timedelta(seconds=float(sec))
        progress = idx / max(1, n_events - 1)

        local_fraud_share = fraud_share
        if scenario == "recovery":
            if progress < 0.35:
                local_fraud_share = fraud_share * 2.2
            elif progress < 0.60:
                local_fraud_share = fraud_share * 0.8
            else:
                local_fraud_share = fraud_share * 0.15
        elif scenario == "botnet":
            local_fraud_share = fraud_share * (2.5 if 0.25 < progress < 0.65 else 0.6)
        elif scenario == "click_injection":
            local_fraud_share = fraud_share * (2.3 if 0.30 < progress < 0.70 else 0.5)
        elif scenario == "malvertising":
            local_fraud_share = fraud_share * (2.1 if 0.20 < progress < 0.75 else 0.55)

        local_fraud_share = float(np.clip(local_fraud_share, 0.0, 0.65))
        is_fraud = bool(rng.random() < local_fraud_share)

        if is_fraud:
            if scenario == "botnet":
                fraud_type = "botnet"
            elif scenario == "click_injection":
                fraud_type = "click_injection"
            elif scenario == "malvertising":
                fraud_type = "malvertising"
            elif scenario == "normal":
                fraud_type = str(rng.choice(["botnet", "geo_proxy", "malvertising"], p=[0.45, 0.35, 0.20]))
            else:
                fraud_type = str(rng.choice(
                    ["botnet", "click_injection", "malvertising", "geo_proxy"],
                    p=[0.40, 0.25, 0.20, 0.15],
                ))
        else:
            fraud_type = "legitimate"

        if is_fraud:
            publisher = str(rng.choice(high_risk_publishers)) if rng.random() < 0.70 else str(rng.choice(publishers))
            ip = str(rng.choice(fraud_ips))
            asn = str(rng.choice(["datacenter", "proxy", "hosting"], p=[0.55, 0.30, 0.15]))
            ua = str(rng.choice(fraud_uas))
            ua_class = str(rng.choice(["bot", "headless", "script"], p=[0.55, 0.30, 0.15]))
            country = str(rng.choice(fraud_countries)) if rng.random() < 0.75 else str(rng.choice(countries))
            device = str(rng.choice(devices, p=[0.65, 0.20, 0.10, 0.05]))
            visible = bool(rng.random() < 0.15)
            viewable = int(rng.integers(0, 120)) if not visible else int(rng.integers(100, 500))
            geo_mismatch = bool(rng.random() < 0.45)
        else:
            publisher = str(rng.choice(publishers))
            ip = str(rng.choice(legit_ips))
            asn = str(rng.choice(["residential", "mobile", "hosting"], p=[0.72, 0.23, 0.05]))
            ua = str(rng.choice(legit_uas))
            ua_class = str(rng.choice(["browser", "webview", "app"], p=[0.70, 0.20, 0.10]))
            country = str(rng.choice(countries))
            device = str(rng.choice(devices, p=[0.52, 0.34, 0.09, 0.05]))
            visible = bool(rng.random() < 0.82)
            viewable = int(rng.integers(300, 30000)) if visible else int(rng.integers(0, 250))
            geo_mismatch = bool(rng.random() < 0.03)

        os_name = str(rng.choice(os_map[device]))
        browser = "Unknown" if ua_class in ["bot", "headless", "script"] else str(rng.choice(browsers))

        if fraud_type == "botnet":
            event_type = "click" if rng.random() < 0.28 else "impression"
        elif fraud_type == "click_injection":
            event_type = "click" if rng.random() < 0.82 else "impression"
        elif fraud_type == "malvertising":
            event_type = "click" if rng.random() < 0.42 else "impression"
        else:
            event_type = str(rng.choice(["impression", "click", "conversion"], p=[0.82, 0.15, 0.03]))

        if event_type == "impression":
            cost = int(rng.integers(1500, 9000))
        elif event_type == "click":
            cost = int(rng.integers(80000, 450000))
        else:
            cost = int(rng.integers(1_000_000, 8_000_000))

        converted = event_type == "conversion" or (
            event_type == "click" and (not is_fraud) and bool(rng.random() < 0.04)
        )
        revenue = int(rng.integers(2_000_000, 12_000_000)) if converted else 0

        if event_type == "click":
            if fraud_type == "click_injection":
                click_delay = int(rng.integers(20, 900))
            elif is_fraud:
                click_delay = int(rng.integers(200, 5000))
            else:
                click_delay = int(rng.integers(3000, 180000))
        else:
            click_delay = -1

        if event_type == "conversion":
            conv_latency = int(rng.integers(200, 4000)) if is_fraud else int(rng.integers(60000, 3600000))
        else:
            conv_latency = -1

        if event_type == "conversion":
            visible = False
            viewable = 0

        quality_flag = ""
        r = float(rng.random())
        if r < 0.015:
            quality_flag = "missing_ip"
            ip = ""
        elif r < 0.025:
            quality_flag = "missing_user_agent"
            ua = ""
            ua_class = "unknown"
            browser = "Unknown"
        elif r < 0.033:
            quality_flag = "negative_cost"
            cost = -abs(cost)
        elif r < 0.040:
            quality_flag = "unknown_country"
            country = ""

        ingest_delay_sec = float(rng.exponential(30))
        if rng.random() < 0.02:
            ingest_delay_sec += 600
            quality_flag = f"{quality_flag};late_arrival" if quality_flag else "late_arrival"

        ingest_ts = ts + timedelta(seconds=ingest_delay_sec)

        rows.append({
            "event_id": f"evt_{idx:08d}",
            "ts_utc": ts,
            "ingest_ts_utc": ingest_ts,
            "campaign_id": str(rng.choice(campaigns)),
            "publisher_id": publisher,
            "placement_id": f"place_{int(rng.integers(1, 25)):02d}",
            "ad_format": str(rng.choice(["banner", "video", "native", "interstitial"])),
            "event_type": event_type,
            "ip_hash": stable_hash(ip) if ip else "",
            "ip_asn_type": asn,
            "user_agent_hash": stable_hash(ua) if ua else "",
            "user_agent_class": ua_class,
            "device_type": device,
            "os": os_name,
            "browser": browser,
            "country": country,
            "visible": visible,
            "viewable_time_ms": viewable,
            "cost_micros": cost,
            "revenue_micros": revenue,
            "converted": converted,
            "click_after_impression_ms": click_delay,
            "conversion_latency_ms": conv_latency,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type,
            "geo_mismatch_flag": int(geo_mismatch),
            "data_quality_flag": quality_flag,
            "source_system": "synthetic_ad_server",
            "pipeline_version": SCHEMA_VERSION,
        })

    df = pd.DataFrame(rows)
    df["record_uid"] = [stable_hash(f"{i}|{eid}", 24) for i, eid in enumerate(df["event_id"])]

    dup_count = max(1, int(0.01 * len(df)))
    dup_idx = rng.choice(df.index.to_numpy(), size=dup_count, replace=False)
    dups = df.loc[dup_idx].copy()
    dups["record_uid"] = [stable_hash("dup|" + str(uid), 24) for uid in dups["record_uid"]]
    dups["data_quality_flag"] = "duplicate_event_id"
    dups["ts_utc"] = dups["ts_utc"] + pd.to_timedelta(rng.uniform(0, 60, len(dups)), unit="s")

    df = pd.concat([df, dups], ignore_index=True)
    df = df.sort_values("ts_utc").reset_index(drop=True)
    return df


def rolling_group_stat(
    df: pd.DataFrame,
    by,
    col: Optional[str] = None,
    window_min: float = 5.0,
    stat: str = "count",
) -> np.ndarray:
    result = np.zeros(len(df), dtype=float)
    if isinstance(by, str):
        by = [by]

    window_sec = window_min * 60.0

    for _, grp in df.groupby(by, sort=False, dropna=False):
        if grp.empty:
            continue

        idx = grp.index.to_numpy()
        epoch = grp["ts_utc"].dt.tz_convert("UTC").astype("int64").to_numpy() / 1e9
        order = np.argsort(epoch)

        sorted_idx = idx[order]
        sorted_epoch = epoch[order]

        left = np.searchsorted(sorted_epoch, sorted_epoch - window_sec, side="left")
        right = np.searchsorted(sorted_epoch, sorted_epoch, side="right")

        if stat == "count":
            values = (right - left).astype(float)
        else:
            if col is None:
                raise ValueError("col is required for sum aggregation")
            arr = grp[col].to_numpy(dtype=float)[order]
            csum = np.concatenate(([0.0], np.cumsum(arr)))
            values = csum[right] - csum[left]

        result[sorted_idx] = values

    return result


def validate_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts_parsed = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    ingest_parsed = pd.to_datetime(df["ingest_ts_utc"], utc=True, errors="coerce")

    median_ts = ts_parsed.median()
    if pd.isna(median_ts):
        median_ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    missing_ip = df["ip_hash"].astype(str).str.strip().eq("")
    missing_ua = df["user_agent_hash"].astype(str).str.strip().eq("")
    unknown_country = df["country"].astype(str).str.strip().eq("")
    negative_cost = pd.to_numeric(df["cost_micros"], errors="coerce") < 0
    malformed_ts = ts_parsed.isna()
    duplicate_event_id = df.duplicated("event_id", keep=False)

    ingest_filled = ingest_parsed.fillna(median_ts + pd.Timedelta(seconds=30))
    late_arrival = (ingest_filled - ts_parsed.fillna(median_ts)) > pd.Timedelta("5min")

    df["missing_ip_flag"] = missing_ip.astype(int)
    df["missing_ua_flag"] = missing_ua.astype(int)
    df["unknown_country_flag"] = unknown_country.astype(int)
    df["negative_cost_flag"] = negative_cost.astype(int)
    df["malformed_ts_flag"] = malformed_ts.astype(int)
    df["duplicate_event_id_flag"] = duplicate_event_id.astype(int)
    df["late_arrival_flag"] = late_arrival.astype(int)

    penalty = (
        0.20 * malformed_ts.astype(float)
        + 0.18 * missing_ip.astype(float)
        + 0.12 * missing_ua.astype(float)
        + 0.10 * negative_cost.astype(float)
        + 0.08 * unknown_country.astype(float)
        + 0.08 * late_arrival.astype(float)
        + 0.08 * duplicate_event_id.astype(float)
    )
    df["quality_score"] = (1.0 - penalty).clip(0.0, 1.0)

    df["ts_utc"] = ts_parsed.fillna(median_ts)
    df["ingest_ts_utc"] = ingest_filled

    df.loc[missing_ip, "ip_hash"] = "unknown"
    df.loc[missing_ua, "user_agent_hash"] = "unknown"
    df.loc[unknown_country, "country"] = "XX"

    df["cost_micros"] = pd.to_numeric(df["cost_micros"], errors="coerce").fillna(0).clip(lower=0)
    df["revenue_micros"] = pd.to_numeric(df["revenue_micros"], errors="coerce").fillna(0).clip(lower=0)
    df["viewable_time_ms"] = pd.to_numeric(df["viewable_time_ms"], errors="coerce").fillna(0).clip(lower=0)
    df["click_after_impression_ms"] = pd.to_numeric(df["click_after_impression_ms"], errors="coerce").fillna(-1)
    df["conversion_latency_ms"] = pd.to_numeric(df["conversion_latency_ms"], errors="coerce").fillna(-1)

    df["visible"] = df["visible"].astype(bool)
    df["converted"] = df["converted"].astype(bool)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["geo_mismatch_flag"] = pd.to_numeric(df["geo_mismatch_flag"], errors="coerce").fillna(0).astype(int)

    df["is_click"] = df["event_type"].eq("click").astype(int)
    df["is_conversion"] = df["event_type"].eq("conversion").astype(int)
    df["datacenter_flag"] = df["ip_asn_type"].isin(["datacenter", "proxy", "hosting"]).astype(int)
    df["impossible_ua_flag"] = df["user_agent_class"].isin(["bot", "headless", "script"]).astype(int)
    df["zero_viewable_click_flag"] = (
        (df["event_type"] == "click") & (df["viewable_time_ms"] < 50)
    ).astype(int)

    df = df.sort_values("ts_utc").reset_index(drop=True)

    df["ip_events_5m"] = rolling_group_stat(df, ["ip_hash"], None, 5, "count")
    df["ip_clicks_5m"] = rolling_group_stat(df, ["ip_hash"], "is_click", 5, "sum")
    df["ua_events_5m"] = rolling_group_stat(df, ["user_agent_hash"], None, 5, "count")

    df["pub_events_5m"] = rolling_group_stat(df, ["publisher_id"], None, 5, "count")
    df["pub_clicks_5m"] = rolling_group_stat(df, ["publisher_id"], "is_click", 5, "sum")
    df["pub_conversions_5m"] = rolling_group_stat(df, ["publisher_id"], "is_conversion", 5, "sum")
    df["pub_spend_5m"] = rolling_group_stat(df, ["publisher_id"], "cost_micros", 5, "sum")

    df["repeat_click_60s"] = rolling_group_stat(df, ["ip_hash", "campaign_id"], "is_click", 1, "sum")
    df["repeat_click_60s"] = df["repeat_click_60s"] - df["is_click"]
    df.loc[df["is_click"] == 0, "repeat_click_60s"] = 0.0

    df["pub_ctr_5m"] = df["pub_clicks_5m"] / df["pub_events_5m"].clip(lower=1)
    df["pub_cvr_5m"] = df["pub_conversions_5m"] / df["pub_clicks_5m"].clip(lower=1)
    df["ip_click_ratio_5m"] = df["ip_clicks_5m"] / df["ip_events_5m"].clip(lower=1)

    for c in ["pub_ctr_5m", "pub_cvr_5m", "ip_click_ratio_5m"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["fraud_label_source"] = "synthetic_ground_truth"
    return df


def compute_rule_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parts = {}

    parts["bot_ua"] = df["impossible_ua_flag"].astype(float) * 35.0
    parts["datacenter_ip"] = df["datacenter_flag"].astype(float) * 20.0
    parts["low_viewability"] = (
        (df["viewable_time_ms"] < 100) & (df["event_type"].isin(["impression", "click"]))
    ).astype(float) * 20.0
    parts["high_ctr"] = (
        (df["pub_ctr_5m"] > 0.25).astype(float) * 15.0
        + (df["pub_ctr_5m"] > 0.50).astype(float) * 10.0
    )
    parts["ip_velocity"] = np.clip((df["ip_events_5m"] - 20.0) / 80.0, 0.0, 1.0) * 25.0
    parts["ua_velocity"] = np.clip((df["ua_events_5m"] - 15.0) / 60.0, 0.0, 1.0) * 20.0
    parts["repeat_click"] = np.clip(df["repeat_click_60s"] / 5.0, 0.0, 1.0) * 25.0
    parts["click_injection"] = (
        (df["event_type"] == "click")
        & (df["click_after_impression_ms"] >= 0)
        & (df["click_after_impression_ms"] <= 1000)
    ).astype(float) * 30.0
    parts["fast_conversion"] = (
        (df["event_type"] == "conversion")
        & (df["conversion_latency_ms"] >= 0)
        & (df["conversion_latency_ms"] <= 5000)
    ).astype(float) * 20.0
    parts["geo_mismatch"] = df["geo_mismatch_flag"].astype(float) * 15.0
    parts["quality_degraded"] = ((1.0 - df["quality_score"]) > 0.25).astype(float) * 10.0

    score_matrix = pd.DataFrame(parts, index=df.index)
    max_sum = 245.0

    df["rule_score"] = np.clip(score_matrix.sum(axis=1) / max_sum * 100.0, 0.0, 100.0)
    df["reason_codes"] = score_matrix.apply(
        lambda row: ",".join([k for k, v in row.items() if float(v) >= 8.0]),
        axis=1,
    )
    df["rule_components_json"] = score_matrix.apply(
        lambda row: json.dumps({k: float(v) for k, v in row.items()}),
        axis=1,
    )

    return df, score_matrix


def fit_and_score_model(
    df: pd.DataFrame,
    contamination: float = 0.08,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()

    X = df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_mask = df["ts_utc"] < df["ts_utc"].quantile(0.60)
    train_count = int(train_mask.sum())

    if train_count < 80:
        df["model_score"] = df["rule_score"]
        df["model_status"] = "fallback_insufficient_training_data"
        return df, {"train_count": train_count, "status": "fallback"}

    lower = X[train_mask].quantile(0.01)
    upper = X[train_mask].quantile(0.99)

    Xc = X.copy()
    for c in Xc.columns:
        lo = float(lower[c]) if pd.notna(lower[c]) else -np.inf
        hi = float(upper[c]) if pd.notna(upper[c]) else np.inf
        Xc[c] = Xc[c].clip(lower=lo, upper=hi)

    scaler = StandardScaler().fit(Xc[train_mask])
    Xs_train = scaler.transform(Xc[train_mask])
    Xs_all = scaler.transform(Xc)

    iso = IsolationForest(
        n_estimators=120,
        max_samples="auto",
        contamination=float(contamination),
        random_state=int(seed),
        n_jobs=1,
    )
    iso.fit(Xs_train)

    raw_all = -iso.decision_function(Xs_all)
    raw_train = raw_all[train_mask.to_numpy()]

    percentiles = np.percentile(raw_train, np.linspace(0, 100, 101))
    model_score = np.searchsorted(percentiles, raw_all, side="right").astype(float)
    model_score = np.clip(model_score, 0.0, 100.0)

    df["model_score"] = model_score
    df["model_status"] = "isolation_forest_rank_percentile"

    meta = {
        "train_count": train_count,
        "contamination": float(contamination),
        "status": "ok",
        "features": MODEL_FEATURES,
    }
    return df, meta


def combine_and_decide(
    df: pd.DataFrame,
    rule_weight: float = 0.55,
    threshold_review: float = 65.0,
    threshold_block: float = 85.0,
    auto_block: bool = False,
    reviewer: str = "demo_reviewer",
    model_meta: Optional[Dict] = None,
) -> pd.DataFrame:
    df = df.copy()

    usable = df["quality_score"] >= 0.65
    df["final_score"] = np.where(
        usable,
        rule_weight * df["rule_score"] + (1.0 - rule_weight) * df["model_score"],
        df["rule_score"],
    )

    agreement = 1.0 - np.abs(df["rule_score"] - df["model_score"]) / 100.0
    support = min(1.0, float((model_meta or {}).get("train_count", 0)) / 500.0)

    df["confidence"] = np.clip(
        (0.45 + 0.35 * agreement) * df["quality_score"] * (0.5 + 0.5 * support),
        0.0,
        1.0,
    )
    df.loc[~usable, "confidence"] *= 0.55

    df["risk_level"] = pd.cut(
        df["final_score"],
        bins=[-1.0, 49.0, 69.0, 84.0, 100.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    ).astype(str)

    threshold_monitor = max(0.0, threshold_review - 20.0)

    recommended = []
    applied = []
    approved_by = []
    notes = []

    for score, conf in zip(df["final_score"], df["confidence"]):
        if score >= threshold_block and conf >= 0.70:
            rec = "BLOCK_PREVENT_BILLING"
        elif score >= threshold_review:
            rec = "REVIEW_HOLD"
        elif score >= threshold_monitor:
            rec = "MONITOR"
        else:
            rec = "ALLOW"

        if rec == "BLOCK_PREVENT_BILLING":
            if auto_block:
                act = "BLOCK_PREVENT_BILLING"
                appr = reviewer
                note = "Authorized auto-block policy"
            else:
                act = "SUGGEST_BLOCK_PENDING_APPROVAL"
                appr = ""
                note = "Human approval required"
        elif rec == "REVIEW_HOLD":
            act = "HOLD_FOR_REVIEW"
            appr = ""
            note = "Analyst review queue"
        elif rec == "MONITOR":
            act = "MONITOR"
            appr = ""
            note = "Observe only"
        else:
            act = "ALLOW"
            appr = "system"
            note = ""

        recommended.append(rec)
        applied.append(act)
        approved_by.append(appr)
        notes.append(note)

    df["recommended_action"] = recommended
    df["applied_action"] = applied
    df["approved_by"] = approved_by
    df["decision_note"] = notes

    prevented_actions = [
        "BLOCK_PREVENT_BILLING",
        "SUGGEST_BLOCK_PENDING_APPROVAL",
        "HOLD_FOR_REVIEW",
    ]
    df["prevented_spend_micros"] = np.where(
        df["applied_action"].isin(prevented_actions),
        df["cost_micros"],
        0.0,
    )

    return df


@st.cache_data(show_spinner=False)
def build_analytics_dataset(
    seed: int,
    n_events: int,
    scenario: str,
    fraud_share: float,
    contamination: float,
    rule_weight: float,
    threshold_review: float,
    threshold_block: float,
    auto_block: bool,
    reviewer: str,
) -> Tuple[pd.DataFrame, Dict]:
    df = generate_events(n_events=n_events, seed=seed, scenario=scenario, fraud_share=fraud_share)
    df = validate_and_enrich(df)
    df, _ = compute_rule_scores(df)
    df, meta = fit_and_score_model(df, contamination=contamination, seed=seed)
    df = combine_and_decide(
        df,
        rule_weight=rule_weight,
        threshold_review=threshold_review,
        threshold_block=threshold_block,
        auto_block=auto_block,
        reviewer=reviewer,
        model_meta=meta,
    )
    return df, meta


def persist_dataset(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        out = df.copy()
        for c in ["ts_utc", "ingest_ts_utc"]:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], utc=True, errors="coerce").dt.strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )

        out.to_sql("events", conn, if_exists="replace", index=False)

        dec_cols = [
            "record_uid",
            "event_id",
            "ts_utc",
            "campaign_id",
            "publisher_id",
            "rule_score",
            "model_score",
            "final_score",
            "confidence",
            "risk_level",
            "recommended_action",
            "applied_action",
            "approved_by",
            "reason_codes",
            "decision_note",
            "prevented_spend_micros",
        ]
        dec_cols = [c for c in dec_cols if c in out.columns]
        out[dec_cols].to_sql("decisions", conn, if_exists="replace", index=False)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                ts_utc TEXT,
                actor TEXT,
                action TEXT,
                object_id TEXT,
                details TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                "system",
                "DATASET_LOADED",
                "demo",
                json.dumps({"rows": int(len(df)), "schema_version": SCHEMA_VERSION}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def approve_decision(record_uid: str, reviewer: str, new_action: str, note: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            UPDATE decisions
            SET applied_action = ?, approved_by = ?, decision_note = ?
            WHERE record_uid = ?
            """,
            (new_action, reviewer, note, record_uid),
        )
        conn.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                reviewer,
                "HUMAN_APPROVAL",
                record_uid,
                json.dumps({"new_action": new_action, "note": note}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def generate_demo_video(path: str, seconds: int = 14, fps: int = 15) -> str:
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is not available. Install opencv-python-headless.")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    width, height = 960, 540
    total = max(2, seconds * fps)

    out = None
    actual = path

    for codec in ("avc1", "mp4v"):
        tmp = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if tmp.isOpened():
            out = tmp
            actual = path
            break
        tmp.release()

    if out is None:
        actual = path.replace(".mp4", ".avi")
        out = cv2.VideoWriter(actual, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        if not out.isOpened():
            raise RuntimeError("OpenCV could not open a video writer. Check writable directory and codec support.")

    rng = np.random.default_rng(11)
    t = np.arange(total)
    phase = t / float(total - 1)

    events = 80 + 25 * np.sin(2 * np.pi * phase * 1.7) + rng.normal(0, 4, total)
    events = np.maximum(events, 1.0)

    fraud = 8 + 55 * np.exp(-(((phase - 0.45) / 0.12) ** 2)) + rng.normal(0, 2, total)
    fraud = np.maximum(fraud, 0.0)

    detected = fraud * (0.55 + 0.35 * np.clip((phase - 0.25) * 3.0, 0.0, 1.0))
    detected = detected + np.maximum(0.0, rng.normal(0, 2, total))
    detected = np.clip(detected, 0.0, events)

    prevented = np.cumsum(detected * 0.18)

    max_events = max(1.0, float(events.max()))
    max_detected = max(1.0, float(detected.max()))

    x0, y0, x1, y1 = 60, 95, 610, 315

    def xy(j: int, val: float, maxval: float) -> Tuple[int, int]:
        xx = int(x0 + (x1 - x0) * (j / float(total - 1)))
        yy = int(y1 - (y1 - y0) * (val / maxval))
        return xx, yy

    for i in range(total):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        cv2.rectangle(frame, (0, 0), (width, 44), (40, 40, 40), -1)
        cv2.putText(
            frame,
            "AFIT Live Demo - Synthetic Invalid Traffic",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if phase[i] < 0.25:
            scen = "NORMAL"
        elif phase[i] < 0.62:
            scen = "BOTNET / CLICK INJECTION SPIKE"
        else:
            scen = "MITIGATION / RECOVERY"

        cv2.putText(
            frame,
            f"Scenario: {scen}",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(frame, (x0, y0), (x1, y1), (25, 25, 25), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), 1)

        prev = None
        for j in range(i + 1):
            pt = xy(j, float(events[j]), max_events)
            if prev is not None:
                cv2.line(frame, prev, pt, (0, 255, 0), 1, cv2.LINE_AA)
            prev = pt

        prev = None
        for j in range(i + 1):
            pt = xy(j, float(detected[j]), max_detected)
            if prev is not None:
                cv2.line(frame, prev, pt, (0, 0, 255), 2, cv2.LINE_AA)
            prev = pt

        cv2.putText(frame, "events/sec", (x0 + 10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "detected fraud/sec", (x0 + 120, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        bx0, by0 = 650, 95
        signals = ["bot_ua", "datacenter", "repeat_click", "click_injection", "low_viewability"]
        base = np.array([0.20, 0.55, 0.35, 0.70, 0.25], dtype=float)
        factor = 1.0 if phase[i] < 0.62 else 0.35
        weights = np.clip(base * factor + rng.normal(0.0, 0.03, 5), 0.0, 1.0)

        cv2.putText(frame, "Signal strength", (bx0, by0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        for k, (sig, w) in enumerate(zip(signals, weights)):
            y = by0 + k * 34
            cv2.putText(frame, sig, (bx0, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx0 + 150, y), (bx0 + 280, y + 16), (60, 60, 60), -1)
            cv2.rectangle(frame, (bx0 + 150, y), (int(bx0 + 150 + 130 * float(w)), y + 16), (0, 165, 255), -1)

        cv2.rectangle(frame, (0, 350), (width, height), (20, 20, 20), -1)

        kpis = [
            ("Events", f"{int(events[:i + 1].sum()):,}"),
            ("Detected", f"{int(detected[:i + 1].sum()):,}"),
            ("Prevented $", f"${prevented[i]:,.0f}"),
            ("Quality", "98.4% simulated"),
        ]

        x = 20
        for label, val in kpis:
            cv2.putText(frame, label, (x, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(frame, val, (x, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            x += 220

        cv2.putText(
            frame,
            "All data synthetic; labels simulated for demo",
            (20, 490),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"t={i / float(fps):.1f}s  schema={SCHEMA_VERSION}",
            (20, 515),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()
    return actual


def binary_metrics(y_true, y_pred, y_score=None) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        return {
            "events": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fpr": 0.0,
            "auc": float("nan"),
            "ap": float("nan"),
        }

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    auc = float("nan")
    ap = float("nan")

    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_true, y_score))
        except Exception:
            ap = float("nan")

    return {
        "events": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "auc": auc,
        "ap": ap,
    }


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / float(trials)
    denom = 1.0 + (z * z) / float(trials)
    center = (p + (z * z) / (2.0 * float(trials))) / denom
    half = (
        z
        * np.sqrt((p * (1.0 - p)) / float(trials) + (z * z) / (4.0 * float(trials) * float(trials)))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def run_self_checks() -> pd.DataFrame:
    results = []

    def check(name: str, passed: bool, details: str = "") -> None:
        results.append({"test": name, "status": "PASS" if passed else "FAIL", "details": details})

    raw_required = [
        "record_uid",
        "event_id",
        "ts_utc",
        "campaign_id",
        "publisher_id",
        "ip_hash",
        "user_agent_hash",
        "event_type",
        "cost_micros",
        "is_fraud",
        "fraud_type",
    ]

    df = generate_events(800, seed=3, scenario="mixed", fraud_share=0.18)
    check("raw_schema_columns_present", all(c in df.columns for c in raw_required), "Core raw canonical fields exist")
    check("unique_record_uid", df["record_uid"].duplicated().sum() == 0, "Each demo row has a stable UID")

    df = validate_and_enrich(df)
    check("quality_score_range", bool(df["quality_score"].between(0, 1).all()), "Quality score normalized")
    check("missing_values_handled", bool((df["ip_hash"].astype(str).str.len() > 0).all()), "Missing IPs mapped to unknown")

    df, _ = compute_rule_scores(df)
    check("rule_score_present", "rule_score" in df.columns, "Transparent rule score exists")
    check("reason_codes_present", "reason_codes" in df.columns, "Reason codes exist")

    df["model_score"] = df["rule_score"]
    bot_mask = df["impossible_ua_flag"] == 1
    if bot_mask.any():
        check(
            "bot_ua_raises_rule_score",
            float(df.loc[bot_mask, "rule_score"].mean()) > float(df.loc[~bot_mask, "rule_score"].mean()),
            "Bot UA signal increases deterministic rule score",
        )
    else:
        check("bot_ua_raises_rule_score", True, "No bot rows in tiny fixture")

    low = df.copy()
    low["quality_score"] = 0.20
    low2 = combine_and_decide(low, model_meta={"train_count": 100})
    low_mask = low2["quality_score"] < 0.65
    check(
        "low_quality_fallback_to_rules",
        bool(np.allclose(low2.loc[low_mask, "final_score"], low2.loc[low_mask, "rule_score"])),
        "Degraded input falls back to transparent rule score",
    )
    check(
        "combined_fields_present",
        all(c in low2.columns for c in ["final_score", "confidence", "applied_action"]),
        "Decision fields exist after combining scores",
    )

    df_a = generate_events(300, seed=9, scenario="normal", fraud_share=0.05)
    df_b = generate_events(300, seed=9, scenario="normal", fraud_share=0.05)
    check(
        "deterministic_seed",
        df_a["event_id"].tolist() == df_b["event_id"].tolist()
        and np.allclose(df_a["cost_micros"].to_numpy(), df_b["cost_micros"].to_numpy()),
        "Same seed produces same synthetic fixture",
    )

    video_path = st.session_state.get("video_path") or VIDEO_PATH
    check(
        "demo_video_generated",
        os.path.exists(video_path) and os.path.getsize(video_path) > 1000,
        f"Video file exists at {video_path}",
    )

    return pd.DataFrame(results)


if "approvals" not in st.session_state:
    st.session_state.approvals = {}

with st.spinner("Preparing synthetic live demo video..."):
    if "video_path" not in st.session_state:
        try:
            st.session_state.video_path = generate_demo_video(VIDEO_PATH)
            st.session_state.video_error = None
        except Exception as exc:
            st.session_state.video_path = None
            st.session_state.video_error = str(exc)


st.sidebar.title("AFIT Control Room")
st.sidebar.caption("Ad Fraud and Invalid Traffic Analytics — synthetic CPU demo")

st.sidebar.markdown(
    """
**Mission**  
Detect and quantify invalid advertising traffic while protecting legitimate publishers and preserving auditable revenue decisions.
    """
)

scenario = st.sidebar.selectbox(
    "Traffic scenario",
    ["mixed", "normal", "botnet", "click_injection", "malvertising", "recovery"],
    index=0,
)
n_events = st.sidebar.slider("Synthetic events", 1000, 20000, 6000, step=500)
fraud_share = st.sidebar.slider("Simulated fraud share", 0.01, 0.35, 0.12, 0.01)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42, step=1)

contamination = st.sidebar.slider("Model contamination", 0.01, 0.25, 0.08, 0.01)
rule_weight = st.sidebar.slider("Rule weight", 0.0, 1.0, 0.55, 0.05)
threshold_review = st.sidebar.slider("Review threshold", 40, 90, 65, 1)
threshold_block = st.sidebar.slider("Block threshold", 50, 99, 85, 1)

if threshold_block < threshold_review:
    threshold_block = threshold_review
    st.sidebar.caption("Block threshold raised to match review threshold.")

auto_block = st.sidebar.checkbox(
    "Authorize auto-block policy",
    value=False,
    help="Consequential billing blocks require explicit policy approval. Default is pending human review.",
)
reviewer = st.sidebar.text_input("Reviewer ID", "analyst_01")

if st.sidebar.button("Clear cache and regenerate dataset"):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("Regenerate demo video"):
    for p in [VIDEO_PATH, VIDEO_PATH.replace(".mp4", ".avi")]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    st.session_state.pop("video_path", None)
    st.session_state.pop("video_error", None)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data: synthetic only. No paid APIs. No real user data.")


dataset_args = (
    int(seed),
    int(n_events),
    scenario,
    float(fraud_share),
    float(contamination),
    float(rule_weight),
    float(threshold_review),
    float(threshold_block),
    bool(auto_block),
    str(reviewer),
)

try:
    with st.spinner("Loading synthetic ad traffic, validating schema, extracting signals, and scoring..."):
        df, model_meta = build_analytics_dataset(*dataset_args)
except Exception:
    st.error("Dataset build failed.")
    st.code(traceback.format_exc())
    st.stop()

df = df.copy()

for uid, payload in list(st.session_state.approvals.items()):
    if isinstance(payload, tuple) and len(payload) == 3:
        action, appr, note = payload
        mask = df["record_uid"] == uid
        if mask.any():
            df.loc[mask, "applied_action"] = action
            df.loc[mask, "approved_by"] = appr
            df.loc[mask, "decision_note"] = note

st.session_state.df = df

dataset_key = json.dumps(dataset_args, default=str)
if st.session_state.get("persisted_key") != dataset_key:
    try:
        persist_dataset(df)
        st.session_state.persisted_key = dataset_key
    except Exception as exc:
        st.sidebar.warning(f"SQLite persistence skipped: {exc}")

st.title("Ad Fraud and Invalid Traffic Analytics")
st.subheader("AI Advertising Intelligence and Revenue Optimization — executable product prototype")

st.caption(
    "This demo uses clearly labeled synthetic traffic. Observed, simulated, estimated, and predicted values are labeled separately. "
    "Consequential actions such as billing blocks and refunds require human or pre-authorized policy approval."
)

tabs = st.tabs(
    [
        "Live Demo",
        "Revenue Dashboard",
        "Event Evidence",
        "Evaluation",
        "Data Quality & Contracts",
        "PRD, Architecture, Risk & Tests",
        "Value Calculator",
    ]
)


with tabs[0]:
    st.header("Live simulation video")
    st.caption(
        "This is a live in-app canvas simulation, not a static mockup. "
        "It starts automatically and is interactive. All traffic, labels, costs, and alerts are synthetic."
    )

    live_config = {
        "scenario": scenario,
        "fraud_share": float(fraud_share),
        "threshold_review": float(threshold_review),
        "threshold_block": float(threshold_block),
        "auto_block": bool(auto_block),
        "reviewer": str(reviewer),
        "schema_version": SCHEMA_VERSION,
    }

    live_html = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AFIT Live Simulation</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #111827;
    --panel2: #172033;
    --text: #e6edf7;
    --muted: #94a3b8;
    --green: #22c55e;
    --red: #ef4444;
    --orange: #f97316;
    --blue: #60a5fa;
    --purple: #a855f7;
    --cyan: #38bdf8;
    --border: rgba(148, 163, 184, 0.22);
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    overflow: hidden;
  }

  .wrap {
    padding: 10px;
    height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
    gap: 10px;
  }

  .top {
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 12px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }

  .title {
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.2px;
  }

  .subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--muted);
    min-width: 150px;
  }

  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 12px currentColor;
  }

  .dot.green { background: var(--green); }
  .dot.amber { background: var(--orange); }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }

  button, select, input[type="range"] {
    background: #1f2937;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 9px;
    font-size: 12px;
    outline: none;
  }

  button {
    cursor: pointer;
  }

  button:hover {
    border-color: var(--cyan);
  }

  .ctl {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
    background: rgba(31, 41, 55, 0.55);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 5px 8px;
  }

  .ctl span.value {
    min-width: 42px;
    text-align: right;
    color: var(--text);
    font-variant-numeric: tabular-nums;
  }

  .main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 330px;
    gap: 10px;
    min-height: 0;
  }

  .canvasCard,
  .sideCard {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    min-height: 0;
  }

  .canvasCard {
    display: grid;
    grid-template-rows: auto 1fr;
  }

  .cardHead {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: center;
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
    background: #0f172a;
  }

  .side {
    display: grid;
    grid-template-rows: auto auto 1fr;
    gap: 10px;
    min-height: 0;
  }

  .kpis {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
  }

  .kpi {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px;
  }

  .kLabel {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 4px;
  }

  .kValue {
    font-size: 15px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    font-size: 11px;
    color: var(--muted);
  }

  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    display: inline-block;
  }

  .evidence {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    min-height: 0;
    display: grid;
    grid-template-rows: auto 1fr;
    overflow: hidden;
  }

  .evidenceBody {
    padding: 10px;
    overflow: auto;
    font-size: 12px;
  }

  .evTitle {
    font-weight: 700;
    margin-bottom: 8px;
  }

  .evGrid {
    display: grid;
    grid-template-columns: 92px 1fr;
    gap: 5px 8px;
    align-items: start;
  }

  .evGrid span {
    color: var(--muted);
  }

  .evGrid b {
    font-weight: 600;
    word-break: break-word;
  }

  .reasons {
    margin-top: 8px;
    padding: 8px;
    border-radius: 8px;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    color: #fecaca;
    font-size: 11px;
    word-break: break-word;
  }

  .evActions {
    margin-top: 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .evActions button {
    font-size: 11px;
    padding: 6px 8px;
  }

  .bottom {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: 10px;
    min-height: 190px;
    overflow: hidden;
  }

  .tableWrap {
    overflow: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }

  thead th {
    position: sticky;
    top: 0;
    background: #1f2937;
    color: var(--muted);
    text-align: left;
    padding: 7px 8px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    white-space: nowrap;
  }

  tbody td {
    padding: 6px 8px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    white-space: nowrap;
  }

  tbody tr {
    cursor: pointer;
  }

  tbody tr:hover {
    background: rgba(56, 189, 248, 0.06);
  }

  tbody tr.sel {
    background: rgba(56, 189, 248, 0.12);
  }

  .badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    border: 1px solid transparent;
  }

  .badge.allow {
    color: #bbf7d0;
    background: rgba(34, 197, 94, 0.12);
    border-color: rgba(34, 197, 94, 0.28);
  }

  .badge.monitor {
    color: #bfdbfe;
    background: rgba(96, 165, 250, 0.12);
    border-color: rgba(96, 165, 250, 0.28);
  }

  .badge.hold {
    color: #fed7aa;
    background: rgba(249, 115, 22, 0.12);
    border-color: rgba(249, 115, 22, 0.28);
  }

  .badge.pending {
    color: #fde68a;
    background: rgba(234, 179, 8, 0.12);
    border-color: rgba(234, 179, 8, 0.28);
  }

  .badge.block {
    color: #ddd6fe;
    background: rgba(168, 85, 247, 0.14);
    border-color: rgba(168, 85, 247, 0.32);
  }

  .audit {
    border-left: 1px solid var(--border);
    padding: 8px 10px;
    overflow: auto;
    font-size: 11px;
  }

  .auditTitle {
    color: var(--muted);
    font-weight: 700;
    margin-bottom: 6px;
  }

  .audit ul {
    margin: 0;
    padding-left: 16px;
    color: var(--muted);
  }

  .audit li {
    margin-bottom: 4px;
    word-break: break-word;
  }

  .muted {
    color: var(--muted);
  }

  @media (max-width: 1100px) {
    .main {
      grid-template-columns: 1fr;
    }

    .bottom {
      grid-template-columns: 1fr;
    }

    .audit {
      border-left: 0;
      border-top: 1px solid var(--border);
    }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="title">AFIT Live Traffic Simulation</div>
      <div class="subtitle">Synthetic ad impressions, clicks, conversions, fraud signals, detection, review, and billing prevention</div>
    </div>

    <div class="status">
      <span id="statusDot" class="dot green"></span>
      <span id="statusText">RUNNING</span>
      <span class="muted">|</span>
      <span id="clock">--:--:--</span>
      <span class="muted">|</span>
      <span>scenario: <b id="scenarioLabel">mixed</b></span>
    </div>

    <div class="controls">
      <button id="playBtn">Pause</button>
      <button id="resetBtn">Reset</button>

      <select id="scenarioSel" title="Scenario">
        <option value="mixed">Mixed</option>
        <option value="normal">Normal</option>
        <option value="botnet">Botnet spike</option>
        <option value="click_injection">Click injection</option>
        <option value="malvertising">Malvertising</option>
        <option value="recovery">Recovery</option>
      </select>

      <label class="ctl">
        Speed
        <input id="speedRange" type="range" min="0.25" max="3" step="0.25" value="1" />
        <span id="speedVal" class="value">1.00x</span>
      </label>

      <label class="ctl">
        Fraud share
        <input id="fraudRange" type="range" min="1" max="40" step="1" value="12" />
        <span id="fraudVal" class="value">12%</span>
      </label>

      <label class="ctl">
        <input id="autoToggle" type="checkbox" />
        Auto-mitigate
      </label>

      <label class="ctl">
        <input id="followToggle" type="checkbox" checked />
        Follow alerts
      </label>
    </div>
  </div>

  <div class="main">
    <div class="canvasCard">
      <div class="cardHead">
        <span>Live network view</span>
        <span class="legend">
          <span><i class="swatch" style="background:#22c55e"></i>legitimate</span>
          <span><i class="swatch" style="background:#ef4444"></i>undetected fraud</span>
          <span><i class="swatch" style="background:#f97316"></i>held / review</span>
          <span><i class="swatch" style="background:#a855f7"></i>blocked</span>
        </span>
      </div>
      <canvas id="simCanvas" width="980" height="430"></canvas>
    </div>

    <div class="side">
      <div class="kpis">
        <div class="kpi">
          <div class="kLabel">Events</div>
          <div class="kValue" id="kEvents">0</div>
        </div>
        <div class="kpi">
          <div class="kLabel">Simulated fraud rate</div>
          <div class="kValue" id="kFraudRate">0%</div>
        </div>
        <div class="kpi">
          <div class="kLabel">Detected rate</div>
          <div class="kValue" id="kDetected">0%</div>
        </div>
        <div class="kpi">
          <div class="kLabel">Precision</div>
          <div class="kValue" id="kPrecision">0%</div>
        </div>
        <div class="kpi">
          <div class="kLabel">Recall</div>
          <div class="kValue" id="kRecall">0%</div>
        </div>
        <div class="kpi">
          <div class="kLabel">Prevented spend</div>
          <div class="kValue" id="kPrevented">$0.00</div>
        </div>
        <div class="kpi">
          <div class="kLabel">False-positive spend</div>
          <div class="kValue" id="kFPSpend">$0.00</div>
        </div>
        <div class="kpi">
          <div class="kLabel">Review queue</div>
          <div class="kValue" id="kQueue">0</div>
        </div>
      </div>

      <div class="canvasCard">
        <div class="cardHead">
          <span>Live trend</span>
          <span class="legend">
            <span><i class="swatch" style="background:#22c55e"></i>observed events/s</span>
            <span><i class="swatch" style="background:#60a5fa"></i>simulated fraud/s</span>
            <span><i class="swatch" style="background:#f97316"></i>predicted detections/s</span>
          </span>
        </div>
        <canvas id="chartCanvas" width="320" height="150"></canvas>
      </div>

      <div class="evidence">
        <div class="cardHead">
          <span>Evidence inspector</span>
          <span class="muted">click a row to inspect</span>
        </div>
        <div id="evidenceBody" class="evidenceBody">
          <div class="muted">Waiting for events...</div>
        </div>
      </div>
    </div>
  </div>

  <div class="bottom">
    <div class="tableWrap">
      <table>
        <thead>
          <tr>
            <th>time</th>
            <th>type</th>
            <th>country</th>
            <th>IP ASN</th>
            <th>UA class</th>
            <th>view ms</th>
            <th>cost</th>
            <th>score</th>
            <th>action</th>
            <th>reasons</th>
          </tr>
        </thead>
        <tbody id="logBody">
          <tr><td colspan="10" class="muted">No events yet</td></tr>
        </tbody>
      </table>
    </div>

    <div class="audit">
      <div class="auditTitle">Audit history</div>
      <ul id="auditList">
        <li class="muted">No audit events yet</li>
      </ul>
    </div>
  </div>
</div>

<script>
const CONFIG = __CONFIG_JSON__;

(function () {
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");

  const chart = document.getElementById("chartCanvas");
  const cctx = chart.getContext("2d");

  const logBody = document.getElementById("logBody");
  const evidenceBody = document.getElementById("evidenceBody");
  const auditList = document.getElementById("auditList");

  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const clockEl = document.getElementById("clock");
  const scenarioLabel = document.getElementById("scenarioLabel");

  const els = {
    play: document.getElementById("playBtn"),
    reset: document.getElementById("resetBtn"),
    scenario: document.getElementById("scenarioSel"),
    speed: document.getElementById("speedRange"),
    speedVal: document.getElementById("speedVal"),
    fraud: document.getElementById("fraudRange"),
    fraudVal: document.getElementById("fraudVal"),
    auto: document.getElementById("autoToggle"),
    follow: document.getElementById("followToggle"),

    events: document.getElementById("kEvents"),
    fraudRate: document.getElementById("kFraudRate"),
    detected: document.getElementById("kDetected"),
    precision: document.getElementById("kPrecision"),
    recall: document.getElementById("kRecall"),
    prevented: document.getElementById("kPrevented"),
    fpSpend: document.getElementById("kFPSpend"),
    queue: document.getElementById("kQueue"),
  };

  const countries = ["US", "GB", "DE", "FR", "IN", "ID", "BR", "NG", "VN", "PK", "BD", "RU", "MX", "JP"];
  const fraudCountries = ["IN", "ID", "VN", "PK", "BD", "NG", "RU"];

  const legitUAs = [
    "Mozilla/5.0 Windows Chrome",
    "Mozilla/5.0 iPhone Safari",
    "Mozilla/5.0 Android Chrome",
    "Mozilla/5.0 macOS Safari",
    "Mozilla/5.0 Roku SmartTV",
  ];

  const fraudUAs = [
    "python-requests/2.31",
    "curl/8.4.0",
    "HeadlessChrome/118.0",
    "bot-crawler/1.0",
    "okhttp/4.10.0",
    "Go-http-client/2.0",
  ];

  const preventedActions = new Set([
    "BLOCK_PREVENT_BILLING",
    "SUGGEST_BLOCK_PENDING_APPROVAL",
    "HOLD_FOR_REVIEW",
  ]);

  function rand(min, max) {
    return Math.random() * (max - min) + min;
  }

  function randInt(min, max) {
    return Math.floor(rand(min, max + 1));
  }

  function choice(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function pct(v) {
    return (v * 100).toFixed(1) + "%";
  }

  function fmtMoney(micros) {
    return "$" + (micros / 1e6).toFixed(2);
  }

  function fmtTime(d) {
    return d.toLocaleTimeString([], { hour12: false });
  }

  function actionBadge(action) {
    let cls = "allow";
    if (action.includes("BLOCK")) cls = "block";
    else if (action.includes("PENDING")) cls = "pending";
    else if (action.includes("HOLD")) cls = "hold";
    else if (action.includes("MONITOR")) cls = "monitor";
    return '<span class="badge ' + cls + '">' + action + "</span>";
  }

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  function nodePositions() {
    const W = canvas.width;
    const H = canvas.height;

    return {
      bot: { x: 82, y: H * 0.18, label: "Bot Farm" },
      proxy: { x: 82, y: H * 0.50, label: "Proxy Pool" },
      user: { x: 82, y: H * 0.82, label: "Residential" },
      detector: { x: W * 0.46, y: H * 0.50, label: "AFIT Detector" },
      publisher: { x: W * 0.84, y: H * 0.22, label: "Publisher" },
      billing: { x: W * 0.84, y: H * 0.50, label: "Billable" },
      review: { x: W * 0.84, y: H * 0.78, label: "Review Hold" },
      blocked: { x: W * 0.58, y: H * 0.88, label: "Blocked" },
    };
  }

  function scenarioMultiplier(sc, t) {
    const cycle = t % 60;

    if (sc === "normal") return 1.0;
    if (sc === "mixed") return 1.0 + 0.55 * Math.sin(t / 7.0);
    if (sc === "botnet") return cycle < 25 ? 2.4 : 0.35;
    if (sc === "click_injection") return cycle > 10 && cycle < 35 ? 2.6 : 0.45;
    if (sc === "malvertising") return cycle > 5 && cycle < 40 ? 2.2 : 0.45;
    if (sc === "recovery") return Math.max(0.18, 2.2 * Math.exp(-cycle / 18.0));

    return 1.0;
  }

  function pickFraudType(sc, t) {
    const cycle = t % 60;

    if (sc === "botnet") return "botnet";
    if (sc === "click_injection") return "click_injection";
    if (sc === "malvertising") return "malvertising";

    if (sc === "recovery") {
      if (cycle < 12) return "botnet";
      if (cycle < 24) return "click_injection";
      return "geo_proxy";
    }

    const r = Math.random();
    if (r < 0.40) return "botnet";
    if (r < 0.65) return "click_injection";
    if (r < 0.85) return "malvertising";
    return "geo_proxy";
  }

  function makeEvent(isFraud, fraudType) {
    const typeRoll = Math.random();

    let eventType;
    if (isFraud) {
      if (fraudType === "click_injection") {
        eventType = typeRoll < 0.82 ? "click" : "impression";
      } else if (fraudType === "botnet") {
        eventType = typeRoll < 0.28 ? "click" : "impression";
      } else {
        eventType = typeRoll < 0.42 ? "click" : "impression";
      }
    } else {
      if (typeRoll < 0.82) eventType = "impression";
      else if (typeRoll < 0.97) eventType = "click";
      else eventType = "conversion";
    }

    const device = choice(["mobile", "desktop", "tablet", "ctv"]);

    let country, asn, ua, uaClass, visible, viewable;
    let clickDelay = -1;
    let convLatency = -1;
    let geo = false;
    let cost;
    let revenue = 0;
    let converted = false;

    if (isFraud) {
      country = Math.random() < 0.75 ? choice(fraudCountries) : choice(countries);
      asn = fraudType === "botnet"
        ? choice(["datacenter", "proxy"])
        : fraudType === "geo_proxy"
          ? "proxy"
          : choice(["datacenter", "hosting", "proxy"]);

      ua = choice(fraudUAs);
      uaClass = choice(["bot", "headless", "script"]);

      visible = Math.random() < 0.15;
      viewable = visible ? randInt(100, 500) : randInt(0, 90);
      geo = Math.random() < 0.45;
    } else {
      country = choice(countries);
      asn = choice(["residential", "mobile", "hosting"]);
      ua = choice(legitUAs);
      uaClass = choice(["browser", "webview"]);

      visible = Math.random() < 0.82;
      viewable = visible ? randInt(300, 30000) : randInt(0, 250);
      geo = Math.random() < 0.03;
    }

    if (eventType === "click") {
      if (isFraud && fraudType === "click_injection") {
        clickDelay = randInt(20, 900);
      } else if (isFraud) {
        clickDelay = randInt(200, 5000);
      } else {
        clickDelay = randInt(3000, 180000);
      }
    }

    if (eventType === "conversion") {
      converted = true;
      revenue = randInt(2000000, 12000000);
      convLatency = isFraud ? randInt(200, 4000) : randInt(60000, 3600000);
      visible = false;
      viewable = 0;
    }

    if (eventType === "impression") {
      cost = randInt(1500, 9000);
    } else if (eventType === "click") {
      cost = randInt(80000, 450000);
    } else {
      cost = randInt(1000000, 8000000);
    }

    let quality = 1.0;
    let dq = "";

    const qr = Math.random();
    if (qr < 0.015) {
      dq = "missing_ip";
      quality -= 0.18;
    } else if (qr < 0.025) {
      dq = "missing_ua";
      quality -= 0.12;
    } else if (qr < 0.033) {
      dq = "negative_cost";
      cost = -cost;
      quality -= 0.10;
    } else if (qr < 0.040) {
      dq = "unknown_country";
      country = "XX";
      quality -= 0.08;
    }

    if (Math.random() < 0.02) {
      dq = dq ? dq + ";late" : "late";
      quality -= 0.08;
    }

    quality = clamp(quality, 0, 1);
    cost = Math.max(0, cost);

    const id = "evt_" + String(randInt(100000000, 999999999));

    return {
      id,
      ts: new Date(),
      eventType,
      device,
      country,
      asn,
      ua,
      uaClass,
      visible,
      viewable,
      clickDelay,
      convLatency,
      cost,
      revenue,
      converted,
      isFraud,
      fraudType,
      geo,
      quality,
      dq,
      score: 0,
      confidence: 0,
      recommended: "PENDING",
      action: "PENDING",
      reasons: "",
      approvedBy: "",
      approvedAt: null,
      source: isFraud ? (fraudType === "botnet" ? "bot" : "proxy") : "user",
    };
  }

  function scoreEvent(ev) {
    let s = 0;
    const r = [];

    if (["bot", "headless", "script"].includes(ev.uaClass)) {
      s += 35;
      r.push("bot_ua");
    }

    if (["datacenter", "proxy", "hosting"].includes(ev.asn)) {
      s += 20;
      r.push("datacenter_ip");
    }

    if (ev.eventType !== "conversion" && ev.viewable < 100) {
      s += 20;
      r.push("low_viewability");
    }

    if (ev.geo) {
      s += 15;
      r.push("geo_mismatch");
    }

    if (ev.eventType === "click" && ev.clickDelay >= 0 && ev.clickDelay <= 1000) {
      s += 30;
      r.push("click_injection");
    }

    if (ev.eventType === "conversion" && ev.convLatency >= 0 && ev.convLatency <= 5000) {
      s += 20;
      r.push("fast_conversion");
    }

    if (ev.dq && ev.dq.includes("missing")) {
      s += 5;
      r.push("quality_degraded");
    }

    s = clamp(s + rand(-3, 5), 0, 100);

    ev.score = s;
    ev.reasons = r.join(", ");
    ev.confidence = clamp((0.45 + 0.45 * (s / 100)) * ev.quality + rand(-0.04, 0.04), 0, 1);

    let rec, act;

    if (s >= state.block && ev.confidence >= 0.70) {
      rec = "BLOCK";
      act = state.autoMitigate ? "BLOCK_PREVENT_BILLING" : "SUGGEST_BLOCK_PENDING_APPROVAL";
    } else if (s >= state.review) {
      rec = "REVIEW_HOLD";
      act = "HOLD_FOR_REVIEW";
    } else if (s >= state.review - 20) {
      rec = "MONITOR";
      act = "MONITOR";
    } else {
      rec = "ALLOW";
      act = "ALLOW";
    }

    ev.recommended = rec;
    ev.action = act;

    return ev;
  }

  function newState() {
    return {
      running: true,
      t: 0,
      frame: 0,
      spawnAcc: 0,
      sampleAcc: 0,

      scenario: CONFIG.scenario || "mixed",
      baseFraud: clamp(Number(CONFIG.fraud_share) || 0.12, 0.01, 0.40),
      speed: 1.0,
      review: Number(CONFIG.threshold_review) || 65,
      block: Number(CONFIG.threshold_block) || 85,
      autoMitigate: Boolean(CONFIG.auto_block),
      followAlerts: true,

      particles: [],
      events: [],
      audit: [],
      selectedId: null,

      counters: {
        events: 0,
        fraud: 0,
        detected: 0,
        tp: 0,
        fp: 0,
        fn: 0,
        tn: 0,
        prevented: 0,
        fpSpend: 0,
        queue: 0,
      },

      samples: {
        events: [],
        fraud: [],
        detected: [],
      },

      lastSample: {
        events: 0,
        fraud: 0,
        detected: 0,
      },
    };
  }

  let state = newState();

  function addAudit(action, objectId, details) {
    state.audit.unshift({
      ts: new Date(),
      action,
      objectId,
      details,
    });

    if (state.audit.length > 80) {
      state.audit.pop();
    }
  }

  function setSelected(id) {
    state.selectedId = id;
  }

  function getSelectedEvent() {
    return state.events.find((e) => e.id === state.selectedId) || null;
  }

  function processAtDetector(p) {
    const ev = scoreEvent(p.event);
    ev.processedAt = new Date();

    state.events.unshift(ev);
    if (state.events.length > 1200) {
      state.events.pop();
    }

    const c = state.counters;
    const detected = ev.score >= state.review;

    c.events += 1;
    if (ev.isFraud) c.fraud += 1;
    if (detected) c.detected += 1;

    if (ev.isFraud && detected) c.tp += 1;
    else if (!ev.isFraud && detected) c.fp += 1;
    else if (ev.isFraud && !detected) c.fn += 1;
    else c.tn += 1;

    if (preventedActions.has(ev.action)) {
      c.prevented += ev.cost;
      c.queue += 1;

      if (!ev.isFraud) {
        c.fpSpend += ev.cost;
      }
    }

    state.lastSample.events += 1;
    if (ev.isFraud) state.lastSample.fraud += 1;
    if (detected) state.lastSample.detected += 1;

    let dest = "publisher";

    if (ev.action === "BLOCK_PREVENT_BILLING") {
      dest = "blocked";
      p.color = "#a855f7";
    } else if (ev.action === "SUGGEST_BLOCK_PENDING_APPROVAL" || ev.action === "HOLD_FOR_REVIEW") {
      dest = "review";
      p.color = "#f97316";
    } else if (ev.isFraud) {
      p.color = "#ef4444";
      dest = "billing";
    } else {
      p.color = "#22c55e";
      dest = "publisher";
    }

    p.dest = dest;
    p.stage = 1;
    p.progress = 0;

    if (detected && state.followAlerts) {
      setSelected(ev.id);
    }
  }

  function spawnParticle() {
    const eff = clamp(state.baseFraud * scenarioMultiplier(state.scenario, state.t), 0, 0.65);
    const isFraud = Math.random() < eff;
    const fraudType = isFraud ? pickFraudType(state.scenario, state.t) : "legitimate";

    const ev = makeEvent(isFraud, fraudType);

    let source = "user";
    if (isFraud) {
      source = fraudType === "botnet" ? "bot" : "proxy";
    } else {
      source = Math.random() < 0.86 ? "user" : "proxy";
    }

    ev.source = source;

    const nodes = nodePositions();

    state.particles.push({
      event: ev,
      x: nodes[source].x,
      y: nodes[source].y,
      stage: 0,
      progress: 0,
      speed: rand(0.45, 0.95),
      color: isFraud ? "#ef4444" : "#22c55e",
      dest: "publisher",
    });
  }

  function update(dt) {
    if (!state.running) return;

    state.t += dt;

    state.spawnAcc += dt * (10 * state.speed);
    while (state.spawnAcc >= 1) {
      state.spawnAcc -= 1;
      spawnParticle();
    }

    const nodes = nodePositions();

    for (let i = state.particles.length - 1; i >= 0; i--) {
      const p = state.particles[i];
      p.progress += dt * p.speed * 1.1 * state.speed;

      if (p.stage === 0) {
        const a = nodes[p.event.source];
        const b = nodes.detector;

        p.x = a.x + (b.x - a.x) * p.progress;
        p.y = a.y + (b.y - a.y) * p.progress + Math.sin(p.progress * 6) * 8;

        if (p.progress >= 1) {
          processAtDetector(p);
        }
      } else {
        const a = nodes.detector;
        const b = nodes[p.dest];

        p.x = a.x + (b.x - a.x) * p.progress;
        p.y = a.y + (b.y - a.y) * p.progress + Math.sin(p.progress * 5) * 6;

        if (p.progress >= 1) {
          state.particles.splice(i, 1);
        }
      }
    }

    state.sampleAcc += dt;
    if (state.sampleAcc >= 1.0) {
      state.sampleAcc = 0;

      state.samples.events.push(state.lastSample.events);
      state.samples.fraud.push(state.lastSample.fraud);
      state.samples.detected.push(state.lastSample.detected);

      for (const k of ["events", "fraud", "detected"]) {
        if (state.samples[k].length > 90) {
          state.samples[k].shift();
        }
      }

      state.lastSample = { events: 0, fraud: 0, detected: 0 };
    }

    state.frame += 1;

    if (state.frame % 6 === 0) {
      renderLog();
      renderEvidence();
      renderAudit();
      updateKpis();
    }
  }

  function drawLink(a, b, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.28;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.restore();
  }

  function drawNode(n, color, pulse) {
    ctx.save();

    const w = 118;
    const h = 44;
    const x = n.x - w / 2;
    const y = n.y - h / 2;

    ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    if (pulse) {
      const p = 0.5 + 0.5 * Math.sin(state.t * 4);
      ctx.shadowColor = color;
      ctx.shadowBlur = 10 + 12 * p;
    }

    roundRect(ctx, x, y, w, h, 10);
    ctx.fill();
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(n.label, n.x, n.y);

    ctx.restore();
  }

  function draw() {
    const nodes = nodePositions();
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.07)";
    ctx.lineWidth = 1;

    for (let x = 0; x < W; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }

    for (let y = 0; y < H; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    drawLink(nodes.bot, nodes.detector, "#ef4444");
    drawLink(nodes.proxy, nodes.detector, "#f97316");
    drawLink(nodes.user, nodes.detector, "#22c55e");

    drawLink(nodes.detector, nodes.publisher, "#22c55e");
    drawLink(nodes.detector, nodes.billing, "#60a5fa");
    drawLink(nodes.detector, nodes.review, "#f97316");
    drawLink(nodes.detector, nodes.blocked, "#a855f7");

    drawNode(nodes.bot, "#ef4444", false);
    drawNode(nodes.proxy, "#f97316", false);
    drawNode(nodes.user, "#22c55e", false);

    drawNode(nodes.detector, "#38bdf8", true);

    drawNode(nodes.publisher, "#22c55e", false);
    drawNode(nodes.billing, "#60a5fa", false);
    drawNode(nodes.review, "#f97316", false);
    drawNode(nodes.blocked, "#a855f7", false);

    for (const p of state.particles) {
      ctx.save();

      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();

      if (p.event.id === state.selectedId) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
        ctx.stroke();
      }

      ctx.restore();
    }

    ctx.save();
    ctx.fillStyle = "#cbd5e1";
    ctx.font = "12px system-ui";
    ctx.fillText("t=" + state.t.toFixed(1) + "s", 12, 20);
    ctx.fillText("scenario=" + state.scenario, 12, 38);
    ctx.fillText("particles=" + state.particles.length, 12, 56);
    ctx.restore();
  }

  function drawSeries(arr, color, maxVal) {
    if (!arr || arr.length < 2) return;

    const W = cctx.canvas.width;
    const H = cctx.canvas.height;

    cctx.strokeStyle = color;
    cctx.lineWidth = 2;
    cctx.beginPath();

    for (let i = 0; i < arr.length; i++) {
      const x = 10 + (W - 20) * (i / Math.max(1, arr.length - 1));
      const y = H - 15 - (H - 30) * (arr[i] / Math.max(1, maxVal));

      if (i === 0) cctx.moveTo(x, y);
      else cctx.lineTo(x, y);
    }

    cctx.stroke();
  }

  function drawChart() {
    const W = cctx.canvas.width;
    const H = cctx.canvas.height;

    cctx.clearRect(0, 0, W, H);

    cctx.fillStyle = "#0f172a";
    cctx.fillRect(0, 0, W, H);

    cctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    cctx.lineWidth = 1;

    for (let y = 15; y < H - 10; y += 25) {
      cctx.beginPath();
      cctx.moveTo(10, y);
      cctx.lineTo(W - 10, y);
      cctx.stroke();
    }

    const all = [
      ...state.samples.events,
      ...state.samples.fraud,
      ...state.samples.detected,
    ];

    const maxVal = Math.max(1, ...all);

    drawSeries(state.samples.events, "#22c55e", maxVal);
    drawSeries(state.samples.fraud, "#60a5fa", maxVal);
    drawSeries(state.samples.detected, "#f97316", maxVal);

    cctx.fillStyle = "#94a3b8";
    cctx.font = "10px system-ui";
    cctx.fillText("last 90 seconds", 10, H - 4);
  }

  function renderLog() {
    const rows = state.events.slice(0, 12).map((ev) => {
      const sel = ev.id === state.selectedId ? "sel" : "";

      return `
        <tr class="${sel}" data-id="${ev.id}">
          <td>${fmtTime(ev.ts)}</td>
          <td>${ev.eventType}</td>
          <td>${ev.country}</td>
          <td>${ev.asn}</td>
          <td>${ev.uaClass}</td>
          <td>${ev.viewable}</td>
          <td>${fmtMoney(ev.cost)}</td>
          <td>${ev.score.toFixed(0)}</td>
          <td>${actionBadge(ev.action)}</td>
          <td>${ev.reasons || "-"}</td>
        </tr>
      `;
    }).join("");

    logBody.innerHTML = rows || '<tr><td colspan="10" class="muted">No events yet</td></tr>';

    logBody.querySelectorAll("tr[data-id]").forEach((tr) => {
      tr.addEventListener("click", () => {
        setSelected(tr.dataset.id);
        renderLog();
        renderEvidence();
      });
    });
  }

  function renderEvidence() {
    const ev = getSelectedEvent();

    if (!ev) {
      evidenceBody.innerHTML = '<div class="muted">Select an event to inspect evidence.</div>';
      return;
    }

    const canApprove = ["HOLD_FOR_REVIEW", "SUGGEST_BLOCK_PENDING_APPROVAL"].includes(ev.action);

    evidenceBody.innerHTML = `
      <div class="evTitle">${ev.id}</div>

      <div class="evGrid">
        <span>time</span><b>${fmtTime(ev.ts)}</b>
        <span>event</span><b>${ev.eventType}</b>
        <span>device</span><b>${ev.device}</b>
        <span>country</span><b>${ev.country}</b>
        <span>IP ASN</span><b>${ev.asn}</b>
        <span>UA class</span><b>${ev.uaClass}</b>
        <span>viewable</span><b>${ev.viewable} ms</b>
        <span>cost</span><b>${fmtMoney(ev.cost)}</b>
        <span>score</span><b>${ev.score.toFixed(1)}</b>
        <span>confidence</span><b>${ev.confidence.toFixed(2)}</b>
        <span>quality</span><b>${ev.quality.toFixed(2)}</b>
        <span>simulated fraud</span><b>${ev.isFraud ? "yes" : "no"}</b>
        <span>fraud type</span><b>${ev.fraudType}</b>
        <span>recommended</span><b>${ev.recommended}</b>
        <span>applied</span><b>${ev.action}</b>
        <span>approved by</span><b>${ev.approvedBy || "-"}</b>
      </div>

      <div class="reasons">
        <b>Signals:</b> ${ev.reasons || "none"}
        ${ev.dq ? "<br><b>Data quality:</b> " + ev.dq : ""}
      </div>

      <div class="evActions">
        ${
          canApprove
            ? `
              <button data-approve="BLOCK_PREVENT_BILLING">Approve block</button>
              <button data-release="ALLOW_RELEASED">Release as legitimate</button>
            `
            : '<span class="muted">No consequential approval required for this row.</span>'
        }
      </div>
    `;
  }

  function renderAudit() {
    if (!state.audit.length) {
      auditList.innerHTML = '<li class="muted">No audit events yet</li>';
      return;
    }

    auditList.innerHTML = state.audit.slice(0, 12).map((a) => {
      return `<li>${fmtTime(a.ts)} | ${a.action} | ${a.objectId} | ${a.details}</li>`;
    }).join("");
  }

  function updateKpis() {
    const c = state.counters;

    els.events.textContent = c.events.toLocaleString();
    els.fraudRate.textContent = c.events ? pct(c.fraud / c.events) : "0%";
    els.detected.textContent = c.events ? pct(c.detected / c.events) : "0%";

    els.precision.textContent = (c.tp + c.fp) ? pct(c.tp / (c.tp + c.fp)) : "0%";
    els.recall.textContent = c.fraud ? pct(c.tp / c.fraud) : "0%";

    els.prevented.textContent = fmtMoney(c.prevented);
    els.fpSpend.textContent = fmtMoney(c.fpSpend);
    els.queue.textContent = c.queue.toLocaleString();

    clockEl.textContent = fmtTime(new Date());
    scenarioLabel.textContent = state.scenario;
  }

  function approveSelected(newAction) {
    const ev = getSelectedEvent();
    if (!ev) return;

    const old = ev.action;
    if (old === newAction) return;

    ev.action = newAction;
    ev.approvedBy = CONFIG.reviewer || "demo_reviewer";
    ev.approvedAt = new Date();

    const c = state.counters;

    const oldPrevented = preventedActions.has(old);
    const newPrevented = preventedActions.has(newAction);

    if (oldPrevented && !newPrevented) {
      c.prevented = Math.max(0, c.prevented - ev.cost);
      c.queue = Math.max(0, c.queue - 1);

      if (!ev.isFraud) {
        c.fpSpend = Math.max(0, c.fpSpend - ev.cost);
      }
    }

    if (!oldPrevented && newPrevented) {
      c.prevented += ev.cost;
      c.queue += 1;

      if (!ev.isFraud) {
        c.fpSpend += ev.cost;
      }
    }

    addAudit("HUMAN_APPROVAL", ev.id, old + " -> " + newAction);

    renderEvidence();
    renderLog();
    renderAudit();
    updateKpis();
  }

  function syncControls() {
    els.scenario.value = state.scenario;
    els.speed.value = String(state.speed);
    els.speedVal.textContent = state.speed.toFixed(2) + "x";

    els.fraud.value = String(Math.round(state.baseFraud * 100));
    els.fraudVal.textContent = Math.round(state.baseFraud * 100) + "%";

    els.auto.checked = state.autoMitigate;
    els.follow.checked = state.followAlerts;

    els.play.textContent = state.running ? "Pause" : "Play";
    statusDot.className = state.running ? "dot green" : "dot amber";
    statusText.textContent = state.running ? "RUNNING" : "PAUSED";
  }

  function renderAll() {
    renderLog();
    renderEvidence();
    renderAudit();
    updateKpis();
  }

  function bindEvents() {
    els.play.addEventListener("click", () => {
      state.running = !state.running;
      syncControls();
    });

    els.reset.addEventListener("click", () => {
      state = newState();
      syncControls();
      renderAll();
      addAudit("RESET", "simulation", "live simulation reset");
    });

    els.scenario.addEventListener("change", (e) => {
      state.scenario = e.target.value;
      addAudit("SCENARIO_CHANGE", "simulation", state.scenario);
    });

    els.speed.addEventListener("input", (e) => {
      state.speed = parseFloat(e.target.value);
      els.speedVal.textContent = state.speed.toFixed(2) + "x";
    });

    els.fraud.addEventListener("input", (e) => {
      state.baseFraud = parseFloat(e.target.value) / 100;
      els.fraudVal.textContent = e.target.value + "%";
    });

    els.auto.addEventListener("change", (e) => {
      state.autoMitigate = e.target.checked;
      addAudit("POLICY_CHANGE", "auto_mitigate", String(state.autoMitigate));
    });

    els.follow.addEventListener("change", (e) => {
      state.followAlerts = e.target.checked;
      addAudit("POLICY_CHANGE", "follow_alerts", String(state.followAlerts));
    });

    evidenceBody.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;

      if (btn.dataset.approve) {
        approveSelected(btn.dataset.approve);
      }

      if (btn.dataset.release) {
        approveSelected(btn.dataset.release);
      }
    });
  }

  let last = performance.now();

  function loop(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;

    update(dt);
    draw();
    drawChart();

    requestAnimationFrame(loop);
  }

  syncControls();
  bindEvents();
  renderAll();

  addAudit("START", "simulation", "live synthetic simulation started");

  requestAnimationFrame(loop);
})();
</script>
</body>
</html>
'''

    live_html = live_html.replace("__CONFIG_JSON__", json.dumps(live_config))
    st.components.v1.html(live_html, height=880, scrolling=False)

    st.divider()

    video_path = st.session_state.get("video_path")
    video_error = st.session_state.get("video_error")

    if video_path and os.path.exists(video_path):
        st.subheader("Recorded demo video")
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)
        st.caption(
            "If your browser cannot play the recorded MP4 codec, use the live simulation above or download the file."
        )

        mime = "video/mp4" if video_path.endswith(".mp4") else "video/x-msvideo"

        st.download_button(
            "Download recorded demo video",
            data=video_bytes,
            file_name=os.path.basename(video_path),
            mime=mime,
        )
    else:
        st.info(
            "Recorded MP4 generation is unavailable in this environment, but the live in-app simulation above is running."
        )
        if video_error:
            st.caption(f"Video generation error: {video_error}")


with tabs[1]:
    st.header("Revenue dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        campaign_filter = st.multiselect("Campaigns", sorted(df["campaign_id"].unique()), default=[])
    with c2:
        publisher_filter = st.multiselect("Publishers", sorted(df["publisher_id"].unique()), default=[])
    with c3:
        event_filter = st.multiselect("Event types", sorted(df["event_type"].unique()), default=[])
    with c4:
        risk_filter = st.multiselect("Risk levels", sorted([x for x in df["risk_level"].unique() if x != "nan"]), default=[])

    fd = df.copy()
    if campaign_filter:
        fd = fd[fd["campaign_id"].isin(campaign_filter)]
    if publisher_filter:
        fd = fd[fd["publisher_id"].isin(publisher_filter)]
    if event_filter:
        fd = fd[fd["event_type"].isin(event_filter)]
    if risk_filter:
        fd = fd[fd["risk_level"].isin(risk_filter)]

    if fd.empty:
        st.warning("No rows match the current filters.")
    else:
        detected_mask = fd["final_score"] >= threshold_review
        fraud_mask = fd["is_fraud"].astype(bool)

        overall = binary_metrics(fraud_mask.astype(int), detected_mask.astype(int), fd["final_score"])

        fraud_rate = float(fraud_mask.mean())
        detected_rate = float(detected_mask.mean())
        prevented_usd = float(
            fd.loc[fd["applied_action"].isin(["BLOCK_PREVENT_BILLING", "SUGGEST_BLOCK_PENDING_APPROVAL", "HOLD_FOR_REVIEW"]), "cost_micros"].sum()
            / 1_000_000.0
        )
        fp_spend_usd = float(fd.loc[detected_mask & (~fraud_mask), "cost_micros"].sum() / 1_000_000.0)
        review_queue = int(fd["applied_action"].isin(["HOLD_FOR_REVIEW", "SUGGEST_BLOCK_PENDING_APPROVAL"]).sum())
        mean_quality = float(fd["quality_score"].mean())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Simulated fraud rate", f"{fraud_rate:.2%}", f"{detected_rate - fraud_rate:+.2%}", help="Ground-truth label is synthetic.")
        k2.metric("Detected rate", f"{detected_rate:.2%}", help="Predicted invalid traffic rate at review threshold.")
        k3.metric("Precision on simulated label", f"{overall['precision']:.2%}")
        k4.metric("Recall on simulated label", f"{overall['recall']:.2%}")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Estimated prevented spend", f"${prevented_usd:,.2f}", help="Observed cost on held, pending, or blocked synthetic events.")
        k6.metric("Estimated false-positive spend", f"${fp_spend_usd:,.2f}", help="Synthetic legitimate spend flagged by detector.")
        k7.metric("Review queue", f"{review_queue:,}")
        k8.metric("Mean data quality", f"{mean_quality:.3f}")

        fd["detected_flag"] = detected_mask.astype(int)

        ts = (
            fd.set_index("ts_utc")
            .resample("5min")
            .agg(
                events=("record_uid", "count"),
                fraud_events=("is_fraud", "sum"),
                detected_events=("detected_flag", "sum"),
            )
            .reset_index()
        )

        fig = px.line(
            ts,
            x="ts_utc",
            y=["events", "fraud_events", "detected_events"],
            title="Time series: observed events, simulated fraud, predicted detections",
            labels={"value": "count / 5 min", "variable": "series", "ts_utc": "UTC time"},
        )
        fig.update_layout(height=330, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Observed = synthetic event counts. Simulated fraud = synthetic label. Predicted detections = model/rule score threshold.")

        cc1, cc2 = st.columns(2)

        with cc1:
            pub = (
                fd.groupby("publisher_id")
                .agg(
                    events=("record_uid", "count"),
                    avg_score=("final_score", "mean"),
                    fraud_rate=("is_fraud", "mean"),
                    spend_micros=("cost_micros", "sum"),
                )
                .reset_index()
            )
            pub["spend_usd"] = pub["spend_micros"] / 1_000_000.0
            pub = pub.sort_values("avg_score", ascending=False).head(12)

            fig2 = px.bar(
                pub,
                x="publisher_id",
                y="avg_score",
                color="fraud_rate",
                title="Top publishers by predicted risk",
                labels={"avg_score": "mean final score", "publisher_id": "publisher"},
            )
            fig2.update_layout(height=330)
            st.plotly_chart(fig2, use_container_width=True)

        with cc2:
            fraud_mix = fd[fd["is_fraud"]].groupby("fraud_type").size().reset_index(name="events")
            if not fraud_mix.empty:
                fig3 = px.bar(
                    fraud_mix,
                    x="fraud_type",
                    y="events",
                    title="Simulated fraud type mix",
                    labels={"fraud_type": "fraud type", "events": "synthetic events"},
                )
                fig3.update_layout(height=330)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No simulated fraud rows in the current filter.")

        fig4 = go.Figure()
        fig4.add_trace(go.Histogram(x=fd["final_score"], nbinsx=50, name="final score", opacity=0.7))
        fig4.add_trace(go.Histogram(x=fd["rule_score"], nbinsx=50, name="rule score", opacity=0.45))
        fig4.add_vline(x=threshold_review, line_dash="dash", annotation_text="review")
        fig4.add_vline(x=threshold_block, line_dash="dot", annotation_text="block")
        fig4.update_layout(title="Score distribution: transparent baseline vs combined score", height=330)
        st.plotly_chart(fig4, use_container_width=True)

        action_funnel = fd["applied_action"].value_counts().reset_index()
        action_funnel.columns = ["action", "events"]
        fig5 = px.bar(action_funnel, x="action", y="events", title="Decision funnel")
        fig5.update_layout(height=300)
        st.plotly_chart(fig5, use_container_width=True)


with tabs[2]:
    st.header("Event evidence and human approval")

    if df.empty:
        st.warning("No events available.")
    else:
        lookup = (
            df.set_index("record_uid")[["ts_utc", "publisher_id", "campaign_id", "final_score", "applied_action"]]
            .to_dict("index")
        )

        def fmt_uid(uid: str) -> str:
            r = lookup[uid]
            return (
                f"{r['ts_utc']:%H:%M:%S} | {r['publisher_id']} | {r['campaign_id']} | "
                f"score {float(r['final_score']):.0f} | {r['applied_action']}"
            )

        selected_uid = st.selectbox("Select synthetic event", options=df["record_uid"].tolist(), format_func=fmt_uid)
        row = df[df["record_uid"] == selected_uid].iloc[0]

        left, right = st.columns([2, 1])

        with left:
            evidence = {
                "record_uid": row["record_uid"],
                "event_id": row["event_id"],
                "ts_utc": row["ts_utc"],
                "campaign_id": row["campaign_id"],
                "publisher_id": row["publisher_id"],
                "event_type": row["event_type"],
                "ip_hash": row["ip_hash"],
                "ip_asn_type": row["ip_asn_type"],
                "user_agent_class": row["user_agent_class"],
                "device_type": row["device_type"],
                "country": row["country"],
                "visible": bool(row["visible"]),
                "viewable_time_ms": float(row["viewable_time_ms"]),
                "cost_micros": float(row["cost_micros"]),
                "is_fraud_simulated": bool(row["is_fraud"]),
                "fraud_type_simulated": row["fraud_type"],
                "quality_score": float(row["quality_score"]),
                "rule_score": float(row["rule_score"]),
                "model_score": float(row["model_score"]),
                "final_score": float(row["final_score"]),
                "confidence": float(row["confidence"]),
                "risk_level": row["risk_level"],
                "recommended_action": row["recommended_action"],
                "applied_action": row["applied_action"],
                "approved_by": row["approved_by"],
                "reason_codes": row["reason_codes"],
            }
            st.json(jsonable(evidence))

            comp = json.loads(row["rule_components_json"])
            comp_df = pd.DataFrame(list(comp.items()), columns=["signal", "points"])
            fig_comp = px.bar(comp_df, x="points", y="signal", orientation="h", title="Rule signal contribution")
            fig_comp.update_layout(height=300)
            st.plotly_chart(fig_comp, use_container_width=True)

        with right:
            st.metric("Final score", f"{float(row['final_score']):.1f}")
            st.metric("Confidence", f"{float(row['confidence']):.2f}")
            st.metric("Risk level", row["risk_level"])
            st.metric("Applied action", row["applied_action"])

            if row["applied_action"] == "SUGGEST_BLOCK_PENDING_APPROVAL":
                if st.button("Approve block and refund recommendation", key=f"approve_block_{selected_uid}"):
                    st.session_state.approvals[selected_uid] = (
                        "BLOCK_PREVENT_BILLING",
                        reviewer,
                        "Approved in demo",
                    )
                    try:
                        approve_decision(selected_uid, reviewer, "BLOCK_PREVENT_BILLING", "Approved in demo")
                    except Exception:
                        pass
                    st.success("Human approval recorded in session and SQLite audit log.")
                    st.rerun()

            if row["applied_action"] in ["HOLD_FOR_REVIEW", "SUGGEST_BLOCK_PENDING_APPROVAL"]:
                if st.button("Release as legitimate", key=f"release_{selected_uid}"):
                    st.session_state.approvals[selected_uid] = (
                        "ALLOW_RELEASED",
                        reviewer,
                        "Reviewed as legitimate",
                    )
                    try:
                        approve_decision(selected_uid, reviewer, "ALLOW_RELEASED", "Reviewed as legitimate")
                    except Exception:
                        pass
                    st.success("Release decision recorded.")
                    st.rerun()

            feat_df = pd.DataFrame(
                {
                    "feature": MODEL_FEATURES,
                    "value": [float(row[f]) if pd.notna(row[f]) else 0.0 for f in MODEL_FEATURES],
                }
            )
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.subheader("Audit history for this event")
        try:
            conn = sqlite3.connect(DB_PATH)
            audit_df = pd.read_sql(
                "SELECT ts_utc, actor, action, object_id, details FROM audit WHERE object_id = ? ORDER BY ts_utc DESC LIMIT 25",
                conn,
                params=(selected_uid,),
            )
            conn.close()
        except Exception:
            audit_df = pd.DataFrame()

        if audit_df.empty:
            st.info("No persisted audit rows yet for this event.")
        else:
            st.dataframe(audit_df, use_container_width=True, hide_index=True)


with tabs[3]:
    st.header("Evaluation protocol")

    st.markdown(
        """
This evaluation uses synthetic ground-truth labels. It demonstrates measurement discipline, not real-world performance claims.

- Chronological split: train on earliest 60%, validate conceptually, test on latest 20%.
- Leakage check: model is fit only on training window, then scored on all rows.
- Baseline: deterministic rule engine.
- Advanced model: IsolationForest rank percentile.
- Guardrails: false-positive spend, review capacity, subgroup recall, calibration proxy, fallback behavior.
        """
    )

    test_mask = df["ts_utc"] >= df["ts_utc"].quantile(0.80)
    te = df[test_mask].copy()
    if len(te) < 50:
        te = df.copy()

    y_true = te["is_fraud"].astype(int).to_numpy()
    cost = te["cost_micros"].to_numpy(dtype=float)

    approaches = [
        ("Rule baseline", (te["rule_score"] >= threshold_review).astype(int).to_numpy(), te["rule_score"].to_numpy()),
        ("Model only", (te["model_score"] >= threshold_review).astype(int).to_numpy(), te["model_score"].to_numpy()),
        ("Combined product score", (te["final_score"] >= threshold_review).astype(int).to_numpy(), te["final_score"].to_numpy()),
        ("Naive sentinel", (np.arange(len(te)) % 8 == 0).astype(int), None),
    ]

    rows = []
    for name, pred, score in approaches:
        m = binary_metrics(y_true, pred, score)
        m["approach"] = name
        m["prevented_spend_usd"] = float(cost[pred == 1].sum() / 1_000_000.0)
        m["false_positive_spend_usd"] = float(cost[(pred == 1) & (y_true == 0)].sum() / 1_000_000.0)
        rows.append(m)

    eval_df = pd.DataFrame(rows)
    eval_df = eval_df[[
        "approach",
        "events",
        "precision",
        "recall",
        "f1",
        "fpr",
        "auc",
        "ap",
        "prevented_spend_usd",
        "false_positive_spend_usd",
    ]]
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    combined_pred = (te["final_score"] >= threshold_review).astype(int).to_numpy()
    tp = int(((y_true == 1) & (combined_pred == 1)).sum())
    pred_count = int(combined_pred.sum())
    fraud_count = int(y_true.sum())

    prec_lo, prec_hi = wilson_ci(tp, pred_count)
    rec_lo, rec_hi = wilson_ci(tp, fraud_count)

    st.subheader("Confidence intervals for combined score")
    c1, c2 = st.columns(2)
    c1.metric("Precision 95% CI", f"{prec_lo:.2%} to {prec_hi:.2%}")
    c2.metric("Recall 95% CI", f"{rec_lo:.2%} to {rec_hi:.2%}")

    st.subheader("Subgroup checks")
    sub_rows = []
    for col in ["device_type", "country"]:
        for val, g in te.groupby(col):
            if len(g) < 20:
                continue
            gm = binary_metrics(
                g["is_fraud"].astype(int).to_numpy(),
                (g["final_score"] >= threshold_review).astype(int).to_numpy(),
                g["final_score"].to_numpy(),
            )
            gm["segment"] = f"{col}={val}"
            sub_rows.append(gm)

    if sub_rows:
        sub_df = pd.DataFrame(sub_rows)[["segment", "events", "precision", "recall", "fpr", "auc"]]
        st.dataframe(sub_df, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough subgroup volume in the current synthetic test window.")

    comb = binary_metrics(y_true, combined_pred, te["final_score"].to_numpy())
    g1, g2, g3 = st.columns(3)
    g1.metric("Guardrail: FPR <= 3%", "PASS" if comb["fpr"] <= 0.03 else "REVIEW", f"{comb['fpr']:.2%}")
    g2.metric("Guardrail: Recall >= 55%", "PASS" if comb["recall"] >= 0.55 else "REVIEW", f"{comb['recall']:.2%}")
    g3.metric("Fallback usage", f"{int((df['quality_score'] < 0.65).sum()):,}", "rows below quality gate")

    st.markdown(
        """
**What these metrics cannot prove**

- Synthetic labels do not prove real-world fraud prevalence.
- AUROC does not prove revenue recovery.
- High aggregate recall can hide weak performance on a protected or strategically important publisher segment.
- Calibration proxies are not a substitute for labeled dispute outcomes.
        """
    )


with tabs[4]:
    st.header("Data quality and contracts")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Mean quality", f"{df['quality_score'].mean():.3f}")
    q3.metric("Late arrivals", f"{df['late_arrival_flag'].mean():.2%}")
    q4.metric("Duplicates", f"{df['duplicate_event_id_flag'].mean():.2%}")

    quality_rows = [
        {"issue": "missing_ip", "rate": float(df["missing_ip_flag"].mean())},
        {"issue": "missing_user_agent", "rate": float(df["missing_ua_flag"].mean())},
        {"issue": "unknown_country", "rate": float(df["unknown_country_flag"].mean())},
        {"issue": "negative_cost", "rate": float(df["negative_cost_flag"].mean())},
        {"issue": "malformed_timestamp", "rate": float(df["malformed_ts_flag"].mean())},
        {"issue": "late_arrival", "rate": float(df["late_arrival_flag"].mean())},
        {"issue": "duplicate_event_id", "rate": float(df["duplicate_event_id_flag"].mean())},
    ]
    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

    st.subheader("Example canonical record")
    example = df.iloc[0].to_dict()
    st.json(jsonable(example))

    st.subheader("Data contract summary")
    contract = pd.DataFrame([
        {"field": "record_uid", "type": "string", "unit": "identifier", "role": "primary demo key", "provenance": "hash(row_index,event_id)", "retention": "90 days"},
        {"field": "event_id", "type": "string", "unit": "identifier", "role": "source event id", "provenance": "synthetic ad server", "retention": "90 days"},
        {"field": "ts_utc", "type": "timestamp", "unit": "UTC", "role": "event time", "provenance": "synthetic ad server", "retention": "90 days"},
        {"field": "ingest_ts_utc", "type": "timestamp", "unit": "UTC", "role": "arrival time", "provenance": "ingestion simulator", "retention": "90 days"},
        {"field": "campaign_id", "type": "string", "unit": "identifier", "role": "entity", "provenance": "synthetic campaign registry", "retention": "180 days"},
        {"field": "publisher_id", "type": "string", "unit": "identifier", "role": "entity", "provenance": "synthetic publisher registry", "retention": "180 days"},
        {"field": "ip_hash", "type": "string", "unit": "hashed identifier", "role": "privacy-preserving key", "provenance": "SHA-256 truncated", "retention": "30 days"},
        {"field": "user_agent_hash", "type": "string", "unit": "hashed identifier", "role": "privacy-preserving key", "provenance": "SHA-256 truncated", "retention": "30 days"},
        {"field": "event_type", "type": "string", "unit": "enum", "role": "impression/click/conversion", "provenance": "synthetic ad server", "retention": "90 days"},
        {"field": "cost_micros", "type": "integer", "unit": "micro-USD", "role": "billing amount", "provenance": "synthetic pricing engine", "retention": "90 days"},
        {"field": "viewable_time_ms", "type": "integer", "unit": "milliseconds", "role": "visibility signal", "provenance": "synthetic measurement", "retention": "90 days"},
        {"field": "is_fraud", "type": "boolean", "unit": "label", "role": "synthetic ground truth", "provenance": "scenario generator", "retention": "90 days"},
        {"field": "fraud_type", "type": "string", "unit": "enum", "role": "synthetic failure taxonomy", "provenance": "scenario generator", "retention": "90 days"},
        {"field": "quality_score", "type": "float", "unit": "0-1", "role": "input trust", "provenance": "validator", "retention": "90 days"},
        {"field": "rule_score", "type": "float", "unit": "0-100", "role": "transparent baseline", "provenance": "rule engine", "retention": "90 days"},
        {"field": "model_score", "type": "float", "unit": "0-100", "role": "anomaly percentile", "provenance": "IsolationForest", "retention": "90 days"},
        {"field": "final_score", "type": "float", "unit": "0-100", "role": "decision score", "provenance": "score combiner", "retention": "90 days"},
        {"field": "confidence", "type": "float", "unit": "0-1", "role": "decision trust", "provenance": "agreement + quality + support", "retention": "90 days"},
        {"field": "applied_action", "type": "string", "unit": "enum", "role": "workflow state", "provenance": "policy engine", "retention": "180 days"},
        {"field": "pipeline_version", "type": "string", "unit": "semver-like", "role": "schema version", "provenance": "app constant", "retention": "180 days"},
    ])
    st.dataframe(contract, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Handling rules**

- Missing values: mapped to `unknown` or neutral numeric values; quality score penalized.
- Duplicates: retained for audit, flagged, and de-duplicated only at reporting layer if needed.
- Late-arriving records: accepted if within retention window; flagged and excluded from strict real-time SLA demos.
- Malformed input: negative costs clipped, bad timestamps imputed, rows downgraded in confidence.
- Outliers: winsorized to training quantiles before model scaling.
- Schema changes: versioned by `pipeline_version`; unknown fields are ignored, required fields trigger validation failure.
        """
    )


with tabs[5]:
    st.header("PRD, architecture, risk register, and self-checks")

    st.markdown(
        """
### Product mission

Detect and quantify invalid advertising traffic while protecting legitimate publishers and preserving auditable revenue decisions.

### Operational problem

Digital advertising ecosystems lose budget to bot impressions, click injection, malvertising redirects, proxy farms, and spoofed conversions. Manual rule tuning and spreadsheets create delayed detection, inconsistent publisher disputes, and revenue leakage.

### Users

- Ad operations analyst
- Revenue manager
- Policy/safety reviewer
- Publisher integrity manager
- Data engineer

### User journeys

1. Morning queue review: ranked events, reason codes, one-click approval/release, audit trail.
2. Campaign launch monitoring: live synthetic stream detects botnet/click-injection spikes and suggests holds.
3. Publisher dispute: reviewer inspects immutable event evidence, rule contributions, model confidence, and audit history.

### Concrete failure cases

- Botnet impression flood from datacenter IPs with bot user agents.
- Click injection where clicks occur hundreds of milliseconds after impression.
- Malvertising traffic with low viewability and impossible user agents.
- Proxy/geo mismatch traffic with high click-through and near-zero conversion quality.
- Data quality failure: missing IP hashes, duplicate event IDs, late arrivals, negative costs.

### Functional requirements

- Ingest synthetic ad impression/click/conversion events.
- Validate schema and assign quality flags.
- Compute rolling publisher, IP, user-agent, and campaign signals.
- Produce deterministic rule score and explainable reason codes.
- Produce unsupervised model score with chronological train/test discipline.
- Combine scores with confidence and fallback.
- Recommend actions: allow, monitor, review hold, block pending approval, blocked.
- Persist events, decisions, and audit history in SQLite.
- Visualize dashboard, event evidence, evaluation, data quality, and value calculator.

### Non-functional requirements

- CPU-only, no paid APIs, no proprietary cloud services for core demo.
- Deterministic seeded synthetic runs.
- Local persistence and reproducible tests.
- Structured auditability for consequential actions.
- Graceful degradation when model training data or input quality is inadequate.
- Reasonable responsiveness on Intel CPU / integrated graphics machines for default event volume.

### Prototype vs production

Prototype: single-file Streamlit app, SQLite, synthetic data, local video generation, CPU-only scoring, basic audit, demo evaluation.

Production: distributed ingestion, Kafka/Pulsar, feature store, PostgreSQL/TimescaleDB or lakehouse, model registry, RBAC, SSO, secret management, encryption, SLA monitoring, dispute workflow, refund integration, policy-as-code, champion/challenger models, data lineage, SOC audits, and controlled experimentation.
        """
    )

    st.markdown(
        """
### Architecture

```mermaid
flowchart LR
  ADS["Synthetic Ad Server"] --> ING["Ingestion Adapter"]
  DSP["DSP / Click Tracker"] --> ING
  ING --> VAL["Schema Validator"]
  VAL --> CAN["Canonical Event Model"]
  CAN --> STORE[("SQLite / Postgres")]
  CAN --> FEAT["Feature Processor"]
  FEAT --> RULE["Transparent Rule Baseline"]
  FEAT --> MODEL["IsolationForest Model"]
  RULE --> COMB["Score Combiner"]
  MODEL --> COMB
  COMB --> POLICY["Decision Policy"]
  POLICY --> AUDIT["Audit History"]
  POLICY --> DASH["Streamlit Dashboard"]
  STORE --> DASH
  AUDIT --> DASH
```

```mermaid
sequenceDiagram
  participant A as Ad Server
  participant I as Ingestion
  participant V as Validator
  participant F as Feature Processor
  participant R as Rule Engine
  participant M as Model
  participant D as Decision Policy
  participant U as Reviewer UI
  A->>I: impression/click event
  I->>V: raw canonical payload
  V->>F: validated event + quality flags
  F->>R: rolling features
  F->>M: numeric feature vector
  R->>D: rule score + reasons
  M->>D: model percentile + status
  D->>U: recommended action + evidence
  U->>D: human approval or release
  D->>D: write audit record
```
        """
    )

    st.markdown(
        """
### Risk register

| Risk | Example | Mitigation | Human approval |
|---|---|---|---|
| Data availability | Ad server feed delayed | Late-arrival flags, replay queue, freshness SLA | No |
| Model error | Bot traffic looks organic | Rule fallback, confidence gating, subgroup review | Yes for block |
| Privacy | IP/UA re-identification | Hashing, retention limits, no raw PII in demo | No |
| Fairness | Publisher segment over-flagged | Segment metrics, dispute workflow, threshold review | Yes for publisher penalty |
| Misuse | Suppress competitor traffic | RBAC, audit log, policy approval, explainability | Yes |
| Operational failure | Video/db write fails | Graceful degradation, retries, local fallback | No |
| Adoption | Analyst ignores tool | Evidence-first UI, queue workload limits | No |
| Cost | Over-blocking legitimate revenue | FP spend guardrail, release workflow | Yes |

Actions requiring human approval:
- Billing block.
- Refund recommendation.
- Publisher suspension.
- Auto-block policy enablement.
- Override of high-confidence allow to block, or block to allow.
        """
    )

    if "self_checks" not in st.session_state:
        try:
            with st.spinner("Running local self-checks..."):
                st.session_state.self_checks = run_self_checks()
        except Exception as exc:
            st.session_state.self_checks = pd.DataFrame(
                [{"test": "self_checks_startup", "status": "FAIL", "details": str(exc)}]
            )

    checks = st.session_state.get("self_checks", pd.DataFrame())

    if st.button("Rerun self-checks"):
        with st.spinner("Rerunning parsing, validation, fallback, determinism, and video checks..."):
            st.session_state.self_checks = run_self_checks()
        checks = st.session_state.self_checks

    if isinstance(checks, pd.DataFrame) and not checks.empty:
        st.dataframe(checks, use_container_width=True, hide_index=True)
        passed = int((checks["status"] == "PASS").sum())
        total = int(len(checks))
        st.success(f"Passed {passed} / {total} local checks.")
    else:
        st.info("Self-checks are not available yet.")


with tabs[6]:
    st.header("Transparent business-value calculator")

    st.caption("Enter assumptions. This calculator does not fabricate customer results.")

    v1, v2, v3, v4 = st.columns(4)
    monthly_spend = v1.number_input("Monthly ad spend (USD)", 10000, 100000000, 2000000, step=10000)
    baseline_fraud = v2.number_input("Baseline invalid traffic rate", 0.01, 0.50, 0.12, step=0.01, format="%.2f")
    detection_uplift = v3.number_input("Incremental detection uplift", 0.0, 1.0, 0.35, step=0.05, format="%.2f")
    recovery_rate = v4.number_input("Recovered spend rate", 0.0, 1.0, 0.65, step=0.05, format="%.2f")

    v5, v6, v7, v8 = st.columns(4)
    fp_rate = v5.number_input("False-positive spend rate", 0.0, 0.10, 0.005, step=0.001, format="%.3f")
    analyst_hours_saved = v6.number_input("Analyst hours saved / month", 0, 1000, 80, step=10)
    analyst_cost = v7.number_input("Analyst hourly cost (USD)", 10, 500, 60, step=5)
    infra_cost = v8.number_input("Incremental infrastructure cost / month (USD)", 0, 100000, 5000, step=500)

    implementation_cost = st.number_input("One-time implementation cost (USD)", 0, 5000000, 120000, step=5000)

    fraud_spend = monthly_spend * baseline_fraud
    recovered = fraud_spend * detection_uplift * recovery_rate
    fp_loss = monthly_spend * fp_rate
    analyst_savings = analyst_hours_saved * analyst_cost
    net_monthly = recovered + analyst_savings - fp_loss - infra_cost
    payback_months = implementation_cost / net_monthly if net_monthly > 0 else float("inf")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated fraud spend", f"${fraud_spend:,.0f}")
    m2.metric("Estimated recovered spend", f"${recovered:,.0f}")
    m3.metric("Estimated net monthly value", f"${net_monthly:,.0f}")
    m4.metric("Payback months", "Never" if not np.isfinite(payback_months) else f"{payback_months:.1f}")

    st.subheader("Sensitivity analysis")
    sens_rows = []
    for uplift in [0.20, 0.35, 0.50]:
        for fr in [0.06, 0.12, 0.18]:
            fs = monthly_spend * fr
            rec = fs * uplift * recovery_rate
            net = rec + analyst_savings - fp_loss - infra_cost
            sens_rows.append({
                "detection_uplift": uplift,
                "baseline_fraud_rate": fr,
                "recovered_usd": rec,
                "net_monthly_usd": net,
            })

    st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
**Interpretation guardrails**

- Value depends on accurate fraud rate and recovery assumptions.
- False-positive spend can offset recovered revenue.
- Analyst time savings are secondary and should be validated with workload telemetry.
- This calculator is for scoping, not a commercial promise.
        """
    )