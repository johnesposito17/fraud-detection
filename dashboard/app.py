"""
Fraud Analytics Dashboard — Streamlit
Analyst-facing tool for threshold tuning, fraud pattern exploration,
and model comparison. Runs entirely from parquet/CSV — no live DB needed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Analytics Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent.parent

MODEL_DISPLAY = {
    "RF_class_weight":  "Random Forest (class weight) ★ best",
    "XGB_class_weight": "XGBoost (class weight)",
    "LR_SMOTE":         "Logistic Regression (SMOTE)",
    "LR_class_weight":  "Logistic Regression (class weight)",
    "LR_undersample":   "Logistic Regression (undersample)",
    "LR_no_handling":   "Logistic Regression (no handling)",
}
MODEL_KEYS = list(MODEL_DISPLAY.keys())

FRAUD_COLOR = "#e74c3c"
LEGIT_COLOR = "#2980b9"
ACCENT      = "#8e44ad"


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/features.parquet")


@st.cache_data
def load_experiment_log() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/experiment_log.csv")


@st.cache_data
def load_sweep(model_key: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/threshold_sweeps" / f"{model_key}.csv")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    st.markdown("---")

    selected_key = st.selectbox(
        "Model",
        MODEL_KEYS,
        format_func=lambda k: MODEL_DISPLAY[k],
        index=0,
    )

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.01, max_value=0.99,
        value=0.50, step=0.01,
        help="Score cutoff above which a transaction is flagged as fraud.",
    )

    st.markdown("**Cost Parameters**")
    fn_cost = st.number_input(
        "Missed Fraud Cost ($)", value=500, min_value=1, step=50,
        help="Cost per false negative — fraud loss + chargeback ops.",
    )
    fp_cost = st.number_input(
        "False Alert Cost ($)", value=5, min_value=1, step=1,
        help="Cost per false positive — analyst review + customer friction.",
    )

    st.markdown("---")
    st.caption("Dataset: Kaggle Credit Card Fraud\n284,807 transactions · 0.17% fraud rate")


# ── Load data ─────────────────────────────────────────────────────────────────
features = load_features()
exp_log  = load_experiment_log()
sweep    = load_sweep(selected_key)

# Recompute cost column dynamically from user-supplied cost params
sweep = sweep.copy()
sweep["total_cost"] = sweep["fn"] * fn_cost + sweep["fp"] * fp_cost

# Snap to nearest pre-computed threshold
idx = (sweep["threshold"] - threshold).abs().idxmin()
row = sweep.loc[idx]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Credit Card Fraud Detection — Analyst Dashboard")
st.caption(
    f"Model: **{MODEL_DISPLAY[selected_key]}** · "
    f"Threshold: **{threshold:.2f}** · "
    f"Cost params: FN=${fn_cost} · FP=${fp_cost}"
)

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
live_cost = int(row["fn"]) * fn_cost + int(row["fp"]) * fp_cost

k1.metric("Precision",     f"{row['precision']*100:.1f}%")
k2.metric("Recall",        f"{row['recall']*100:.1f}%")
k3.metric("F1 Score",      f"{row['f1']:.3f}")
k4.metric("Fraud Caught",  f"{row['fraud_caught_pct']:.1f}%")
k5.metric("Total Cost",    f"${live_cost:,}")
k6.metric("Review Queue",  f"{int(row['tp']) + int(row['fp']):,} txns")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊 Threshold Analysis", "🔎 Fraud Patterns", "📈 Model Comparison"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Threshold Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Precision · Recall · Cost vs. Threshold")

    col_left, col_right = st.columns(2)

    # ── Precision-Recall curve ────────────────────────────────────────────────
    with col_left:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep["precision"],
            name="Precision", line=dict(color=LEGIT_COLOR, width=2),
        ))
        fig_pr.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep["recall"],
            name="Recall", line=dict(color=FRAUD_COLOR, width=2),
        ))
        fig_pr.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep["f1"],
            name="F1", line=dict(color=ACCENT, width=2, dash="dot"),
        ))
        # Marker for current threshold
        fig_pr.add_vline(
            x=threshold, line_dash="dash", line_color="gray", line_width=1,
            annotation_text=f"t={threshold:.2f}",
            annotation_position="top right",
        )
        fig_pr.update_layout(
            title="Precision / Recall / F1 vs. Threshold",
            xaxis_title="Threshold", yaxis_title="Score",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", y=-0.2),
            height=380, margin=dict(l=40, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Cost curve ───────────────────────────────────────────────────────────
    with col_right:
        min_cost_idx = sweep["total_cost"].idxmin()
        min_cost_t   = sweep.loc[min_cost_idx, "threshold"]
        min_cost_val = sweep.loc[min_cost_idx, "total_cost"]

        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep["total_cost"],
            name="Total Cost", line=dict(color="#e67e22", width=2),
            fill="tozeroy", fillcolor="rgba(230,126,34,0.08)",
        ))
        # Minimum cost marker
        fig_cost.add_trace(go.Scatter(
            x=[min_cost_t], y=[min_cost_val],
            mode="markers+text",
            marker=dict(size=10, color="green", symbol="diamond"),
            text=[f"Min cost<br>t={min_cost_t:.2f}"],
            textposition="top center",
            name="Min Cost",
        ))
        # Current threshold
        fig_cost.add_vline(
            x=threshold, line_dash="dash", line_color="gray", line_width=1,
        )
        fig_cost.update_layout(
            title=f"Total Cost vs. Threshold (FN=${fn_cost}, FP=${fp_cost})",
            xaxis_title="Threshold", yaxis_title="Cost ($)",
            legend=dict(orientation="h", y=-0.2),
            height=380, margin=dict(l=40, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    st.subheader("Confusion Matrix at Selected Threshold")

    tp, fp, fn, tn = int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"])
    total_fraud = tp + fn

    cm_col1, cm_col2, cm_col3 = st.columns([1, 1, 1])

    with cm_col1:
        fig_cm = go.Figure(go.Heatmap(
            z=[[tp, fn], [fp, tn]],
            x=["Predicted Fraud", "Predicted Legit"],
            y=["Actual Fraud", "Actual Legit"],
            colorscale=[[0, "#f8f9fa"], [1, FRAUD_COLOR]],
            showscale=False,
            text=[[f"TP: {tp}", f"FN: {fn}"], [f"FP: {fp}", f"TN: {tn}"]],
            texttemplate="%{text}",
            textfont=dict(size=16),
        ))
        fig_cm.update_layout(
            title="Confusion Matrix (test set)",
            height=260, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_col2:
        st.markdown("**At this threshold:**")
        st.markdown(f"- 🟢 **Caught:** {tp} of {total_fraud} fraud cases ({tp/max(total_fraud,1)*100:.1f}%)")
        st.markdown(f"- 🔴 **Missed:** {fn} fraud cases (${fn * fn_cost:,} in losses)")
        st.markdown(f"- ⚠️ **False alerts:** {fp} legitimate transactions flagged")
        st.markdown(f"- 💰 **Total cost:** ${live_cost:,}")
        st.markdown(f"- 🗂️ **Review queue:** {tp + fp} transactions")

    with cm_col3:
        st.markdown("**Business interpretation:**")
        review_rate = (tp + fp) / max(tn + tp + fp + fn, 1) * 100
        st.markdown(f"- {review_rate:.2f}% of all transactions sent to review")
        precision_pct = tp / max(tp + fp, 1) * 100
        st.markdown(f"- {precision_pct:.1f}% of flagged transactions are real fraud")
        baseline_cost = total_fraud * fn_cost
        savings = baseline_cost - live_cost
        st.markdown(f"- Saves **${savings:,}** vs. no model (catch-nothing baseline)")
        st.markdown(f"- Min-cost threshold: **{min_cost_t:.2f}** (${min_cost_val:,.0f})")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Fraud Patterns
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    fraud_df = features[features["class"] == 1]
    legit_df = features[features["class"] == 0]

    row2_l, row2_r = st.columns(2)

    # ── Fraud rate by hour ────────────────────────────────────────────────────
    with row2_l:
        hourly = features.groupby("hour_of_day").agg(
            total=("class", "count"),
            fraud=("class", "sum"),
        ).reset_index()
        hourly["fraud_rate"] = hourly["fraud"] / hourly["total"] * 100

        fig_hour = go.Figure()
        fig_hour.add_trace(go.Bar(
            x=hourly["hour_of_day"], y=hourly["fraud_rate"],
            marker_color=[FRAUD_COLOR if r > 0.3 else "#aab7b8" for r in hourly["fraud_rate"]],
            name="Fraud Rate %",
        ))
        fig_hour.add_hline(
            y=features["class"].mean() * 100,
            line_dash="dot", line_color="gray",
            annotation_text="Baseline",
        )
        fig_hour.update_layout(
            title="Fraud Rate by Hour of Day",
            xaxis_title="Hour", yaxis_title="Fraud Rate (%)",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            height=340, margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    # ── Fraud rate by amount bucket ───────────────────────────────────────────
    with row2_r:
        bucket_order = ["micro", "small", "medium", "large"]
        bucket = features.groupby("amount_bucket").agg(
            total=("class", "count"),
            fraud=("class", "sum"),
        ).reindex(bucket_order).reset_index()
        bucket["fraud_rate"] = bucket["fraud"] / bucket["total"] * 100

        fig_bucket = px.bar(
            bucket, x="amount_bucket", y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale=["#aab7b8", FRAUD_COLOR],
            labels={"amount_bucket": "Amount Bucket", "fraud_rate": "Fraud Rate (%)"},
            title="Fraud Rate by Transaction Amount Tier",
            text=bucket["fraud_rate"].round(2).astype(str) + "%",
        )
        fig_bucket.update_traces(textposition="outside")
        fig_bucket.update_layout(
            height=340, margin=dict(l=40, r=20, t=50, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bucket, use_container_width=True)

    row3_l, row3_r = st.columns(2)

    # ── Amount distribution: fraud vs. legit ──────────────────────────────────
    with row3_l:
        sample_legit = legit_df["amount"].sample(5000, random_state=42)

        fig_amt = go.Figure()
        fig_amt.add_trace(go.Histogram(
            x=np.log1p(sample_legit),
            name=f"Legit (sample n=5k)",
            opacity=0.65, marker_color=LEGIT_COLOR,
            nbinsx=60, histnorm="probability density",
        ))
        fig_amt.add_trace(go.Histogram(
            x=np.log1p(fraud_df["amount"]),
            name=f"Fraud (n={len(fraud_df):,})",
            opacity=0.75, marker_color=FRAUD_COLOR,
            nbinsx=60, histnorm="probability density",
        ))
        fig_amt.update_layout(
            barmode="overlay",
            title="Transaction Amount Distribution (log scale)",
            xaxis_title="log(Amount + 1)", yaxis_title="Density",
            legend=dict(orientation="h", y=-0.2),
            height=340, margin=dict(l=40, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_amt, use_container_width=True)

    # ── Night vs. day ────────────────────────────────────────────────────────
    with row3_r:
        night = features.groupby("is_night").agg(
            total=("class", "count"),
            fraud=("class", "sum"),
        ).reset_index()
        night["fraud_rate"] = night["fraud"] / night["total"] * 100
        night["label"] = night["is_night"].map({0: "Day (06:00–23:59)", 1: "Night (00:00–05:59)"})

        fig_night = go.Figure()
        fig_night.add_trace(go.Bar(
            x=night["label"], y=night["total"] - night["fraud"],
            name="Legitimate", marker_color=LEGIT_COLOR,
        ))
        fig_night.add_trace(go.Bar(
            x=night["label"], y=night["fraud"],
            name="Fraud", marker_color=FRAUD_COLOR,
        ))
        for _, r in night.iterrows():
            fig_night.add_annotation(
                x=r["label"], y=r["total"],
                text=f"{r['fraud_rate']:.2f}% fraud",
                showarrow=False, yshift=10, font=dict(size=12),
            )
        fig_night.update_layout(
            barmode="stack",
            title="Transaction Volume and Fraud Rate: Night vs. Day",
            yaxis_title="Transactions",
            legend=dict(orientation="h", y=-0.2),
            height=340, margin=dict(l=40, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_night, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("All Experiments — Side-by-Side")

    # Recompute cost with current user params
    log = exp_log.copy()
    log["cost_recomputed"] = log["fn_optimal"] * fn_cost + log["fp_optimal"] * fp_cost

    display_cols = {
        "label":           "Model",
        "imbalance":       "Imbalance Method",
        "pr_auc":          "PR-AUC",
        "f1_at_0.5":       "F1 @ 0.50",
        "f1_optimal":      "F1 @ Opt Thresh",
        "optimal_threshold":"Opt Threshold",
        "cost_recomputed": f"Cost @ Opt (FN=${fn_cost}/FP=${fp_cost})",
    }
    tbl = log[list(display_cols.keys())].rename(columns=display_cols)
    tbl[f"Cost @ Opt (FN=${fn_cost}/FP=${fp_cost})"] = \
        tbl[f"Cost @ Opt (FN=${fn_cost}/FP=${fp_cost})"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        tbl.style.highlight_max(
            subset=["PR-AUC", "F1 @ 0.50", "F1 @ Opt Thresh"],
            color="#d5f5e3",
        ).highlight_min(
            subset=[f"Cost @ Opt (FN=${fn_cost}/FP=${fp_cost})"],
            color="#d5f5e3",
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    # ── PR-AUC comparison ────────────────────────────────────────────────────
    with chart_col1:
        fig_auc = px.bar(
            log.sort_values("pr_auc"),
            x="pr_auc", y="label",
            orientation="h",
            color="pr_auc",
            color_continuous_scale=["#aab7b8", LEGIT_COLOR],
            labels={"pr_auc": "PR-AUC", "label": ""},
            title="Precision-Recall AUC by Model",
            text=log.sort_values("pr_auc")["pr_auc"].round(4).astype(str),
        )
        fig_auc.update_traces(textposition="outside")
        fig_auc.update_layout(
            xaxis=dict(range=[0, 1]),
            coloraxis_showscale=False,
            height=350, margin=dict(l=20, r=60, t=50, b=40),
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    # ── Cost comparison ──────────────────────────────────────────────────────
    with chart_col2:
        fig_cost_cmp = px.bar(
            log.sort_values("cost_recomputed", ascending=False),
            x="cost_recomputed", y="label",
            orientation="h",
            color="cost_recomputed",
            color_continuous_scale=[LEGIT_COLOR, FRAUD_COLOR],
            labels={"cost_recomputed": f"Cost ($)", "label": ""},
            title=f"Total Cost at Optimal Threshold (FN=${fn_cost}, FP=${fp_cost})",
            text=log.sort_values("cost_recomputed", ascending=False)["cost_recomputed"]
                    .apply(lambda x: f"${x:,.0f}"),
        )
        fig_cost_cmp.update_traces(textposition="outside")
        fig_cost_cmp.update_layout(
            coloraxis_showscale=False,
            height=350, margin=dict(l=20, r=80, t=50, b=40),
        )
        st.plotly_chart(fig_cost_cmp, use_container_width=True)

    # ── Imbalance strategy comparison (LR only) ───────────────────────────────
    st.subheader("Imbalance Strategy Tradeoff (Logistic Regression)")
    lr_models = log[log["label"].str.startswith("LR")].copy()

    fig_imb = go.Figure()
    fig_imb.add_trace(go.Bar(
        x=lr_models["label"], y=lr_models["pr_auc"],
        name="PR-AUC", marker_color=LEGIT_COLOR,
        yaxis="y",
    ))
    fig_imb.add_trace(go.Scatter(
        x=lr_models["label"], y=lr_models["cost_recomputed"],
        name=f"Cost ($)", marker=dict(size=10, color=FRAUD_COLOR),
        line=dict(color=FRAUD_COLOR), yaxis="y2", mode="lines+markers",
    ))
    fig_imb.update_layout(
        title="LR: Imbalance Strategy — PR-AUC vs. Cost",
        yaxis=dict(title="PR-AUC", range=[0, 1]),
        yaxis2=dict(title="Cost ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.2),
        height=340, margin=dict(l=40, r=60, t=50, b=60),
    )
    st.plotly_chart(fig_imb, use_container_width=True)

    st.info(
        "**Key takeaway:** Random Forest with class weighting achieves the highest PR-AUC (0.820) "
        "and best F1 (0.857). XGBoost is close (PR-AUC 0.794) and trains 10× faster. "
        "Among LR variants, SMOTE delivers the lowest cost ($7,825) but requires resampling "
        "time. No-handling LR has the highest cost despite decent F1 — threshold miscalibration "
        "at 0.5 causes it to miss more fraud."
    )
