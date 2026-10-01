"""Batch distillation anomaly review — research decision support, not batch release."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="배치 증류 이상신호 의사결정 지원", page_icon="⚗️", layout="wide")

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
BG, SURFACE, SURFACE2 = "#071822", "#0C2531", "#10313F"
INK, MUTED, LINE = "#F3F8FA", "#AFC2CB", "#244652"
TEAL, CYAN, AMBER, RED, GREEN = "#43D1C8", "#73C8EB", "#F2C66D", "#FF766F", "#63D39B"

st.markdown(f"""
<style>
.stApp{{background:{BG};color:{INK}}}
.main .block-container{{max-width:1500px;padding:1.1rem 2.4rem 3rem}}
h1,h2,h3,h4,p,li,label{{color:{INK}}} h1{{font-size:2rem!important;letter-spacing:-.035em;margin:.15rem 0!important}}
[data-testid="stSidebar"]{{background:#06131C;border-right:1px solid {LINE}}}
[data-testid="stSidebar"] *{{color:{INK}}}
.eyebrow{{color:{TEAL};font-size:.72rem;font-weight:850;letter-spacing:.12em}}
.subtitle{{color:{MUTED};font-size:.94rem;margin:.25rem 0 1rem}}
.hero{{background:linear-gradient(120deg,#0C2B38,#0A202B);border:1px solid {LINE};border-left:4px solid {TEAL};border-radius:14px;padding:1rem 1.2rem;margin:.7rem 0 1rem}}
.hero .k{{color:{TEAL};font-size:.7rem;font-weight:850;letter-spacing:.08em}}.hero .v{{font-size:1.15rem;font-weight:850;margin:.2rem 0}}.hero .d{{color:{MUTED};font-size:.8rem}}
.section-head{{margin:1.15rem 0 .55rem}}.section-head b{{font-size:1.22rem}}.section-head span{{display:block;color:{MUTED};font-size:.78rem;margin-top:.18rem}}
.help{{background:#0A222D;border-left:3px solid {TEAL};border-radius:0 9px 9px 0;padding:.7rem .85rem;color:{MUTED};font-size:.78rem;line-height:1.55;margin:.45rem 0 .9rem}}
.help b{{color:{INK}}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:.55rem 0 1rem}}
.kpi{{background:{SURFACE};border:1px solid {LINE};border-radius:12px;padding:.8rem .9rem}}
.kpi .k{{color:{MUTED};font-size:.68rem;font-weight:750}}.kpi .v{{font-size:1.15rem;font-weight:900;margin:.12rem 0}}.kpi .d{{color:{MUTED};font-size:.67rem}}
.state{{border-left:4px solid {TEAL}}}.state.good{{border-left-color:{GREEN}}}.state.warn{{border-left-color:{AMBER}}}.state.bad{{border-left-color:{RED}}}
.decision{{background:{SURFACE};border:1px solid {LINE};border-radius:14px;padding:1rem 1.05rem;margin:.55rem 0}}
.decision h3{{margin:0 0 .25rem;font-size:1.12rem}}.decision p{{color:{MUTED};font-size:.8rem;line-height:1.55;margin:.2rem 0}}
.decision.good{{box-shadow:inset 4px 0 {GREEN}}}.decision.warn{{box-shadow:inset 4px 0 {AMBER}}}.decision.bad{{box-shadow:inset 4px 0 {RED}}}
.priority{{display:flex;gap:.6rem;align-items:flex-start;background:{SURFACE2};border-radius:10px;padding:.7rem .8rem;margin:.45rem 0}}
.priority .n{{min-width:28px;height:28px;border-radius:50%;background:{TEAL};color:#05202A;display:flex;align-items:center;justify-content:center;font-weight:900}}
.priority b{{font-size:.82rem}}.priority span{{display:block;color:{MUTED};font-size:.72rem;margin-top:.12rem}}
.nav-note{{color:{MUTED};font-size:.72rem;margin:.3rem 0 .5rem}}
[data-testid="stMetric"]{{background:{SURFACE};border:1px solid {LINE};border-radius:11px;padding:.72rem .82rem;min-height:94px}}
[data-testid="stMetricLabel"]{{color:{MUTED}}}[data-testid="stMetricValue"]{{color:{INK};font-size:1.32rem}}
[data-testid="stMetricDelta"]{{font-size:.72rem}}
div[data-testid="stDataFrame"]{{border:1px solid {LINE};border-radius:10px;overflow:hidden}}
div[data-testid="stDataFrame"] iframe{{background:{SURFACE}!important}}
.dark-table{{width:100%;border-collapse:separate;border-spacing:0;background:{SURFACE};border:1px solid {LINE};border-radius:11px;overflow:hidden;font-size:.76rem}}
.dark-table th{{background:{SURFACE2};color:{CYAN};text-align:left;padding:.62rem .7rem;border-bottom:1px solid {LINE}}}
.dark-table td{{color:{INK};padding:.6rem .7rem;border-bottom:1px solid {LINE};vertical-align:top}}
.dark-table tr:last-child td{{border-bottom:0}}.dark-table td:nth-child(3){{color:{TEAL};font-weight:750}}
.source{{background:{SURFACE};border:1px solid {LINE};border-radius:10px;padding:.75rem .85rem;margin:.4rem 0;color:{MUTED};font-size:.73rem;line-height:1.5}}
.source b{{color:{CYAN}}}.source a{{color:{TEAL}!important}}
.footer{{border-top:1px solid {LINE};color:{MUTED};font-size:.68rem;padding-top:.8rem;margin-top:1.2rem}}
.panel{{background:{SURFACE};border:1px solid {LINE};border-radius:14px;padding:1rem;margin:.35rem 0 .8rem}}
.panel-title{{font-weight:850;font-size:.95rem;margin-bottom:.25rem}}.panel-copy{{color:{MUTED};font-size:.74rem;line-height:1.5}}
.compare{{display:grid;grid-template-columns:1fr 34px 1fr;gap:8px;align-items:center;margin:.6rem 0}}
.compare .box{{background:{SURFACE2};border:1px solid {LINE};border-radius:11px;padding:.75rem}}
.compare .arrow{{text-align:center;color:{TEAL};font-size:1.2rem}}.compare .label{{color:{MUTED};font-size:.67rem}}.compare .value{{font-size:1.05rem;font-weight:900}}
.badge{{display:inline-block;border-radius:999px;padding:.2rem .55rem;font-size:.68rem;font-weight:850;background:#143746;color:{CYAN}}}
.badge.good{{background:#12392D;color:{GREEN}}}.badge.warn{{background:#493F16;color:{AMBER}}}.badge.bad{{background:#4A2527;color:{RED}}}
div[role="radiogroup"]{{gap:.35rem;background:#0A202B;border:1px solid {LINE};padding:.35rem;border-radius:12px}}
div[role="radiogroup"] label{{background:transparent;border-radius:8px;padding:.35rem .75rem}}
div[role="radiogroup"] label:has(input:checked){{background:#123745}}
.stSlider [data-baseweb="slider"] div{{color:{TEAL}}}
@media(max-width:900px){{.kpi-row{{grid-template-columns:1fr 1fr}}.main .block-container{{padding:1rem}}}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="검증 모델을 불러오는 중입니다.")
def load_model():
    path = ART / "model_v3.joblib"
    if not path.exists():
        return None, "artifacts/model_v3.joblib 파일이 없습니다."
    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"모델 로딩 실패: {type(exc).__name__}: {exc}"


@st.cache_data
def load_csv(name):
    path = ART / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_sample(kind):
    path = ART / f"sample_{kind}.csv.gz"
    return pd.read_csv(path) if path.exists() else None


MODEL, error = load_model()
if MODEL is None:
    st.error(error)
    st.stop()

M = MODEL["metrics"]
FEATS = MODEL["feature_names"]
SUMMARY = pd.DataFrame(MODEL["model_summary"])
DET = SUMMARY[SUMMARY.task == "y_detect"].set_index("model")
FC = SUMMARY[SUMMARY.task == "y_forecast"].set_index("model")
ROC = load_csv("roc_curve.csv")
SENS = load_csv("sensitivity.csv")


def section(title, description):
    st.markdown(f'<div class="section-head"><b>{title}</b><span>{description}</span></div>', unsafe_allow_html=True)


def style_plot(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(t=55, b=45, l=25, r=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        font=dict(family="Arial, Noto Sans KR, sans-serif", color=INK, size=13),
        title_font=dict(color=INK, size=16), legend=dict(orientation="h", y=1.13, x=0),
        hoverlabel=dict(bgcolor="#102F3C", font_color=INK, bordercolor=LINE))
    fig.update_xaxes(gridcolor=LINE, linecolor=LINE, tickfont_color=INK, title_font_color=INK, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, linecolor=LINE, tickfont_color=INK, title_font_color=INK, zeroline=False)
    for annotation in fig.layout.annotations:
        annotation.font.color = INK
    return fig


def spe_score(x):
    pca = MODEL["pca"]
    residual = x - pca.inverse_transform(pca.transform(x))
    return (residual ** 2).sum(axis=1), residual


def operating_point(target):
    valid = ROC[ROC.fpr <= target] if ROC is not None else pd.DataFrame()
    if valid.empty:
        return np.nan, np.nan
    row = valid.loc[valid.tpr.idxmax()]
    return float(row.tpr), float(row.thr)


def continuous_spe_limit(target):
    """Interpolate the locked calibration limits so the UI is not restricted to five presets."""
    limits = MODEL.get("run_spe_limits", MODEL["spe_limits"])
    xp = np.array([.01, .03, .05, .10, .20])
    yp = np.array([limits[f"{v:.2f}"] for v in xp], dtype=float)
    return float(np.exp(np.interp(target, xp, np.log(yp))))


def apply_virtual_perturbation(x, t, tag, mode, intensity, start_pct, duration_sec, direction):
    """Inject a transparent sensitivity perturbation in standardized feature space."""
    out = x.copy()
    cols = (np.ones(len(FEATS), dtype=bool) if tag == "전체 센서" else
            np.array([str(f).split("|")[0] == tag for f in FEATS]))
    if not cols.any() or intensity == 0:
        return out, np.zeros(len(t), dtype=bool), np.zeros(len(t))
    start = float(t.min() + (t.max() - t.min()) * start_pct / 100)
    mask = (t >= start) & (t <= start + duration_sec)
    pos = np.flatnonzero(mask)
    profile = np.zeros(len(t), dtype=float)
    if len(pos):
        u = np.linspace(0, 1, len(pos))
        if mode == "완만한 드리프트":
            profile[pos] = u
        elif mode == "일시적 스파이크":
            profile[pos] = np.exp(-.5 * ((u - .5) / .13) ** 2)
        elif mode == "변동성 증가":
            profile[pos] = np.sin(np.linspace(0, 8 * np.pi, len(pos)))
        else:
            profile[pos] = 1
    if mode == "정상범위 복귀":
        alpha = min(intensity / 6, 1.0)
        out[:, cols] *= (1 - alpha * profile[:, None])
    else:
        out[:, cols] += direction * intensity * profile[:, None]
    return out, mask, profile


def scenario_metrics(x, spe_limit, prob_limit, persistence):
    spe, residual = spe_score(x)
    prob = MODEL["rf"].predict_proba(x)[:, 1]
    p_mask = sustained_mask(spe, spe_limit, persistence)
    r_mask = sustained_mask(prob, prob_limit, persistence) if np.isfinite(prob_limit) else np.zeros(len(spe), dtype=bool)
    return dict(spe=spe, residual=residual, prob=prob, p_mask=p_mask, r_mask=r_mask,
                p_events=count_events(p_mask), r_events=count_events(r_mask), q95=float(np.quantile(spe, .95)),
                peak_prob=float(prob.max()))


def tag_context(tag):
    for prefix, text in [("PDI", "차압 계통"), ("FT", "유량 계통"), ("FYI", "유량 계통"),
                         ("LS", "액위 계통"), ("AV", "밸브 계통"), ("TV", "밸브 계통")]:
        if str(tag).startswith(prefix):
            return text
    if str(tag).startswith("T"):
        return "온도 계통"
    if str(tag).startswith("H"):
        return "가열·조작 계통"
    if str(tag).startswith("P"):
        return "압력 계통"
    return "공정 센서"


def count_events(mask):
    mask = np.asarray(mask, dtype=bool)
    return int(np.sum(mask & ~np.r_[False, mask[:-1]]))


def sustained_mask(values, threshold, persistence):
    raw = np.asarray(values) >= threshold
    if persistence <= 1:
        return raw
    # Mark the full confirmed block, not only its centre.
    hit = np.convolve(raw.astype(int), np.ones(persistence, dtype=int), mode="valid") >= persistence
    out = np.zeros_like(raw, dtype=bool)
    for start in np.flatnonzero(hit):
        out[start:start + persistence] = True
    return out


def risk_state(pca_events, rf_events):
    if pca_events > 0 and rf_events > 0:
        return "위험", "bad", "즉시 확인", "두 모델에서 지속 이탈이 함께 확인됨"
    if pca_events > 0 or rf_events > 0:
        return "주의", "warn", "교차 확인", "한 모델에서만 지속 이탈이 확인됨"
    return "안정", "good", "추세 관찰", "현재 기준에서 지속 초과 이벤트 없음"


def decision_text(state, ratio, top_tag, peak_time, pca_events, rf_events):
    if state == "위험":
        return (f"{peak_time:,.0f}초 부근에서 관리한계 대비 {ratio:.2f}배의 이탈이 확인되었습니다. "
                f"{top_tag} 계통을 우선 확인하고 배치기록·알람·설비 상태를 같은 시간축으로 대조하십시오. "
                "모델 결과만으로 공정을 조정하거나 배치를 판정하지 않습니다.")
    if state == "주의":
        return (f"PCA-SPE 이벤트 {pca_events}건, RF 이벤트 {rf_events}건으로 두 지표가 일치하지 않습니다. "
                f"{top_tag} 신호의 지속시간과 반복 여부를 확인한 뒤, 단발성 노이즈이면 종료하고 반복되면 조사 기록을 개시하십시오.")
    return ("현재 설정에서 즉시 조사가 필요한 지속 이탈은 확인되지 않았습니다. "
            "정상 판정이나 출하 승인을 의미하지 않으며 기존 IPC·시험·배치기록 검토는 그대로 수행합니다.")


# Sidebar: only input controls.
with st.sidebar:
    st.markdown('<div class="eyebrow">BATCH REVIEW</div>', unsafe_allow_html=True)
    st.header("분석 조건")
    source = st.radio("Run 데이터", ["정상 예시", "이상 예시", "CSV 업로드"],
                      help="예시 Run은 화면 동작 확인용이며 실제 생산 판정에 사용할 수 없습니다.")
    uploaded = st.file_uploader("186개 Window Feature", type=["csv", "gz"],
                                help="학습 파이프라인과 동일한 186개 특성 및 t_end가 필요합니다.") if source == "CSV 업로드" else None
    st.divider()
    st.markdown("**센서 민감도 가상 주입**")
    all_tags = list(dict.fromkeys(str(f).split("|")[0] for f in FEATS))
    ranked_tags = (pd.Series(MODEL["rf"].feature_importances_, index=FEATS)
                   .groupby(lambda s: str(s).split("|")[0]).sum().sort_values(ascending=False).index.tolist())
    scenario_enabled = st.toggle("가상 교란 적용", value=False,
                                 help="선택 센서의 표준화 특성에 가상 변화를 주입하고 모델 반응을 다시 계산합니다.")
    scenario_tag = st.selectbox("대상 공정 태그", ["전체 센서"] + (ranked_tags or all_tags), index=1)
    scenario_mode = st.selectbox("교란 형태", ["완만한 드리프트", "센서 편향", "일시적 스파이크", "변동성 증가", "정상범위 복귀"])
    direction_label = st.radio("변화 방향", ["증가", "감소"], horizontal=True)
    scenario_intensity = st.slider("교란 강도 (표준편차)", 0.0, 6.0, 2.0, .25,
                                   help="물리 단위가 아닌 학습 정상분포의 표준편차 단위입니다.")
    scenario_start = st.slider("주입 시작 (Run 진행률)", 0, 90, 45, 5, format="%d%%")
    scenario_duration = st.slider("지속시간 (초)", 30, 4800, 300, 30)
    st.caption("실제 공정 거동 예측이 아니라 센서 변화에 대한 모델 민감도 시험입니다.")
    st.divider()
    st.markdown("**경보 판정 규칙**")
    fpr_pct = st.slider(
        "목표 Window 오탐률", min_value=.5, max_value=20.0, value=5.0, step=.5,
        format="%.1f%%",
        help="Calibration 정상 데이터에서 허용할 Window 단위 오탐률입니다.")
    fpr_target = fpr_pct / 100
    persistence = st.slider("연속 확인 Window", 1, 6, 3,
                            help="기준 초과가 연속으로 유지될 때만 이벤트로 묶습니다. 5초 간격이므로 3개는 약 15초입니다.")
    st.caption("교란 조건은 입력과 모델 출력을, 판정 규칙은 경보 민감도를 각각 바꿉니다.")
    st.divider()
    st.caption(f"검증 데이터  {M['n_runs']} Run · {M['n_windows']:,} windows\n\n120초 Window · 5초 간격 · {M['n_splits']}개 외부 분할")
    st.warning("주의 · 실험 검증용\n\nDeviation·출하 판정 아님")

if source == "정상 예시":
    df = load_sample("normal")
elif source == "이상 예시":
    df = load_sample("anomalous")
elif uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV를 읽지 못했습니다: {exc}")
        df = None
else:
    df = None

recall_est, probability_threshold = operating_point(fpr_target)
spe_limit = continuous_spe_limit(fpr_target)
st.markdown('<div class="eyebrow">BATCH DISTILLATION · PROCESS ANALYTICS</div>', unsafe_allow_html=True)
st.title("배치 증류 공정 이상탐지 및 품질조사 지원")
st.markdown('<div class="subtitle">Run 선별 → 이탈 시점 확인 → 공정 태그 우선순위 → 담당자 조사 기록</div>', unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="k">참조</div><div class="v">현재 이상상태 탐지는 성립 · 60초 사전예측은 미입증</div>'
            '<div class="d">실험 검증 결과에 따른 적용 범위입니다. 자동 공정제어, 품질 일탈 확정, 출하 판정을 수행하지 않습니다.</div></div>', unsafe_allow_html=True)

page = st.radio("화면 이동", ["01 Run 판독", "02 모델 검증", "03 사전예측 검토", "04 현장 전환"],
                horizontal=True, label_visibility="collapsed")
st.markdown('<div class="nav-note">각 화면은 ‘현재 Run 판단 → 근거 확인 → 조기예측 한계 → 현장 적용 요건’ 순으로 독립적으로 사용할 수 있습니다.</div>', unsafe_allow_html=True)


if page == "01 Run 판독":
    section("Run 검토 및 민감도 시험", "기준 Run과 센서 교란 시나리오를 같은 모델에 통과시켜 상태 변화와 조사 우선순위를 비교합니다.")
    if df is None:
        st.info("왼쪽에서 예시 Run을 선택하거나 CSV를 업로드하세요.")
    else:
        missing = [c for c in FEATS if c not in df.columns]
        if missing:
            st.error(f"필수 특성 {len(missing)}개가 없습니다. 누락값을 임의 보정하지 않고 판정을 중단합니다.")
        else:
            x = MODEL["scaler"].transform(df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0))
            t = df["t_end"].to_numpy() if "t_end" in df else np.arange(len(df)) * MODEL["stride_sec"]
            direction = 1 if direction_label == "증가" else -1
            applied_strength = scenario_intensity if scenario_enabled else 0.0
            x_scenario, injection_mask, profile = apply_virtual_perturbation(
                x, t, scenario_tag, scenario_mode, applied_strength, scenario_start, scenario_duration, direction)
            base = scenario_metrics(x, spe_limit, probability_threshold, persistence)
            sim = scenario_metrics(x_scenario, spe_limit, probability_threshold, persistence)
            peak_idx = int(np.argmax(sim["spe"]))
            ratio, peak_time = sim["q95"] / spe_limit, float(t[peak_idx])
            contrib = pd.Series(sim["residual"][peak_idx] ** 2, index=FEATS)
            tags = contrib.groupby(contrib.index.str.split("|").str[0]).sum().sort_values(ascending=False).head(6)
            share = tags / max(contrib.sum(), 1e-12)
            top_tag = str(share.index[0]) if len(share) else "상위 태그"
            if sim["p_events"] and sim["r_events"]:
                state, css, action, reason = "위험", "bad", "즉시 교차 확인", "PCA와 RF가 모두 지속 이탈"
            elif sim["p_events"] or sim["r_events"]:
                state, css, action, reason = "조사", "warn", "태그·기록 대조", "한 모델에서 지속 이탈"
            elif sim["q95"] > base["q95"] * 1.30 or sim["peak_prob"] > base["peak_prob"] + .10:
                state, css, action, reason = "관찰", "warn", "추세 재확인", "기준 미만이지만 모델 반응 증가"
            else:
                state, css, action, reason = "안정", "good", "기존 절차 유지", "지속 초과와 유의한 증가 없음"

            run_id = str(df["run_id"].dropna().iloc[0]) if "run_id" in df and not df["run_id"].dropna().empty else "업로드 Run"
            chemical_system = run_id.split("|")[0].replace("batch_dist_ternary_", "").replace("+", " + ")
            source_name = source.replace(" 예시", " 예시 Run")

            delta_q95 = sim["q95"] - base["q95"]
            delta_prob = sim["peak_prob"] - base["peak_prob"]
            st.markdown(f'<div class="kpi-row"><div class="kpi state {css}"><div class="k">시나리오 판독</div><div class="v">{state}</div><div class="d">{reason}</div></div>'
                        f'<div class="kpi"><div class="k">PCA q95 변화</div><div class="v">{base["q95"]:,.1f} → {sim["q95"]:,.1f}</div><div class="d">{delta_q95:+,.1f} · 한계 {spe_limit:,.1f}</div></div>'
                        f'<div class="kpi"><div class="k">RF 최고확률 변화</div><div class="v">{base["peak_prob"]:.1%} → {sim["peak_prob"]:.1%}</div><div class="d">{delta_prob:+.1%}p</div></div>'
                        f'<div class="kpi"><div class="k">지속 이벤트</div><div class="v">PCA {sim["p_events"]} · RF {sim["r_events"]}</div><div class="d">{persistence} Window 연속 기준</div></div></div>', unsafe_allow_html=True)

            scenario_desc = (f'{scenario_tag} · {scenario_mode} · {direction_label} {applied_strength:.2f}σ · '
                             f'{scenario_start}%부터 {scenario_duration}초') if scenario_enabled else '가상 교란 없음 · 원본 Run 재현'
            st.markdown(f'<div class="decision {css}"><h3>{state} · {action}</h3><p><b>시험 조건:</b> {scenario_desc}<br>'
                        f'최대 반응은 {peak_time:,.0f}초, 우선 확인 태그는 {top_tag}입니다. '
                        '이 결과는 민감도 시험이며 실제 공정 원인이나 조정량을 확정하지 않습니다.</p></div>', unsafe_allow_html=True)

            left, right = st.columns([1.75, 1], gap="large")
            with left:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.16,
                                    subplot_titles=("정상 패턴 이탈 배수", "RF 이상확률"), row_heights=[.56, .44])
                fig.add_trace(go.Scatter(x=t, y=base["spe"] / spe_limit, name="기준 Run", line=dict(color="#708B97", width=1.7)), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=sim["spe"] / spe_limit, name="가상 시나리오", line=dict(color=TEAL, width=2.5),
                                         fill="tozeroy", fillcolor="rgba(67,209,200,.08)"), row=1, col=1)
                fig.add_hline(y=1, line=dict(color=RED, dash="dash"), row=1, col=1)
                fig.add_trace(go.Scatter(x=t, y=base["prob"], name="기준 RF", line=dict(color="#708B97", width=1.7)), row=2, col=1)
                fig.add_trace(go.Scatter(x=t, y=sim["prob"], name="시나리오 RF", line=dict(color=CYAN, width=2.5)), row=2, col=1)
                if np.isfinite(probability_threshold):
                    fig.add_hline(y=probability_threshold, line=dict(color=AMBER, dash="dot"), row=2, col=1)
                if injection_mask.any():
                    x0, x1 = float(t[injection_mask][0]), float(t[injection_mask][-1])
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(242,198,109,.10)", line_width=0, row="all", col=1)
                fig.update_yaxes(title="관리한계 배수", range=[0, max(1.35, np.quantile(sim["spe"] / spe_limit, .995) * 1.15)], row=1, col=1)
                fig.update_yaxes(title="확률", range=[0, 1], tickformat=".0%", row=2, col=1)
                fig.update_xaxes(title="Run 경과시간(초)", row=2, col=1)
                style_plot(fig, 455); st.plotly_chart(fig, width="stretch")
            with right:
                st.markdown('<div class="panel"><div class="panel-title">변화 해석</div><div class="panel-copy">회색은 원본 Run, 청록색은 선택한 교란을 주입한 결과입니다. 음영은 교란 구간입니다.</div></div>', unsafe_allow_html=True)
                for i, (tag, val) in enumerate(share.head(3).items(), 1):
                    st.markdown(f'<div class="priority"><div class="n">{i}</div><div><b>{tag} · {tag_context(tag)}</b>'
                                f'<span>최대 이탈시점 기여 {val:.1%} · 동일 시점의 원신호·조작기록 대조</span></div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="panel"><span class="badge {css}">{state}</span><div class="panel-title" style="margin-top:.55rem">판독 근거</div>'
                            f'<div class="panel-copy">PCA {sim["p_events"]}건 · RF {sim["r_events"]}건 · q95 한계 대비 {ratio:.2f}배. '
                            '두 모델의 일치 여부와 지속성을 함께 사용했습니다.</div></div>', unsafe_allow_html=True)

            section("교란 강도별 모델 반응", "현재 선택한 태그와 형태를 유지하고 강도만 0~6σ로 바꿔 상태 전환점을 확인합니다.")
            levels = np.linspace(0, 6, 13)
            qrat, peaks, states = [], [], []
            for lv in levels:
                xx, _, _ = apply_virtual_perturbation(x, t, scenario_tag, scenario_mode, lv, scenario_start, scenario_duration, direction)
                mm = scenario_metrics(xx, spe_limit, probability_threshold, persistence)
                qrat.append(mm["q95"] / spe_limit); peaks.append(mm["peak_prob"])
                states.append("위험" if mm["p_events"] and mm["r_events"] else "조사" if mm["p_events"] or mm["r_events"] else "안정")
            response = make_subplots(specs=[[{"secondary_y": True}]])
            response.add_trace(go.Scatter(x=levels, y=qrat, mode="lines+markers", name="PCA q95 / 한계", line=dict(color=TEAL, width=3)), secondary_y=False)
            response.add_trace(go.Scatter(x=levels, y=peaks, mode="lines+markers", name="RF 최고확률", line=dict(color=CYAN, width=3)), secondary_y=True)
            response.add_hline(y=1, line=dict(color=RED, dash="dash"), secondary_y=False)
            response.add_vline(x=applied_strength, line=dict(color=AMBER, width=3), annotation_text="현재 강도")
            response.update_xaxes(title="가상 교란 강도(σ)")
            response.update_yaxes(title="PCA 관리한계 배수", secondary_y=False)
            response.update_yaxes(title="RF 최고확률", tickformat=".0%", range=[0, 1], secondary_y=True)
            style_plot(response, 315); st.plotly_chart(response, width="stretch")
            st.caption("강도를 높여도 출력이 변하지 않으면 해당 태그에 대한 모델 민감도가 낮다는 뜻입니다. 물리적 안전 한계나 희석·조작 처방으로 해석하지 않습니다.")

            with st.expander("데이터·화학계 및 해석 제한"):
                st.markdown(f"**데이터:** {source_name}  ·  **화학계:** `{chemical_system}`\n\n"
                            "가상 주입은 표준화된 센서 Feature를 변화시키는 모델 민감도 분석입니다. 실제 몰분율·압력·VLE/NRTL 계수가 없으므로 희석률, 운전조건 또는 제품 품질 변화를 계산하지 않습니다. "
                            "실제 처방 기능은 물질수지·열역학 모델과 현장 실험 검증을 별도로 연결해야 합니다.")


elif page == "02 모델 검증":
    section("현재 이상상태 분류 성능", "AUPRC는 실제 이상 구간을 상위 경보로 얼마나 잘 모았는지 보여주며, 불균형 데이터에서 AUROC보다 우선합니다.")
    st.markdown('<div class="help"><b>도메인 해석</b> · 이 성능은 센서가 이미 이상을 관찰한 이후의 상태 구분입니다. '
                '공정 이상을 60초 전에 예측하거나 제품 품질을 판정했다는 의미가 아닙니다.</div>', unsafe_allow_html=True)
    order = ["ConvAE", "RF", "XGBoost", "TimeOnly"]
    labels = ["ConvAE\n정상학습", "RF\n지도학습", "XGBoost\n지도학습", "TimeOnly\n시간대조군"]
    vals = [DET.loc[x, "auprc"] for x in order]
    errs = [DET.loc[x, "auprc_sd"] for x in order]
    colors = [GREEN, TEAL, CYAN, "#81939D"]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors, error_y=dict(type="data", array=errs),
                           text=[f"{v:.3f}" for v in vals], textposition="outside"))
    fig.add_hline(y=M["detect_prevalence"], line=dict(color=RED, dash="dash"))
    fig.update_yaxes(title="AUPRC", range=[0, .8])
    style_plot(fig, 390)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"점선은 무작위 기준인 이상 구간 비율 {M['detect_prevalence']:.3f}입니다. 막대 위 선은 15개 분할 간 표준편차입니다.")
    st.markdown(f'<div class="kpi-row"><div class="kpi state good"><div class="k">ConvAE</div><div class="v">{DET.loc["ConvAE","auprc"]:.3f}</div><div class="d">정상 데이터만 학습</div></div>'
                f'<div class="kpi"><div class="k">Random Forest</div><div class="v">{DET.loc["RF","auprc"]:.3f}</div><div class="d">변수 영향 확인 가능</div></div>'
                f'<div class="kpi"><div class="k">TimeOnly</div><div class="v">{DET.loc["TimeOnly","auroc"]:.3f}</div><div class="d">AUROC가 무작위 수준</div></div>'
                f'<div class="kpi"><div class="k">검증 반복</div><div class="v">3 × 5</div><div class="d">3 Seeds · 5 Folds</div></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="decision good"><h3>검증 판단 · 현재 이상상태 탐지는 성립</h3><p>ConvAE와 RF의 차이는 분할 간 편차보다 작습니다. '
                '특정 모델의 우월성보다 센서 시계열에 현재 이상상태 정보가 존재한다는 근거로 해석합니다.</p></div>', unsafe_allow_html=True)

    if SENS is not None:
        section("결론 강건성", "라벨 경계와 Recovery 정의를 바꿨을 때 결론이 뒤집히는지 확인합니다.")
        boundary = SENS[SENS.analysis == "label_boundary"]
        recovery = SENS[SENS.analysis == "recovery"]
        l, r = st.columns(2)
        with l:
            f = go.Figure(go.Scatter(x=boundary.setting, y=boundary.auprc, mode="lines+markers+text",
                                     line=dict(color=TEAL, width=3), marker=dict(size=9),
                                     text=[f"{v:.3f}" for v in boundary.auprc], textposition="top center"))
            f.update_layout(title="라벨 경계 제외 민감도")
            f.update_yaxes(title="AUPRC", range=[.46, .53])
            style_plot(f, 290); st.plotly_chart(f, width="stretch")
            st.caption("±0~30초 제외 변화폭이 작아 경계 오차만으로 성능을 설명하기 어렵습니다.")
        with r:
            f = go.Figure(go.Bar(x=recovery.setting, y=recovery.auprc, marker_color=[TEAL, AMBER],
                                 text=[f"{v:.3f}" for v in recovery.auprc], textposition="outside"))
            f.update_layout(title="Recovery 포함 여부")
            f.update_yaxes(title="AUPRC", range=[0, .75])
            style_plot(f, 290); st.plotly_chart(f, width="stretch")
            st.caption("Recovery 포함 시 정답 범위도 넓어지므로 모델 개선으로 해석하지 않습니다.")


elif page == "03 사전예측 검토":
    section("60초 사전예측 타당성", "센서 관찰 전 예측이 시간대조군보다 우수한지와 운영 가능한 오경보 수준을 함께 확인합니다.")
    det_best, det_time = DET.drop(index=["TimeOnly"]).auroc.max(), DET.loc["TimeOnly", "auroc"]
    fc_best, fc_time = FC.drop(index=["TimeOnly", "Dummy"]).auroc.max(), FC.loc["TimeOnly", "auroc"]
    delta = fc_best - fc_time
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["현재 이상탐지", "60초 사전예측"], y=[det_best, fc_best], name="센서 모델", marker_color=TEAL,
                         text=[f"{det_best:.3f}", f"{fc_best:.3f}"], textposition="outside"))
    fig.add_trace(go.Bar(x=["현재 이상탐지", "60초 사전예측"], y=[det_time, fc_time], name="시간대조군", marker_color=AMBER,
                         text=[f"{det_time:.3f}", f"{fc_time:.3f}"], textposition="outside"))
    fig.add_hline(y=.5, line=dict(color=RED, dash="dot"))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(title="AUROC", range=[0, 1])
    style_plot(fig, 400); st.plotly_chart(fig, width="stretch")
    st.caption("빨간 점선은 무작위 AUROC 0.5입니다. 사전예측에서는 시간대조군이 센서 모델보다 높아 선행신호 근거가 성립하지 않았습니다.")

    st.markdown(f'<div class="decision bad"><h3>미입증 · 센서 모델이 시간대조군보다 {abs(delta):.3f} 낮음</h3>'
                '<p>이 결과에서 60초 전 자동경보를 운영하면 공정 선행신호가 아니라 배치 진행시간을 학습한 경보가 될 수 있습니다. '
                '현재 단계에서는 조기경보 기능을 비활성화하고 사후 원인조사 지원에 한정합니다.</p></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="kpi-row"><div class="kpi state warn"><div class="k">관찰 이전 탐지</div><div class="v">{M["early_hits"]} / {M["early_total"]}</div><div class="d">{M["early_rate"]:.1%}</div></div>'
                f'<div class="kpi"><div class="k">120초 이내 탐지</div><div class="v">{M["within120"]} / {M["early_total"]}</div><div class="d">{M["within120_rate"]:.1%}</div></div>'
                f'<div class="kpi state bad"><div class="k">정상구간 오경보</div><div class="v">{M["fa_per_hour"]:.2f}회/h</div><div class="d">사전 기준 1.0회/h 초과</div></div>'
                f'<div class="kpi state bad"><div class="k">최종 Gate</div><div class="v">미입증</div><div class="d">조기경보 비활성</div></div></div>', unsafe_allow_html=True)
    section("의사결정 지원", "조기예측 결과에 따른 운영 위험과 다음 조치를 명시합니다.")
    for n, title, desc in [
        (1, "시간 Shortcut 제거", "상대 진행시간·Operating Point만으로 동일 평가를 반복하고 센서 모델이 이를 명확히 초과해야 합니다."),
        (2, "Event onset 재정의", "explicit/inferred onset을 구분하고 추정 onset은 신뢰도와 허용 오차를 기록합니다."),
        (3, "Matched normal 확대", "동일 chemical system·OP·진행률의 정상 Run으로 오경보를 다시 계산합니다."),
    ]:
        st.markdown(f'<div class="priority"><div class="n">{n}</div><div><b>{title}</b><span>{desc}</span></div></div>', unsafe_allow_html=True)


else:
    section("현장 전환 선결조건", "연구용 모델을 품질시스템에 연결하기 전에 문서·검증·책임·변경관리 기준을 확정합니다.")
    st.markdown('<div class="help"><b>적용 원칙</b> · 본 화면은 초기에는 Shadow Mode로 운영합니다. 담당자의 기존 판단을 대체하지 않고, '
                '경보 결과와 실제 조사 결과를 축적해 기준을 재검증한 뒤 사용 범위를 단계적으로 확대합니다.</div>', unsafe_allow_html=True)

    steps = [
        ("01", "의도된 용도·책임", "사용 목적, 사용 금지 범위, QA·공정·IT 승인 책임을 SOP에 명시"),
        ("02", "데이터 무결성", "원본 센서, 시간 동기화, 사용자·변경·판정 이력을 감사추적으로 보존"),
        ("03", "잠금 검증", "독립 기간·새 Run에서 고정 모델과 임계값의 재현성·오경보를 사전 승인 기준으로 평가"),
        ("04", "경보 처리", "확인→설비·배치기록 대조→QA escalation→종결 근거의 순서를 표준화"),
        ("05", "변경·주기 검토", "센서·설비·원료·코드·모델 변경 시 영향평가, 재검증, rollback 기준 운영"),
    ]
    for n, title, desc in steps:
        st.markdown(f'<div class="priority"><div class="n">{n}</div><div><b>{title}</b><span>{desc}</span></div></div>', unsafe_allow_html=True)

    section("경보 대응 SOP", "현장 절차로 전환할 때 필요한 최소 단계입니다. 사업장 품질시스템과 규정에 맞춰 승인해야 합니다.")
    sop = pd.DataFrame([
        ["1. 수신", "Run·시간·모델버전·임계값·상태 저장", "자동", "누락 없는 기록"],
        ["2. 데이터 확인", "결측·시간동기·센서 상태·정비 여부 확인", "공정/IT", "데이터 이상이면 모델 판정 보류"],
        ["3. 공정 대조", "상위 태그, 조작변수, 알람, 배치기록, P&ID 대조", "공정", "조사 근거 기록"],
        ["4. QA 평가", "제품·공정 영향과 일탈 조사 필요성 결정", "QA", "승인된 QRM 기준"],
        ["5. 종결", "오경보/공정이상/센서이상 분류 및 CAPA·변경 연계", "QA", "전자서명·감사추적"],
        ["6. 주기 검토", "오경보, 누락, drift, 모델 버전, 임계값 추세 검토", "QA/공정/IT", "재검증·중단 기준"],
    ], columns=["단계", "필수 수행", "책임", "완료 근거"])
    st.markdown(sop.to_html(index=False, classes="dark-table", border=0, escape=True), unsafe_allow_html=True)

    section("규제·가이드라인 근거", "자동 승인 규칙이 아니라 현장 SOP와 검증계획을 설계하기 위한 공식 참고자료입니다.")
    st.markdown('''
    <div class="source"><b>FDA Process Validation (2011)</b> · 공정 성능을 제품 생애주기 전반에서 지속적으로 확인하고, Stage 3에서 공정이 관리상태를 유지하는지 추세를 검토하는 근거.<br>
    <a href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/process-validation-general-principles-and-practices" target="_blank">공식 문서 열기</a></div>
    <div class="source"><b>ICH Q9(R1) Quality Risk Management</b> · 경보 수준, 누락 위해도, 조사 우선순위와 검토 수준을 위험 기반으로 정하고 문서화하는 근거.<br>
    <a href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/q9r1-quality-risk-management" target="_blank">공식 문서 열기</a></div>
    <div class="source"><b>ICH Q10 Pharmaceutical Quality System</b> · 공정 성능·제품 품질 모니터링, CAPA, 변경관리, 경영검토와 연결하는 근거.<br>
    <a href="https://www.ich.org/page/quality-guidelines" target="_blank">ICH Quality Guidelines 열기</a></div>
    <div class="source"><b>FDA 21 CFR Part 11 Scope and Application</b> · 전자기록의 신뢰성, 감사추적, 보존, 변경 이력과 위험기반 컴퓨터시스템 검증 범위를 설정하는 근거.<br>
    <a href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application" target="_blank">공식 문서 열기</a></div>
    ''', unsafe_allow_html=True)
    st.warning("현재 모델은 연구 검증 단계입니다. 위 SOP를 구현해도 별도의 컴퓨터화 시스템 검증, 현장 성능 검증, 사용자 교육과 QA 승인이 완료되기 전에는 GMP 판정에 사용할 수 없습니다.")

    with st.expander("모델·데이터 상세"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Run", f"{M['n_runs']}개")
        c2.metric("Window", f"{M['n_windows']:,}개")
        c3.metric("Feature", f"{M['n_features']}개")
        c4.metric("외부 분할", f"{M['n_splits']}개")
        st.caption(f"과거 {MODEL['window_sec']}초 Window · {MODEL['stride_sec']}초 간격 · 초기 {MODEL['warmup_sec']}초 제외 · System+Operating Point 그룹 분할")

st.markdown('<div class="footer">연구용 의사결정 지원 · 일탈·OOS·출하 판정은 승인된 시험법, 배치기록, 위험평가와 품질시스템 절차에 따라 권한자가 수행합니다.</div>', unsafe_allow_html=True)
