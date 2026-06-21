import base64
import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data/raw/dataset_merged.csv"
ML_DIR = ROOT / "models/ml"
DL_MODEL_PATH = ROOT / "models/dl/best_dl_model.pt"
DL_META_PATH = ROOT / "models/dl/best_dl_model_meta.json"
CLUSTER_MODEL_REGISTRY = ROOT / "models/clustering/multi_view_clustering_models.csv"
IMAGE_DIR = ROOT / "app/image"

SOCIAL_MAP = {"low": 0, "medium": 1, "high": 2}
GENDER_MAP = {"male": 0, "female": 1}
PLATFORMS = ["Both", "Instagram", "Other", "TikTok"]


st.set_page_config(page_title="Dự đoán sức khỏe tinh thần", layout="wide")

st.markdown(
    """
    <style>
    .result-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 0.85rem;
        width: 100%;
        margin-top: 0.9rem;
    }
    .result-card {
        border: 1px solid #e6e8ef;
        border-radius: 12px;
        padding: 0.78rem 0.9rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        min-height: 88px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        overflow-wrap: anywhere;
    }
    .result-label {
        color: #6b7280;
        font-size: 0.78rem;
        margin-bottom: 0.25rem;
    }
    .result-value {
        color: #111827;
        font-size: 1.02rem;
        font-weight: 650;
        line-height: 1.25;
    }
    .result-note {
        color: #6b7280;
        font-size: 0.76rem;
        margin-top: 0.25rem;
    }
    .persona-card {
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        background: #eff6ff;
        margin-bottom: 1rem;
    }
    .persona-title {
        color: #1e3a8a;
        font-size: 0.84rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }
    .persona-value {
        color: #111827;
        font-size: 1rem;
        font-weight: 650;
        line-height: 1.45;
    }
    .prediction-panel {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        margin-top: 0.85rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        display: grid;
        grid-template-columns: 150px minmax(0, 1fr);
        gap: 1rem;
        align-items: center;
        width: 100%;
        overflow: hidden;
    }
    .prediction-image-wrap {
        width: 150px;
        max-width: 100%;
        text-align: center;
    }
    .prediction-image {
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 14px;
        display: block;
    }
    .prediction-image-caption {
        color: #6b7280;
        font-size: 0.74rem;
        line-height: 1.35;
        margin-top: 0.45rem;
        overflow-wrap: anywhere;
    }
    .prediction-content {
        min-width: 0;
    }
    .prediction-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.72rem;
        font-size: 0.78rem;
        font-weight: 750;
        margin-bottom: 0.55rem;
    }
    .prediction-badge.safe {
        color: #166534;
        background: #dcfce7;
        border: 1px solid #bbf7d0;
    }
    .prediction-badge.risk {
        color: #991b1b;
        background: #fee2e2;
        border: 1px solid #fecaca;
    }
    .prediction-title {
        font-size: 1rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.35rem;
    }
    .prediction-text {
        font-size: 0.9rem;
        color: #374151;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }
    .prediction-mini {
        color: #6b7280;
        font-size: 0.78rem;
        margin-top: 0.35rem;
        overflow-wrap: anywhere;
    }
    .user-marker-note {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border: 1px solid #fed7aa;
        background: #fff7ed;
        color: #9a3412;
        border-radius: 999px;
        padding: 0.22rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 0.15rem 0 0.25rem;
    }
    .user-marker-star {
        color: #f97316;
        font-size: 0.86rem;
        line-height: 1;
    }
    @media (max-width: 760px) {
        .prediction-panel {
            grid-template-columns: 1fr;
            gap: 0.75rem;
        }
        .prediction-image-wrap {
            width: 120px;
        }
        .result-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if torch is not None:

    class FeatureTokenizer(nn.Module):
        def __init__(self, n_features, d_token):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.feature_embedding = nn.Parameter(torch.empty(n_features, d_token))
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.feature_embedding)

        def forward(self, x):
            return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0) + self.feature_embedding.unsqueeze(0)


    class FTTransformerClassifier(nn.Module):
        def __init__(self, n_features, d_token=48, n_heads=4, n_layers=2, dropout=0.15):
            super().__init__()
            self.tokenizer = FeatureTokenizer(n_features, d_token)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=d_token * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, d_token),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token, 1),
            )
            nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, x):
            tokens = self.tokenizer(x)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            encoded = self.encoder(tokens)
            return self.head(encoded[:, 0])


@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_feature_stats():
    raw = load_raw_data().copy()
    feature_df = build_feature_frame(raw)
    continuous_cols = [
        "age",
        "daily_social_media_hours",
        "sleep_hours",
        "screen_time_before_sleep",
        "academic_performance",
        "physical_activity",
        "social_interaction_level",
        "stress_level",
        "anxiety_level",
        "sm_sleep_ratio",
        "screen_sleep_ratio",
    ]
    return {
        "mean": feature_df[continuous_cols].mean(),
        "std": feature_df[continuous_cols].std(ddof=0).replace(0, 1),
        "continuous_cols": continuous_cols,
    }


@st.cache_resource
def load_ml_artifacts():
    artifacts = {}
    for path in sorted(ML_DIR.glob("top*.pkl")):
        obj = joblib.load(path)
        label = obj.get("candidate_key", path.stem)
        test_f1 = obj.get("test_metrics", {}).get("test_f1")
        display = f"ML: {label}"
        if test_f1 is not None:
            display += f" (test F1 {test_f1:.3f})"
        artifacts[display] = obj
    return artifacts


@st.cache_resource
def load_dl_artifact():
    if torch is None or not DL_MODEL_PATH.exists():
        return None
    checkpoint = torch.load(DL_MODEL_PATH, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    model = FTTransformerClassifier(n_features=metadata.get("input_dim", 18), d_token=48, n_heads=4, n_layers=2, dropout=0.15)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return {"model": model, "metadata": metadata}


@st.cache_resource
def load_cluster_artifacts():
    registry = pd.read_csv(CLUSTER_MODEL_REGISTRY)
    artifacts = {}
    for row in registry.itertuples(index=False):
        artifact = joblib.load(ROOT / row.artifact_path)
        artifacts[row.view] = {"registry": row._asdict(), "artifact": artifact}
    return artifacts


def encode_social(value):
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return float(SOCIAL_MAP[str(value).strip().lower()])


def build_feature_frame(df):
    out = pd.DataFrame()
    out["age"] = df["age"].astype(float)
    out["gender"] = df["gender"].astype(str).str.lower().map(GENDER_MAP).fillna(0).astype(int)
    out["daily_social_media_hours"] = df["daily_social_media_hours"].astype(float)
    out["sleep_hours"] = df["sleep_hours"].astype(float)
    out["screen_time_before_sleep"] = df["screen_time_before_sleep"].astype(float)
    out["academic_performance"] = df["academic_performance"].astype(float)
    out["physical_activity"] = df["physical_activity"].astype(float)
    out["social_interaction_level"] = df["social_interaction_level"].map(encode_social)
    out["stress_level"] = df["stress_level"].astype(float)
    out["anxiety_level"] = df["anxiety_level"].astype(float)

    platform = df["platform_usage"].astype(str)
    for name in PLATFORMS:
        out[f"platform_{name}"] = (platform == name).astype(int)

    out["sm_sleep_ratio"] = out["daily_social_media_hours"] / (out["sleep_hours"] + 1)
    out["screen_sleep_ratio"] = out["screen_time_before_sleep"] / (out["sleep_hours"] + 1)
    out["heavy_sm_user"] = (out["daily_social_media_hours"] >= 5).astype(int)
    out["low_sleep"] = (out["sleep_hours"] < 6).astype(int)
    return out


def preprocess_for_depression(user_input, feature_cols):
    feature_df = build_feature_frame(pd.DataFrame([user_input]))
    stats = load_feature_stats()
    feature_df[stats["continuous_cols"]] = (
        feature_df[stats["continuous_cols"]] - stats["mean"]
    ) / stats["std"]
    return feature_df.reindex(columns=feature_cols, fill_value=0)


def predict_ml(artifact, user_input):
    feature_cols = artifact["feature_columns"]
    X = preprocess_for_depression(user_input, feature_cols)
    model = artifact["model"]
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = int(prob >= 0.5)
    return prob, pred, 0.5


def predict_dl(artifact, user_input):
    if artifact is None:
        raise RuntimeError("PyTorch or DL artifact is unavailable.")
    metadata = artifact["metadata"]
    X = preprocess_for_depression(user_input, metadata["feature_cols"])
    tensor = torch.tensor(X.values.astype("float32"))
    with torch.no_grad():
        prob = float(torch.sigmoid(artifact["model"](tensor).squeeze(1))[0].item())
    threshold = float(metadata.get("threshold", 0.5))
    pred = int(prob >= threshold)
    return prob, pred, threshold


def make_user_form():
    with st.form("user_input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Tuổi", 13, 19, 16)
            gender = st.selectbox("Giới tính", ["female", "male"], format_func=lambda x: {"female": "Nữ", "male": "Nam"}[x])
            platform = st.selectbox("Nền tảng chính", ["Instagram", "TikTok", "Both", "Other"])
            social = st.selectbox(
                "Mức tương tác xã hội",
                ["low", "medium", "high"],
                index=1,
                format_func=lambda x: {"low": "Thấp", "medium": "Trung bình", "high": "Cao"}[x],
            )
        with c2:
            sm_hours = st.slider("Số giờ dùng mạng xã hội/ngày", 0.0, 10.0, 5.0, 0.1)
            sleep = st.slider("Số giờ ngủ/ngày", 3.0, 10.0, 6.5, 0.1)
            screen_sleep = st.slider("Thời gian nhìn màn hình trước ngủ", 0.0, 4.0, 1.5, 0.1)
            activity = st.slider("Vận động thể chất giờ/ngày", 0.0, 3.0, 1.0, 0.1)
        with c3:
            academic = st.slider("Kết quả học tập / GPA", 1.0, 4.0, 3.0, 0.01)
            stress = st.slider("Mức stress", 1, 10, 5)
            anxiety = st.slider("Mức lo âu", 1, 10, 5)

        submitted = st.form_submit_button("Chạy dự đoán")

    user_input = {
        "age": age,
        "gender": gender,
        "platform_usage": platform,
        "daily_social_media_hours": sm_hours,
        "sleep_hours": sleep,
        "screen_time_before_sleep": screen_sleep,
        "academic_performance": academic,
        "physical_activity": activity,
        "social_interaction_level": social,
        "stress_level": stress,
        "anxiety_level": anxiety,
    }
    return user_input, submitted


def risk_label(prob, pred):
    if pred:
        return "Lớp 1: Có dấu hiệu trầm cảm", "Mẫu này vượt ngưỡng nguy cơ của model"
    if prob >= 0.35:
        return "Lớp 0: Không có dấu hiệu trầm cảm", "Chưa vượt ngưỡng, nhưng nên theo dõi thêm"
    return "Lớp 0: Không có dấu hiệu trầm cảm", "Thấp hơn ngưỡng nguy cơ"


def class_description(pred):
    if pred:
        return {
            "badge": "Lớp 1",
            "title": "Có dấu hiệu trầm cảm",
            "subtitle": "Model đánh giá mẫu này thuộc nhóm có nguy cơ cao hơn.",
            "css": "risk",
        }
    return {
        "badge": "Lớp 0",
        "title": "Không có dấu hiệu trầm cảm",
        "subtitle": "Model đánh giá mẫu này chưa vượt ngưỡng nguy cơ.",
        "css": "safe",
    }


def class_image_path(pred):
    image_path = IMAGE_DIR / f"class{pred}.png"
    return image_path if image_path.exists() else None


def image_data_uri(image_path):
    if image_path is None:
        return ""
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


CLUSTER_NAME_VI = {
    "Light / Rested Users": "Người dùng nhẹ, ngủ tương đối đủ",
    "Heavy Daytime Social Media Users": "Dùng mạng xã hội nhiều ban ngày",
    "Heavy Night Social Media Users": "Dùng mạng xã hội nhiều về đêm",
    "Healthy Lifestyle Users": "Lối sống lành mạnh",
    "Lifestyle Risk Users": "Lối sống có rủi ro",
    "Socially Isolated Users": "Tương tác xã hội thấp",
    "Low Activity Moderate Social Users": "Tương tác vừa phải, ít vận động",
    "Moderately Social Users": "Tương tác xã hội trung bình",
    "Low Stress Users": "Nhóm stress thấp",
    "High Stress Users": "Nhóm stress cao",
}


VIEW_NAME_VI = {
    "Usage Intensity": "Cường độ sử dụng",
    "Lifestyle Health": "Lối sống",
    "Social Behavior": "Tương tác xã hội",
    "Stress Pattern": "Stress và lo âu",
}


def vi_cluster_name(name):
    return CLUSTER_NAME_VI.get(name, name)


def predict_cluster_view(artifact, user_input):
    user_df = pd.DataFrame([user_input]).copy()
    user_df["social_interaction_score"] = user_df["social_interaction_level"].map(encode_social)
    X_user = artifact["scaler"].transform(user_df[artifact["features"]])
    if artifact["reducer"] is not None:
        X_user = artifact["reducer"].transform(X_user)
    cluster_id = int(artifact["kmeans"].predict(X_user)[0])
    return cluster_id, artifact["cluster_name_map"][cluster_id], X_user


def plot_cluster_with_user(view_name, artifact, user_input):
    raw = load_raw_data().dropna().copy()
    raw["social_interaction_score"] = raw["social_interaction_level"].map(encode_social)
    X = artifact["scaler"].transform(raw[artifact["features"]])
    if artifact["reducer"] is not None:
        X_plot = artifact["reducer"].transform(X)
    else:
        X_plot = X
    if X_plot.shape[1] > 2:
        X_plot = artifact["plot_pca"].transform(X_plot) if "plot_pca" in artifact else X_plot[:, :2]

    labels = artifact["kmeans"].predict(X_plot if artifact["reducer"] is not None else X)
    names = [vi_cluster_name(artifact["cluster_name_map"][int(label)]) for label in labels]
    cluster_id, cluster_name, X_user = predict_cluster_view(artifact, user_input)
    cluster_name_vi = vi_cluster_name(cluster_name)
    user_point = X_user[0]
    if user_point.shape[0] > 2:
        user_point = user_point[:2]

    fig = go.Figure()
    for name in sorted(set(names)):
        mask = np.array(names) == name
        fig.add_trace(
            go.Scattergl(
                x=X_plot[mask, 0],
                y=X_plot[mask, 1],
                mode="markers",
                name=name,
                marker={"size": 5, "opacity": 0.42},
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Trục 1: %{x:.2f}<br>"
                    "Trục 2: %{y:.2f}<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[user_point[0]],
            y=[user_point[1]],
            mode="markers",
            name="Người dùng hiện tại",
            marker={
                "symbol": "star",
                "size": 17,
                "color": "#f97316",
                "line": {"color": "#7c2d12", "width": 1.6},
            },
            hovertemplate=(
                "<b>Người dùng hiện tại</b><br>"
                f"Nhóm: {cluster_name_vi}<br>"
                "Trục 1: %{x:.2f}<br>"
                "Trục 2: %{y:.2f}<extra></extra>"
            ),
        )
    )
    view_title = VIEW_NAME_VI.get(artifact["display_name"], artifact["display_name"])
    fig.update_layout(
        title={"text": f"{view_title}: {cluster_name_vi}", "font": {"size": 13}},
        height=330,
        margin={"l": 14, "r": 12, "t": 44, "b": 92},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10},
        },
        xaxis={"title": {"text": "Trục 1", "standoff": 18}, "automargin": True},
        yaxis={"title": {"text": "Trục 2", "standoff": 12}, "automargin": True},
    )
    return fig, cluster_name_vi


st.title("Dự đoán sức khỏe tinh thần & phân cụm hành vi mạng xã hội")
st.caption("Ứng dụng demo: dự đoán nguy cơ trầm cảm và phân cụm hành vi theo nhiều góc nhìn.")

user_input, submitted = make_user_form()

ml_artifacts = load_ml_artifacts()
dl_artifact = load_dl_artifact()

model_options = list(ml_artifacts.keys())
if dl_artifact is not None:
    model_name = dl_artifact["metadata"].get("model_name", "Best DL")
    model_options.insert(0, f"DL: {model_name}")

tab_predict, tab_cluster = st.tabs(["Dự đoán trầm cảm", "Phân cụm hành vi"])

with tab_predict:
    st.subheader("Chọn model và dự đoán nguy cơ")
    selected_model = st.selectbox("Model dự đoán", model_options)

    if st.button("Dự đoán nguy cơ trầm cảm", type="primary") or submitted:
        if selected_model.startswith("DL:"):
            prob, pred, threshold = predict_dl(dl_artifact, user_input)
        else:
            prob, pred, threshold = predict_ml(ml_artifacts[selected_model], user_input)

        label, note = risk_label(prob, pred)
        class_info = class_description(pred)
        selected_model_safe = html.escape(selected_model)
        label_safe = html.escape(label)
        note_safe = html.escape(note)
        st.markdown(
            (
                '<div class="result-grid">'
                '<div class="result-card">'
                '<div class="result-label">Xác suất lớp 1</div>'
                f'<div class="result-value">{prob:.1%}</div>'
                '<div class="result-note">Có dấu hiệu trầm cảm</div>'
                '</div>'
                '<div class="result-card">'
                '<div class="result-label">Xác suất lớp 0</div>'
                f'<div class="result-value">{1 - prob:.1%}</div>'
                '<div class="result-note">Không có dấu hiệu trầm cảm</div>'
                '</div>'
                '<div class="result-card">'
                '<div class="result-label">Ngưỡng quyết định</div>'
                f'<div class="result-value">{threshold:.2f}</div>'
                '<div class="result-note">P(lớp 1) ≥ ngưỡng thì dự đoán lớp 1</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        image_path = class_image_path(pred)
        image_uri = image_data_uri(image_path)
        image_html = ""
        if image_uri:
            image_html = (
                '<div class="prediction-image-wrap">'
                f'<img class="prediction-image" src="{image_uri}" alt="{label_safe}">'
                f'<div class="prediction-image-caption">{label_safe}</div>'
                '</div>'
            )

        st.markdown(
            (
                '<div class="prediction-panel">'
                f'{image_html}'
                '<div class="prediction-content">'
                f'<div class="prediction-badge {class_info["css"]}">{html.escape(class_info["badge"])}</div>'
                f'<div class="prediction-title">{html.escape(class_info["title"])}</div>'
                '<div class="prediction-text">'
                f'{html.escape(class_info["subtitle"])} {note_safe}.<br>'
                f'Xác suất thuộc <b>lớp 1</b> là <b>{prob:.1%}</b>; '
                f'ngưỡng quyết định là <b>{threshold:.2f}</b>. '
                f'Vì vậy kết quả cuối cùng là <b>{label_safe}</b>.'
                '</div>'
                f'<div class="prediction-mini">Model đang chọn: {selected_model_safe}</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        st.progress(min(max(prob, 0), 1))
        st.caption(
            "Xác suất được lấy từ `predict_proba` của model ML hoặc sigmoid của model DL. "
            "Đây là xác suất model gán cho lớp 1. Nếu xác suất lớp 1 thấp hơn ngưỡng, kết quả là lớp 0."
        )
        st.info("Đây là demo mô hình, không phải chẩn đoán y khoa.")

with tab_cluster:
    st.subheader("Phân cụm người dùng theo nhiều góc nhìn")
    cluster_artifacts = load_cluster_artifacts()

    if st.button("Gán cụm hành vi", type="primary") or submitted:
        persona_parts = []
        cluster_items = list(cluster_artifacts.items())
        rendered_views = []
        for view_name, pack in cluster_items:
            artifact = pack["artifact"]
            fig, cluster_name = plot_cluster_with_user(view_name, artifact, user_input)
            persona_parts.append(cluster_name)
            rendered_views.append((artifact, fig, cluster_name))

        st.markdown(
            f"""
            <div class="persona-card">
                <div class="persona-title">Persona tổng hợp</div>
                <div class="persona-value">{" | ".join(persona_parts)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        grid = st.columns(2)
        for idx, (artifact, fig, cluster_name) in enumerate(rendered_views):
            with grid[idx % 2]:
                st.markdown(f"**{VIEW_NAME_VI.get(artifact['display_name'], artifact['display_name'])}**")
                st.caption(cluster_name)
                st.markdown(
                    '<div class="user-marker-note"><span class="user-marker-star">★</span>Người dùng hiện tại</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"Model: {artifact['method']} | số cụm k={artifact['k']}")
    else:
        st.write("Nhập thông tin và bấm **Gán cụm hành vi** để xem người dùng nằm ở cụm nào trong từng góc nhìn.")
