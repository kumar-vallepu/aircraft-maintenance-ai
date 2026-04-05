import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Aircraft AI", layout="wide")

# ---------- NASA STYLE HERO ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0b0f14;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #11161c;
}
.hero {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.9)),
                url('https://images.unsplash.com/photo-1446776811953-b23d57bd21aa');
    background-size: cover;
    background-position: center;
    padding: 120px 50px;
    border-radius: 15px;
    margin-bottom: 30px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 700;
    color: white;
}
.hero p {
    font-size: 18px;
    color: #c9d1d9;
    max-width: 600px;
}
[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 15px;
    border-radius: 10px;
}
</style>

<div class="hero">
    <h1>Aircraft Predictive Maintenance</h1>
    <p>
    AI-powered engine monitoring system for predicting failures,
    optimizing maintenance schedules, and improving operational safety.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    cols = ['unit', 'time', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]

    df = pd.read_csv(
        'data/train_FD001.txt',
        sep=r"\s+",
        header=None,
        engine='python',
        on_bad_lines='skip'
    )

    df = df.dropna(axis=1)
    df.columns = cols

    rul = df.groupby('unit')['time'].max().reset_index()
    rul.columns = ['unit', 'max_time']
    df = df.merge(rul, on='unit')
    df['RUL'] = df['max_time'] - df['time']

    return df

df = load_data()

# ---------- ML MODEL ----------
df['failure'] = df['RUL'].apply(lambda x: 1 if x < 30 else 0)

features = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']
X = df[features]
y = df['failure']

# ✅ NEW: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ NEW: Predictions
y_pred = model.predict(X_test)

# ✅ NEW: Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ---------- NAVIGATION ----------
section = st.sidebar.radio(
    "Navigation",
    ["Overview", "Analytics", "Model Insights"]
)

# ---------- SIDEBAR ----------
engine_id = st.sidebar.selectbox("Select Engine", df['unit'].unique())
engine_data = df[df['unit'] == engine_id]
latest_rul = int(engine_data['RUL'].values[-1])

# ---------- OVERVIEW ----------
if section == "Overview":

    st.markdown("### System Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Engine ID", engine_id)
    col2.metric("Cycle", int(engine_data['time'].values[-1]))
    col3.metric("Remaining Life", latest_rul)

    fig_rul = px.line(engine_data, x='time', y='RUL',
                      title="Remaining Useful Life")

    st.plotly_chart(fig_rul, use_container_width=True)

# ---------- ANALYTICS ----------
elif section == "Analytics":

    st.markdown("### Data Analytics")

    sensor = st.selectbox("Select Sensor", [f'sensor{i}' for i in range(1, 22)])

    fig_sensor = px.line(engine_data, x='time', y=sensor,
                         title=f"{sensor} Trend")

    health_counts = df['failure'].value_counts()

    fig_pie = go.Figure(data=[go.Pie(
        labels=['Healthy', 'Failure Soon'],
        values=[health_counts.get(0, 0), health_counts.get(1, 0)],
        hole=0.5
    )])

    fig_pie.update_layout(title="Fleet Health Distribution")

    trend = df.groupby('time')['failure'].mean().reset_index()

    fig_trend = px.line(trend, x='time', y='failure',
                        title="Failure Probability Trend")

    st.plotly_chart(fig_sensor, use_container_width=True)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------- MODEL INSIGHTS ----------
elif section == "Model Insights":

    st.markdown("### Model Insights")

    latest_data = engine_data.iloc[-1][features].values.reshape(1, -1)
    prediction = model.predict(latest_data)[0]
    proba = model.predict_proba(latest_data)[0][1]

    if prediction == 1:
        st.error(f"Failure risk: {proba*100:.2f}%")
    else:
        st.success(f"Normal operation (risk: {proba*100:.2f}%)")

    # ✅ NEW: Accuracy
    st.markdown("### Model Performance")
    st.metric("Accuracy", f"{accuracy*100:.2f}%")

    # ✅ NEW: Confusion Matrix
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # ✅ NEW: Classification Report
    st.text("Classification Report")
    st.text(report)

    # Existing
    importances = model.feature_importances_

    feat_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig_imp = px.bar(feat_df, x='Feature', y='Importance',
                     title="Feature Importance")

    fig_hist = px.histogram(df, x='RUL', nbins=50,
                            title="RUL Distribution")

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------- ALERT ----------
if latest_rul < 20:
    st.warning("Immediate maintenance required")
elif latest_rul < 50:
    st.info("Maintenance recommended soon")