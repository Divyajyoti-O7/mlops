import streamlit as st
import requests
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Customer Segmentation Demo")
st.write("Enter customer details to see which segment they belong to.")

@st.cache_data
def load_data():
    return pd.read_csv("data/raw/mall_customers_v1.csv")

df = load_data()

age = st.slider("Age", 18, 70, 30)
income = st.slider("Annual Income (k$)", 10, 150, 60)
spending = st.slider("Spending Score", 1, 100, 50)

if st.button("Predict Segment"):
    response = requests.post(
        "http://127.0.0.1:8001/predict",
        json={"features": [age, income, spending]}
    )

    if response.status_code == 200:
        result = response.json()

        st.success(f"Customer Segment: {result['cluster']}")
        st.info(
            f"Distance to Segment Center: "
            f"{result['distance_to_nearest_cluster']:.2f}"
        )

        st.subheader("📊 Customer Position vs Existing Customers")

        scatter_fig = px.scatter(
            df,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            opacity=0.4,
            title="Customer Distribution"
        )

        scatter_fig.data[0].name = "Existing Customers"
        scatter_fig.data[0].showlegend = True

        scatter_fig.add_scatter(
            x=[income],
            y=[spending],
            mode="markers",
            marker=dict(size=14, symbol="x"),
            name="New Customer"
        )

        scatter_fig.update_layout(
            xaxis_title="Annual Income (k$)",
            yaxis_title="Spending Score",
            legend_title_text="Legend"
        )

        st.plotly_chart(scatter_fig, use_container_width=True)

        st.subheader("📈 Distance to Nearest Segment")

        distance_df = pd.DataFrame({
            "Metric": ["Distance to Nearest Cluster"],
            "Value": [result["distance_to_nearest_cluster"]]
        })

        bar_fig = px.bar(
            distance_df,
            x="Metric",
            y="Value",
            text="Value",
            title="Cluster Distance Indicator"
        )

        bar_fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        bar_fig.update_layout(
            yaxis_title="Distance",
            xaxis_title=""
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.error("API not reachable or request failed. Is backend running?")


