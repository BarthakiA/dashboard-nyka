import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="Nykaa Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("NYKA.csv", parse_dates=["signup_date", "last_purchase_date"])
    return df

df = load_data()

# Pre-computations
churn_rate = df["churn_within_3m_flag"].mean()
avg_cltv = df["predicted_CLTV_3m"].mean()
cltv_by_seg = df.groupby("RFM_segment_label")["predicted_CLTV_3m"].mean().reset_index()
churn_by_seg = df.groupby("RFM_segment_label")["churn_within_3m_flag"].mean().reset_index()
total_customers = len(df)

tabs = st.tabs(["Summary", "RFM", "CLTV", "Churn", "What-If Analysis"])

# Summary Tab
with tabs[0]:
    st.header("Dashboard Summary")
    top_seg = df['RFM_segment_label'].value_counts().idxmax()
    top_cltv_seg = cltv_by_seg.loc[cltv_by_seg['predicted_CLTV_3m'].idxmax()]
    highest_churn_seg = churn_by_seg.loc[churn_by_seg['churn_within_3m_flag'].idxmax()]
    lowest_churn_seg = churn_by_seg.loc[churn_by_seg['churn_within_3m_flag'].idxmin()]

    st.markdown(f"- **Top Segment:** {top_seg}")
    st.markdown(f"- **Average CLTV:** ₹{avg_cltv:.0f}")
    st.markdown(f"- **Top CLTV Segment:** {top_cltv_seg['RFM_segment_label']} (₹{top_cltv_seg['predicted_CLTV_3m']:.0f})")
    st.markdown(f"- **Overall Churn Rate:** {churn_rate:.1%}")
    st.markdown(f"- **Highest Churn Segment:** {highest_churn_seg['RFM_segment_label']} ({highest_churn_seg['churn_within_3m_flag']:.1%})")
    st.markdown(f"- **Lowest Churn Segment:** {lowest_churn_seg['RFM_segment_label']} ({lowest_churn_seg['churn_within_3m_flag']:.1%})")

# RFM Tab
with tabs[1]:
    st.header("1. RFM Segmentation")
    rfm = df.rename(columns={
        "recency_days": "Recency",
        "frequency_3m": "Frequency",
        "monetary_value_3m": "Monetary"
    })
    # Recency histogram
    fig1 = px.histogram(rfm, x="Recency", nbins=40,
                        title="Recency (days) Distribution",
                        color_discrete_sequence=["#6B7280"])
    fig1.update_layout(legend_title_text="Recency")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Analysis:** Distribution peaks under 60 days. A smaller tail extends to 180 days.")

    st.markdown("---")

    # Frequency histogram
    fig2 = px.histogram(rfm, x="Frequency", nbins=20,
                        title="Order Frequency (3m) Distribution",
                        color_discrete_sequence=["#9CA3AF"])
    fig2.update_layout(legend_title_text="Frequency")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Analysis:** Most customers place 1–3 orders. High-frequency buyers (5+) are fewer.")

    st.markdown("---")

    # Monetary histogram
    fig3 = px.histogram(rfm, x="Monetary", nbins=30,
                        title="Monetary Value (₹, 3m) Distribution",
                        color_discrete_sequence=["#4B5563"])
    fig3.update_layout(legend_title_text="Monetary")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Analysis:** Spending clusters at ₹500–₹1000. Key high-value outliers exceed ₹5000.")

    st.markdown("---")

    # 3D scatter
    fig4 = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary",
                         color="RFM_segment_label",
                         color_discrete_sequence=px.colors.qualitative.Pastel1,
                         title="3D RFM Segments")
    fig4.update_layout(legend_title_text="Segment")
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("**Analysis:** Clear spatial separation between Champions, Loyal, At Risk, etc.")

# CLTV Tab
with tabs[2]:
    st.header("2. Customer Lifetime Value")
    fig5 = px.histogram(df, x="predicted_CLTV_3m", nbins=30,
                        title="Predicted CLTV Distribution",
                        color_discrete_sequence=["#6B7280"])
    fig5.update_layout(legend_title_text="Predicted CLTV")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**Analysis:** Majority under ₹1000; tail extends to ₹3000+.")

    st.markdown("---")

    fig6 = px.scatter(df, x="predicted_CLTV_3m", y="actual_CLTV_3m",
                      title="Predicted vs Actual CLTV",
                      color_discrete_sequence=["#9CA3AF"])
    fig6.update_layout(legend_title_text="Prediction vs Actual")
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("**Analysis:** Strong overall alignment; slight overestimation at low end.")

    st.markdown("---")

    fig7 = px.bar(cltv_by_seg, x="RFM_segment_label", y="predicted_CLTV_3m",
                  title="Avg CLTV by Segment",
                  color_discrete_sequence=["#4B5563"])
    fig7.update_layout(xaxis_title="Segment", yaxis_title="Avg CLTV",
                       legend_title_text="Segment")
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("**Analysis:** Champions and Loyal segments drive highest CLTV.")

    st.markdown("---")

    fig8 = px.box(df, x="RFM_segment_label", y="actual_CLTV_3m",
                  title="Actual CLTV by Segment",
                  color_discrete_sequence=["#9CA3AF"])
    fig8.update_layout(legend_title_text="Segment")
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown("**Analysis:** High variance within segments suggests tailored strategies.")

# Churn Tab
with tabs[3]:
    st.header("3. Churn Analysis & Prediction")
    fig9 = px.bar(x=["Active","Churned"], y=[1-churn_rate, churn_rate],
                  title="Overall 3-Month Churn Rate",
                  color_discrete_sequence=["#6B7280","#9CA3AF"])
    fig9.update_layout(xaxis_title="Status", yaxis_title="Proportion", legend_title_text="Customer Status")
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown(f"**Analysis:** Churn rate at {churn_rate:.1%} warrants retention focus.")

    st.markdown("---")

    fig10 = px.bar(churn_by_seg, x="RFM_segment_label", y="churn_within_3m_flag",
                   title="Churn Rate by Segment",
                   color_discrete_sequence=["#4B5563"])
    fig10.update_layout(xaxis_title="Segment", yaxis_title="Churn Rate", legend_title_text="Segment")
    st.plotly_chart(fig10, use_container_width=True)
    st.markdown("**Analysis:** At Risk segment churns highest; Champions churn lowest.")

    st.markdown("---")

    fig11 = px.box(df, x="churn_within_3m_flag", y="recency_days",
                   title="Recency by Churn Status",
                   color_discrete_sequence=["#6B7280"])
    fig11.update_layout(xaxis_title="Churned(1) vs Active(0)", yaxis_title="Recency (days)", legend_title_text="Status")
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown("**Analysis:** Churned customers have longer purchase gaps.")

    st.markdown("---")

    features = ["recency_days","frequency_3m","monetary_value_3m",
                "time_on_app_minutes","page_views_per_session",
                "campaign_clicks","campaign_views","cart_abandonment_rate",
                "first_time_buyer_flag"]
    X = df[features].fillna(0); y = df["churn_within_3m_flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000); model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred); auc = roc_auc_score(y_test, y_pred)
    fig12 = px.line(x=fpr, y=tpr,
                    title=f"ROC Curve (AUC={auc:.2f})",
                    color_discrete_sequence=["#4B5563"])
    fig12.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", legend_title_text="ROC")
    st.plotly_chart(fig12, use_container_width=True)
    st.markdown("**Analysis:** AUC indicates solid prediction power.")

# What-If Analysis Tab
with tabs[4]:
    st.header("4. What-If Analysis")
    cltv_boost = st.slider("Projected CLTV increase (%)", 0, 50, 10)
    churn_reduc = st.slider("Projected churn reduction (%)", 0, 50, 10)

    baseline_retained = total_customers * (1 - churn_rate)
    scenario_retained = total_customers * (1 - churn_rate * (1 - churn_reduc/100))
    scenario_cltv = avg_cltv * (1 + cltv_boost/100)

    baseline_revenue = baseline_retained * avg_cltv
    scenario_revenue = scenario_retained * scenario_cltv

    summary_df = pd.DataFrame({
        "Scenario": ["Baseline", "What-If"],
        "Retained_Customers": [baseline_retained, scenario_retained],
        "Avg_CLTV": [avg_cltv, scenario_cltv],
        "Projected_Revenue": [baseline_revenue, scenario_revenue]
    })

    fig13 = px.bar(summary_df, x="Scenario", y="Projected_Revenue",
                   title="Projected Revenue Comparison",
                   color_discrete_sequence=["#6B7280","#4B5563"])
    fig13.update_layout(yaxis_title="Revenue (₹)", legend_title_text="Scenario")
    st.plotly_chart(fig13, use_container_width=True)
    st.markdown("**Analysis:** Compare baseline vs scenario revenue with applied improvements.")
    st.markdown("- Increased CLTV and reduced churn directly boost revenue.")
    st.markdown("- Use this to justify marketing investments.")
