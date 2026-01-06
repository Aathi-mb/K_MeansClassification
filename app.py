import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------- Page Config --------------------
st.set_page_config(page_title="K-Means Classification", layout="centered")

st.title("🔵 K-Means Classification App")
st.write("Upload dataset and perform K-Means clustering")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # -------------------- Feature Selection --------------------
    st.subheader("🎯 Select Features for Clustering")
    feature_cols = st.multiselect(
        "Choose TWO features",
        df.columns,
        default=df.columns[:2]
    )

    if len(feature_cols) == 2:
        X = df[feature_cols]

        # -------------------- Scaling --------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -------------------- Choose K --------------------
        st.subheader("⚙️ K-Means Settings")
        k = st.slider("Select number of clusters (K)", 2, 10, 3)

        # -------------------- Train Model --------------------
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Predicted_Cluster"] = kmeans.fit_predict(X_scaled)

        st.success("✅ Clustering Completed")

        # -------------------- Cluster Output --------------------
        st.subheader("📊 Clustered Data")
        st.dataframe(df.head())

        # -------------------- Plot --------------------
        st.subheader("📈 Cluster Visualization")

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=df["Predicted_Cluster"]
        )

        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker="X",
            s=200
        )

        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_title("K-Means Clustering")

        st.pyplot(fig)

        # -------------------- New Data Prediction --------------------
        st.subheader("🧪 Predict Cluster for New Data")

        val1 = st.number_input(f"Enter {feature_cols[0]}")
        val2 = st.number_input(f"Enter {feature_cols[1]}")

        if st.button("Predict Cluster"):
            new_data = [[val1, val2]]
            new_data_scaled = scaler.transform(new_data)
            pred = kmeans.predict(new_data_scaled)

            st.success(f"🔮 Predicted Cluster: {pred[0]}")

    else:
        st.warning("⚠️ Please select exactly TWO features")
