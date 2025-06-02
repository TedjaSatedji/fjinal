import streamlit as st
import pandas as pd
import numpy as np
import re

@st.cache_data
def load_data():
    data = pd.read_csv("gpu_with_vendors.csv", encoding="utf-8", on_bad_lines="skip")

    # Clean price
    data["price"] = data["price"].replace('[\$,]', '', regex=True)
    data = data[data["price"].str.strip().str.upper() != "N/A"]
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    # Clean clock speed
    data["clock_speed"] = data["clock_speed"].astype(str).str.replace(" MHz", "", regex=False)
    data["clock_speed"] = pd.to_numeric(data["clock_speed"], errors="coerce")

    # Clean memory
    data["memory"] = data["memory"].astype(str).apply(lambda x: re.sub(r"[^\d.]", "", x))
    data["memory"] = pd.to_numeric(data["memory"], errors="coerce")

    # Drop incomplete rows
    data = data.dropna(subset=["price", "clock_speed", "memory"])
    return data

# Load the cleaned data
df = load_data()

st.title("🎯 GPU Recommender (SAW Model)")
st.markdown("Find the best GPU based on your budget and preferences using the Simple Additive Weighting method.")

# User input: Budget
budget = st.number_input("💰 Enter your budget ($):", min_value=100, max_value=10000, value=1500)

# User input: Preferred brands
vendor = df["vendor"].dropna().unique().tolist()
preferred_brands = st.multiselect("🏷️ Preferred Brands (optional):", options=vendor, default=[])

# User input: Weights (auto-normalized)
st.subheader("⚖️ Set Importance (0–100) — Weights are normalized automatically")
col1, col2, col3 = st.columns(3)
with col1:
    raw_price_weight = st.slider("Price Importance", 0, 100, 40)
with col2:
    raw_clock_weight = st.slider("Clock Speed Importance", 0, 100, 40)
with col3:
    raw_memory_weight = st.slider("Memory Importance", 0, 100, 20)

weight_sum = raw_price_weight + raw_clock_weight + raw_memory_weight

if weight_sum == 0:
    st.error("Total weight cannot be 0. Please adjust at least one slider.")
else:
    # Normalize weights
    weights = {
        "price": raw_price_weight / weight_sum,
        "clock_speed": raw_clock_weight / weight_sum,
        "memory": raw_memory_weight / weight_sum
    }

    # Filter by budget and brand
    filtered_df = df[(df["price"] <= budget) & (df["vendor"].isin(preferred_brands))].reset_index(drop=True)

    if filtered_df.empty:
        st.warning("No GPUs match your budget and brand preferences.")
    else:
        # Normalize data for SAW (min-max normalization)
        norm_df = pd.DataFrame()
        norm_df["price"] = filtered_df["price"].min() / filtered_df["price"]  # lower = better
        norm_df["clock_speed"] = filtered_df["clock_speed"] / filtered_df["clock_speed"].max()
        norm_df["memory"] = filtered_df["memory"] / filtered_df["memory"].max()

        # Calculate SAW score
        saw_scores = (
            norm_df["price"] * weights["price"] +
            norm_df["clock_speed"] * weights["clock_speed"] +
            norm_df["memory"] * weights["memory"]
        )
        filtered_df["SAW Score"] = saw_scores

        # Top recommendation
        best_gpu = filtered_df.loc[saw_scores.idxmax()]

        st.subheader("🏆 Best GPU Match")
        st.markdown(f"**Model**: {best_gpu['model']}")
        st.markdown(f"**Brand**: {best_gpu['brand']}")
        st.markdown(f"**Memory**: {best_gpu['memory']} GB")
        st.markdown(f"**Clock Speed**: {best_gpu['clock_speed']} MHz")
        st.markdown(f"**Price**: ${best_gpu['price']:.2f}")
        st.markdown(f"[🔗 View Product]({best_gpu['item_url']})")

        # Table of all matches
        st.subheader("📋 All Matching GPUs")
        st.dataframe(filtered_df.sort_values("SAW Score", ascending=False)[
            ["vendor", "brand", "model", "memory", "clock_speed", "price", "SAW Score"]
        ])
