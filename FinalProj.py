import streamlit as st
import pandas as pd
import numpy as np
import re
import requests

import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCBGjgvI6yfK-zTH8QjLd4x1Yzrg-Qhc7Q"

genai.configure(api_key=GEMINI_API_KEY)

SERPAPI_KEY = "d3c7eccccf78af74e7fb50878650049bc17e1e15eacce3d8a41040c71a080b91"

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
            url("https://r4.wallpaperflare.com/wallpaper/58/397/197/nvidia-gpus-technology-pc-gaming-wallpaper-f93682b61b3a6fea60bb226aa8078573.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

    # Clean length
    if "length" in data.columns:
        data["length"] = data["length"].astype(str).apply(lambda x: re.sub(r"[^\d.]", "", x))
        data["length"] = pd.to_numeric(data["length"], errors="coerce")
    else:
        data["length"] = np.nan  # fallback if column missing   

    # Drop incomplete rows
    data = data.dropna(subset=["price", "clock_speed", "memory", "length"])
    return data

def get_gpu_image_url(query):
    try:
        params = {
            "q": query,
            "tbm": "isch",
            "ijn": "0",
            "api_key": SERPAPI_KEY,
        }
        response = requests.get("https://serpapi.com/search", params=params)
        results = response.json()
        if "images_results" in results and len(results["images_results"]) > 0:
            return results["images_results"][0]["thumbnail"]
    except Exception as e:
        st.warning(f"Image scraping failed: {e}")
    return None

# Load the cleaned data
df = load_data()

st.title("🤔 GPU Recommender (SAW Model)")
st.markdown("Find the best GPU based on your budget and preferences using the Simple Additive Weighting method.")

# User input: Budget
budget = st.number_input("💰 Enter your budget ($):", min_value=100, max_value=10000, value=1500)

# User input: Preferred brands
vendor = df["vendor"].dropna().unique().tolist()
preferred_brands = st.multiselect("🏷️ Preferred Brands (optional):", options=vendor, default=vendor)

# User input: Weights (auto-normalized)
st.subheader("⚖️ Set Importance (0–100) — Weights are normalized automatically")
col1, col2, col3, col4 = st.columns(4)
with col1:
    raw_price_weight = st.slider("Price Importance (Higher is cheaper)", 0, 100, 30)
with col2:
    raw_clock_weight = st.slider("Clock Speed Importance", 0, 100, 30)
with col3:
    raw_memory_weight = st.slider("Memory Importance", 0, 100, 20)
with col4:
    raw_length_weight = st.slider("Length Importance (Higher is shorter)", 0, 100, 20)

weight_sum = raw_price_weight + raw_clock_weight + raw_memory_weight + raw_length_weight

if weight_sum == 0:
    st.error("Total weight cannot be 0. Please adjust at least one slider.")
else:
    # Normalize weights
    weights = {
        "price": raw_price_weight / weight_sum,
        "clock_speed": raw_clock_weight / weight_sum,
        "memory": raw_memory_weight / weight_sum,
        "length": raw_length_weight / weight_sum
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
        norm_df["length"] = filtered_df["length"].min() / filtered_df["length"]  # shorter = better

        # Calculate SAW score
        saw_scores = (
            norm_df["price"] * weights["price"] +
            norm_df["clock_speed"] * weights["clock_speed"] +
            norm_df["memory"] * weights["memory"] +
            norm_df["length"] * weights["length"]
        )
        filtered_df["SAW Score"] = saw_scores

        # Top recommendation
        top_gpus = filtered_df.sort_values("SAW Score", ascending=False).head(5).reset_index(drop=True)

        st.subheader("🏆 Top 5 GPU Matches")
        for idx, gpu in top_gpus.iterrows():
            st.markdown(f"### {idx+1}. {gpu['model']} ({gpu['brand']})")
            image_url = get_gpu_image_url(f"{gpu['brand']} {gpu['model']} GPU")
            if image_url:
                st.image(image_url, width=300, caption=f"{gpu['brand']} {gpu['model']}")
            else:
                st.info("No image found.")

            st.markdown(f"- **SAW Score**: {gpu['SAW Score']:.4f}")
            st.markdown(f"- **Memory**: {gpu['memory']} GB")
            st.markdown(f"- **Clock Speed**: {gpu['clock_speed']} MHz")
            st.markdown(f"- **Length**: {gpu['length']} mm")
            st.markdown(f"- **Price**: ${gpu['price']:.2f}")
            st.markdown(f"- [🔗 View Product]({gpu['item_url']})")
            st.markdown("---")
            
             # Button for Gemini AI recommendation
            if st.button(f"{gpu['model']} PC Build Reccomendation Powered by Gemini AI✨", key=f"gemini_{idx}"):
                with st.spinner("Contacting Gemini AI..."):
                    def call_gemini_api(gpu_info):
                        prompt = (
                            f"Suggest a balanced PC build including CPU, RAM, motherboard, PSU, case, and storage "
                            f"to pair with the following GPU for gaming and productivity, maintain balance.\n\n"
                            f"GPU details:\n"
                            f"- Model: {gpu_info['model']}\n"
                            f"- Brand: {gpu_info['brand']}\n"
                            f"- Memory: {gpu_info['memory']} GB\n\n"
                            f"Keep it only 1 paragraph\n"
                        )

                        model = genai.GenerativeModel("gemini-2.0-flash")
                        response = model.generate_content(prompt)
                        return response.text
                    
                    recommendation = call_gemini_api({
                        "model": gpu['model'],
                        "brand": gpu['brand'],
                        "memory": gpu['memory'],
                        "clock_speed": gpu['clock_speed'],
                        "length": gpu['length'],
                        "price": gpu['price'],
                    })
                    st.success(recommendation)
            st.markdown("---")

        # Table of all matches
        st.subheader("📋 All Matching GPUs")
        st.dataframe(filtered_df.sort_values("SAW Score", ascending=False)[
            ["vendor", "brand", "model", "memory", "clock_speed", "length", "price", "SAW Score",]
        ])