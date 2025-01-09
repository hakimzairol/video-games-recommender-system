import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from scipy.sparse.linalg import svds
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the split datasets
part1 = pd.read_csv("C:\\Users\\user\\Documents\\Sem 5\\Data Science Project\\cleaned_dataset_part1.csv")
part2 = pd.read_csv("C:\\Users\\user\\Documents\\Sem 5\\Data Science Project\\cleaned_dataset_part2.csv")
df_cleaned = pd.concat([part1, part2], ignore_index=True)

image_data = pd.read_csv('C:\\Users\\user\\Documents\\Sem 5\\Data Science Project\\Image.csv', encoding='latin1')
user_data = pd.read_csv('C:\\Users\\user\\Documents\\Sem 5\\Data Science Project\\user_data.csv')


user_features = user_data.groupby('UserID').agg(
    Total_Hours=('Hours', 'sum'),
    Avg_Hours=('Hours', 'mean'),
    Num_Games=('Game', 'nunique'),
    Avg_Metacritic=('Metacritic score', 'mean'),
    Avg_Achievements=('Achievements', 'mean'),
    Avg_Recommendations=('Recommendations', 'mean')
).reset_index()

# Normalize features for clustering
scaler = StandardScaler()
normalized_features = scaler.fit_transform(user_features.iloc[:, 1:])

# Apply KMeans clustering
optimal_k = 5  # Number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
user_features['Cluster'] = kmeans.fit_predict(normalized_features)

def apply_svd_in_cluster(cluster_id, user_data, user_features):
    # Filter users in the cluster
    cluster_users = user_features[user_features['Cluster'] == cluster_id]['UserID']
    cluster_data = user_data[user_data['UserID'].isin(cluster_users)]

    if cluster_data.empty:
        return pd.DataFrame()  # Return empty DataFrame if no users in the cluster

    # Create weighted ratings
    cluster_data['Weighted_Rating'] = (
        cluster_data['Hours'] * 0.5 +
        cluster_data['Metacritic score'] * 0.2 +
        cluster_data['Achievements'] * 0.2 +
        cluster_data['Recommendations'] * 0.1
    )

    # Normalize ratings
    min_rating = cluster_data['Weighted_Rating'].min()
    max_rating = cluster_data['Weighted_Rating'].max()
    cluster_data['Rating'] = 1 + 4 * (cluster_data['Weighted_Rating'] - min_rating) / (max_rating - min_rating)

    # Create user-item matrix for the cluster
    user_item_matrix = cluster_data.pivot_table(index='UserID', columns='Game', values='Rating', fill_value=0)

    if user_item_matrix.empty:
        return pd.DataFrame()  # Return empty DataFrame if no items for the cluster

    np.random.seed(42)  # Set a fixed random seed
    U, sigma, Vt = svds(user_item_matrix.values, k=3)
    sigma = np.diag(sigma)

    # Reconstruct the Predicted Ratings Matrix
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    predicted_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

    return predicted_df


cluster_predictions = {}
for cluster_id in range(optimal_k):
    cluster_predictions[cluster_id] = apply_svd_in_cluster(cluster_id, user_data, user_features)

# Combine predictions from all clusters
all_predictions = pd.concat([pred for pred in cluster_predictions.values() if not pred.empty], axis=0)

# Ensure user IDs are integers for consistent matching
all_predictions.index = all_predictions.index.astype(int)



def recommend_games_cf(user_id, original_data, predictions_df, num_recommendations=15):
    if user_id not in predictions_df.index:
        return None  # Return None if user_id is not in predictions_df

    user_played = original_data[original_data['UserID'] == user_id]['Game'].tolist()
    recommendations = predictions_df.loc[user_id].drop(index=user_played, errors='ignore').sort_values(ascending=False)
    top_recommendations = recommendations.head(num_recommendations)

    recommended_games = []
    for game in top_recommendations.index:
        game_details = df_cleaned[df_cleaned['Name'] == game]
        if not game_details.empty:
            recommended_games.append({
                "Name": game,
                "Price": game_details['Price'].values[0],
                "Genres": game_details['Genres'].values[0],
                "Categories": game_details['Categories'].values[0],
                "Image Source": image_data.loc[image_data['Name'] == game, 'Image Source'].values
            })
    return recommended_games





# Combine features for vectorization
df_cleaned['combined_features'] = (
    df_cleaned['Genres'] + " " +
    df_cleaned['Tags'] + " " +
    df_cleaned['Categories'] + " " +
    df_cleaned['About the game']
)

# Vectorize combined features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned['combined_features'])

# Function to recommend based on game name
def recommend_by_game_name(game_name, top_n=15):
    if game_name not in df_cleaned['Name'].values:
        return f"Error: Game '{game_name}' not found in the dataset."

    idx = df_cleaned[df_cleaned['Name'] == game_name].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n + 1]]
    return df_cleaned.iloc[top_indices][['Name', 'Price', 'Genres', 'Categories']]

# Function to recommend based on preferences
def recommend_by_preferences(genres, tags, categories, top_n=15):
    filtered_df = df_cleaned[
        (df_cleaned['Genres'].str.contains('|'.join(genres), case=False, na=False)) &
        (df_cleaned['Tags'].str.contains('|'.join(tags), case=False, na=False)) &
        (df_cleaned['Categories'].str.contains('|'.join(categories), case=False, na=False))
    ]

    if filtered_df.empty:
        return "Error: No games match the selected preferences."

    tfidf_matrix_filtered = tfidf_vectorizer.transform(filtered_df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix_filtered, tfidf_matrix_filtered)
    sim_scores = np.mean(cosine_sim, axis=0)
    top_indices = np.argsort(sim_scores)[::-1][:top_n]

    return filtered_df.iloc[top_indices][['Name', 'Price', 'Genres', 'Categories']]

# Generate unique options for genres, tags, and categories
def preprocess_options(df_cleaned, column_name):
    options = df_cleaned[column_name].dropna().str.split(',')
    unique_options = set([item.strip() for sublist in options for item in sublist])
    return sorted(unique_options)

unique_genres = preprocess_options(df_cleaned, 'Genres')
unique_tags = preprocess_options(df_cleaned, 'Tags')
unique_categories = preprocess_options(df_cleaned, 'Categories')



# Streamlit App
st.title("🎮 Game Recommendation System")

# Main sidebar
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Content-Based Filtering (CBF)"):
    st.session_state.page = "CBF"
if st.sidebar.button("Collaborative Filtering (CF)"):
    st.session_state.page = "CF"

# Dashboard sidebar
st.sidebar.title("📊 Data Dashboard")
if st.sidebar.button("Show Dashboard"):
    st.session_state.page = "Dashboard"

if st.session_state.page == "Home":
    st.header("Discover Games")
    st.markdown("### Featured Games")
    random_games = df_cleaned.sample(12)

    for i in range(0, len(random_games), 4):
        cols = st.columns(4)
        for col, (_, row) in zip(cols, random_games.iloc[i:i+4].iterrows()):
            with col:
                game_image = image_data.loc[image_data['Name'] == row['Name'], 'Image Source'].values
                if len(game_image) > 0:
                    st.image(game_image[0], use_container_width=True, caption=row['Name'])
                else:
                    st.text("No Image Available")
                st.markdown(f"**Price:** {row['Price']}")
                st.markdown(f"**Genres:** {row['Genres']}")

elif st.session_state.page == "CBF":
    st.header("CBF: Enter Game Name or Select Preferences")

    # Unified input section
    col1, col2 = st.columns(2)

    with col1:
        game_name = st.text_input("Enter a Game Name:")

    with col2:
        selected_genres = st.multiselect("Select Genres:", unique_genres)
        selected_tags = st.multiselect("Select Tags:", unique_tags)
        selected_categories = st.multiselect("Select Categories:", unique_categories)

    if game_name:
        recommendations = recommend_by_game_name(game_name,top_n=15)
        
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            
            for _, row in recommendations.iterrows():
                game_image = image_data.loc[image_data['Name'] == row['Name'], 'Image Source'].values
                col1, col2 = st.columns([1, 4])

                with col1:
                    if len(game_image) > 0:
                        st.image(game_image[0], use_container_width=True)
                    else:
                        st.text("No Image Available")

                with col2:
                    st.markdown(f"### {row['Name']}")
                    st.markdown(f"**Price:** {row['Price']}")
                    st.markdown(f"**Genres:** {row['Genres']}")
                    st.markdown(f"**Categories:** {row['Categories']}")

    elif selected_genres or selected_tags or selected_categories:
        recommendations = recommend_by_preferences(selected_genres, selected_tags, selected_categories)
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            for _, row in recommendations.iterrows():
                game_image = image_data.loc[image_data['Name'] == row['Name'], 'Image Source'].values
                col1, col2 = st.columns([1, 4])

                with col1:
                    if len(game_image) > 0:
                        st.image(game_image[0], use_container_width=True)
                    else:
                        st.text("No Image Available")

                with col2:
                    st.markdown(f"### {row['Name']}")
                    st.markdown(f"**Price:** {row['Price']}")
                    st.markdown(f"**Genres:** {row['Genres']}")
                    st.markdown(f"**Categories:** {row['Categories']}")
    else:
        st.info("Enter a game name or select preferences to get recommendations.")

elif st.session_state.page == "CF":
    st.header("CF: Enter User ID")
    user_id = st.text_input("Enter User ID:")

    if user_id:
        try:
            user_id = int(user_id)

            if user_id not in all_predictions.index:
                st.error("User ID not found in predictions.")
            else:
                recommendations = recommend_games_cf(user_id, user_data, all_predictions, num_recommendations=15    )
                if not recommendations:
                    st.info("No recommendations found for this user.")
                else:
                    for game in recommendations:
                        col1, col2 = st.columns([1, 4])

                        with col1:
                            if "Image Source" in game and isinstance(game["Image Source"], np.ndarray) and len(game["Image Source"]) > 0:
                                st.image(game["Image Source"][0], use_container_width=True)
                            else:
                                st.text("No Image Available")

                        with col2:
                            st.markdown(f"### {game['Name']}")
                            st.markdown(f"**Price:** {game['Price']} USD")
                            st.markdown(f"**Genres:** {game['Genres']}")
                            st.markdown(f"**Categories:** {game['Categories']}")
        except ValueError:
            st.error("Please enter a valid User ID.")






elif st.session_state.page == "Dashboard":
    st.header("📊 Data Dashboard")

    # Visualization: Release Date Distribution
    df_cleaned['Release date'] = pd.to_datetime(df_cleaned['Release date'], errors='coerce')  # Let Pandas infer format
    st.markdown("### Game Release Date Distribution")
    
    if df_cleaned['Release date'].notna().any():
        fig, ax = plt.subplots()
        df_cleaned['Release date'].dt.year.dropna().hist(bins=20, ax=ax)  # Adjust bin count based on data
        ax.set_xlabel('Release Year')
        ax.set_ylabel('Number of Games')
        ax.set_title('Release Date Distribution')
        st.pyplot(fig)
    else:
        st.warning("No valid release dates found for visualization.")

    # Visualization: Genre Distribution (Pie Chart)
    st.markdown("### Top 10 Game Genres")
    all_genres = []
    genres = df_cleaned["Genres"].dropna().values
    for g in genres:
        all_genres.extend(g.split(","))

    genres_df = pd.DataFrame(all_genres, columns=["genres"])
    genres_count = genres_df.groupby("genres")["genres"].count().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(genres_count.head(10), labels=genres_count.head(10).index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Top 10 Genres")
    st.pyplot(fig)

    # Visualization: Category Distribution (Pie & Bar Charts)
    st.markdown("### Top 10 Game Categories")
    all_categories = []
    categories = df_cleaned["Categories"].dropna().values
    for c in categories:
        all_categories.extend(c.split(","))

    categories_df = pd.DataFrame(all_categories, columns=["Categories"])
    categories_count = categories_df.groupby("Categories")["Categories"].count().sort_values(ascending=False)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    axs[0].pie(categories_count.head(10), labels=categories_count.head(10).index, autopct='%1.1f%%', startangle=90)
    axs[0].set_title("Top 10 Categories (Pie Chart)")

    categories_count.head(10).plot.bar(ax=axs[1], color="skyblue")
    axs[1].set_title("Top 10 Categories (Bar Chart)")
    axs[1].set_xlabel("Categories")
    axs[1].set_ylabel("Count")
    plt.xticks(rotation=45)

    st.pyplot(fig)
