import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model_path = "rf_recommendation_model.pkl"
drive_file_id = "1OMkO3-85lhz50hGh9Hu_iJVERxVc_23E" # replace with your real ID
download_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    gdown.download(download_url, model_path, quiet=False)
else:
    print("✅ Model already exists locally.")

# === Helper to safely load files ===
def safe_load(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ File not found: {filename}. Please place it in the same folder as app.py.")
    return path

# === Load Models and Data ===
try:
    model = joblib.load(safe_load("rf_recommendation_model.pkl"))
    user_enc = joblib.load(safe_load("user_encoder.pkl"))
    event_enc = joblib.load(safe_load("event_encoder.pkl"))
    type_enc = joblib.load(safe_load("type_encoder.pkl"))
    subtype_enc = joblib.load(safe_load("subtype_encoder.pkl"))
    df = pd.read_csv(safe_load("event_management_with_clicks_v2_consistent_ids.csv"))
except Exception as e:
    raise RuntimeError(f"❌ Error loading model/data files: {e}")

# === Create Event Similarity Matrix ===
try:
    user_event_matrix = df.pivot_table(
        index='user_id',
        columns='event_id',
        values='event_average_rating',
        fill_value=0
    )

    event_similarity = cosine_similarity(user_event_matrix.T)
    event_similarity_df = pd.DataFrame(
        event_similarity,
        index=user_event_matrix.columns,
        columns=user_event_matrix.columns
    )
except Exception as e:
    raise RuntimeError(f"❌ Error building event similarity matrix: {e}")

# === Recommendation Functions ===

def recommend_for_existing_user(user_id, n=5):
    if user_id not in df['user_id'].unique():
        return []

    user_id_enc = user_enc.transform([user_id])[0]
    known_events = df[df['user_id'] == user_id]['event_id'].unique()
    unseen_events = [e for e in df['event_id'].unique() if e not in known_events]

    X_pred = []
    for event_id in unseen_events:
        event_row = df[df['event_id'] == event_id].iloc[0]
        event_id_enc = event_enc.transform([event_id])[0]
        event_type_enc = type_enc.transform([event_row['event_type']])[0]
        event_sub_event_enc = subtype_enc.transform([event_row['event_sub_event']])[0]
        adj_attendance = event_row['adjusted_attendance_rate']
        clicks = event_row['clicks']

        X_pred.append([
            user_id_enc, event_id_enc, event_type_enc,
            event_sub_event_enc, adj_attendance, clicks
        ])

    X_pred = pd.DataFrame(X_pred, columns=[
        'user_encoded', 'event_encoded', 'event_type_encoded',
        'event_sub_event_encoded', 'adjusted_attendance_rate', 'clicks'
    ])

    preds = model.predict(X_pred)
    top_n_idx = np.argsort(preds)[::-1][:n]
    top_n_events = np.array(unseen_events)[top_n_idx]
    top_n_scores = preds[top_n_idx]

    event_info = df.drop_duplicates('event_id').set_index('event_id')[['event_type', 'event_sub_event']].to_dict('index')

    results = []
    for eid, score in zip(top_n_events, top_n_scores):
        info = event_info.get(eid, {'event_type': 'Unknown', 'event_sub_event': 'Unknown'})
        results.append({
            'event_id': eid,
            'predicted_rating': round(float(score), 3),
            'event_type': info['event_type'],
            'event_sub_event': info['event_sub_event']
        })

    return results


def recommend_similar_event(event_id, n=5):
    if event_id not in event_similarity_df.index:
        return []

    similar_events = event_similarity_df[event_id].sort_values(ascending=False)[1:n+1]
    event_info = df.drop_duplicates('event_id').set_index('event_id')[['event_type', 'event_sub_event']].to_dict('index')

    results = []
    for eid, sim in similar_events.items():
        info = event_info.get(eid, {'event_type': 'Unknown', 'event_sub_event': 'Unknown'})
        results.append({
            'event_id': eid,
            'similarity_score': round(float(sim), 3),
            'event_type': info['event_type'],
            'event_sub_event': info['event_sub_event']
        })

    return results


def weighted_hybrid_recommend(user_id=None, event_id=None, n=5, alpha=0.7):
    if user_id and user_id in user_event_matrix.index:
        collab_recs = recommend_for_existing_user(user_id, n=n)
        collab_df = pd.DataFrame(collab_recs)

        top_event = df[df['user_id'] == user_id].sort_values(
            'event_average_rating', ascending=False
        )['event_id'].iloc[0]

        content_recs = recommend_similar_event(top_event, n=n)
        content_df = pd.DataFrame(content_recs)

        merged = pd.merge(
            collab_df, content_df,
            on='event_id', how='outer',
            suffixes=('_collab', '_content')
        ).fillna(0)

        merged['final_score'] = alpha * merged['predicted_rating'] + (1 - alpha) * merged['similarity_score']
        merged = merged.sort_values('final_score', ascending=False).head(n)

        results = merged[['event_id', 'final_score', 'event_type_collab', 'event_sub_event_collab']].rename(
            columns={
                'event_type_collab': 'event_type',
                'event_sub_event_collab': 'event_sub_event'
            }
        ).to_dict('records')

        return results

    elif event_id:
        return recommend_similar_event(event_id, n)

    else:
        return []

# === API Endpoints ===

@app.route("/")
def home():
    return jsonify({"message": "✅ Event Recommendation API is running!"})

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id")
    event_id = request.args.get("event_id")
    recs = weighted_hybrid_recommend(user_id=user_id, event_id=event_id, n=5, alpha=0.8)
    return jsonify(recs)

if __name__ == "__main__":
    app.run(debug=True)
