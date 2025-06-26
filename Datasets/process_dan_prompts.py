import pandas as pd


# 1. Load your data
df = pd.read_csv("jailbreak_prompts_2023_12_25.csv")

# 2. Define the top-11 community names exactly as they appear in your dataset
top_11_names = [
    "Advanced",
    "Toxic",
    "Basic",
    "Start Prompt",
    "Exception",
    "Anarchy",
    "Narrative",
    "Opposite",
    "Guidelines",
    "Fictional",
    "Virtualization"
]

# 3. Filter to keep only those 11
df = df[df["community"].isin(top_11_names)]



df['date'] = pd.to_datetime(df['date'], errors='coerce')
grouped = df.groupby('community_id')



def get_earliest_prompt(group_df):
    sorted_df = group_df.sort_values('date')
    return sorted_df.iloc[0]['prompt']

def get_latest_prompt(group_df):
    sorted_df = group_df.sort_values('date')
    return sorted_df.iloc[-1]['prompt']


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def compute_embeddings(prompts):
    return model.encode(prompts, convert_to_numpy=True)

def find_most_central_prompt(prompts):
    embeddings = compute_embeddings(prompts)
    sim_matrix = cosine_similarity(embeddings)
    avg_sims = sim_matrix.mean(axis=1)
    idx_central = np.argmax(avg_sims)
    return prompts[idx_central]


import random

def get_random_prompts(group_df, k=2, seed=42):
    # If a community has fewer than k prompts, take them all.
    if len(group_df) <= k:
        return group_df['prompt'].tolist()
    else:
        return group_df['prompt'].sample(n=k, random_state=seed).tolist()



def remove_near_duplicates(prompts, threshold=0.95):
    emb = compute_embeddings(prompts)
    sim_matrix = cosine_similarity(emb)
    selected_indices = []
    used = set()

    for i in range(len(prompts)):
        if i in used:
            continue
        selected_indices.append(i)
        # Mark all near-duplicates
        duplicates = np.where(sim_matrix[i] >= threshold)[0]
        for d in duplicates:
            used.add(d)

    return [prompts[i] for i in selected_indices]



final_selection = []

for community_id, group_df in grouped:
    if group_df.empty:
        continue

    # 1) Earliest & Latest
    earliest = get_earliest_prompt(group_df)
    latest = get_latest_prompt(group_df)

    # 2) Most central
    all_prompts = group_df['prompt'].tolist()
    central = find_most_central_prompt(all_prompts)

    # 3) Random prompt(s)
    randoms = get_random_prompts(group_df, k=2, seed=42)

    # Combine
    candidates = list({earliest, latest, central} | set(randoms))

    # 4) Deduplicate
    deduped_candidates = remove_near_duplicates(candidates, threshold=0.95)

    # Optional: If you want a cap, e.g., max 5 or 6 prompts from each community:
    # deduped_candidates = deduped_candidates[:6]

    # Store
    for p in deduped_candidates:
        final_selection.append({
            'community_id': community_id,
            'prompt': p
            
        })

# Convert final selection to DataFrame for review
selected_df = pd.DataFrame(final_selection)
print(selected_df)
