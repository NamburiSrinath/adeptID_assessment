import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import faiss
import numpy as np
import ast

# Load data and model
train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Process input sentences from training data. Use only BODY and ONET_NAME
train_examples = []
n_examples = train_data.shape[0]
for i in range(n_examples):
  train_examples.append(InputExample(texts=[train_data['BODY'][i], train_data['ONET_NAME'][i]]))

# Load these examples in dataloader and use Multiple Negatives Ranking Loss as we don't exactly have labels
# And we only have +ve examples 
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=256)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train and save the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
model.save('fine_tuned_model')

# Read the Model and use this on test data!
model = SentenceTransformer('fine_tuned_model')

# Encode each example from test data. 
# The expectation is that, these embeddings should be similar to the ONET_EMBEDDINGS we
# trained on a contrastive loss objective

test_data_preds = model.encode(test_data['BODY'])
unique_onet_names_embeds = model.encode(train_data['ONET_NAME'].unique())

# Optional: saved ONET embeddings corresponding to the ONET_NAMES
final_df = pd.DataFrame()
final_df['ONET_NAME'] = train_data['ONET_NAME'].unique()
final_df['ONET_EMBED'] = list(unique_onet_names_embeds)
final_df.to_csv('onet_embeddings.csv', index=False)

# Use FAISS search to get top 10 ONET_NAMES
# Idea
# 1. index the ONET_NAMES
# 2. For every test data point, get the 10 closest embeddings and thus the 10 closest categories

d = unique_onet_names_embeds.shape[1]
index = faiss.IndexFlatL2(d)
index.add(unique_onet_names_embeds)

def faiss_search(query_embedding, top_k=10):
    _, top_indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
    top_categories = train_data['ONET_NAME'].unique()[top_indices[0]].tolist()
    return top_categories

test_embeddings = model.encode(test_data['BODY'].tolist())
test_predictions = []
for query_embedding in test_embeddings:
    top_categories = faiss_search(query_embedding, top_k=10)
    test_predictions.append(top_categories)

# Optional: Saving predicted data as a new column
df = pd.DataFrame()
df['ID'] = test_data['ID']
df['ONET_NAME'] = test_data['ONET_NAME']
df['PREDICTIONS'] = test_predictions
# df['PREDICTIONS'] = df['PREDICTIONS'].apply(ast.literal_eval)
df.to_csv('predictions.csv')

# Computing accuracy, if the actual ONET_NAME is there in predicted list, then it's appropriate
# Can do Acc@1, Acc@10 metrics (top-1, top-10), but currently this is a simpler version of accuracy computation! 
correct_predictions = df.apply(lambda row: row['ONET_NAME'] in row['PREDICTIONS'], axis=1)
accuracy = correct_predictions.mean()
print(f'Final Accuracy: {accuracy * 100:.2f}%')