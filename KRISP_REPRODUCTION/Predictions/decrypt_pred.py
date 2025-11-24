import json

# Replace with the actual path from your mm_predict logs
results_path = "/data1/souradeepd/Krisp/mmf/Predictions/save/okvqa_krisp_29402777/reports/okvqa_run_test_2025-11-23T09:52:21.json"

with open(results_path, 'r') as f:
    predictions = json.load(f)

# Print the first 5 predictions
for p in predictions[:5]:
    print(f"Question ID: {p['question_id']} -> Predicted Answer: {p['answer']}")