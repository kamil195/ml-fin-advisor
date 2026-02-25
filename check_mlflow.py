"""Query latest MLflow run metrics."""
import mlflow
mlflow.set_tracking_uri("sqlite:///D:/Projects/ML Model for Fin-Tech/mlflow.db")
runs = mlflow.search_runs(experiment_names=["fin-advisor-v1"], order_by=["start_time DESC"], max_results=1)
for _, r in runs.iterrows():
    print(f"Run ID: {r['run_id']}")
    print(f"Start:  {r['start_time']}")
    print()
    for k, v in sorted(r.items()):
        if k.startswith("metrics.") and str(v) != "nan":
            name = k.replace("metrics.", "")
            print(f"  {name:40s} = {v}")
