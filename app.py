import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

try:
    try:
    from .dataset import generate_synthetic_transactions, extract_rfm_features, prepare_clustering_matrix
try:
    try:
    from .clustering import KMeansFromScratch, HierarchicalAgglomerativeFromScratch, GaussianMixtureFromScratch
try:
    try:
    from .validation import calculate_silhouette_score, run_elbow_and_silhouette_validation, SegmentProfiler
try:
    try:
    from .personalization import PersonalizationEngine
try:
    try:
    from .visualization import compute_pca_2d_projection

app = FastAPI(title="Customer Segmentation + Personalization Engine", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global Dataset State
transactions_df = generate_synthetic_transactions(n_customers=500, seed=42)
rfm_df = extract_rfm_features(transactions_df)
X_scaled, feature_cols, scaler_params = prepare_clustering_matrix(rfm_df)
personalization_engine = PersonalizationEngine()

class SegmentRequest(BaseModel):
    algorithm: str = "kmeans"  # 'kmeans', 'hierarchical', 'gmm'
    n_clusters: int = 4
    linkage: str = "average"   # For hierarchical

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/dataset_info")
async def api_dataset_info():
    return {
        "total_transactions": len(transactions_df),
        "total_customers": len(rfm_df),
        "date_range": f"{transactions_df['invoice_date'].min().strftime('%Y-%m-%d')} to {transactions_df['invoice_date'].max().strftime('%Y-%m-%d')}",
        "features": feature_cols,
        "rfm_summary": {
            "mean_recency": round(float(rfm_df['recency'].mean()), 1),
            "mean_frequency": round(float(rfm_df['frequency'].mean()), 1),
            "mean_monetary": round(float(rfm_df['monetary'].mean()), 2)
        }
    }

@app.get("/api/validation")
async def api_validation():
    validation_results = run_elbow_and_silhouette_validation(X_scaled, k_range=(2, 8))
    return validation_results

@app.post("/api/segment")
async def api_segment(req: SegmentRequest):
    algo = req.algorithm.lower()
    k = req.n_clusters

    if algo == "kmeans":
        model = KMeansFromScratch(n_clusters=k, seed=42)
        labels, centroids, inertia = model.fit_predict(X_scaled)
    elif algo == "hierarchical":
        model = HierarchicalAgglomerativeFromScratch(n_clusters=k, linkage=req.linkage)
        labels = model.fit_predict(X_scaled)
    elif algo == "gmm":
        model = GaussianMixtureFromScratch(n_clusters=k, seed=42)
        labels, probs, means = model.fit_predict(X_scaled)
    else:
        return JSONResponse(status_code=400, content={"error": f"Unknown algorithm: {algo}"})

    # Calculate Silhouette Score
    sil_score = round(calculate_silhouette_score(X_scaled, labels), 4)

    # Segment Business Profiling
    profiles = SegmentProfiler.profile_segments(rfm_df, labels)

    # Marketing Recommendations
    recommendations = personalization_engine.generate_segment_recommendations(profiles)

    # Revenue Impact Simulation
    revenue_simulation = personalization_engine.simulate_revenue_impact(rfm_df, labels, profiles)

    # PCA 2D Projection
    customer_ids = rfm_df["customer_id"].tolist()
    pca_data = compute_pca_2d_projection(X_scaled, labels, customer_ids)

    return {
        "algorithm": algo,
        "n_clusters": k,
        "silhouette_score": sil_score,
        "profiles": recommendations,
        "revenue_simulation": revenue_simulation,
        "pca_projection": pca_data
    }
