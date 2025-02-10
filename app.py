import os
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset from a relative path.
try:
    df = pd.read_csv("Mall_Customers.csv")
except Exception as e:
    print("Error loading dataset. Ensure 'Mall_Customers.csv' is in the working directory.")
    raise e

# For clustering, we will use two key features: Annual Income and Spending Score.
features = ["Annual Income (k$)", "Spending Score (1-100)"]

# Define a function to filter the data based on dynamic filters (Gender and Age range)
def filter_data(df, gender, age_range):
    dff = df.copy()
    if gender != "All":
        dff = dff[dff["Genre"] == gender]
    dff = dff[(dff["Age"] >= age_range[0]) & (dff["Age"] <= age_range[1])]
    return dff

# Initialize Dash app
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server  # for deployment with gunicorn

# App layout with only the Clustering dashboard.
app.layout = dbc.Container([
    html.H1("Clustering Dashboard", style={'textAlign': 'center', 'marginTop': 20}),
    html.H5("Dynamic Filters"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Gender:"),
            dcc.Dropdown(
                id="gender-filter",
                options=[
                    {"label": "All", "value": "All"},
                    {"label": "Male", "value": "Male"},
                    {"label": "Female", "value": "Female"}
                ],
                value="All"
            )
        ], md=4),
        dbc.Col([
            html.Label("Select Age Range:"),
            dcc.RangeSlider(
                id="age-range",
                min=int(df["Age"].min()),
                max=int(df["Age"].max()),
                value=[int(df["Age"].min()), int(df["Age"].max())],
                marks={age: str(age) for age in range(int(df["Age"].min()),
                                                      int(df["Age"].max())+1, 5)}
            )
        ], md=8)
    ], style={'marginTop': 20}),
    html.H5("Clustering Options", style={'marginTop': 30}),
    dbc.Row([
        dbc.Col([
            html.Label("Select Clustering Algorithm:"),
            dcc.Dropdown(
                id="algo-selector",
                options=[
                    {"label": "K-Means", "value": "kmeans"},
                    {"label": "Hierarchical", "value": "hierarchical"},
                    {"label": "DBSCAN", "value": "dbscan"}
                ],
                value="kmeans"
            )
        ], md=4)
    ], style={'marginTop': 20}),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Select Number of Clusters:"),
                dcc.Slider(
                    id="num-clusters",
                    min=2,
                    max=10,
                    step=1,
                    value=4,
                    marks={i: str(i) for i in range(2, 11)}
                )
            ], id="num-clusters-container")
        ], md=4),
        dbc.Col([
            html.Div([
                html.Label("Epsilon (eps):"),
                dcc.Input(id="dbscan-eps", type="number", value=5, step=0.5)
            ], id="dbscan-eps-container", style={"display": "none"}),
            html.Div([
                html.Label("Min Samples:"),
                dcc.Input(id="dbscan-min-samples", type="number", value=5, step=1)
            ], id="dbscan-min-samples-container", style={"display": "none"})
        ], md=8)
    ], style={'marginTop': 20}),
    html.Hr(),
    html.Div(id="clustering-output")
], fluid=True)

# Callback to toggle visibility of clustering parameter containers based on algorithm selection
@app.callback(
    [Output("num-clusters-container", "style"),
     Output("dbscan-eps-container", "style"),
     Output("dbscan-min-samples-container", "style")],
    Input("algo-selector", "value")
)
def toggle_params(algo):
    if algo in ["kmeans", "hierarchical"]:
        return {"display": "block"}, {"display": "none"}, {"display": "none"}
    elif algo == "dbscan":
        return {"display": "none"}, {"display": "block"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}

# Callback for updating clustering output based on dynamic filters and algorithm parameters.
@app.callback(
    Output("clustering-output", "children"),
    [Input("algo-selector", "value"),
     Input("num-clusters", "value"),
     Input("dbscan-eps", "value"),
     Input("dbscan-min-samples", "value"),
     Input("gender-filter", "value"),
     Input("age-range", "value")]
)
def update_clustering(algo, num_clusters, dbscan_eps, dbscan_min_samples, gender, age_range):
    # Filter the data based on selected gender and age range.
    dff = filter_data(df, gender, age_range)
    if dff.empty:
        return html.Div("No data available for the selected filters.", style={'color': 'red'})

    X = dff[features].values
    labels = None
    metric_text = ""
    cluster_fig = None

    if algo == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        metric_text = f"K-Means Inertia: {inertia:.2f}"
    elif algo == "hierarchical":
        hc = AgglomerativeClustering(n_clusters=num_clusters)
        labels = hc.fit_predict(X)
        metric_text = f"Hierarchical Clustering: {num_clusters} clusters selected."
    elif algo == "dbscan":
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        labels = db.fit_predict(X)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        metric_text = f"DBSCAN identified {n_clusters_found} clusters (noise labeled as -1)."
    else:
        return html.Div("Invalid algorithm selected.", style={'color': 'red'})

    dff["Cluster"] = labels.astype(str)

    # Calculate silhouette score if applicable (only when more than 1 cluster exists)
    unique_labels = set(labels)
    if algo != "dbscan" or (len(unique_labels - {-1}) > 1):
        try:
            sil_score = silhouette_score(X, labels)
            metric_text += f" | Silhouette Score: {sil_score:.2f}"
        except Exception:
            pass

    # Create scatter plot for clustering results.
    cluster_fig = px.scatter(
        dff, x=features[0], y=features[1], color="Cluster",
        title=f"Clustering Result using {algo.capitalize()}",
        color_discrete_sequence=px.colors.qualitative.Set1,
        hover_data=["Age", "Genre"]
    )
    cluster_fig.update_layout(template="plotly_white")

    # For K-Means, add an Elbow Method plot.
    elbow_fig = None
    if algo == "kmeans":
        inertias = []
        cluster_range = range(1, 11)
        for k in cluster_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            inertias.append(km.inertia_)
        elbow_fig = go.Figure(data=go.Scatter(
            x=list(cluster_range), y=inertias, mode="lines+markers"
        ))
        elbow_fig.update_layout(
            title="Elbow Method for Optimal Number of Clusters",
            xaxis_title="Number of Clusters",
            yaxis_title="Inertia",
            template="plotly_white"
        )

    output_components = [
        html.H5("Clustering Evaluation Metric", style={'marginTop': 20}),
        html.P(metric_text, style={'fontSize': '16px', 'textAlign': 'center'}),
        dcc.Graph(figure=cluster_fig)
    ]
    if elbow_fig:
        output_components.append(html.H5("Elbow Method", style={'marginTop': 30}))
        output_components.append(dcc.Graph(figure=elbow_fig))

    return html.Div(output_components)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
