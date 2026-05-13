import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from clearml import PipelineController

# Ваши S3 пути (с правильным бакетом)
DATASET_S3_PATH = "s3://r-mlops-bucket-8-1-11-22209764/raw_data/dataset.csv"
DATA_S3_PATH = "s3://r-mlops-bucket-8-1-11-22209764/raw_data/data.csv"

def prepare_data(dataset_path, data_path):
    storage_options = {
        "client_kwargs": {"endpoint_url": "http://storage.yandexcloud.net"}
    }
    df = pd.read_csv(dataset_path, storage_options=storage_options).drop(
        columns="Unnamed: 0"
    )
    df_year = pd.read_csv(data_path, storage_options=storage_options)[["id", "year"]]
    df_year["track_id"] = df_year["id"]
    df_year.drop(columns="id", inplace=True)
    df = pd.merge(df, df_year, on="track_id")
    return df

def train_model(df):
    xtab_song = pd.crosstab(df["track_id"], df["track_genre"]) * 2
    xtab_song.reset_index(inplace=True)
    df_distinct = (
        df.drop_duplicates("track_id").sort_values("track_id").reset_index(drop=True)
    )
    data_encoded = pd.concat(
        [df_distinct, xtab_song.drop(columns=["track_id"])], axis=1
    )
    numerical_features = [
        "explicit", "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "year",
    ]
    scaler = MinMaxScaler()
    data_encoded[numerical_features] = scaler.fit_transform(
        data_encoded[numerical_features]
    )
    similarity_matrix = cosine_similarity(
        data_encoded[numerical_features + list(xtab_song.columns[1:])]
    )
    return data_encoded, similarity_matrix

def recommend_song(track_title: str, data_encoded, similarity_matrix, N=5):
    indices = pd.Series(
        data_encoded.index, index=data_encoded["track_name"]
    ).drop_duplicates()
    if track_title not in indices:
        return []
    idx = indices[track_title]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : N + 1]
    song_indices = [i[0] for i in sim_scores]
    recommended = data_encoded[["track_name", "artists", "album_name"]].iloc[
        song_indices
    ]
    return recommended.to_dict(orient="records")

# Создание пайплайна
pipe = PipelineController(
    name="Train recommend model",
    project="mlops",
    version="0.0.1",
    docker="python:3.12-slim",
)

pipe.add_parameter(
    name="dataset_path",
    description="path to dataset.csv",
    default=DATASET_S3_PATH,
)

pipe.add_parameter(
    name="data",
    description="path to data.csv",
    default=DATA_S3_PATH,
)

pipe.set_default_execution_queue("default")

pipe.add_function_step(
    name="prepare_data",
    function=prepare_data,
    function_kwargs=dict(
        dataset_path="${pipeline.dataset_path}", data_path="${pipeline.data}"
    ),
    function_return=["df"],
    docker="python:3.12-slim",
    packages=[
        "clearml[s3]==2.0.2",
        "pandas>=2.3.2",
        "scikit-learn>=1.7.1",
        "s3fs==2025.9.0",
        "fsspec==2025.9.0",
    ],
)

pipe.add_function_step(
    name="train_model",
    function=train_model,
    function_kwargs=dict(df="${prepare_data.df}"),
    function_return=["data_encoded", "similarity_matrix"],
    docker="python:3.12-slim",
)

pipe.add_function_step(
    name="recommend",
    function=recommend_song,
    function_kwargs=dict(
        track_title="Hold On",
        data_encoded="${train_model.data_encoded}",
        similarity_matrix="${train_model.similarity_matrix}",
    ),
    function_return=["recommended"],
    docker="python:3.12-slim",
)

# Запуск на агенте в очереди default
pipe.start(queue="default")