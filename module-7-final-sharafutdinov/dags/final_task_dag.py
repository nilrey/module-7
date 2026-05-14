# Постройте DAG на airflow для запуска ML pipeline 
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import io
import tempfile
import os

# Определение схемы 
def strict_datetime_column(series):
    return series.str.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC').all()

taxi_schema = DataFrameSchema(
    columns={
        "pickup_datetime": Column(
            str,
            required=True,
            checks=Check(strict_datetime_column, 
                         error="pickup_datetime: не все значения соответствуют формату 'YYYY-MM-DD HH:MM:SS UTC'")
        ),
        "fare_amount": Column(
            float,
            required=True,
            checks=[Check.ge(0), Check.le(100)]
        ),
        "pickup_longitude": Column(
            float,
            required=True,
            checks=Check.in_range(-180, 180)
        ),
        "pickup_latitude": Column(
            float,
            required=True,
            checks=Check.in_range(-180, 180)
        ),
        "dropoff_longitude": Column(
            float,
            required=True,
            checks=[Check.ge(-180), Check.le(180)]
        ),
        "dropoff_latitude": Column(
            float,
            required=True,
            checks=[Check.ge(-180), Check.le(180)]
        ),
        "passenger_count": Column(
            int,
            required=True,
            checks=[Check.ge(1), Check.lt(10)]
        ),
    },
    strict=True,
    ordered=False
)

# Конфигурация
S3_BUCKET = "r-mlops-bucket-8-1-11-22209764"
S3_KEY_INPUT = "uber.csv"
S3_KEY_OUTPUT = "uber_processed.csv"

# def load_and_clean_csv(**context):
#     """Загрузка CSV из S3, удаление пустого первого столбца (если есть)"""
#     s3_hook = S3Hook(aws_conn_id="yandex_s3_connection")
    
#     # Загрузка файла
#     content = s3_hook.read_key(key=S3_KEY_INPUT, bucket_name=S3_BUCKET)
#     df = pd.read_csv(io.StringIO(content))
    
#     # Удаление безымянного столбца-индекса (если есть)
#     if 'Unnamed: 0' in df.columns:
#         df = df.drop(columns=['Unnamed: 0'])
    
#     # Сохраняем в /tmp для следующего шага
#     tmp_path = '/tmp/uber_cleaned.csv'
#     df.to_csv(tmp_path, index=False)
    
#     context['task_instance'].xcom_push(key='cleaned_data_path', value=tmp_path)
#     return f"Загружено {len(df)} записей"

# def validate_data(**context):
#     """Валидация DataFrame через Pandera"""
#     tmp_path = context['task_instance'].xcom_pull(key='cleaned_data_path', task_ids='load_and_clean_csv')
#     df = pd.read_csv(tmp_path)
    
#     # Валидация
#     try:
#         taxi_schema.validate(df, lazy=True)
#         print(f"Валидация пройдена: {len(df)} записей")
#     except pa.errors.SchemaErrors as e:
#         print(f"Ошибки валидации: {e}")
#         raise
    
#     context['task_instance'].xcom_push(key='validated_data_path', value=tmp_path)

def load_and_clean_csv(**context):
    """Загрузка CSV из S3 с очисткой данных"""
    s3_hook = S3Hook(aws_conn_id="yandex_s3_connection")
    
    # Загрузка файла
    content = s3_hook.read_key(key=S3_KEY_INPUT, bucket_name=S3_BUCKET)
    df = pd.read_csv(io.StringIO(content))
    
    # 1. Удаляем лишнюю колонку key
    if 'key' in df.columns:
        df = df.drop(columns=['key'])
        print("Удалена колонка 'key'")
    
    # 2. Удаляем безымянный столбец-индекса
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("Удалена колонка 'Unnamed: 0'")
    
    # 3. Очистка fare_amount (должна быть от 0 до 100)
    df = df[(df['fare_amount'] >= 0) & (df['fare_amount'] <= 100)]
    print(f"После фильтрации fare_amount: {len(df)} записей")
    
    # 4. Очистка координат (должны быть в диапазоне -180..180)
    for col in ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']:
        df = df[(df[col] >= -180) & (df[col] <= 180)]
    print(f"После фильтрации координат: {len(df)} записей")
    
    # 5. Очистка passenger_count (должен быть от 1 до 9)
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 9)]
    print(f"После фильтрации passenger_count: {len(df)} записей")
    
    # 6. Удаляем строки с null значениями в важных колонках
    required_cols = ['pickup_datetime', 'fare_amount', 'pickup_longitude', 
                     'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 
                     'passenger_count']
    df = df.dropna(subset=required_cols)
    print(f"После удаления null: {len(df)} записей")
    
    # Сохраняем очищенные данные
    tmp_path = '/tmp/uber_cleaned.csv'
    df.to_csv(tmp_path, index=False)
    
    context['task_instance'].xcom_push(key='cleaned_data_path', value=tmp_path)
    return f"Загружено и очищено {len(df)} записей"

def validate_data(**context):
    """Валидация DataFrame через Pandera"""
    tmp_path = context['task_instance'].xcom_pull(key='cleaned_data_path', task_ids='load_and_clean_csv')
    df = pd.read_csv(tmp_path)
    
    print(f"Валидация {len(df)} записей")
    print(f"Диапазон fare_amount: {df['fare_amount'].min()} - {df['fare_amount'].max()}")
    print(f"Диапазон passenger_count: {df['passenger_count'].min()} - {df['passenger_count'].max()}")
    
    # Валидация
    try:
        taxi_schema.validate(df, lazy=True)
        print(f"✓ Валидация пройдена: {len(df)} записей")
    except pa.errors.SchemaErrors as e:
        print(f"✗ Ошибки валидации: {e}")
        print(f"Количество некорректных записей: {len(e.failure_cases)}")
        raise
    
    context['task_instance'].xcom_push(key='validated_data_path', value=tmp_path)


def transform_data(**context):
    """Трансформация: парсинг даты, добавление часа"""
    tmp_path = context['task_instance'].xcom_pull(key='validated_data_path', task_ids='validate_data')
    df = pd.read_csv(tmp_path)
    
    # Преобразование pickup_datetime в datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
    
    # Добавление часа
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    
    # Сохраняем обработанные данные
    output_path = '/tmp/uber_processed.csv'
    df.to_csv(output_path, index=False)
    
    context['task_instance'].xcom_push(key='processed_data_path', value=output_path)
    print(f"Трансформация завершена. Добавлен столбец 'pickup_hour'")

def upload_to_s3(**context):
    """Выгрузка обработанного файла в S3"""
    tmp_path = context['task_instance'].xcom_pull(key='processed_data_path', task_ids='transform_data')
    
    s3_hook = S3Hook(aws_conn_id="yandex_s3_connection")
    s3_hook.load_file(
        filename=tmp_path,
        key=S3_KEY_OUTPUT,
        bucket_name=S3_BUCKET,
        replace=True
    )
    
    print(f"Файл загружен в s3://{S3_BUCKET}/{S3_KEY_OUTPUT}")

# Определение DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'final_task_dag',
    default_args=default_args,
    description='DAG для обработки данных такси',
    schedule='@hourly',
    catchup=False,
    tags=['taxi', 'pandera', 's3']
)

with dag:
    load_task = PythonOperator(
        task_id='load_and_clean_csv',
        python_callable=load_and_clean_csv
    )
    
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data
    )
    
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )
    
    upload_task = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3
    )
    
    load_task >> validate_task >> transform_task >> upload_task