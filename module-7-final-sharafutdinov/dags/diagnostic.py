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
import chardet

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
    strict=True,  # Запрещает лишние колонки
    ordered=False
)

# Конфигурация
S3_BUCKET = "r-mlops-bucket-8-1-11-22209764"
S3_KEY_INPUT = "uber.csv"
S3_KEY_OUTPUT = "uber_processed.csv"

def detect_encoding(file_path):
    """Определение кодировки файла"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] if result['encoding'] else 'utf-8'

def load_and_clean_csv(**context):
    """Загрузка CSV из S3 - диагностическая версия"""
    s3_hook = S3Hook(aws_conn_id="yandex_s3_connection")
    
    # Получаем объект из S3
    s3_object = s3_hook.get_key(key=S3_KEY_INPUT, bucket_name=S3_BUCKET)
    
    # Читаем бинарные данные
    binary_data = s3_object.get()['Body'].read()
    
    # Показываем первые 500 байт для диагностики
    print(f"Первые 200 байт файла: {binary_data[:200]}")
    
    # Пробуем декодировать и посмотреть, что получится
    decodings = ['utf-8-sig', 'utf-16', 'latin1', 'cp1251']
    
    for encoding in decodings:
        try:
            decoded = binary_data.decode(encoding)
            print(f"\n=== Кодировка {encoding} ===")
            print(f"Первые 500 символов:\n{decoded[:500]}")
            
            # Пробуем прочитать CSV с разными разделителями
            for sep in [',', ';', '\t']:
                try:
                    from io import StringIO
                    df = pd.read_csv(StringIO(decoded), sep=sep, nrows=5)
                    print(f"\nРазделитель '{sep}':")
                    print(f"Колонки: {list(df.columns)}")
                    print(f"Первая строка:\n{df.head(1)}")
                    
                    if len(df.columns) > 3:  # Если получили больше 3 колонок
                        # Сохраняем этот df для дальнейшей работы
                        break
                except:
                    continue
            else:
                continue
            break
        except Exception as e:
            print(f"Кодировка {encoding} не подошла: {e}")
            continue
    else:
        raise ValueError("Не удалось прочитать файл")
    
    # Если дошли сюда, df должен быть определен
    if df is None:
        raise ValueError("Не удалось распарсить CSV")
    
    print(f"\nФинальные колонки: {list(df.columns)}")
    
    # Сохраняем для следующего шага
    cleaned_path = '/tmp/uber_cleaned.csv'
    df.to_csv(cleaned_path, index=False)
    
    context['task_instance'].xcom_push(key='cleaned_data_path', value=cleaned_path)
    return f"Загружено {len(df)} записей"

def validate_data(**context):
    """Валидация DataFrame через Pandera"""
    tmp_path = context['task_instance'].xcom_pull(key='cleaned_data_path', task_ids='load_and_clean_csv')
    df = pd.read_csv(tmp_path)
    
    print(f"Валидация: {len(df)} записей")
    print(f"Колонки для валидации: {list(df.columns)}")
    
    # Валидация
    try:
        taxi_schema.validate(df, lazy=True)
        print(f"Валидация пройдена: {len(df)} записей")
    except pa.errors.SchemaErrors as e:
        print(f"Ошибки валидации: {e}")
        print(f"Количество ошибок: {len(e.failure_cases)}")
        print(f"Примеры ошибок:\n{e.failure_cases.head()}")
        raise
    
    context['task_instance'].xcom_push(key='validated_data_path', value=tmp_path)

def transform_data(**context):
    """Трансформация: парсинг даты, добавление часа"""
    tmp_path = context['task_instance'].xcom_pull(key='validated_data_path', task_ids='validate_data')
    df = pd.read_csv(tmp_path)
    
    # Очистка строки даты (удаление миллисекунд если есть)
    df['pickup_datetime'] = df['pickup_datetime'].str.replace(r'\.\d+', '', regex=True)
    
    # Преобразование pickup_datetime в datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
    
    # Добавление часа
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    
    # Добавление дня недели (опционально)
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    
    print(f"Добавлены колонки: pickup_hour, pickup_dayofweek")
    print(f"Диапазон дат: {df['pickup_datetime'].min()} - {df['pickup_datetime'].max()}")
    
    # Сохраняем обработанные данные
    output_path = '/tmp/uber_processed.csv'
    df.to_csv(output_path, index=False)
    
    context['task_instance'].xcom_push(key='processed_data_path', value=output_path)
    print(f"Трансформация завершена. Итоговых записей: {len(df)}")

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
    'start_date': datetime(2025, 1, 1),  # Исправлено на прошедшую дату
    'retries': 2,  # Увеличил количество попыток
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'diagnostic_dag',
    default_args=default_args,
    description='DAG для diagnostic',
    schedule='@hourly',
    catchup=False,
    tags=['module-7', 'uber', 'ETL']
) as dag:
    
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