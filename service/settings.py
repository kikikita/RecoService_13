import os
import pickle
import zipfile

import dill
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

API_KEY = os.getenv("API_KEY")
zip_data = zipfile.ZipFile('service/data/data.zip', 'r')
knn_model = dill.load(zip_data.open('knn_bm25_item.dill'))
pop_model = dill.load(zip_data.open('pop_model_7.dill'))
users_list = pickle.load(zip_data.open('users_list.pickle'))


class Config(BaseSettings):

    class Config:
        case_sensitive = False


class LogConfig(Config):
    level: str = "INFO"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False
        fields = {
            "level": {
                "env": ["log_level"]
            },
        }


class ServiceConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10

    log_config: LogConfig


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
