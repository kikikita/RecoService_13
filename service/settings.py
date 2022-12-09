import os
import pickle

import dill
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

API_KEY = os.getenv("API_KEY")
knn_model = dill.load(open('service/data/knn_bm25_item.dill', 'rb'))
pop_model = dill.load(open('service/data/pop_model_7.dill', 'rb'))
users_list = pickle.load(open('service/data/users_list.pickle', 'rb'))


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
