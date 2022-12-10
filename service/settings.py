import os
import pickle
import zipfile

import dill
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

API_KEY = os.getenv("API_KEY")
zip_knn = zipfile.ZipFile('service/data/knn/knn_bm25_item.zip', 'r')
knn_model = dill.load(zip_knn.open('knn_bm25_item.dill'))
zip_pop = zipfile.ZipFile('service/data/knn/pop_model_7.zip', 'r')
pop_model = dill.load(zip_pop.open('pop_model_7.dill'))
us_zip = zipfile.ZipFile('service/data/knn/users_list.zip', 'r')
users_list = pickle.load(us_zip.open('users_list.pickle'))
emb_zip = zipfile.ZipFile('service/data/lightFM/emb_maps.zip', 'r')
embeds_maps = pickle.load(emb_zip.open('emb_maps.pickle'))


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
