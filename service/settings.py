from pydantic import BaseSettings


class Config(BaseSettings):

    class Config:
        case_sensitive = False
        env_file = 'service/.env'


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
    api_key: str
    zip_models_path: str = 'models/models.zip'
    knn_model: str = 'knn_bm25_item.dill'
    pop_model: str = 'pop_model_7.dill'
    users_list: str = 'users_list.pickle'
    emb_maps: str = 'emb_maps.pickle'
    knows_items: str = 'known_items.pickle'

    log_config: LogConfig


def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
