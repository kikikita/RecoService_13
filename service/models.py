import pickle

import typing as tp
import zipfile

import dill
import numpy as np
from pydantic import BaseModel

from service.settings import get_config

config = get_config()


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class BaseRecModel:
    def get_reco(self, user_id: int, k_recs: int) -> tp.List[int]:
        pass


class DummyModel(BaseRecModel):
    def __init__(self) -> None:
        pass

    def get_reco(self, user_id: int, k_recs: int) -> tp.List[int]:
        return list(range(k_recs))


class KNNModel(BaseRecModel):
    def __init__(self) -> None:
        self.unzip_knn = zipfile.ZipFile(config.zip_knn_path, 'r')
        self.knn_model = dill.load(self.unzip_knn.open(config.knn_model))
        self.pop_model = dill.load(self.unzip_knn.open(config.pop_model))
        self.users_list = pickle.load(self.unzip_knn.open(config.users_list))

    def get_reco(self, user_id: int, k_recs: int = 10) -> tp.List[int]:
        """
        сначала проводится проверка холодный ли юзер,
        есть ли он в списке юзеров из трейна
        если да - то модель KNN выдает рекомендации,
        если нет - то выдаем ему популярное
        """
        if user_id in self.users_list:
            recs = self.knn_model.similar_items(user_id)
            if recs:
                recs = [x[0] for x in recs if not np.isnan(x[0])]

                if len(recs) < k_recs:
                    pop = self.pop_model.recommend(k_recs)
                    recs.extend(pop[:k_recs])
                    recs = list(dict.fromkeys(recs))
                    recs = recs[:k_recs]

                return recs
        return list(self.pop_model.recommend(k_recs))


ALL_MODELS = {'dummy_model': DummyModel(), 'knn_model': KNNModel()}


def get_models() -> tp.Dict[str, BaseRecModel]:
    return ALL_MODELS
