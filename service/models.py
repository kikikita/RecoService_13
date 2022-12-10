import random
import typing as tp

import numpy as np
from pydantic import BaseModel

from service.settings import embeds_maps, knn_model, pop_model, users_list


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class OurModels:
    def get_reco(self, user_id) -> list:
        pass


class DummyModel(OurModels):
    def __init__(self) -> None:
        pass

    def get_reco(self, user_id) -> list:
        return random.sample(range(1, 20), 10)


class KNNModel(OurModels):
    def __init__(self) -> None:
        pass

    def get_reco(self, user_id, N=10):
        """
        check if user is in users list
        if yes - return knn recs and add pop recs if knn recs < 10
        if no - return pop recs
        """
        pop = pop_model.recommend(N)

        if user_id in users_list:
            try:
                recs = knn_model.similar_items(user_id)
                if recs:
                    recs = [x[0] for x in recs]
                    recs = [x for x in recs if not np.isnan(x)]
                    if len(recs) < N:
                        recs.extend(pop[:N])
                        recs = recs[:N]
                    return recs.tolist()
                return pop.tolist()
            except AttributeError:
                return pop.tolist()
        else:
            return pop.tolist()


class LightFMModel(OurModels):
    def __init__(self) -> None:
        pass

    def get_reco(self, user_id, K=10):
        """
        check if user is in users list
        if true - return lightfm recs
        if false - return popular recs
        """
        emb_users_list = embeds_maps['user_id_map'].index

        if user_id in emb_users_list:
            output = embeds_maps['user_embeddings'][
                    embeds_maps['user_id_map'][user_id], :]\
                .dot(embeds_maps['item_embeddings'].T)
            recs = (-output).argsort()[:10]
            return [embeds_maps['item_id_map'][item_id]for item_id in recs]
        return pop_model.recommend(K).tolist()


ALL_MODELS = {
    'dummy_model': DummyModel(),
    'knn_model': KNNModel(),
    'lightfm_model': LightFMModel(),
    }


def get_models() -> tp.Dict[str, OurModels]:
    return ALL_MODELS
