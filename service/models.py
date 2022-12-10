import random
import typing as tp

from pydantic import BaseModel


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class RecoModel:
    def get_reco(self, user_id) -> list:
        pass


class DummyModel(RecoModel):
    def __init__(self) -> None:
        pass

    def get_reco(self, user_id) -> list:
        return random.sample(range(1, 20), 10)


ALL_MODELS = {'dummy_model': DummyModel()}


def get_models() -> tp.Dict[str, DummyModel]:
    return ALL_MODELS
