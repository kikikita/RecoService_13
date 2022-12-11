import typing as tp

from pydantic import BaseModel


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


ALL_MODELS = {'dummy_model': DummyModel()}


def get_models() -> tp.Dict[str, DummyModel]:
    return ALL_MODELS
