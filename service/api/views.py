from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import (
    ModelNotFoundError,
    NotAuthorizedError,
    UserNotFoundError,
)
from service.log import app_logger
from service.models import get_models


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


MODELS = get_models()
router = APIRouter()


token_bearer = HTTPBearer(auto_error=False)


async def get_api_key(
    token: HTTPAuthorizationCredentials = Security(token_bearer),
) -> str:
    if not token:
        raise NotAuthorizedError(
            error_message="Missing bearer token",
        )
    return token.credentials


def check_api_key(expected: str, actual: str) -> None:
    if expected != actual:
        raise NotAuthorizedError(
            error_message="Invalid token",
        )


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    ''' Health check '''
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={404: {"description": "User or model not found"},
               401: {"description": "Not authorized"}},
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    api_key: str = Depends(get_api_key)
) -> RecoResponse:
    ''' Get recommendations for a user '''
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    check_api_key(request.app.state.api_key, api_key)

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name not in MODELS.keys():
        raise ModelNotFoundError(
            error_message=f"Model {model_name} not found"
        )

    reco_list = MODELS[model_name].get_reco(user_id, request.app.state.k_recs)
    return RecoResponse(user_id=user_id, items=reco_list)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
