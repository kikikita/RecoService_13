import os

import uvicorn
import sentry_sdk

from service.api.app import create_app
from service.settings import get_config

config = get_config()
app = create_app(config)


if __name__ == "__main__":

    host = os.getenv("HOST", "213.202.219.36")
    port = int(os.getenv("PORT", "8002"))

    uvicorn.run(app, host=host, port=port)
