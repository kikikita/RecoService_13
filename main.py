import os

import uvicorn
import sentry_sdk

from service.api.app import create_app
from service.settings import get_config

config = get_config()
app = create_app(config)


if __name__ == "__main__":
    sentry_sdk.init(
        dsn="http://c385d49b389e44f59fe3f96b4f02fe8c@213.202.219.36:9000/1",
        traces_sample_rate=1.0,
    )

    host = os.getenv("HOST", "213.202.219.36")
    port = int(os.getenv("PORT", "8002"))

    uvicorn.run(app, host=host, port=port)
