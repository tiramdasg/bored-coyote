import time

from fastapi.middleware.wsgi import WSGIMiddleware
import uvicorn

from fastapi import FastAPI

from create_dash_app import create_dash_app

fast_api_app = FastAPI()


@fast_api_app.get("/")
def read_main():
    return {
        "routes": [
            {"method": "GET", "path": "/", "summary": "Landing"},
            {"method": "GET", "path": "/status", "summary": "App status"},
            {"method": "GET", "path": "/dash", "summary": "Sub-mounted Dash application"},
        ]
    }


@fast_api_app.get("/status")
def get_status():
    return {"status": "ok"}


requests_pathname_prefix = "/dash/"
app = create_dash_app(requests_pathname_prefix)
fast_api_app.mount("/dash", WSGIMiddleware(app.server))

if __name__ == '__main__':
    time.sleep(3)
    uvicorn.run(fast_api_app, host="0.0.0.0", port=8001)
