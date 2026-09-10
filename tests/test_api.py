import requests
from pathlib import Path


API_URL = "http://127.0.0.1:8000/predict"

IMAGE_PATH = Path(
    "../dataset/seg_test/forest/20057.jpg"
)


def test_prediction():

    with open(IMAGE_PATH, "rb") as image:

        response = requests.post(
            API_URL,
            files={
                "file": (
                    IMAGE_PATH.name,
                    image,
                    "image/jpeg"
                )
            }
        )

    print("Status:", response.status_code)
    print("Response:", response.json())


if __name__ == "__main__":
    test_prediction()