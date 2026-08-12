import os

import requests


class AssessmentAPIError(Exception):
    pass


def assess_image(filename, content_type, image_bytes, session=requests):
    api_url = os.getenv("CAR_DAMAGE_API_URL", "http://127.0.0.1:18000").rstrip("/")
    api_key = os.getenv("CAR_DAMAGE_API_KEY", "")
    timeout = float(os.getenv("CAR_DAMAGE_API_TIMEOUT_SECONDS", "60"))
    if not api_key:
        raise AssessmentAPIError("CAR_DAMAGE_API_KEY is not configured.")

    try:
        response = session.post(
            f"{api_url}/assess",
            headers={"x-api-key": api_key},
            files={"image": (filename, image_bytes, content_type)},
            timeout=timeout,
        )
    except requests.RequestException as error:
        raise AssessmentAPIError("Assessment API is unreachable.") from error

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "60")
        raise AssessmentAPIError(f"Rate limit exceeded. Try again in {retry_after} seconds.")
    if response.status_code == 401:
        raise AssessmentAPIError("Assessment API key is invalid or inactive.")
    if response.status_code == 413:
        raise AssessmentAPIError("Image file exceeds API upload limit.")
    if response.status_code == 503:
        raise AssessmentAPIError("Assessment API is temporarily unavailable.")
    try:
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise AssessmentAPIError("Assessment API returned an invalid response.") from error
