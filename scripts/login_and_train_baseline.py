import os
import sys
import json
import requests


def main():
    base_url = os.environ.get("BASE_URL", "http://127.0.0.1:5000")
    username = os.environ.get("DEMO_USER", "demo")
    password = os.environ.get("DEMO_PASS", "demo1234")
    csv_path = os.environ.get("CSV_PATH", os.path.join("data", "nasa_battery_synth.csv"))

    print(f"Logging in to {base_url} ...")
    r = requests.post(f"{base_url}/api/login", json={"username": username, "password": password})
    if r.status_code != 200:
        print("Login failed:", r.status_code, r.text)
        sys.exit(1)
    token = r.json().get("access_token")
    if not token:
        print("No token in response:", r.text)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}
    print("Starting baseline training with:", csv_path)
    with open(csv_path, "rb") as f:
        resp = requests.post(f"{base_url}/api/baseline/train", headers=headers, files={"file": f})
    print("Status:", resp.status_code)
    print(resp.text)


if __name__ == "__main__":
    main()


