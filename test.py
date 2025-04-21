import requests

url = "http://ocr-api-production-d7df.up.railway.app"
files = {"image": open(r"C:\Users\faten\OneDrive\Pictures\Camera Roll\WIN_20250417_07_41_59_Pro.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())