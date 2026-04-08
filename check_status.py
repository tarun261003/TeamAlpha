import urllib.request
import json
import traceback

try:
    url = 'https://huggingface.co/api/spaces/tarun8477/truth_seeker_env'
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    print("Space API Status Response:")
    print("Stage:", data.get("runtime", {}).get("stage"))
    print("Error:", data.get("runtime", {}).get("error"))
except Exception as e:
    print("Error reaching API:")
    traceback.print_exc()
