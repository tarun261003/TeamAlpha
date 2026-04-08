import urllib.request

try:
    health_url = "https://tarun8477-truth-seeker-env.hf.space/health"
    req = urllib.request.Request(health_url)
    resp = urllib.request.urlopen(req)
    print("Health response:", resp.read().decode())
except Exception as e:
    print("Health check failed:", e)

try:
    root_url = "https://tarun8477-truth-seeker-env.hf.space/"
    req = urllib.request.Request(root_url)
    resp = urllib.request.urlopen(req)
    print("Root response:", resp.read().decode())
except BaseException as e:
    print("Root hit exception:", e)
