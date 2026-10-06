import json
import urllib.request

payload = {
    "narrative": ("I have been getting repeated phone calls from a debt collector about "
                  "an account I do not recognize. They claim I owe money from years ago "
                  "but have never sent me anything in writing despite my requests."),
    "product": "Debt collection",
    "issue": "Cont'd attempts collect debt not owed",
    "company": "missing",
    "draft": True,
}
req = urllib.request.Request("http://127.0.0.1:8000/v1/triage",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req)), indent=2))