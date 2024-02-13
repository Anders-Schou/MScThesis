import json

def load_json(path: str):
    f = open(path, "r")
    j = json.loads(f.read())
    f.close()
    return j


json_dict = load_json("./settings.json")

print(json_dict)
print(json_dict["id"])
print(type(json_dict))