import argparse
import json

def load_json(path: str):
    f = open(path, "r")
    j = json.loads(f.read())
    f.close()
    return j

parser = argparse.ArgumentParser()
parser.add_argument("--settings", type=str, required=True)
args = parser.parse_args()

json_dict = load_json(args.settings)

print(json_dict)
print(json_dict["id"])
print(type(json_dict))