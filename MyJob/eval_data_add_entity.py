import json

with open("./data/entity_predict.json", "r", encoding="utf8") as f:
    entity_predict = json.load(f)
print(len(entity_predict))

with open("./data/dev_data_add_entity.json", "w", encoding="utf8") as f1:
    for line in open("./data/dev.json", "r", encoding="utf8"):
        dev = json.loads(line)
        for i in range(len(dev["input"])):
            # temp_dev = dev["input"][i]
            for j in range(len(dev["input"][i])):
                _entity_predict = entity_predict.pop(0)
                if _entity_predict == ["others"]:
                    join_part = "【】"
                else:
                    _entity_predict = [i for i in _entity_predict if i != "others"]
                    join_part = "【##" + "$，##".join(_entity_predict) + "$】"
                dev["input"][i][j] = join_part + dev["input"][i][j]
        f1.write(json.dumps(dev, ensure_ascii=False) + "\n")