"""生成确定性排序键和模型初始化种子。"""
import hashlib
import json


def digest(config, purpose, domain, counter, object_id):
    value = [config["protocol"], config["seed"], purpose, config["profile"],
             domain, counter, object_id]
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_order(items, config, purpose, domain, counter):
    """按稳定哈希键排序对象。"""
    return sorted(items, key=lambda x: (digest(config, purpose, domain, counter, x), x))


def model_seed(config, purpose, domain, repeat, module):
    return int(digest(config, purpose, domain, repeat, module)[:8], 16)
