import yaml
import munch

def load():
    config_file = open("config.yml", "r")
    yaml_config = yaml.safe_load(config_file)
    config = munch.munchify(yaml_config)
    return config