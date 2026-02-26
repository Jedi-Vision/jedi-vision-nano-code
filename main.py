from jv import Driver
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config",
    type=str,
    default="config/default.yaml",
    help="Configuration file containing arguments for Driver."
)
parser.add_argument(
    "-a", "--args",
    type=str,
    action="store",
    nargs='*',
    help="Arguments to replace any config field in 'key:value' style."
)
args = parser.parse_args()

extra_args = {}
if args.args:
    for arg in args.args:
        k, v = arg.split(":")
        extra_args[k] = v

with open(args.config) as f:
    config = yaml.safe_load(f)

driver_kwargs = config['driver']
object_kwargs = config['object']

bino: bool = driver_kwargs['binocular']

if bino:
    depth_kwargs = config['stereo']
else:
    depth_kwargs = config['mono']

gstreamer_kwargs = config['gstreamer']


def infer_type(value):
    try:
        return float(value)
    except ValueError:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return value


for key, value in extra_args.items():
    typed_value = infer_type(value)
    if key in driver_kwargs:
        driver_kwargs[key] = typed_value
    elif key in object_kwargs:
        object_kwargs[key] = typed_value
    elif key in depth_kwargs:
        depth_kwargs[key] = typed_value
    elif key in gstreamer_kwargs:
        gstreamer_kwargs[key] = typed_value

driver = Driver(**driver_kwargs, **(depth_kwargs if bino else {}), gstreamer_kwargs=gstreamer_kwargs)
driver.run(object_kwargs, depth_kwargs if not bino else {})
