from dataclasses import fields, dataclass
from typing import Dict, Any


def json_data(cls):
    """
    The decorator used for JSON parser.

    Examples::

        @json_data
        class Point:
            x: int
            y: int

        # Use for tensorneko read
        point = tensorneko.io.read.text.of_json("path/to/file.json", cls=Point)

        # Use for the Python json loads
        with open("path/to/file.json", "r") as f:
            point = json.loads(f.read(), object_hook=Point)

        print(point.x, point.y)

    """

    def constructor(self, d: Dict[str, Any]):
        for field in fields(self.__class__):
            k = field.name
            t = field.type
            if "typing.List[" in str(t):
                inner_type = t.__args__[0]
                if "typing.List[" in str(inner_type):
                    inner_inner_type = inner_type.__args__[0]
                    try:
                        is_json_data = inner_inner_type.is_json_data
                    except AttributeError:
                        is_json_data = False
                    if is_json_data:
                        v = [[inner_inner_type(e) for e in each] for each in d[k]]
                    else:
                        v = d[k]
                else:
                    try:
                        is_json_data = inner_type.is_json_data
                    except AttributeError:
                        is_json_data = False

                    if is_json_data:
                        v = list(map(inner_type, d[k]))
                    else:
                        v = d[k]
            else:
                try:
                    is_json_data = t.is_json_data
                except AttributeError:
                    is_json_data = False

                if is_json_data:
                    v = t(d[k])
                else:
                    v = d[k]

            self.__dict__[k] = v

    def to_dict(self) -> dict:
        out = {}
        for field in fields(self):
            value = self.__dict__[field.name]
            if "is_json_data" in type(value).__dict__ and type(value).is_json_data:
                value = value.to_dict()
            out[field.name] = value
        return out

    processed_class: type = dataclass(cls)
    processed_class.__init__ = constructor
    processed_class.to_dict = to_dict
    processed_class.is_json_data = True
    return processed_class
