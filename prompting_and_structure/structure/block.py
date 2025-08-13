import json
from scipy.spatial.transform import Rotation as R


def _round_list(ls):
    return [round(x) for x in ls]


class Block:
    def __init__(
        self,
        id: int,
        block_name: str,
        gpt_name: str,
        shape: str,
        dimensions: list,
        position: list,
        orientation: list,
        color: list = [1.0, 0, 0, 1.0],
    ) -> None:
        assert len(color) == 4, "color is not length 4"
        assert (
            shape == "cuboid" or shape == "cylinder" or shape == "cone"
        ), f"shape must be cuboid or cylinder or cone: {shape}"
        if shape == "cuboid":
            assert (
                len(dimensions) == 3
            ), f"dimensions for cuboid not length 3: {dimensions}"
        elif shape == "cylinder" or shape == "cone":
            assert (
                len(dimensions) == 2
            ), f"dimensions for cylinder not length 2: {dimensions}"
        assert (
            len(orientation) == 4 or len(orientation) == 3
        ), f"orientation must be length 4 quaternion or length 3 euler angles: {orientation}"
        assert len(color) == 4, f"color must be rgba: {color}"

        if type(dimensions) == dict:
            dimensions = list(dimensions.values())

        self._id = id
        self._block_name = block_name
        self._gpt_name = gpt_name
        self._shape = shape
        self._dimensions = dimensions
        self._position = position
        self.orientation = orientation
        self._color = color

    def get_json(self):
        dimensions = (
            {"x": self.dimensions[0], "y": self.dimensions[1], "z": self.dimensions[2]}
            if self.shape == "cuboid"
            else {"radius": self.dimensions[0], "height": self.dimensions[1]}
        )

        return {
            "id": self.id,
            "block_name": self.block_name,
            "gpt_name": self.gpt_name,
            "shape": self.shape,
            "dimensions": dimensions,
            "position": self.position,
            "orientation": self.orientation,
            "color": self.color,
        }

    @staticmethod
    def from_json(input_json: json):
        data = json.loads(input_json)
        return Block(
            id=data["id"],
            block_name=data["block_name"],
            gpt_name=data["gpt_name"],
            shape=data["shape"],
            dimensions=data["dimensions"],
            position=data["position"],
            orientation=data["orientation"],
            color=data["color"],
        )

    @staticmethod
    def from_gpt_output(gpt_output, available_blocks):
        gpt_name = list(gpt_output.keys())[1]
        data = gpt_output[gpt_name]
        block_name = data["block_type"]
        available_blocks_data = available_blocks[block_name]

        return Block(
            id=999,  # id gets updated by place blocks call, otherwise it's unknown
            gpt_name=gpt_name,
            block_name=block_name,
            shape=available_blocks_data["shape"],
            dimensions=available_blocks_data["dimensions"],
            position=data["position"],
            orientation=data["orientation"],
            color=data["color"]
            # could do color based on available blocks json or gpt, but doesn't exist for now (defualts)
        )

    def move(self, position: list, delta=True):
        x, y, z = self.position
        if delta:
            delta_x, delta_y, delta_z = position
            self.position = [x + delta_x, y + delta_y, z + delta_z]
        else:
            self.position = position

    def rotate(self, euler_angles: list, delta=False):
        x, y, z = self.position
        if delta:
            delta_x, delta_y, delta_z = euler_angles
            self.euler_orientation = [x + delta_x, y + delta_y, z + delta_z]
        else:
            self.euler_orientation = euler_angles

    @property
    def euler_orientation(self):
        return R.from_quat(self.orientation).as_euler("xyz", degrees=True)

    @euler_orientation.setter
    def euler_orientation(self, value):
        self._orientation = R.from_euler(value).as_quat()

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def block_name(self):
        return self._block_name

    @block_name.setter
    def block_name(self, value):
        self._block_name = value

    @property
    def gpt_name(self):
        return self._gpt_name

    @gpt_name.setter
    def gpt_name(self, value):
        self._gpt_name = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = _round_list(value)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        # self._position = _round_list(value)
        self._position = value

    def get_round_position(self):
        return _round_list(self._position)

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        self._dimensions = _round_list(value)

    def __str__(self):
        return f"""
            BLOCK: 
            shape: {self.shape},
            color: {self.color},
            dims: {self.dimensions},
            pos: {self.position}
        """
