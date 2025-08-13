import numpy as np
import pickle
import os
import json

from typing import List
from dataclasses import dataclass
from prompting_and_structure.structure import Structure


@dataclass
class Assembly:
    structure: Structure
    to_build: str
    structure_directory: str
    isometric_image: np.ndarray
    available_blocks_json: dict
    assembly_num: int
    eval_rating: int | None = None
    eval_guesses: List[str] | None = None
    pre_perturbation_image: np.ndarray | None = (
        None  # isometric image before perturbation if perturbation is performed.
    )

    def save_to_structure_dir(self, name=None):
        if name is None:
            self.save(
                f"{self.structure_directory}/assembly_obj_{self.assembly_num}.pkl"
            )
        else:
            self.save(f"{self.structure_directory}/{name}")

    @classmethod
    def load_assembly_from_structure_dir(cls, structure_dir, assembly_num):
        cls.load(f"{structure_dir}/assembly_obj_{assembly_num}.pkl")

    @classmethod
    def load_all_assemblies_from_structure_dir(cls, structure_dir):
        pkl_files = [
            f
            for f in os.listdir(structure_dir)
            if f.endswith("pkl") and f.startswith("assembly_obj_")
        ]
        return [
            cls.load_assembly_from_structure_dir(structure_dir, i)
            for i in range(len(pkl_files))
        ]

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def human_friendly_save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save(os.path.join(save_dir, "assembly.pkl"))
        self.isometric_image.save(os.path.join(save_dir, "isometric_image.png"))

        if self.pre_perturbation_image is not None:
            self.pre_perturbation_image.save(
                os.path.join(save_dir, "pre_perturbation_image.png")
            )

        save_dict = {
            "to_build": self.to_build,
            "structure_directory": self.structure_directory,
            "assembly_num": self.assembly_num,
            "eval_rating": self.eval_rating,
            "eval_guesses": self.eval_guesses,
        }
        with open(os.path.join(save_dir, "save_dict.json"), "w") as json_file:
            json.dump(save_dict, json_file, indent=4)

        structure_json = self.structure.get_json()
        with open(os.path.join(save_dir, "structure.json"), "w") as json_file:
            json.dump(structure_json, json_file, indent=4)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    import os

    test_assembly = Assembly(
        Structure(), "gpt_caching", "train", np.array([5, 5, 3]), {}, 10
    )
    test_assembly.eval_guesses = ["train", "car"]
    test_assembly.eval_rating = 4
    test_assembly.save("assembly_test.pkl")

    load_assembly = test_assembly.load("assembly_test.pkl")
    print(load_assembly)

    os.remove("assembly_test.pkl")
