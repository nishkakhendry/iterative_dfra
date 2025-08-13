import json
import os
import numpy as np
import pybullet as p

import matplotlib.pyplot as plt
from PIL import Image

from prompting_and_structure.utils.file_and_parse_utils import (
    get_last_json_as_dict,
    save_to_json,
    slugify,
    markdown_json,
)
from prompting_and_structure.pybullet.pybullet_images import get_imgs
from prompting_and_structure.prompts.prompt import prompt_with_caching
from prompting_and_structure.structure import Structure, Block, Assembly
from prompting_and_structure.utils.file_and_parse_utils import load_from_json, extract_code_from_response


SAVE_DIR = "gpt_caching"

########### IDfRA prompts ############
# Judge module
def make_missing_suggestion_prompts(to_build, available_blocks,structure_list,missing_blocks):
    system_prompt = f"""
    "You are a precise and critical evaluator of the performance of a robotic arm in building assembly structures using blocks. The simulated physical environment used follows Newtonian physics.
     Your role is to assess each block's semantic contribution to the structure, which blocks are missing, how accurately the resulting structure resembles the intended design 
    {to_build} and which key features should be added, removed, or enhanced.
     
    You must reason step-by-step, comparing the planned actions with what occurs in the visual input. You must consider block availability constraints to identify use of unavailable blocks in the 
    proposed plan. Clearly state that such unavailable blocks must not be used in subsequent designs.
    
    At the end, you must return a structured JSON containing your full analysis under a 'description' field."
    """.strip()

    user_prompt = f"""
    The robotic arm was tasked with building a structure that resembles a {to_build} using blocks. The attempt is captured in this GIF.
    
    The plan to build this structure is given here as a list of actions: {structure_list}.
    
    **Block availability anaylsis**:
    Here is a list of missing blocks: {missing_blocks}.
    If the list is non-empty, elements are in the format "<x>-<y>-<z>-<block type>_<index>" where <x>,<y>,<z> are the dimensions of each missing 
    block and <index> is the unavailable block index for a block with dimensions <x>,<y>,<z>. Validate block unavailability by comparing the used blocks in {structure_list} against the pool in {available_blocks}. 
    Map each missing block to the corresponding action using its <index> (e.g. <index> = 5 means the 5th time a block of dimension <x>,<y>,<z> is required, it is unavailable. This can be at any index in the 
    action list). **Clearly state that the missing blocks are unavailable and should not be used in future designs**.

    **Step-by-step**:
    1. Reason about which part of the structure each block forms and whether it contributes meaningfully towards resembling a {to_build}. 
    2. For each action, compare the plan to what actually happens visually in the GIF and your knowledge about which blocks are missing - reason about which actions are skipped due to unavilable blocks.
    
    **Once all steps are analyzed, observe the final structure**:
    - Judge how closely the final structure resembles a {to_build}
    - Critically assess the proportions and positioning of blocks in the design
    - List what features are missing or could be added to improve resemblance to {to_build}
    - Identify extra features that reduce structural integrity or resemblance which should be removed

    Finally, remove any comment about stability and return your full evaluation in this JSON format:
    {markdown_json({"description_missing_suggestions": "In this GIF..."})}
    """.strip()
    return system_prompt, user_prompt

# Replan submodule prompt
def make_replan_prompt(to_build, available_blocks, outcome_reason, plans_so_far):
    return [
        f""" 
    I'm working on constructing a block tower that represents a(n) {to_build}.
    I have the following blocks available: 
    {markdown_json(available_blocks)}
    
    Here are descriptions of plans I have tried so far:
    {markdown_json(plans_so_far)}
    
    Here is feedback from a judge about unavailable blocks which should not be used in the next plan, suggestions to improve the design, and whether any obvious topples occurred:   
    {markdown_json(outcome_reason)}.
    
    Your task is to generate a **new structural plan** for assembling a {to_build} using the available blocks.
    
    **Important Requirements**:
    - You must **physically alter the structure** when a new feature (like legs, head, tail, etc.) is added. It is **not acceptable** to reuse an existing block and merely rename it to represent a new function.
    - If the feedback suggests adding any new component, your design must contain **distinct, additional blocks** that clearly represent those features.
    - You are encouraged to **start from scratch** or rearrange the blocks entirely if necessary to reflect the added or removed features.
    - Avoid duplicating block roles (e.g., do not say a block is both "body base" and "legs") unless the geometry clearly supports that dual role.
    - Respect available quantity and dimensions of blocks. 
    - **Be flexible and creative** - Use larger or smaller blocks if they are available to match major semantic features better (e.g choose a larger block as a base or body if it is available and better suited). 
    
    The layout should prioritize **semantic recognizability**, not just stability or symmetry. You are encouraged to adjust:
    - **Placement** of blocks (e.g., moving the neck or tail **slightly** towards either side rather than stacking everything centrally - but **carefully** so that entire neck or tail blocks are supported by the blocks below)
    - **Block selection** and **Relative sizing and direction** to match the conceptual structure (e.g., from the pool of available blocks - using smaller blocks for minor features like tail and larger blocks for major features like body, legs, base) 
    - Pay attention to **relative sizes of different features** when choosing dimensions of blocks.
    
    **Design Constraints**:
    - Prioritize stability, simplicity, and visual clarity. Do not overcomplicate the design or make it excessively tall or wide. Use a minimal amount of blocks and keep it simple, just enough so that it looks like a {to_build}.
    - The description should emphasize overall proportions, key features (like legs, body), and how each block contributes to the recognizable structure.
    - Respect available quantity and dimensions of blocks. 
    - Blocks **cannot** be cut to a smaller size under any circumstances. In such cases, use a smaller block if required or remove the feature.
    
    **Block Placement Instructions**:
    - Explain which blocks to use, their shape, and dimensions.
    - Decide a semantic name for the block for the role it represents in the structure. 
    - Describe exactly how the blocks are placed relative to each other (they can also just be placed on the ground). 
    - Explain the overall orientation of the structure. 
            
    The dimensions of a cuboid are given as x, y, and z, which define the size of the block.  You can rearrange these dimensions to fit your design requirements. For instance, if you need to place a block "vertically" with its longest side along the z-axis, but the dimensions are listed as x: 30, y: 30, z: 10, you can adjust them to x: 10, y: 30, z: 30 to achieve the desired orientation. Ensure the x and y dimensions are also consistent with the rest of the design.
    
    Cylinders are always positioned "upright," with the flat sides parallel to the ground and their radius extending along the x and y axes.  
    """
    ]

# Order submodule prompt
def order_blocks_prompts_re(to_build, replan_response):
    return f"""
    Here is the overall unordered plan (called unordered_plan) for stacking blocks to assemble a {to_build} structure: {replan_response}
        
    Given the blocks described in the plan, I will place and stack these blocks one at a time by lowering them from a very tall height.
    
    Please describe the sequence in which the blocks in unordered_plan should be placed to correctly form a {to_build} structure. This means that blocks at the bottom should be placed first, followed by the higher blocks, so that the upper blocks can be stacked on top of the lower ones. 
    
    IMPORTANT: If any previously suggested placement order was illogical (e.g. placing legs AFTER the body, or a door AFTER a roof), **you must correct it** by reordering the assembly steps (e.g. switch the plan steps to place the 
    logically LOWER feature first in the order - e.g. the legs before the body, or the door before the roof). 
    
    For each block, specify whether it will be placed on the ground or on top of another block. If a block will be supported by multiple blocks, mention all of them. Ensure that the blocks are placed in a way that they remain stable and won't topple over when the physics simulation is run. 
    
    Blocks cannot hover without support and are very unstable when only half their bottom area is support by a singular block below. Clearly mention that these scenarios should be avoided.    
    """.strip()

# Position submodule prompt
def decide_positions_prompts_re(to_build,order_plan):
    return f"""
    With the stacking order determined, you need to compute the x, y, and z coordinates, as well as the yaw angle for each block to build a {to_build} structure according to this plan: {order_plan}.
    
    The x, y, and z coordinates represent the center of each block. The yaw angle refers to the rotation around the z-axis in degrees. Remember, you can swap the dimensions of blocks to adjust their configuration.
    
    **Consider the dimensions of the blocks itself as well as dimensions of other blocks when determining the x, y positions**. 
    
    The **z position must be computed precisely** so the **bottom face of each block rests flush** on the top face of the block(s) beneath it (or on the ground if it's a base block). Calculate this precisely using the z dimension or height of blocks below. Incorrect height values will lead to **floating blocks or overlaps**, which are invalid.

    The **x and y positions MUST be computed accurately** so the centre of each block is calculated based on the blocks supporting it - ensuring that its bottom face should be placed **either fully on the ground** or **fully on top of a support polygon formed by one or more other block(s)**. 
    **You must NEVER place a block with only partial support for its bottom face**, such as having half of it on the ground and the other half on another block. This leads to **unstable configurations** and must be avoided at all costs. For example, avoid configurations where a block is simultaneously supported by the ground and another block, as their heights differ and that leads to instability. Explain how each x and y position is calculated based on relative positioning and dimensions of other blocks.

    **Below are two negative examples in which the bottom face of the final block is not fully supported and falls down**. This is especially crucial when the bottom face of a block is only supported by one other block: 
    INVALID Example 1: In the first negative example in JSON format below, a block is placed with only half of its bottom face supported — the other half hangs over empty space. This violates the full-support rule, where the entire bottom face of a block must lie completely within the support surface beneath it. Such configurations are unstable and **must be strictly avoided at all costs**. 
    {markdown_json(
            [
                {
                    "name": "block1",
                    "shape": "cuboid",
                    "dimensions": {"x": 25, "y": 25, "z": 25},
                    "color": [0.5, 0.5, 0.5, 1],
                    "position": {"x": 0, "y": 0, "z": 12.5},
                    "yaw": 0
                },
                {
                    "name": "block2",
                    "shape": "cuboid",
                    "dimensions": {"x": 25, "y": 25, "z": 25},
                    "color": [0.5, 0.5, 0.5, 1],
                    "position": {"x": 12.5, "y": 0, "z": 37.5},
                    "yaw": 0
                }
            ]
        )} 
    
    INVALID Example 2: In the second negative example, two blocks are both placed with only half their bottom faces supported on a single small block beneath them. The combined base areas of the upper blocks extend beyond the boundaries of the support surface, resulting in instability. **EVERY** block's entire bottom face must be fully contained within the top surfaces of its support blocks — overhangs like these are NEVER allowed.
    {markdown_json([
            {
            "name": "head",
            "shape": "cuboid",
            "dimensions": {"x": 25,"y": 25,"z": 25},
            "color": [ 0.5, 0.5, 0.5,1],
            "position": {"x": 0, "y": 0,"z": 157.5},
            "yaw": 0
        },
        {
            "name": "ear_left",
            "shape": "cuboid",
            "dimensions": {"x": 25,"y": 25,"z": 25},
            "color": [0.5,0.5,0.5,1],
            "position": {"x": -12.5,"y": 0,"z": 170},
            "yaw": 0
        },
        {
            "name": "ear_right",
            "shape": "cuboid",
            "dimensions": {"x": 25,"y": 25,"z": 25},
            "color": [0.5,0.5,0.5,1],
            "position": {"x": 12.5,"y": 0,"z": 170},
            "yaw": 0
        }
        ])}
    

    ### Critical Stability Constraints (Must Be Strictly Followed While Determining x, y, z Coordinates):
    - A block must only be placed if the entire 2D rectangle of its bottom face lies fully within the support polygon formed by the top faces of the blocks beneath it. The support area must fully contain the projected base area of the block above. Even a 5mm overhang is disallowed.
    - A block **must be entirely supported** by:
        - A single larger or equal-sized block **directly beneath it**, OR
        - Multiple adjacent blocks where its **center of mass lies entirely over the combined support area**.
    - DO NOT allow any part of a block to **overhang unsupported space**. This includes any "half-on, half-off" placements.
    - Ensure **no collisions or overlaps** in x, y, or z.

    For each block, explain exactly which block(s) provide support, and show how its bottom face lies entirely within the supporting block's top face(s) or support area. State whether the support is from a single block or a combination, and explicitly verify that no part of the base overhangs.

    IMPORTANT: Before you output the JSON plan, verify for each block:
    1. Its entire bottom face lies within the top face(s) of its support blocks.
    2. Its center of mass is directly above the support area.
    3. There is no overlap or height misalignment.
    4. The structure does not include any partial, asymmetric, or mixed ground/raised supports.

    If ANY of these 4 conditions are not met, the block is invalid . Please re-position it such that its bottom face is supported fully before formulating your final JSON output. 

    Output your plan in the following JSON format. Make sure all the keys are present:
    {markdown_json(
        [
            {
                "name": "support1",
                "shape": "cylinder",
                "dimensions": {"radius": 20, "height": 40},
                "color": [0.5, 0.5, 0.5, 1],
                "position": {"x": -50, "y": 0, "z": 0},
                "yaw": 0,
            },
            {
                "name": "support2",
                "shape": "cylinder",
                "dimensions": {"radius": 20, "height": 40},
                "color": [0.5, 0.5, 0.5, 1],
                "position": {"x": 50, "y": 0, "z": 0},
                "yaw": 0,
            },
            {
                "name": "deck",
                "shape": "cuboid",
                "dimensions": {"x": 100, "y": 50, "z": 20},
                "color": [0.5, 0.5, 0.5, 1],
                "position": {"x": 0, "y": 0, "z": 50},
                "yaw": 45,
            },
        ]
    )}
    """

########### BloxNet prompts for initial assembly ############
def make_description_prompt(to_build):
    return f"""
I'm working on constructing a block tower that represents a(n) {to_build}. I need a concise, qualitative description of the design that captures its essence in a minimalistic style. The design should focus on simplicity, avoiding unnecessary complexity while still conveying the key features of a(n) {to_build}. The description should highlight the overall structure and proportions, emphasizing how the block arrangement reflects the object's shape and form. However the design shouldn't be too large, too wide, or too tall. 
""".strip()

def make_plan_prompt(to_build, blockset, description):
    return [
        f"""
Here's a description of the layout of a {to_build}:
{description}

You have the following blocks available: 
{markdown_json(blockset)}
Write a plan for how to assemble a {to_build} using the available blocks. Use blocks as needed while respecting the number available constraint. 

Explain which blocks to use and their shape and dimensions. 

Explain the overall orientation of the structure.

Explain each block's role in the structure. 

Explain how the blocks should stack on top of each other (they can also just be placed on the ground). 

Do not overcomplicate the design. Try to use a minimal number of blocks to represent the key components of a {to_build}. Avoid making structures that are too tall, wide, or complex.

Only consider the main key components of a {to_build}, not minor details that are hard to represent with blocks. 
Use a minimal amount of blocks and keep it simple, just enough so that it looks like a {to_build}.

The dimensions of a cuboid are given as x, y, and z, which define the size of the block. You can rearrange these dimensions to fit your design requirements. For instance, if you need to place a block "vertically" with its longest side along the z-axis, but the dimensions are listed as x: 30, y: 30, z: 10, you can adjust them to x: 10, y: 30, z: 30 to achieve the desired orientation. Ensure the x and y dimensions are also consistent with the rest of the design.

Cylinders are always positioned "upright," with the flat sides parallel to the ground and their radius extending along the x and y axes.

Cones are always positioned with their flat side down and their pointed tip facing upwards. This means the base of the cone lies parallel to the ground plane, with the cone's height extending along the z-axis and the radius along the x and y axes.

Decide a semantic name for the block for the role it represents in the structure. 
Decide the colors of each block to look like a {to_build}. Color is an rgba array with values from 0 to 1.
"""
    ]

def order_blocks_prompts(to_build):
    return f"""
Given the blocks described in the plan, I will place and stack these blocks one at a time by lowering them from a very tall height.

Please describe the sequence in which the blocks should be placed to correctly form a {to_build} structure. This means that blocks at the bottom should be placed first, followed by the higher blocks, so that the upper blocks can be stacked on top of the lower ones. Also note that it is difficult to stack blocks on top of a cone, so avoid placing blocks directly on top of cones.

For each block, specify whether it will be placed on the ground or on top of another block. If a block will be supported by multiple blocks, mention all of them. Ensure that the blocks are placed in a way that they remain stable and won't topple over when the physics simulation is run. Blocks cannot hover without support.
""".strip()

def decide_positions_prompts(to_build):
    return f"""
With the stacking order determined, I now need to know the x and y positions, as well as the yaw angle (in degrees), for each block to build a {to_build} structure.

The x and y coordinates should represent the center of each block. The yaw angle refers to the rotation around the z-axis in degrees. Remember, you can swap the dimensions of blocks to adjust their configuration.

Ensure that blocks at similar heights in the structure are spaced out in x and y so that they don't collide.

Make sure the structure is roughly centered at the origin (0, 0), and that each block stacks correctly on the specified blocks (or the ground). Every block must have a stable base to prevent it from falling. 

Consider the dimensions of the blocks when determining the x, y positions. Provide your reasoning for the chosen x and y positions and the yaw angle for each block.

Output a JSON following this format:
{markdown_json(
    [
        {
            "name": "support1",
            "shape": "cylinder",
            "dimensions": {"radius": 20, "height": 40},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": -50, "y": 0},
            "yaw": 0,
        },
        {
            "name": "support2",
            "shape": "cylinder",
            "dimensions": {"radius": 20, "height": 40},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": 50, "y": 0},
            "yaw": 0,
        },
        {
            "name": "deck",
            "shape": "cuboid",
            "dimensions": {"x": 100, "y": 50, "z": 20},
            "color": [0.5, 0.5, 0.5, 1],
            "position": {"x": 0, "y": 0},
            "yaw": 45,
        },
    ]
)}
"""

# Functions to parse JSONs to extract block information
def process_available_blocks(blocks):
    available_blocks = []
    for block_name, block in blocks.items():
        block_shape = block["shape"]
        block_dimensions = block["dimensions"]
        number_available = block["number_available"]
        available_blocks.append(
            {
                "shape": block_shape,
                "dimensions": block_dimensions,
                "number_available": number_available,
            }
        )
    return available_blocks
def blocks_from_json(json_data):
    blocks = []
    for block_data in json_data:
        if block_data["shape"] == "cuboid":
            dimensions = [
                block_data["dimensions"]["x"],
                block_data["dimensions"]["y"],
                block_data["dimensions"]["z"],
            ]
        elif block_data["shape"] == "cylinder" or block_data["shape"] == "cone":
            dimensions = [
                block_data["dimensions"]["radius"],
                block_data["dimensions"]["height"],
            ]
        else:
            raise ValueError(f"Invalid shape {block_data['shape']}")

        try:
            ori = p.getQuaternionFromEuler([0, 0, np.radians(block_data["yaw"])])
        except:
            ori = block_data["orientation"]

        try:
            name = block_data["name"]
        except:
            name = block_data["gpt_name"]

        try:
            position = [block_data["position"]["x"],block_data["position"]["y"],block_data["position"]["z"]]
        except:
            try:
                position = [block_data["position"][0],block_data["position"][1],1 * 1000]
            except:
                position = [block_data["position"]["x"], block_data["position"]["y"], 1 * 1000]
        block = Block(
            id=999,  # id gets updated by place blocks call, otherwise it's unknown
            gpt_name=name,
            block_name="",
            shape=block_data["shape"],
            dimensions=dimensions,
            position=position,
            orientation=ori,
            color=block_data["color"],
        )
        blocks.append(block)
    return blocks

# Full Judge module functionality
def get_outcome_reason(to_build, gif_path, json_file,seq_iter,structure_list,missing_blocks):
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)

    # Prepare blockset
    blockset = load_from_json(json_file)
    available_blocks = process_available_blocks(blockset)

    # Prompt GPT with GIF + system and user prompts
    system_prompt, user_prompt = make_missing_suggestion_prompts(to_build, available_blocks,structure_list=structure_list,missing_blocks=missing_blocks)
    response, _ = prompt_with_caching(
        messages_and_images=user_prompt,
        context=[],
        save_dir=structure_dir,
        name="missing_suggestions",
        cache=True,
        temperature=0.4, #0.25
        i=seq_iter,
        img_path=gif_path[0],
        system_message=system_prompt
    )
    missing_suggestions = extract_code_from_response(response, lang="json", last_block_only=False)
    print(" ============================== MISSING + SUGGESTIONS ================================== ")
    print(missing_suggestions)
    print(" ================================================================ ")
    return missing_suggestions

# Full Replanner module functionality
def replan_assembly(to_build,json_file,outcome_reason,plans_so_far,seq_iter):
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)

    # Prepare blockset
    blockset = load_from_json(json_file)
    available_blocks = process_available_blocks(blockset)

    # 1 - Replan submodule
    prompt = make_replan_prompt(to_build, available_blocks, outcome_reason, plans_so_far)
    response, main_context = prompt_with_caching(
        prompt,
        [],
        structure_dir,
        "re_plan",
        cache=True,
        temperature=0.5,
        i=seq_iter,
    )
    print(" ============================== REPLAN ================================== ")
    print(response)
    print(" ================================================================ ")

    # 2 - Order submodule
    prompt = order_blocks_prompts_re(to_build, replan_response=response)
    response, main_context = prompt_with_caching(
        prompt,
        main_context,
        structure_dir,
        "order_replan",
        cache=True,
        temperature=0.5,
        i=seq_iter,
    )
    print(response)

    # 3 - Position submodule
    prompt = decide_positions_prompts_re(to_build,order_plan=response)
    response, main_context = prompt_with_caching(
        prompt,
        main_context,
        structure_dir,
        "positions_replan",
        cache=True,
        temperature=0.25,
        i=seq_iter
        )
    print(response)

    json_output = get_last_json_as_dict(response)
    structure_dir_name = (to_build.lower()).replace(' ', '-')
    os.makedirs(f"./gpt_caching/{structure_dir_name}/replan_assembly_{seq_iter}/", exist_ok=True)
    save_to_json(json_output, f"./gpt_caching/{structure_dir_name}/replan_assembly_{seq_iter}/position_plan.json")

    blocks = blocks_from_json(json_output)
    structure = Structure()
    structure.add_blocks(blocks)
    structure.place_blocks()
    isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
    img = Image.fromarray(isometric_img)
    # img.save(f"{structure_dir}/{to_build_slug}_{seq_iter}.png")

    assembly = Assembly(
        structure=structure,
        structure_directory=structure_dir,
        to_build=to_build,
        isometric_image=img,
        available_blocks_json=available_blocks,
        assembly_num=seq_iter,
        eval_rating=None,
        eval_guesses=None,
    )
    assembly.save_to_structure_dir()
    return assembly

# Generate initial BloxNet assembly
def generate_structure(to_build, available_blocks, iter=0):
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)
    os.makedirs(structure_dir, exist_ok=True)

    # Prepare blockset
    available_blocks = process_available_blocks(available_blocks)

    # 1- make description
    prompt = make_description_prompt(to_build)
    response, _ = prompt_with_caching(
        prompt,
        [],
        structure_dir,
        "description",
        cache=True,
        temperature=0.5,
        i=iter,
    )
    print(response)

    # 2 - make plan
    prompt = make_plan_prompt(to_build, available_blocks, response)
    response, main_context = prompt_with_caching(
        prompt,
        [],
        structure_dir,
        "main_plan",
        cache=True,
        temperature=0.5,
        i=iter,
    )
    print(response)

    # 3.1 - decide ordering of blocks
    prompt = order_blocks_prompts(to_build)
    response, main_context = prompt_with_caching(
        prompt,
        main_context,
        structure_dir,
        "order_plan",
        cache=True,
        temperature=0.5,
        i=iter,
    )
    print(response)

    # 3.2 - decide positions
    prompt = decide_positions_prompts(to_build)
    response, main_context = prompt_with_caching(
        prompt,
        main_context,
        structure_dir,
        "positions_plan",
        cache=True,
        temperature=0.5,
        i=iter,
    )
    print(response)

    json_output = get_last_json_as_dict(response)
    blocks = blocks_from_json(json_output)

    structure = Structure()
    structure.add_blocks(blocks)
    structure.place_blocks()
    isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
    img = Image.fromarray(isometric_img)
    img.save(f"{structure_dir}/{to_build_slug}_{iter}.png")

    save_to_json(structure.get_json(), f"{structure_dir}/{to_build_slug}.json")

    assembly = Assembly(
        structure=structure,
        structure_directory=structure_dir,
        to_build=to_build,
        isometric_image=img,
        available_blocks_json=available_blocks,
        assembly_num=iter,
        eval_rating=None,
        eval_guesses=None,
    )
    assembly.save_to_structure_dir()
    return assembly

# Custom function to save rendered assembly - all figures in report
def save_pybullet_img_position_plan(json_output, structure_dir_name,seq_iter):
    blocks = blocks_from_json(json_output)
    structure = Structure()
    structure.add_blocks(blocks)
    structure.place_blocks()
    isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
    img = Image.fromarray(isometric_img)
    img.save(f"./gpt_caching/{structure_dir_name}/{structure_dir_name}_pp_{seq_iter}.png")