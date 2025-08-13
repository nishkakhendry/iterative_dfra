import json
import os
import numpy as np
import pybullet as p

import matplotlib.pyplot as plt
from PIL import Image

from bloxnet.utils.utils import (
    get_last_json_as_dict,
    save_to_json,
    slugify,
    markdown_json,
)
from bloxnet.pybullet.pybullet_images import get_imgs
from bloxnet.prompts.prompt import prompt_with_caching
from bloxnet.structure import Structure, Block, Assembly
from bloxnet.utils.utils import load_from_json, extract_code_from_response


SAVE_DIR = "gpt_caching"

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

# def make_topple_prompt(to_build,structure_list):
#     system_prompt = f"""
#         You are a highly skilled vision and reasoning assistant. You analyze sequences of simulation frames of a robotic arm assembling a {to_build} using blocks to detect block topples with precision.
#         Only report a topple if it clearly occurs after placement and the block was expected to remain elevated based on the plan. Be cautious, deliberate, and grounded in visible evidence.
#     """.strip()
#     user_prompt = f"""
#     Observe the frames of this video carefully. Ignore all blocks away from the centre of the table and reason about whether any of the blocks placed by the robot topple out of place after their placement.
#     A topple is defined as a block falling down to the table when the following plan states its target position is at a height above the table on top of the structure greater than 30: {structure_list}
#
#     Say all blocks are stable if no topple occurs. If any topple occurs, is it towards the beginning or the end of the video? And roughly where with reference to the structure?
#
#     Finally, look at the last frame. Critically assess whether any blocks have fallen away from the structure and are now lying near it on the table, and comment on the final state of the block assembly of a {to_build}.
#
#     Return your evaluation in this JSON format:
#     {markdown_json({"description_topple": "In this video..."})}
#     """.strip()
#     return system_prompt, user_prompt

def make_topple_prompt(to_build,structure_list):
    system_prompt = f"""You are a visual reasoning assistant tasked with evaluating the structural stability of a block assembly performed by a robotic arm.

        You will be shown two sequences of synchronized images:
        - One from the front-left diagonal corner of the table.
        - One from the back-right diagonal corner of the table.

        These sequences show the assembly process from two opposing viewpoints.

        You are also given a JSON action plan that lists the intended block placements, including the position and height of each block.

        Your task:
        - Carefully analyze the progression of both image sequences.
        - Focus only on blocks placed at or near the center of the table.
        - Determine if any block placed by the robot **topples** after being placed. A topple is defined as:
          - A block falling from its intended resting position to the tabletop surface.
          - Particularly when the plan specifies its position at a height (`z`) **greater than 10 units**.
        - Use **both viewpoints** to resolve occlusions and confirm motion or instability.
        - Report a topple if it is clearly observed from one view, and ideally cross-confirmed from both.
        - If no topple occurs, clearly state that all blocks remain stable.

        Return your assessment in the following JSON format:
        {markdown_json({"description_stability": "<your judgement here>"})}
    """.strip()
    user_prompt = f"""
    Here are synchronized image sequences from two diagonal corners of the assembly scene. Use both views to check whether any of the blocks placed by the robot fall or topple out of place after being placed. Only consider blocks
    located near the center of the table, and use the action plan to determine expected target heights.

    A topple is defined as a block visibly falling down to the table, especially when its planned position has a height (`z`) greater than 30 units.

    Here is the JSON list of planned actions: {structure_list}
    """.strip()
    return system_prompt, user_prompt

# def make_topple_prompt(to_build,structure_list):
#     system_prompt = f"""
#     You are a highly skilled robotic vision and reasoning assistant. You analyze sequences of simulation frames showing a robotic arm assembling a {to_build} using green blocks.
#
#     The input is a visual grid of video frames captured during a simulation. Standing facing the robot on the opposite end of the table - one video is captured from the front right corner while the other is captured from the left of the table.
#     Each frame shows a moment in time, ordered left to right, top to bottom. The robot sequentially places blocks as part of a construction plan.
#
#     Your task is to carefully identify any topple events. A topple is defined as a block that falls to the table after being placed, **despite being planned to stay elevated above a height of 30**. Only consider blocks near the center of the table and ignore stray blocks placed far from the structure.
#
#     Be precise, cautious, and only report a topple if it is clearly visible in the frame sequence. Compare the final frame to the intended plan to assess the overall assembly state.
#     """.strip()
#
#     user_prompt = f"""
#     You are shown two stitched sequences of simulation frames that capture a robotic arm building a structure using green blocks. One video is captured from the front right corner while the other is captured from the left of the table
#     facing the robot. Two videos are provided to reduce ambiguity in detecting topples and assessing final structure. The frames are ordered from top-left to bottom-right in temporal order, like reading a page.
#
#     Each image shows the robot's progress. The robot is building the structure in the centre of the table.
#
#     Use the following construction plan, where any block with a **z-position greater than 30** is intended to be placed above the base and remain elevated: {structure_list}
#
#     Your task:
#     1. Identify if any placed blocks **fall to the table** later in the sequence, even though their target z-position in the plan is greater than 30.
#     2. Say "all blocks are stable" if no such topple occurs.
#     3. If a topple does occur:
#        - Specify whether it happens toward the beginning or end of the sequence.
#        - Describe where it occurs with respect to the structure.
#
#     Finally, critically analyze the **last frame** in the sequence:
#     - Are any blocks visibly fallen near the structure that were supposed to remain in place?
#     - Is the overall structure complete and stable, or are there missing or toppled elements?
#
#     Return your full evaluation in this JSON format:
#     {markdown_json({"description_topple": "In this video..."})}
#     """.strip()
#
#     return system_prompt, user_prompt

# def make_topple_prompt(to_build, structure_list):
#     system_prompt = f"""You are a visual reasoning assistant tasked with evaluating the structural stability of a block assembly performed by a robotic arm.
#
#         You will be shown two sequences of synchronized images:
#         - One from the front-left diagonal corner of the table.
#         - One from the back-right diagonal corner of the table.
#
#         These sequences show the assembly process from two opposing viewpoints.
#
#         You are also given a JSON action plan that lists the intended block placements, including the position and height of each block.
#
#         Your task:
#         - Carefully analyze the progression of both image sequences.
#         - Focus only on blocks placed at or near the center of the table.
#         - Use **both viewpoints** to resolve occlusions and confirm motion or instability.
#
#         Compare the final positions of each block to the target positions in this plan: {structure_list}. Mention any blocks which have a target height of greater than 30 units which have their final position on the table.
#
#         Return your assessment in the following JSON format:
#         {markdown_json({"description_stability": "<your judgement here>"})}
#     """.strip()
#     user_prompt = f"""
#     Here are synchronized image sequences from two diagonal corners of the assembly scene. Use the final frame in both views to check whether any of the blocks placed by the robot fell or toppled out of place after being placed. Only consider blocks
#     located near the center of the table, and use the action plan to determine expected target heights.
#
#     A topple is defined as a block visibly falling down to the table, especially when its planned position has a height (`z`) greater than 30 units. Mention any blocks which have a target height of greater than 30 units which have their final position on the table.
#
#     Here is the JSON list of planned actions: {structure_list}
#     """.strip()
#     return system_prompt, user_prompt

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


def get_stability_correction(
    to_build, unstable_block: Block, pos_delta, structure_json, x_img, y_img
):
    return [
        f"""
{markdown_json(structure_json)}

While building the {to_build} by placing blocks one at a time in the order you specified by the JSON above, I noticed that block {unstable_block.gpt_name} is unstable and falls. 
The block moved by {pos_delta[0]:.2f} mm in the x direction and {pos_delta[1]:.2f} mm in the y direction.
Please adjust the position of block {unstable_block.gpt_name} (And potentially other blocks) to make the structure more stable.
Make sure every block has a stable base to rest on.

Output the JSON with your corrections following the same format and provide some reasoning for the changes you made. Feel free to correct other parts of the structure if they appear incorrect or to add, change, or remove blocks.

Here is an orthographic image of the side view of the structure with the y-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
""",
        x_img,
        f"""
Here is an orthographic image of the side view of the structure with the x-axis pointing to the right and the z-axis pointing up. {unstable_block.gpt_name} is highlighted in red while the other blocks are colored in white.
""",
        y_img,
        f"""
Describe what you see in these images and use them to help inform your correction. Then, provide the ouptut JSON in the proper format.
""",
    ]


def get_structure_info(img):
    return [
        img,
        f"""
I am currently building a structure made out of toy blocks shown in the given image. Describe in detail as much as you can about this image. Please list the top 10 things that the structure most resembles in order of similarity.

After describing the image in detail and providing some initial thoughts, answer as JSON in the following format providing 10 guesses. Your guesses should mostly be single words like "bottle" and never use adjectives like "toy_bottle". 
{
    markdown_json({"guesses": ["guess_1", "guess_2", "guess_3", "guess_4", "guess_5", "guess_6", "guess_7", "guess_8", "guess_9", "guess_10"]})
}
""".strip(),
    ]


def get_structure_rating(to_build):
    return f"""
Given your description of the block structure, how well does the structure in the image use blocks to resemble a {to_build} considering that the structure is made from  a limited set of toy blocks? 
Rate the resemblance of the block structure to a {to_build} on a scale of
 1 to 5 defined by the following:
1 - the structure in the image has no resemblance to the intended structure. It's missing all key features and appears incoherent
2 - the structure in the image has a small amount of resemblance to the intented structure. It has at least 1 key feature and shows an attempt at the intended structure
3 - the structure has clear similarities to the intended structure and appears coherent. It has at least 1 key feature and shows similarities in other ways as well.
4 - the structure represents multiple key features of the intended design and shows a decent overall resemblance.
5 - the structure in the image is a good block reconstruction of the intended structure, representing multiple key features and showing an overall resemblance to the intended structure.

Provide a brief explanation of your thought process then provide your final response as JSON in the following format:
{markdown_json({"rating": 5})}
"""


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
            # position=[
            #     block_data["position"]["x"],
            #     block_data["position"]["y"],
            #     1 * 1000,
            # ],
            orientation=ori,
            color=block_data["color"],
        )
        blocks.append(block)

    return blocks


def stability_check(blocks, debug=False):
    for i in range(len(blocks)):
        structure = Structure()
        structure.add_blocks(blocks[: i + 1])
        structure.place_blocks()

        last_block = blocks[i]

        x_img, y_img = get_imgs(
            keys=["x", "y"], axes=False, labels=False, highlight_id=last_block.id
        )

        stable, pos_delta, rot_delta = structure.check_stability(
            blocks[i].id, debug=debug
        )

        pos_delta = 1000 * np.array(pos_delta)

        if not stable:
            return False, last_block, pos_delta, x_img, y_img

    return True, None, None, x_img, y_img


def get_system_message():
    return f"""
You are an expert in creating block constructions with experience building many objects and creating stable structures.

Cuboid dimensions x, y, z define the size of the block, but you can swap them around to adjust the block's orientation. For example, if a block needs to be placed "vertically" then it's longest axis should be the z-axis. So, if a block is listed as having dimensions {{x: 50, y: 10, z: 10}} you can instead write the dimensions as {{x: 10, y: 10, z: 50}} listing z as the longest axis. It can also be important to pay attention to switching x and y to conform with the rest of your structure.

Cylinders are always upright with the z-axis as the height and the radius is the same along the x and y axes. 

Don't make structures which are too complex or try to show fine detail. Instead, focus on showing the broader structure.
""".strip()


TEMPERATURE = 0.5

def get_outcome_reason(to_build, gif_path, json_file,seq_iter,structure_list,missing_blocks):
    # make directory to save structure
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)

    # prepare blockset
    blockset = load_from_json(json_file)
    available_blocks = process_available_blocks(blockset)

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

def replan_assembly(to_build,json_file,outcome_reason,plans_so_far,seq_iter):
    # make directory to save structure
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)

    # prepare blockset
    blockset = load_from_json(json_file)
    available_blocks = process_available_blocks(blockset)

    # 1 - re plan
    prompt = make_replan_prompt(to_build, available_blocks, outcome_reason, plans_so_far)
    response, main_context = prompt_with_caching(
        prompt,
        [],
        structure_dir,
        "re_plan",
        cache=True,
        temperature=TEMPERATURE,
        i=seq_iter,
    )
    print(" ============================== REPLAN ================================== ")
    print(response)
    print(" ================================================================ ")

    # 2.1 - decide ordering of blocks
    prompt = order_blocks_prompts_re(to_build, replan_response=response)
    response, main_context = prompt_with_caching(
        prompt,
        main_context,
        structure_dir,
        "order_replan",
        cache=True,
        temperature=TEMPERATURE,
        i=seq_iter,
    )
    print(response)

    # 2.2 - decide positions
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
    img.save(f"{structure_dir}/{to_build_slug}_{seq_iter}.png")

    save_to_json(structure.get_json(), f"{structure_dir}/{to_build_slug}.json")

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

def generate_structure(to_build, available_blocks, iter=0):
    # make directory to save structure
    to_build_slug = slugify(to_build)
    structure_dir = os.path.join(SAVE_DIR, to_build_slug)
    os.makedirs(structure_dir, exist_ok=True)

    # prepare blockset
    available_blocks = process_available_blocks(available_blocks)

    # 1- make description
    prompt = make_description_prompt(to_build)
    response, _ = prompt_with_caching(
        prompt,
        [],
        structure_dir,
        "description",
        cache=True,
        temperature=TEMPERATURE,
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
        temperature=TEMPERATURE,
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
        temperature=TEMPERATURE,
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
        temperature=TEMPERATURE,
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

    # for i in range(2):
    #
    #     stable, unstable_block, pos_delta, x_img, y_img = stability_check(
    #         blocks, debug=True
    #     )
    #
    #     if stable:
    #         break
    #
    #     prompt = get_stability_correction(
    #         to_build, unstable_block, pos_delta, json_output, x_img, y_img
    #     )
    #     response, stability_context = prompt_with_caching(
    #         prompt, [], structure_dir, f"stability_correction_{iter}", cache=True, i=i
    #     )
    #
    #     json_output = get_last_json_as_dict(response)
    #     blocks = blocks_from_json(json_output)
    #
    #     structure = Structure()
    #     structure.add_blocks(blocks)
    #     structure.place_blocks()
    #     isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
    #     img = Image.fromarray(isometric_img)
    #     img.save(
    #         f"{structure_dir}/{to_build_slug}_stability_correction_{iter}_{i}.png"
    #     )

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


def save_pybullet_img_position_plan(json_output, structure_dir_name,seq_iter):
    blocks = blocks_from_json(json_output)
    structure = Structure()
    structure.add_blocks(blocks)
    structure.place_blocks()
    isometric_img = get_imgs(keys=["isometric"], axes=True, labels=False)
    img = Image.fromarray(isometric_img)
    img.save(f"./gpt_caching/{structure_dir_name}/{structure_dir_name}_pp_{seq_iter}.png")