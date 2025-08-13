import random
import ast
import openai
import base64
import os
import re

# List of objects, including target structures and related items across different themes
objects = [
    # Provided example objects
    "Giraffe", "Taj Mahal", "Jungle", "Ceiling Fan", "Robot", "House", "Sofa", "Shark", "Ship",
    "Piano", "Eiffel Tower", "Sheep", "Bridge", "Bicycle", "Dining Table Set",
    "Leaning Tower of Pisa", "Filament Roll", "Table", "Wedding Cake",

    # Related to Giraffe
    "Elephant", "Zebra", "Lion", "Cheetah", "Rhino", "Antelope", "Hippo", "Gazelle", "Savanna", "Acacia Tree",

    # Related to Taj Mahal
    "Pyramids of Giza", "Statue of Liberty", "Great Wall of China", "Sydney Opera House", "Big Ben", 
    "Petra", "Machu Picchu", "Angkor Wat", "Colosseum", "Burj Khalifa",

    # Related to Jungle
    "Monkey", "Tiger", "Parrot", "Toucan", "Snake", "Frog", "Liana Vine", "Bamboo", "Waterfall", "Orchid",

    # Related to Ceiling Fan
    "Table Lamp", "Chandelier", "Air Conditioner", "Heater", "Light Bulb", "Wall Clock", "Curtains", "Window", "Door", "Carpet",

    # Related to Robot
    "Drone", "3D Printer", "Factory Arm", "Vacuum Robot", "AI Chip", "Circuit Board", "Sensor", "Remote Control", "Battery Pack", "Power Adapter",

    # Related to House
    "Cottage", "Apartment", "Bungalow", "Mansion", "Garage", "Fence", "Mailbox", "Driveway", "Garden", "Roof",

    # Related to Sofa
    "Armchair", "Coffee Table", "Bookshelf", "TV Stand", "Side Table", "Rug", "Floor Lamp", "Cushion", "Blanket", "Footstool",

    # Related to Shark
    "Whale", "Dolphin", "Octopus", "Coral Reef", "Sea Turtle", "Jellyfish", "Squid", "Seal", "Penguin", "Clownfish",

    # Related to Ship
    "Sailboat", "Yacht", "Cargo Ship", "Submarine", "Cruise Liner", "Fishing Boat", "Harbor", "Anchor", "Lifebuoy", "Dock",

    # Related to Piano
    "Guitar", "Violin", "Drums", "Flute", "Trumpet", "Harp", "Cello", "Saxophone", "Clarinet", "Microphone",

    # Related to Eiffel Tower
    "Arc de Triomphe", "Notre Dame", "Louvre", "Paris Street Lamp", "French Café Table", "Metro Sign", "Carousel", "Seine River", "French Flag", "Baguette",

    # Related to Sheep
    "Cow", "Goat", "Horse", "Pig", "Chicken", "Duck", "Farmhouse", "Tractor", "Hay Bale", "Barn",

    # Related to Bridge
    "Suspension Bridge", "Drawbridge", "Footbridge", "Aqueduct", "Arch Bridge", "Stone Bridge", "Wooden Bridge", "Bridge Tower", "Bridge Cables", "Bridge Deck",

    # Related to Bicycle
    "Motorcycle", "Tricycle", "Scooter", "Unicycle", "BMX", "Helmet", "Bike Lock", "Bike Rack", "Pump", "Water Bottle",

    # Related to Dining Table Set
    "Chair", "Dinner Plate", "Glass", "Wine Glass", "Knife", "Fork", "Spoon", "Napkin", "Tablecloth", "Serving Bowl",

    # Related to Leaning Tower of Pisa
    "Bell Tower", "Clock Tower", "Cathedral", "Plaza", "Tourist Camera", "Souvenir", "Guidebook", "Italian Flag", "Pasta Dish", "Gelato",

    # Related to Table
    "Desk", "Workbench", "Countertop", "Picnic Table", "Folding Table", "Sideboard", "Kitchen Island", "Study Table", "Console Table", "Vanity",

    # Related to Wedding Cake
    "Cupcake", "Macaron Tower", "Champagne Glass", "Bouquet", "Wedding Dress", "Tuxedo", "Wedding Ring", "Bride", "Groom", "Dance Floor",

    # Misc
    "Banquet Hall", "Stage Lights", "City Skyline", "Orchestra", "Picnic Basket", "Ice Sculpture", "Lantern", "Fireworks", "Street Food Stall", "Fountain"
]

# Function to choose x unique random items from a given list items
def choose_random_items(items, x):
    if x > len(items):
        raise ValueError("x cannot be greater than the number of items in the list.")
    return random.sample(items, x)

# Function to prompt GPT-4o with an image and list of objects, and return ranked resemblance
def prompt_gpt(img_path, obj_list):
    # Encode image in base64
    with open(img_path, "rb") as img:
        b64 = base64.b64encode(img.read()).decode()

    # Prompt GPT-4o with image and unordered object list
    response = openai.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": 'You are an expert at visually recognising structures in images. Given an image and a list of object names, sort the list in descending order of semantic resemblance to the image. The first item must be the most similar, the last item the least similar. ALWAYS output **ONLY** this reordered list in Python list format -- with no markers padding or explanation text at all.'},
            {"role": "user", "content": [
                                {"type": "text", "text":  f"Object list to be ranked: {obj_list}"},
                                {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{b64}", 'detail':"high"} } ] }
    ])
    
    try:
        # Try to directly parse the response as a Python list
        parsed = ast.literal_eval(response.choices[0].message.content)
    except:
        # If GPT wraps output in a code block, extract and parse the code block
        pattern = rf"```{'python'}\s*\n(.*?)```"
        code_blocks = re.findall(pattern, response.choices[0].message.content, re.DOTALL)
        parsed = ast.literal_eval(code_blocks[-1])
    print("Ranked list: ", parsed)
    return parsed

if __name__ == "__main__":
    # Directory containing rendered assembly images
    # img_dir = "C:/Users/nishk/OneDrive/Desktop/DT/Writing/chosen_assemblies_title_names/"
    img_dir = "C:/Users/nishk/OneDrive/Desktop/DT/Writing/bloxnet_assemblies/"

    # List of representative target structures to evaluate
    assembly_obj_list = ["Bridge", "Burger","Ceiling Fan", "Couch","Dining Table Set", "Eiffel Tower", "Giraffe", "House","Jungle","Piano","Robot","Shark","Sheep","Ship", "Taj Mahal"] 

    x = 20  # Number of choices (including target) to present in ranking

    obj_ranks = {}
    # Iterate over each target structure
    for object_to_keep in assembly_obj_list:
        # Construct image file path for current assembly
        img_path = os.path.join(img_dir,object_to_keep+".png")

        # Create a list with the target and x-1 random distractors
        selected = [object_to_keep] + choose_random_items(objects, x-1)
        random.shuffle(selected)
        print("Object: ", object_to_keep)
        print("Random list: ", selected)

        # Run the GPT-4o ranking and get position of target structure
        ranked = prompt_gpt(img_path=img_path, obj_list=selected)
        obj_rank = ranked.index(object_to_keep)
        print("Assigned Rank: ", obj_rank+1)

        # Store rank for evaluation
        obj_ranks[object_to_keep] = obj_rank+1

    print("All rankings: ", obj_ranks)

    # Compute and print evaluation metrics
    top1_count = sum(1 for r in obj_ranks.values() if r == 1)
    total = len(obj_ranks)
    top1_accuracy = top1_count / total
    print(f"Top-1 Accuracy %: {top1_accuracy*100}")

    avg_rank = sum(obj_ranks.values()) / len(obj_ranks)
    print("Average rank: ", avg_rank) 

    rel_rank = sum(v / x for v in obj_ranks.values()) / len(obj_ranks)
    print("Relative rank %: ", rel_rank*100) 