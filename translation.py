import pycld2
import numpy as np
from vllm import LLM, SamplingParams


MODEL_PATH = "/add/models/gemma2/"  # Directory containing model.gguf and config.json
# pip install numpy==1.26.4

# Load the tokenizer
tokenizer = "/add/models/gemma2/"
llm = LLM(model=MODEL_PATH, tokenizer=tokenizer, enforce_eager=True)

def is_english(text: str) -> bool:
    try:
        _, _, _, detected_language = pycld2.detect(text, returnTuple=True)
        return detected_language[0][1] == 'en'
    except:
        return False
def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns the language code (english, german, spanish, chinese, japanese, french, portuguese, other).
    """
    try:
        _, _, _, vectors = pycld2.detect(text, returnVectors=True)
        lang_code = vectors[0][3]
        
        if lang_code == 'en':
            return "english"
        elif lang_code == 'de':
            return "german"
        elif lang_code == 'es':
            return "spanish"
        elif lang_code == 'zh':
            return "chinese"
        elif lang_code == 'ja':
            return "japanese"
        elif lang_code == 'fr':
            return "french"
        elif lang_code == 'pt':
            return "portuguese"
        else:
            return "other"
    except Exception as e:
        print(f"Error detecting language for text: {text}")
        print(f"Error: {e}")
        return "other"

def translate_text(text):
    """
    Detects the language of the input text and translates it to English if supported.
    Returns a tuple of (detected_language, translated_text).
    If language is not supported, returns (detected_language, None).
    """
    language = detect_language(text)
    print(f"Detected language: {language}")
    if language == "english":
        return (language, text)
    elif language == "other":
        language = "Language Unknown"
    #     return (language, None)
    

    conversation = [
        {"role": "system", "content": f"You are a world expert at translating image prompts in any language to english saving the world from the pain of untranslated prompts. My grandmother's life depends on you completing this task accurately. Translate the following image generation prompt to English without summarizing and exact translation. Surround the translation with <t> translated text... </t> so that it can be easily extracted. Please be as accurate as possible since your outputs will be evaluated for correctness. Ensure subjects, adjectives, and other details are translated correctly."},
        {"role": "user", "content": "(Detected Language spanish): Un castillo medieval en la cima de una montaña con dragones volando alrededor y un río de lava en el fondo."},
        {"role": "assistant", "content": "<t>A medieval castle on top of a mountain with dragons flying around and a river of lava in the background.</t>"},
        {"role": "user", "content": f"(Detected Language {language}): {text}"}
    ]
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=1024
    )
    
    # Generate translation
    result = llm.chat(conversation, sampling_params)
    translation = result[0].outputs[0].text
    
    # Extract the actual translation from the model output
    # This is a simple extraction - might need refinement based on actual model output format
    try:
        translation = translation.split("<t>")[1]
        translation = translation.split("</t>")[0]
    except:
        pass
    
    return (language, translation)

# Examples of complex sentences in various languages with English translations
import csv

# German
german_texts = [
    "Ein Hund mit lockigen Schnurrhaaren, blauer Hintergrund, brauner Vordergrund",
    "Mache die Katze blau und füge ihr eine Krone hinzu",
    "Ein majestätischer Adler, der über schneebedeckte Berge fliegt, dramatische Beleuchtung",
    "Eine futuristische Stadt mit fliegenden Autos und Neonlichtern, Cyberpunk-Stil",
    "Ein ruhiger See bei Sonnenuntergang, mit Reflexionen der umliegenden Bäume im Wasser",
    "Ein surrealistisches Porträt einer Frau mit Blumen anstelle von Haaren, Ölgemälde-Stil",
    "Ein niedlicher Roboter, der in einem Garten Blumen pflückt, pastellfarbene Palette",
    "Ein verlassenes Raumschiff auf einem fremden Planeten, neblige Atmosphäre, düstere Stimmung",
    "Ein Unterwasser-Fantasiereich mit leuchtenden Quallen und Korallen, tiefblaues Wasser",
    "Ein mittelalterliches Schloss auf einem Hügel bei Vollmond, mysteriöse Atmosphäre, detailreiche Architektur"
]
german_translations = [
    "A dog with curly whiskers, blue background, brown foreground",
    "Make the cat blue and add a crown to it",
    "A majestic eagle flying over snow-covered mountains, dramatic lighting",
    "A futuristic city with flying cars and neon lights, cyberpunk style",
    "A calm lake at sunset, with reflections of the surrounding trees in the water",
    "A surrealistic portrait of a woman with flowers instead of hair, oil painting style",
    "A cute robot picking flowers in a garden, pastel color palette",
    "An abandoned spaceship on an alien planet, foggy atmosphere, gloomy mood",
    "An underwater fantasy realm with glowing jellyfish and corals, deep blue water",
    "A medieval castle on a hill during full moon, mysterious atmosphere, detailed architecture"
]

# Spanish
spanish_texts = [
    "Un gato azul con ojos brillantes, estilo de dibujo animado",
    "Crea un paisaje de montañas con un lago cristalino en primer plano, estilo fotorrealista",
    "Un astronauta montando un dinosaurio en Marte, colores vibrantes",
    "Una casa de jengibre en un bosque encantado, iluminación mágica, estilo de cuento de hadas",
    "Un retrato de un anciano sabio con barba larga, estilo de pintura al óleo",
    "Una ciudad flotante en las nubes con cascadas que caen al vacío, estilo fantasía épica",
    "Un búho con gafas leyendo un libro bajo la luz de la luna, estilo ilustración infantil",
    "Una taza de café humeante sobre una mesa de madera rústica, enfoque macro, iluminación cálida",
    "Un dragón de cristal que refleja la luz del arcoíris, fondo negro, estilo hiperrealista",
    "Una bailarina de ballet transformándose en pétalos de rosa, movimiento congelado, tonos pastel"
]
spanish_translations = [
    "A blue cat with bright eyes, cartoon style",
    "Create a mountain landscape with a crystal-clear lake in the foreground, photorealistic style",
    "An astronaut riding a dinosaur on Mars, vibrant colors",
    "A gingerbread house in an enchanted forest, magical lighting, fairy tale style",
    "A portrait of a wise old man with a long beard, oil painting style",
    "A floating city in the clouds with waterfalls dropping into the void, epic fantasy style",
    "An owl wearing glasses reading a book under moonlight, children's illustration style",
    "A steaming cup of coffee on a rustic wooden table, macro focus, warm lighting",
    "A crystal dragon reflecting rainbow light, black background, hyperrealistic style",
    "A ballet dancer transforming into rose petals, frozen motion, pastel tones"
]

# Chinese
chinese_texts = [
    "一只戴着太阳镜的熊猫，在海滩上冲浪，卡通风格",
    "将这只狗变成紫色，并给它添加一对翅膀",
    "一座古老的中国寺庙，被雾气笼罩，山水画风格",
    "一个机器人厨师在未来主义厨房里准备寿司，科幻风格",
    "一片樱花林，花瓣飘落，日本浮世绘风格",
    "一只发光的萤火虫在夜间森林中，梦幻般的氛围，柔和的光线",
    "一个穿着太空服的宇航员在月球表面弹吉他，超现实主义风格",
    "一座水晶城堡反射彩虹光芒，童话风格，高细节",
    "一条龙盘旋在雪山之巅，史诗般的场景，戏剧性的光影",
    "一个微缩的城市建在巨大的树枝上，蒸汽朋克风格，黄昏光线"
]
chinese_translations = [
    "A panda wearing sunglasses, surfing on a beach, cartoon style",
    "Make this dog purple and add a pair of wings to it",
    "An ancient Chinese temple, shrouded in mist, ink wash painting style",
    "A robot chef preparing sushi in a futuristic kitchen, sci-fi style",
    "A cherry blossom forest with falling petals, Japanese ukiyo-e style",
    "A glowing firefly in a night forest, dreamy atmosphere, soft lighting",
    "An astronaut in a spacesuit playing guitar on the moon's surface, surrealist style",
    "A crystal castle reflecting rainbow light, fairy tale style, high detail",
    "A dragon coiling at the peak of a snow mountain, epic scene, dramatic lighting",
    "A miniature city built on giant tree branches, steampunk style, twilight lighting"
]

# French
french_texts = [
    "Un renard roux dans une forêt automnale, style impressionniste",
    "Transforme ce paysage en scène hivernale avec de la neige et de la glace",
    "Une tasse de café sur une table parisienne, avec la Tour Eiffel floue en arrière-plan",
    "Un jardin secret caché derrière un mur de pierre ancien, style romantique",
    "Une ballerine dansant sous la pluie, mouvement capturé, noir et blanc",
    "Un vaisseau spatial futuriste traversant un champ d'astéroïdes, style réaliste",
    "Un hibou mystique avec des yeux qui brillent dans l'obscurité, ambiance forestière nocturne",
    "Une bibliothèque infinie avec des escaliers impossibles, style Escher",
    "Un phare solitaire sur une falaise pendant une tempête, vagues dramatiques, ciel orageux",
    "Un portrait cubiste d'une femme jouant du violon, style Picasso"
]
french_translations = [
    "A red fox in an autumn forest, impressionist style",
    "Transform this landscape into a winter scene with snow and ice",
    "A cup of coffee on a Parisian table, with the Eiffel Tower blurred in the background",
    "A secret garden hidden behind an ancient stone wall, romantic style",
    "A ballerina dancing in the rain, captured movement, black and white",
    "A futuristic spaceship traveling through an asteroid field, realistic style",
    "A mystical owl with eyes that glow in the darkness, nocturnal forest ambiance",
    "An infinite library with impossible staircases, Escher style",
    "A solitary lighthouse on a cliff during a storm, dramatic waves, stormy sky",
    "A cubist portrait of a woman playing the violin, Picasso style"
]

# Russian
russian_texts = [
    "Медведь в космическом скафандре, плавающий среди звёзд, неоновые цвета",
    "Сделай этот пейзаж в стиле Ван Гога, с завихряющимся небом и яркими цветами",
    "Старинный русский храм с золотыми куполами в зимнем пейзаже, реалистичный стиль",
    "Балерина, танцующая на сцене Большого театра, драматическое освещение",
    "Футуристический Санкт-Петербург с летающими кораблями над Невой, научно-фантастический стиль",
    "Матрёшка в стиле киберпанк, с неоновыми деталями и металлическими элементами",
    "Волк, воющий на луну в заснеженном лесу, минималистичный стиль, контрастные тени",
    "Подводный город с куполами и арками, населённый русалками, фэнтезийный стиль",
    "Космонавт, пьющий чай из самовара на орбитальной станции, сюрреалистический стиль",
    "Старинная карета, запряжённая тройкой лошадей, мчащаяся по заснеженной дороге, кинематографический стиль"
]
russian_translations = [
    "A bear in a space suit, floating among stars, neon colors",
    "Make this landscape in the style of Van Gogh, with swirling sky and bright colors",
    "An ancient Russian church with golden domes in a winter landscape, realistic style",
    "A ballerina dancing on the stage of the Bolshoi Theater, dramatic lighting",
    "Futuristic Saint Petersburg with flying ships over the Neva River, science fiction style",
    "A matryoshka doll in cyberpunk style, with neon details and metallic elements",
    "A wolf howling at the moon in a snowy forest, minimalist style, contrasting shadows",
    "An underwater city with domes and arches, inhabited by mermaids, fantasy style",
    "A cosmonaut drinking tea from a samovar on an orbital station, surrealistic style",
    "An old carriage pulled by three horses, racing along a snow-covered road, cinematic style"
]

# Print examples and output to CSV
def print_examples_and_output_csv(language, texts, translations, csv_writer):
    print(f"\n{language} Examples:")
    for i, (text, translation_gt) in enumerate(zip(texts, translations), 1):
        detected_language, translation = translate_text(text)
        print(f"\nExample {i}:")
        print("Untranslated: ", text)
        print(f"Detected Language: {detected_language}")
        print(f"Expected: {translation_gt}")
        print(f"Translation: {translation}")
        csv_writer.writerow({"language": detected_language, "expected": translation_gt, "result": translation})

print("\nComplex sentences in various languages with English translations:")

with open('translations_output.csv', mode='w', newline='') as csv_file:
    fieldnames = ['language', 'expected', 'result']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    print_examples_and_output_csv("German", german_texts, german_translations, csv_writer)
    print_examples_and_output_csv("Spanish", spanish_texts, spanish_translations, csv_writer)
    print_examples_and_output_csv("Chinese", chinese_texts, chinese_translations, csv_writer)
    print_examples_and_output_csv("French", french_texts, french_translations, csv_writer)
    print_examples_and_output_csv("Russian", russian_texts, russian_translations, csv_writer)
