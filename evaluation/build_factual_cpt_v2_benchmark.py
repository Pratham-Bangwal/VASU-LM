"""Build the frozen, independently written factual-CPT v2 benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


OUTPUT = Path("evaluation/benchmarks/factual_cpt_v2.json")
SEED = 42
CREATED_AT = "2026-07-19T00:00:00+05:30"

# Each fact is independently phrased for this benchmark. The four options are
# deliberately short so raw and length-normalized likelihood remain auditable.
FACTS = [
    ("science", "Water freezes at zero degrees on the", " Celsius scale.", "Which temperature scale places water's freezing point at zero degrees?", ["Celsius", "Kelvin", "Fahrenheit", "Rankine"]),
    ("science", "Earth completes one orbit around the", " Sun.", "What body does Earth orbit?", ["Sun", "Moon", "Mars", "Polaris"]),
    ("science", "The molecule that carries hereditary information is", " DNA.", "Which molecule carries hereditary information?", ["DNA", "glucose", "water", "oxygen"]),
    ("science", "The planet commonly called the Red Planet is", " Mars.", "Which planet is called the Red Planet?", ["Mars", "Venus", "Mercury", "Neptune"]),
    ("science", "The atomic number of carbon is", " six.", "What is carbon's atomic number?", ["six", "eight", "twelve", "fourteen"]),
    ("science", "During photosynthesis, plants take in", " carbon dioxide.", "Which gas do plants take in during photosynthesis?", ["carbon dioxide", "helium", "argon", "hydrogen"]),
    ("science", "The largest organ of the human body is the", " skin.", "What is the largest human organ?", ["skin", "heart", "liver", "lung"]),
    ("science", "At standard sea-level pressure, water boils at", " 100 degrees Celsius.", "At sea level, at what Celsius temperature does water boil?", ["100 degrees", "0 degrees", "50 degrees", "212 degrees"]),
    ("science", "Light in a vacuum travels at about", " 300,000 kilometres per second.", "Approximately how fast does light travel in a vacuum?", ["300,000 kilometres per second", "300 kilometres per second", "30,000 kilometres per hour", "3,000 metres per second"]),
    ("science", "The green pigment important to photosynthesis is", " chlorophyll.", "Which green pigment is central to photosynthesis?", ["chlorophyll", "melanin", "keratin", "haemoglobin"]),
    ("history", "Magna Carta was sealed in the year", " 1215.", "In what year was Magna Carta sealed?", ["1215", "1066", "1492", "1776"]),
    ("history", "The Second World War ended in", " 1945.", "In which year did the Second World War end?", ["1945", "1918", "1939", "1963"]),
    ("history", "The ancient pyramids at Giza were built in", " Egypt.", "In which country are the pyramids at Giza?", ["Egypt", "Greece", "Mexico", "India"]),
    ("history", "The main language of ancient Rome was", " Latin.", "Which language was central in ancient Rome?", ["Latin", "Sanskrit", "Mandarin", "Arabic"]),
    ("history", "Johannes Gutenberg is associated with the movable-type", " printing press.", "Which invention is associated with Johannes Gutenberg?", ["printing press", "steam engine", "telephone", "telescope"]),
    ("history", "The French Revolution began in", " 1789.", "In what year did the French Revolution begin?", ["1789", "1689", "1815", "1914"]),
    ("history", "The first president of the United States was", " George Washington.", "Who was the first United States president?", ["George Washington", "Abraham Lincoln", "Thomas Edison", "Benjamin Franklin"]),
    ("history", "The European Renaissance began first in", " Italy.", "In which country did the European Renaissance begin?", ["Italy", "Norway", "Canada", "Japan"]),
    ("geography", "The capital city of Japan is", " Tokyo.", "What is Japan's capital?", ["Tokyo", "Kyoto", "Seoul", "Beijing"]),
    ("geography", "The largest ocean on Earth is the", " Pacific Ocean.", "Which is Earth's largest ocean?", ["Pacific Ocean", "Atlantic Ocean", "Indian Ocean", "Arctic Ocean"]),
    ("geography", "The Nile River is located in", " Africa.", "On which continent is the Nile River?", ["Africa", "Europe", "Australia", "South America"]),
    ("geography", "Mount Everest belongs to the", " Himalayas.", "Which mountain range contains Mount Everest?", ["Himalayas", "Andes", "Alps", "Rockies"]),
    ("geography", "The Sahara Desert lies in", " Africa.", "On which continent is the Sahara Desert?", ["Africa", "Asia", "Europe", "Antarctica"]),
    ("geography", "Brazil is part of", " South America.", "On which continent is Brazil?", ["South America", "North America", "Africa", "Asia"]),
    ("geography", "The capital of Australia is", " Canberra.", "What is Australia's capital city?", ["Canberra", "Sydney", "Melbourne", "Perth"]),
    ("geography", "The equator separates the Northern and", " Southern Hemispheres.", "Which two hemispheres are separated by the equator?", ["Northern and Southern", "Eastern and Western", "land and ocean", "polar and tropical"]),
    ("technology", "Binary notation uses the digits zero and", " one.", "Which digits are used in binary notation?", ["zero and one", "one and two", "zero through nine", "two and three"]),
    ("technology", "The abbreviation CPU means central processing", " unit.", "What does CPU stand for?", ["central processing unit", "computer power utility", "central program user", "coded processing update"]),
    ("technology", "HTTP is a protocol used to transfer resources on the", " web.", "What is HTTP primarily used for?", ["transferring web resources", "charging batteries", "printing photographs", "measuring temperature"]),
    ("technology", "RAM normally provides temporary", " working memory.", "What role does RAM normally serve?", ["temporary working memory", "permanent paper storage", "network cabling", "audio amplification"]),
    ("technology", "Python is a general-purpose programming", " language.", "What kind of tool is Python?", ["programming language", "database cable", "image sensor", "battery chemistry"]),
    ("technology", "TCP and IP are core protocols of the", " internet.", "TCP/IP is most closely associated with what?", ["internet communication", "mechanical engines", "paper publishing", "medical imaging"]),
    ("technology", "A transistor can act as an electronic", " switch.", "What can a transistor act as?", ["electronic switch", "wooden lever", "chemical fuel", "optical lens"]),
    ("technology", "A URL identifies the address of an online", " resource.", "What does a URL identify?", ["online resource", "processor temperature", "battery voltage", "keyboard layout"]),
    ("biography", "Marie Curie conducted pioneering research on", " radioactivity.", "What field is strongly associated with Marie Curie's research?", ["radioactivity", "plate tectonics", "linguistics", "computer graphics"]),
    ("biography", "Albert Einstein developed the theory of", " relativity.", "Which theory is associated with Albert Einstein?", ["relativity", "cell theory", "germ theory", "plate tectonics"]),
    ("biography", "Charles Darwin explained evolution through natural", " selection.", "Which mechanism is central to Darwin's account of evolution?", ["natural selection", "radioactive decay", "continental drift", "electromagnetism"]),
    ("biography", "Isaac Newton formulated laws of motion and", " gravity.", "Which subject is associated with Isaac Newton?", ["gravity", "vaccination", "radioactivity", "genetics"]),
    ("biography", "Ada Lovelace wrote an early published", " algorithm.", "What is Ada Lovelace noted for writing?", ["an algorithm", "a symphony", "a constitution", "a medical vaccine"]),
    ("biography", "Alan Turing made foundational contributions to", " computing.", "Which field received foundational work from Alan Turing?", ["computing", "geology", "botany", "architecture"]),
    ("biography", "Galileo Galilei used telescopic observations in", " astronomy.", "In which science did Galileo use telescopic observations?", ["astronomy", "ecology", "chemistry", "anatomy"]),
    ("biography", "Alexander Fleming discovered", " penicillin.", "What did Alexander Fleming discover?", ["penicillin", "insulin", "electricity", "helium"]),
    ("definitions", "A community of organisms interacting with its environment is an", " ecosystem.", "What term describes organisms interacting with their environment?", ["ecosystem", "algorithm", "parliament", "isotope"]),
    ("definitions", "A system in which citizens exercise political power is a", " democracy.", "What political system gives citizens political power?", ["democracy", "monopoly", "ecosystem", "equation"]),
    ("definitions", "The change of a liquid into a gas is called", " evaporation.", "What is the change from liquid to gas called?", ["evaporation", "freezing", "condensation", "melting"]),
    ("definitions", "A finite sequence of steps for solving a problem is an", " algorithm.", "What is a finite problem-solving procedure called?", ["algorithm", "ecosystem", "molecule", "currency"]),
    ("definitions", "Two or more atoms chemically bonded form a", " molecule.", "What can chemically bonded atoms form?", ["molecule", "continent", "protocol", "democracy"]),
    ("definitions", "A broad rise in the general price level is called", " inflation.", "What term describes a broad rise in prices?", ["inflation", "evaporation", "migration", "photosynthesis"]),
    ("definitions", "The variety of living organisms in an area is its", " biodiversity.", "What term means the variety of living organisms?", ["biodiversity", "velocity", "democracy", "encryption"]),
    ("definitions", "Energy from a naturally replenished source is called", " renewable energy.", "What is energy from a replenished natural source called?", ["renewable energy", "fossil fuel", "static charge", "thermal waste"]),
]

CONTINUATIONS = [
    ("science", "Careful observation is useful because", " it allows claims to be checked against evidence."),
    ("science", "When a measurement is repeated,", " the results can be compared for consistency."),
    ("science", "A clear scientific explanation should", " distinguish evidence from speculation."),
    ("history", "Historical records can disagree, so", " researchers compare sources and their context."),
    ("history", "A change in technology may affect society by", " altering work, communication, and daily life."),
    ("history", "To understand an old document, a reader should", " consider who wrote it and why."),
    ("geography", "Rivers reshape landscapes over time by", " eroding rock and carrying sediment downstream."),
    ("geography", "Climate varies between regions because", " latitude, elevation, oceans, and winds differ."),
    ("geography", "A reliable map should include", " a scale, labels, and a clear legend."),
    ("technology", "A secure password is stronger when", " it is long, unique, and difficult to guess."),
    ("technology", "Computers exchange data by", " following agreed rules called protocols."),
    ("technology", "Before installing software, a user should", " verify its source and review the requested permissions."),
    ("definitions", "A useful definition identifies", " the essential features that distinguish a concept."),
    ("definitions", "An example supports an explanation when", " it clearly illustrates the stated principle."),
    ("definitions", "A comparison is easiest to follow when", " the same criteria are applied to both subjects."),
    ("general language", "The library became quiet after", " the final group of students left for the evening."),
    ("general language", "Although the road was wet,", " the driver slowed down and arrived safely."),
    ("general language", "The meeting ended with", " a short summary of decisions and next steps."),
    ("general language", "After checking the instructions twice,", " Maya assembled the shelf without difficulty."),
    ("general language", "The gardener covered the young plants because", " a cold night was expected."),
    ("general language", "A well-organized report begins with", " a clear statement of its purpose."),
    ("general language", "When the train was delayed,", " passengers waited for an updated departure time."),
    ("general language", "The teacher asked for evidence, so", " the student cited the results of the experiment."),
    ("general language", "Because the room was too dark,", " Arun opened the curtains to let in daylight."),
    ("general language", "The recipe was simple enough that", " a beginner could follow every step."),
]

OPEN_TOPICS = [
    ("science", "Why do shadows change length during the day?"),
    ("science", "Describe one difference between weather and climate."),
    ("science", "Explain why clean water matters to health."),
    ("science", "What is one way evidence can challenge a hypothesis?"),
    ("history", "Why do historians compare several sources?"),
    ("history", "Describe one effect printing had on communication."),
    ("history", "What can an old map reveal about its time?"),
    ("history", "Why should dates in a historical claim be checked?"),
    ("geography", "How can a mountain range influence rainfall?"),
    ("geography", "Explain how a river can form a delta."),
    ("geography", "Why are coastal climates often milder?"),
    ("geography", "What information does a map scale provide?"),
    ("technology", "Explain the difference between hardware and software."),
    ("technology", "Why should important files be backed up?"),
    ("technology", "What does a network protocol do?"),
    ("technology", "Describe one risk of reusing a password."),
    ("biography", "What makes a scientific biography trustworthy?"),
    ("biography", "How can a biographer separate evidence from legend?"),
    ("biography", "Why is historical context useful in a biography?"),
    ("definitions", "Define biodiversity in ordinary language."),
    ("definitions", "Explain what an algorithm is with a simple example."),
    ("definitions", "What is the difference between observation and inference?"),
    ("general language", "Continue a short paragraph about preparing for a storm."),
    ("general language", "Write a coherent continuation about finding a lost key."),
    ("general language", "Continue a paragraph about a delayed journey."),
]


def build_benchmark() -> dict[str, Any]:
    cloze: list[dict[str, Any]] = []
    multiple_choice: list[dict[str, Any]] = []
    for index, (category, stem, target, question, options) in enumerate(FACTS):
        for variant in range(2):
            item_id = index * 2 + variant
            prompt = stem if variant == 0 else f"Complete the factual statement: {stem}"
            cloze.append({
                "id": f"cloze_{item_id:03d}",
                "category": category,
                "prompt": prompt,
                "target": target,
            })
            rotated = options[variant:] + options[:variant]
            answer_index = rotated.index(options[0])
            mc_prompt = question if variant == 0 else f"Select the correct answer. {question}"
            multiple_choice.append({
                "id": f"mc_{item_id:03d}",
                "category": category,
                "prompt": mc_prompt,
                "options": [f" {option}" for option in rotated],
                "answer_index": answer_index,
            })

    continuations = []
    for index, (category, context, target) in enumerate(CONTINUATIONS):
        for variant in range(2):
            prefix = context if variant == 0 else f"In clear prose, {context[0].lower()}{context[1:]}"
            continuations.append({
                "id": f"continuation_{index * 2 + variant:03d}",
                "category": category,
                "prompt": prefix,
                "target": target,
            })

    open_ended = []
    for index, (category, prompt) in enumerate(OPEN_TOPICS):
        open_ended.extend((
            {"id": f"open_{index * 2:03d}", "category": category, "prompt": prompt},
            {
                "id": f"open_{index * 2 + 1:03d}",
                "category": category,
                "prompt": f"Give a brief, cautious response: {prompt}",
            },
        ))
    payload = {
        "format_version": "vasu_factual_cpt_benchmark_v2",
        "benchmark_id": "factual_cpt_v2",
        "created_at": CREATED_AT,
        "seed": SEED,
        "provenance": "Independently written prompts; no Wikimedia sentences copied.",
        "cloze": cloze,
        "multiple_choice": multiple_choice,
        "continuations": continuations,
        "open_ended": open_ended,
    }
    assert len(cloze) == len(multiple_choice) == 100
    assert len(continuations) == len(open_ended) == 50
    return payload


def benchmark_bytes() -> bytes:
    return (json.dumps(build_benchmark(), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_benchmark(path: Path = OUTPUT) -> str:
    content = benchmark_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    digest = write_benchmark()
    print(f"benchmark={OUTPUT}")
    print(f"sha256={digest}")
    print("examples=300 (100 cloze, 100 multiple-choice, 50 continuation, 50 open)")


if __name__ == "__main__":
    main()
