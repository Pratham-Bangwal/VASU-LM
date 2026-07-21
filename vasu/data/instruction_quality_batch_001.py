"""Deterministic, purpose-written authoring for instruction-quality batch 001."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable
from urllib.parse import urlparse

from vasu.data.alpaca_masked_v2 import encode_source_example
from vasu.data.instruction_quality import (
    exact_duplicate_groups,
    near_duplicate_candidates,
    normalize_for_comparison,
    quality_score,
    validate_records,
    write_jsonl,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


AUTHORING_VERSION = "vasu_instruction_quality_batch_001_authoring_v1"
CREATED_AT = "2026-07-19T00:00:00Z"
EXPECTED_COUNTS = {
    "short_factual_qa": 125,
    "beginner_explanation": 100,
    "exact_format_following": 100,
    "rewriting_transformation": 75,
    "lists_structured_output": 50,
    "json_schema_output": 25,
    "uncertainty_honest_fallback": 25,
}
EXPECTED_DIFFICULTY = {"easy": 350, "medium": 150}


# URLs are aligned with FACT_CATALOG rows. Each page is about the exact place,
# concept, organ, event, unit, or dictionary entry used by that row. This avoids
# the broad-category source binding rejected by the first human quality gate.
FACT_DIRECT_SOURCES: dict[str, list[str]] = {
    "geography": [
        "https://www.britannica.com/place/Pacific-Ocean", "https://www.britannica.com/place/Sahara-desert-Africa", "https://www.britannica.com/place/Nile-River", "https://www.britannica.com/place/Ottawa", "https://www.britannica.com/place/Tokyo", "https://www.britannica.com/place/Brazil", "https://www.britannica.com/place/Canberra", "https://www.britannica.com/place/Mount-Everest", "https://www.britannica.com/place/Mediterranean-Sea", "https://www.britannica.com/place/Antarctica", "https://www.britannica.com/place/Rome", "https://www.britannica.com/place/Cairo", "https://www.britannica.com/place/Wellington-New-Zealand",
    ],
    "science": [
        "https://science.nasa.gov/mercury/", "https://www.britannica.com/science/photosynthesis", "https://www.britannica.com/science/gravity-physics", "https://www.britannica.com/science/water", "https://science.nasa.gov/sun/", "https://www.britannica.com/science/solid-state-of-matter", "https://www.britannica.com/science/electron", "https://www.britannica.com/science/evaporation", "https://science.nasa.gov/saturn/", "https://www.britannica.com/technology/thermometer", "https://www.britannica.com/science/atomic-nucleus", "https://www.energy.gov/eere/wind/how-do-wind-turbines-work", "https://science.nasa.gov/moon/",
    ],
    "body": [
        "https://www.britannica.com/science/heart", "https://www.britannica.com/science/human-respiratory-system", "https://www.britannica.com/science/human-skin", "https://www.britannica.com/science/skull", "https://www.britannica.com/science/tendon", "https://www.britannica.com/science/stomach", "https://www.britannica.com/science/hemoglobin", "https://www.niddk.nih.gov/health-information/kidney-disease/kidneys-how-they-work", "https://www.britannica.com/science/nail-anatomy", "https://www.britannica.com/science/elbow-anatomy", "https://www.britannica.com/science/human-skeleton", "https://www.nei.nih.gov/learn-about-eye-health/healthy-vision/how-eyes-work", "https://www.britannica.com/science/esophagus",
    ],
    "technology": [
        "https://www.cloudflare.com/learning/network-layer/what-is-a-router/", "https://developer.mozilla.org/en-US/docs/Glossary/URL", "https://www.bluetooth.com/learn-about-bluetooth/tech-overview/", "https://oceanservice.noaa.gov/education/tutorial_geodesy/geo09_gps.html", "https://www.cisa.gov/news-events/news/data-backup-options", "https://www.cisa.gov/secure-our-world/use-strong-passwords", "https://www.wi-fi.org/discover-wi-fi", "https://www.cisa.gov/news-events/news/understanding-anti-virus-software", "https://developer.mozilla.org/en-US/docs/Glossary/Hyperlink", "https://www.merriam-webster.com/dictionary/scanner", "https://developer.mozilla.org/en-US/docs/Glossary/Browser", "https://www.ibm.com/think/topics/cloud-storage",
    ],
    "time": [
        "https://www.britannica.com/science/week", "https://www.britannica.com/science/month", "https://www.britannica.com/science/hour", "https://www.nist.gov/pml/special-publication-330/sp-330-section-4", "https://www.merriam-webster.com/dictionary/April", "https://www.britannica.com/topic/Saturday-day", "https://www.britannica.com/science/day", "https://www.merriam-webster.com/dictionary/decade", "https://www.merriam-webster.com/dictionary/century", "https://www.merriam-webster.com/dictionary/January", "https://www.merriam-webster.com/dictionary/February", "https://www.merriam-webster.com/dictionary/evening", "https://www.britannica.com/science/year",
    ],
    "measurement": [
        "https://www.bipm.org/en/si-base-units/metre", "https://www.bipm.org/en/si-base-units/kilogram", "https://www.bipm.org/en/si-base-units/second", "https://www.nist.gov/pml/owm/si-units-length", "https://www.nist.gov/pml/owm/si-units-length", "https://www.bipm.org/en/publications/si-brochure", "https://www.merriam-webster.com/dictionary/ruler", "https://www.britannica.com/technology/protractor", "https://www.nist.gov/pml/owm/si-units-length", "https://www.merriam-webster.com/dictionary/scale", "https://www.bipm.org/en/si-base-units/ampere", "https://www.bipm.org/en/si-base-units/kelvin", "https://www.bipm.org/en/si-base-units/mole",
    ],
    "history": [
        "https://www.britannica.com/topic/Pyramids-of-Giza", "https://www.britannica.com/biography/Johannes-Gutenberg", "https://www.britannica.com/place/Pompeii", "https://www.britannica.com/topic/papyrus-writing-material", "https://www.britannica.com/topic/Senate-Roman-history", "https://www.britannica.com/topic/Silk-Road-trade-route", "https://www.britannica.com/topic/democracy/Ancient-Greece", "https://www.britannica.com/biography/William-Shakespeare", "https://www.britannica.com/technology/compass-navigational-instrument", "https://www.nasa.gov/history/sputnik/sputorig.html", "https://www.britannica.com/place/Machu-Picchu", "https://www.britannica.com/topic/Magna-Carta",
    ],
    "computing": [
        "https://www.britannica.com/technology/central-processing-unit", "https://www.britannica.com/technology/RAM-computing", "https://developer.mozilla.org/en-US/docs/Glossary/Variable", "https://developer.mozilla.org/en-US/docs/Glossary/Loop", "https://www.britannica.com/science/algorithm", "https://www.merriam-webster.com/dictionary/bit", "https://www.britannica.com/technology/operating-system", "https://www.britannica.com/technology/debugging", "https://www.britannica.com/technology/compiler", "https://www.britannica.com/technology/database", "https://developer.mozilla.org/en-US/docs/Glossary/HTML", "https://developer.mozilla.org/en-US/docs/Glossary/Function",
    ],
    "environment": [
        "https://www.epa.gov/recycle/recycling-basics-and-benefits", "https://www.epa.gov/ghgemissions/overview-greenhouse-gases", "https://www.energy.gov/eere/solar/how-does-solar-work", "https://www.nrcs.usda.gov/resources/education-and-teaching-materials/soil-erosion", "https://www.noaa.gov/education/resource-collections/weather-atmosphere/water-cycle", "https://oceanservice.noaa.gov/education/tutorial_corals/coral01_intro.html", "https://www.epa.gov/enviroatlas/biodiversity-conservation", "https://www.noaa.gov/jetstream/layers-of-atmosphere", "https://www.noaa.gov/education/resource-collections/weather-atmosphere/water-cycle", "https://www.energy.gov/eere/renewable-energy", "https://www.epa.gov/environmental-topics/air-topics", "https://www.epa.gov/watersense/start-saving",
    ],
    "definitions": [
        "https://www.merriam-webster.com/dictionary/transparent", "https://www.merriam-webster.com/dictionary/synonym", "https://www.merriam-webster.com/dictionary/antonym", "https://www.merriam-webster.com/dictionary/biodegradable", "https://www.merriam-webster.com/dictionary/habitat", "https://www.merriam-webster.com/dictionary/evidence", "https://www.merriam-webster.com/dictionary/concise", "https://www.merriam-webster.com/dictionary/prediction", "https://www.merriam-webster.com/dictionary/resource", "https://www.merriam-webster.com/dictionary/optional", "https://www.merriam-webster.com/dictionary/summary", "https://www.merriam-webster.com/dictionary/durable",
    ],
}


# Each tuple is (instruction, response, independently checkable claim).
FACT_CATALOG: dict[str, list[tuple[str, str, str]]] = {
    "geography": [
        ("What is the largest ocean on Earth?", "The Pacific Ocean.", "The Pacific Ocean is Earth's largest ocean."),
        ("Which continent contains the Sahara Desert?", "Africa.", "The Sahara Desert is in Africa."),
        ("What river flows through Egypt toward the Mediterranean Sea?", "The Nile River.", "The Nile flows through Egypt to the Mediterranean Sea."),
        ("What is the capital of Canada?", "Ottawa.", "Ottawa is the capital of Canada."),
        ("Which country has Tokyo as its capital?", "Japan.", "Tokyo is the capital of Japan."),
        ("On which continent is Brazil?", "South America.", "Brazil is in South America."),
        ("What is the capital of Australia?", "Canberra.", "Canberra is the capital of Australia."),
        ("Which mountain range includes Mount Everest?", "The Himalayas.", "Mount Everest is in the Himalayas."),
        ("What sea separates Europe and Africa?", "The Mediterranean Sea.", "The Mediterranean Sea lies between Europe and Africa."),
        ("Which continent is the South Pole on?", "Antarctica.", "The South Pole is on Antarctica."),
        ("What is the capital of Italy?", "Rome.", "Rome is the capital of Italy."),
        ("Which country contains the city of Cairo?", "Egypt.", "Cairo is in Egypt."),
        ("What is the capital of New Zealand?", "Wellington.", "Wellington is the capital of New Zealand."),
    ],
    "science": [
        ("Which planet is closest to the Sun?", "Mercury.", "Mercury is the closest planet to the Sun."),
        ("What gas do plants take in during photosynthesis?", "Carbon dioxide.", "Plants take in carbon dioxide during photosynthesis."),
        ("What force pulls objects toward Earth?", "Gravity.", "Gravity pulls objects toward Earth."),
        ("What is the common name for H2O?", "Water.", "H2O is water."),
        ("Which star is nearest to Earth?", "The Sun.", "The Sun is the nearest star to Earth."),
        ("What state of matter has a fixed shape and volume?", "A solid.", "A solid has a fixed shape and volume."),
        ("What particle has a negative electric charge?", "An electron.", "An electron has a negative electric charge."),
        ("What is the process of liquid becoming gas called?", "Evaporation.", "Evaporation changes liquid into gas."),
        ("Which planet is known for its prominent rings?", "Saturn.", "Saturn is known for its prominent rings."),
        ("What instrument measures temperature?", "A thermometer.", "A thermometer measures temperature."),
        ("What is the center of an atom called?", "The nucleus.", "The nucleus is at the center of an atom."),
        ("What type of energy comes from moving air?", "Wind energy.", "Moving air can provide wind energy."),
        ("What natural satellite orbits Earth?", "The Moon.", "The Moon is Earth's natural satellite."),
    ],
    "body": [
        ("Which organ pumps blood through the body?", "The heart.", "The heart pumps blood through the body."),
        ("Which organs are mainly used for breathing?", "The lungs.", "The lungs are the main organs of breathing."),
        ("What is the largest organ of the human body?", "The skin.", "The skin is the body's largest organ."),
        ("Which body part contains the brain?", "The skull.", "The skull contains and protects the brain."),
        ("What tissue connects muscle to bone?", "A tendon.", "A tendon connects muscle to bone."),
        ("Which organ helps digest food using acid?", "The stomach.", "The stomach uses acid during digestion."),
        ("What carries oxygen in red blood cells?", "Hemoglobin.", "Hemoglobin carries oxygen in red blood cells."),
        ("Which organs filter waste from the blood?", "The kidneys.", "The kidneys filter waste from the blood."),
        ("What hard structures protect the ends of fingers and toes?", "Nails.", "Nails protect the ends of fingers and toes."),
        ("Which joint connects the upper and lower arm?", "The elbow.", "The elbow connects the upper and lower arm."),
        ("What is the body's framework of bones called?", "The skeleton.", "The skeleton is the body's framework of bones."),
        ("Which sense organ detects light?", "The eye.", "The eye detects light."),
        ("What tube carries food from the mouth to the stomach?", "The esophagus.", "The esophagus carries food to the stomach."),
    ],
    "technology": [
        ("What device directs data between computer networks?", "A router.", "A router directs data between networks."),
        ("What does URL stand for?", "Uniform Resource Locator.", "URL stands for Uniform Resource Locator."),
        ("What technology connects nearby devices without cables over short distances?", "Bluetooth.", "Bluetooth connects nearby devices wirelessly over short distances."),
        ("What does GPS help a device determine?", "Its location.", "GPS helps a device determine location."),
        ("What is a digital copy stored for recovery called?", "A backup.", "A backup is a copy stored for recovery."),
        ("What protects an online account besides a password by requiring another check?", "Multi-factor authentication.", "Multi-factor authentication requires an additional verification factor."),
        ("What does Wi-Fi provide to compatible devices?", "Wireless network access.", "Wi-Fi provides wireless network access."),
        ("What type of software detects and removes malicious programs?", "Antivirus software.", "Antivirus software detects and removes malicious software."),
        ("What is a clickable connection to another web resource called?", "A hyperlink.", "A hyperlink connects to another web resource."),
        ("What device converts a paper document into a digital image?", "A scanner.", "A scanner converts paper documents into digital images."),
        ("What does a web browser display?", "Web pages.", "A web browser displays web pages."),
        ("What is remote storage accessed through the internet commonly called?", "Cloud storage.", "Cloud storage is remote storage accessed through a network."),
    ],
    "time": [
        ("How many days are in a standard week?", "Seven days.", "A standard week has seven days."),
        ("How many months are in a calendar year?", "Twelve months.", "A calendar year has twelve months."),
        ("How many minutes are in one hour?", "Sixty minutes.", "One hour has sixty minutes."),
        ("How many seconds are in one minute?", "Sixty seconds.", "One minute has sixty seconds."),
        ("Which month follows March?", "April.", "April follows March."),
        ("Which day comes after Friday?", "Saturday.", "Saturday follows Friday."),
        ("How many hours are in one day?", "Twenty-four hours.", "One day has twenty-four hours."),
        ("What is a period of ten years called?", "A decade.", "A decade is ten years."),
        ("What is a period of one hundred years called?", "A century.", "A century is one hundred years."),
        ("Which month begins the calendar year?", "January.", "January begins the calendar year."),
        ("How many days does February usually have?", "Twenty-eight days.", "February usually has twenty-eight days."),
        ("What part of the day comes after afternoon?", "Evening.", "Evening comes after afternoon."),
        ("How many days are in a common calendar year?", "Three hundred sixty-five days.", "A common calendar year has 365 days."),
    ],
    "measurement": [
        ("What SI unit measures length?", "The metre.", "The metre is the SI base unit of length."),
        ("What SI unit measures mass?", "The kilogram.", "The kilogram is the SI base unit of mass."),
        ("What SI unit measures time?", "The second.", "The second is the SI base unit of time."),
        ("How many centimetres are in one metre?", "One hundred centimetres.", "One metre equals one hundred centimetres."),
        ("How many metres are in one kilometre?", "One thousand metres.", "One kilometre equals one thousand metres."),
        ("What unit is commonly used to measure liquid volume in the metric system?", "The litre.", "The litre is a metric unit of volume."),
        ("What tool measures the length of a small object?", "A ruler.", "A ruler measures length."),
        ("What unit marks angles on a standard school protractor?", "Degrees.", "A standard school protractor marks angles in degrees."),
        ("How many millimetres are in one centimetre?", "Ten millimetres.", "One centimetre equals ten millimetres."),
        ("What does a scale measure?", "Weight or mass.", "A scale measures weight or mass, depending on its design."),
        ("What SI unit measures electric current?", "The ampere.", "The ampere is the SI base unit of electric current."),
        ("What SI unit measures temperature?", "The kelvin.", "The kelvin is the SI base unit of thermodynamic temperature."),
        ("What SI unit measures amount of substance?", "The mole.", "The mole is the SI base unit of amount of substance."),
    ],
    "history": [
        ("Which ancient civilization built the pyramids at Giza?", "Ancient Egyptians.", "Ancient Egyptians built the pyramids at Giza."),
        ("Who developed the movable-type printing press in Europe?", "Johannes Gutenberg.", "Johannes Gutenberg developed movable-type printing in Europe."),
        ("Which city was buried by Mount Vesuvius in 79 CE?", "Pompeii.", "Pompeii was buried by the eruption of Vesuvius in 79 CE."),
        ("What writing material did ancient Egyptians make from a river plant?", "Papyrus.", "Ancient Egyptians made papyrus from the papyrus plant."),
        ("Which civilization used a senate in its republic?", "Ancient Rome.", "The Roman Republic had a senate."),
        ("What trade route linked East Asia and Europe?", "The Silk Road.", "The Silk Road linked East Asia and Europe through trade routes."),
        ("Which ancient people created democracy in Athens?", "The ancient Greeks.", "Ancient Athenians developed a form of democracy."),
        ("Who wrote the plays Hamlet and Macbeth?", "William Shakespeare.", "William Shakespeare wrote Hamlet and Macbeth."),
        ("Which invention used magnetic direction to aid navigation?", "The compass.", "The magnetic compass aided navigation."),
        ("What was the name of the first artificial satellite?", "Sputnik 1.", "Sputnik 1 was the first artificial satellite."),
        ("Which civilization built Machu Picchu?", "The Inca.", "The Inca built Machu Picchu."),
        ("What document was signed in England in 1215 to limit royal power?", "Magna Carta.", "Magna Carta was sealed in England in 1215."),
    ],
    "computing": [
        ("What does CPU stand for?", "Central Processing Unit.", "CPU stands for Central Processing Unit."),
        ("What does RAM temporarily store?", "Data and instructions in active use.", "RAM temporarily stores data and instructions in active use."),
        ("What is a named storage location in a program called?", "A variable.", "A variable is a named storage location for a value."),
        ("What structure repeats instructions in a program?", "A loop.", "A loop repeats instructions."),
        ("What is a step-by-step procedure for solving a problem called?", "An algorithm.", "An algorithm is a step-by-step procedure."),
        ("Which binary digit values can a bit hold?", "Zero or one.", "A bit can hold zero or one."),
        ("What software manages a computer's hardware and applications?", "An operating system.", "An operating system manages hardware and applications."),
        ("What is an error in a computer program commonly called?", "A bug.", "A program error is commonly called a bug."),
        ("What command translates source code into another executable form?", "A compiler.", "A compiler translates source code."),
        ("What collection organizes related data in tables or records?", "A database.", "A database organizes related data."),
        ("What does HTML describe on a web page?", "Its structure and content.", "HTML describes web-page structure and content."),
        ("What is a reusable block of program instructions called?", "A function.", "A function is a reusable block of program instructions."),
    ],
    "environment": [
        ("What process turns used materials into new products?", "Recycling.", "Recycling processes used materials into new products."),
        ("Which gas is most associated with human-caused climate warming?", "A major one is carbon dioxide.", "Carbon dioxide is a major greenhouse gas from human activities."),
        ("What renewable energy source uses sunlight?", "Solar energy.", "Solar energy uses sunlight."),
        ("What do tree roots help prevent by holding soil?", "Soil erosion.", "Tree roots help reduce soil erosion."),
        ("What is water falling from clouds called?", "Precipitation.", "Water falling from clouds is precipitation."),
        ("What habitat contains salt water and coral colonies?", "A coral reef.", "Coral reefs are saltwater habitats built by coral colonies."),
        ("What is the variety of living things in an area called?", "Biodiversity.", "Biodiversity is the variety of living things in an area."),
        ("What layer of gases surrounds Earth?", "The atmosphere.", "The atmosphere is the layer of gases around Earth."),
        ("What process moves water through evaporation, condensation, and precipitation?", "The water cycle.", "The water cycle includes evaporation, condensation, and precipitation."),
        ("What kind of resource can naturally replenish over time?", "A renewable resource.", "A renewable resource replenishes naturally over time."),
        ("What term describes harmful material released into the environment?", "Pollution.", "Pollution is harmful material introduced into the environment."),
        ("What practice uses only the water needed for a task?", "Water conservation.", "Water conservation reduces unnecessary water use."),
    ],
    "definitions": [
        ("What does transparent mean when describing a material?", "It lets light pass through clearly.", "Transparent material lets light pass through clearly."),
        ("What is a synonym?", "A word with the same or a similar meaning.", "A synonym has the same or a similar meaning as another word."),
        ("What is an antonym?", "A word with an opposite meaning.", "An antonym has an opposite meaning."),
        ("What does biodegradable mean?", "Able to break down naturally.", "Biodegradable material can break down naturally."),
        ("What is a habitat?", "The natural home of an organism.", "A habitat is an organism's natural home."),
        ("What is evidence?", "Information that supports or challenges a claim.", "Evidence supports or challenges a claim."),
        ("What does concise mean?", "Brief and clear.", "Concise means brief and clear."),
        ("What is a prediction?", "A statement about what may happen.", "A prediction states what may happen."),
        ("What is a resource?", "Something useful for a purpose.", "A resource is something useful for a purpose."),
        ("What does optional mean?", "Not required.", "Optional means not required."),
        ("What is a summary?", "A short account of the main points.", "A summary gives the main points briefly."),
        ("What does durable mean?", "Able to last a long time.", "Durable means able to last a long time."),
    ],
}


EXPLANATIONS: list[tuple[str, str]] = [
    ("gravity", "Gravity is the attraction between objects with mass. Earth has enough mass to pull nearby things toward its center, so dropped objects fall and people stay on the ground."),
    ("photosynthesis", "Photosynthesis is how plants use light energy to make sugar from water and carbon dioxide. Oxygen is released during the process, and the sugar helps the plant grow."),
    ("electric current", "Electric current is the movement of electric charge through a material. In a closed circuit, a power source pushes charge along a path so devices such as lamps can work."),
    ("a programming variable", "A variable is a named place where a program stores a value. The program can read that value and, when allowed, replace it with a new one."),
    ("a function in programming", "A function is a named group of instructions that performs a specific task. Programs can call it whenever that task is needed, which reduces repeated code."),
    ("a loop in programming", "A loop repeats a group of instructions while a condition holds or for a set number of times. It is useful when the same action must happen repeatedly."),
    ("a database", "A database stores related information in an organized form. Software can search, add, change, and remove records without reading every piece of data manually."),
    ("machine learning", "Machine learning uses examples in data to adjust a model so it can make predictions or identify patterns. Its results depend on the data, the method, and careful evaluation."),
    ("the internet", "The internet is a worldwide network of connected computer networks. Devices exchange data using shared rules, allowing services such as websites, email, and video calls to work."),
    ("cloud storage", "Cloud storage keeps files on remote computers operated by a service provider. You reach the files through a network instead of relying only on storage inside your own device."),
    ("a strong password", "A strong password is long, unique, and difficult to guess. Using a different password for each account limits the damage if one service is compromised."),
    ("recycling", "Recycling collects and processes suitable used materials so they can become new products. Correct sorting matters because contamination can prevent materials from being recycled."),
    ("weather", "Weather describes short-term conditions in the atmosphere, such as temperature, wind, rain, and cloud cover. It can change from hour to hour or day to day."),
    ("a fraction", "A fraction represents part of a whole or a ratio between quantities. The top number is the numerator, and the bottom number is the denominator."),
    ("probability", "Probability describes how likely an event is. A probability of zero means impossible, while a probability of one means certain under the stated conditions."),
    ("computer memory", "Computer memory holds data and instructions that a processor needs. Some memory is fast and temporary, while other storage keeps information after power is removed."),
    ("an operating system", "An operating system manages hardware, files, memory, and running applications. It also gives programs standard ways to use devices such as screens, keyboards, and storage."),
    ("an algorithm", "An algorithm is a clear sequence of steps for completing a task or solving a problem. Good algorithms state what information they need and what result they produce."),
    ("supply and demand", "Supply is how much sellers offer, while demand is how much buyers want. Their interaction can influence prices, although real markets are also affected by costs, rules, and competition."),
    ("sleep hygiene", "Sleep hygiene means habits that support regular, restful sleep. Examples include keeping a consistent schedule, limiting late caffeine, and making the sleeping area quiet and comfortable."),
    ("a web browser", "A web browser requests web resources, interprets formats such as HTML and CSS, and displays the result. It also provides controls for navigation, downloads, privacy, and saved pages."),
    ("a search engine", "A search engine indexes information from many web pages and ranks results for a query. Its ranking is an estimate of relevance, not proof that every result is correct."),
    ("two-factor authentication", "Two-factor authentication asks for two different forms of proof before allowing access. A password plus a temporary code is a common example and is safer than a password alone."),
    ("encryption", "Encryption transforms readable data into a coded form using a key. Authorized users can reverse the transformation, while others should not be able to read the protected data easily."),
    ("a computer file", "A file is a named collection of data stored by a computer. Its format tells software how to interpret the contents, such as text, an image, or audio."),
    ("a folder", "A folder organizes files and other folders. It helps users group related work and locate items without searching the entire storage device."),
    ("a computer network", "A computer network connects devices so they can exchange data and share resources. The connection may use cables, radio signals, or both."),
    ("a router", "A router examines network addresses and forwards data toward its destination. Home routers commonly connect a local network to an internet service."),
    ("a software update", "A software update replaces or adds program files. Updates may correct errors, close security weaknesses, improve compatibility, or add features."),
    ("a backup", "A backup is a separate copy of important data kept for recovery. A useful backup should be tested and protected from the same failure that could damage the original."),
    ("debugging", "Debugging is the process of finding and correcting problems in software. It often involves reproducing the problem, examining evidence, testing a cause, and confirming the fix."),
    ("source code", "Source code is the human-readable set of instructions written in a programming language. A compiler or interpreter helps a computer execute those instructions."),
    ("an integer", "An integer is a whole number with no fractional part. Integers include negative numbers, zero, and positive numbers."),
    ("an average", "An average is a value used to summarize a group of numbers. The arithmetic mean is found by adding the values and dividing by how many values there are."),
    ("a percentage", "A percentage expresses an amount out of one hundred. For example, twenty-five percent means twenty-five out of every one hundred equal parts."),
    ("a decimal", "A decimal uses place value and a decimal point to represent whole numbers and fractions. Digits to the right represent tenths, hundredths, and smaller parts."),
    ("a prime number", "A prime number is a whole number greater than one with exactly two positive divisors: one and itself. Two is the smallest prime number."),
    ("area", "Area measures how much two-dimensional surface a shape covers. It is reported in square units, such as square metres or square centimetres."),
    ("perimeter", "Perimeter is the total distance around a two-dimensional shape. You find it by adding the lengths of all outer sides."),
    ("a graph", "A graph displays information visually so patterns and comparisons are easier to see. Labels and scales are needed to interpret it correctly."),
    ("the water cycle", "The water cycle moves water through evaporation, condensation, precipitation, and collection. Solar energy and gravity help drive this continuous movement."),
    ("condensation", "Condensation happens when water vapor cools and changes into liquid droplets. It can form clouds or appear on the outside of a cold glass."),
    ("erosion", "Erosion moves soil or rock from one place to another through water, wind, ice, or gravity. It differs from weathering, which breaks material down in place."),
    ("an ecosystem", "An ecosystem includes living organisms and the nonliving surroundings they interact with. Energy and materials move among its plants, animals, microbes, air, water, and soil."),
    ("a food chain", "A food chain shows one path by which energy moves between organisms. It usually starts with a producer and continues through organisms that eat other organisms."),
    ("renewable energy", "Renewable energy comes from sources that naturally replenish, such as sunlight, wind, and flowing water. Its environmental effects still depend on how systems are built and used."),
    ("air pollution", "Air pollution is the presence of harmful substances in the air. Sources include smoke, vehicle exhaust, industry, fires, and some natural events."),
    ("climate", "Climate describes typical weather patterns in a place over long periods. Weather is what happens over shorter periods, such as today or this week."),
    ("composting", "Composting lets microorganisms break down suitable food scraps and plant material. The finished compost can add organic matter to soil."),
    ("biodiversity", "Biodiversity is the variety of life, including genes, species, and ecosystems. Greater variety can help ecosystems continue functioning when conditions change."),
    ("the heart", "The heart is a muscular organ that pumps blood through blood vessels. This circulation delivers oxygen and nutrients and carries away waste products."),
    ("the lungs", "The lungs exchange gases between air and blood. Oxygen enters the blood, while carbon dioxide moves out to be exhaled."),
    ("digestion", "Digestion breaks food into smaller substances that the body can absorb and use. It involves physical movement, digestive chemicals, and several organs."),
    ("hydration", "Hydration means having enough water available for normal body functions. Needs vary with activity, weather, health, and diet."),
    ("a balanced diet", "A balanced diet includes varied foods that provide energy, protein, vitamins, minerals, fiber, and suitable fats. Individual needs can differ."),
    ("handwashing", "Handwashing with soap removes dirt and many germs from the skin. Scrubbing all hand surfaces before rinsing makes the cleaning more effective."),
    ("a pulse", "A pulse is the pressure wave felt in an artery each time the heart pumps. Its rate is commonly counted in beats per minute."),
    ("body temperature", "Body temperature reflects the balance between heat production and heat loss. It varies somewhat by person, time, activity, and measurement method."),
    ("a vaccine", "A vaccine trains the immune system to recognize a specific disease-causing threat. This can reduce the chance or severity of future illness from that threat."),
    ("first aid", "First aid is immediate basic care given before full medical help is available. It should match the situation and should not replace professional care when that is needed."),
    ("a noun", "A noun names a person, place, thing, or idea. In a sentence, it can serve roles such as subject or object."),
    ("a verb", "A verb expresses an action, occurrence, or state. Its form can change to show tense and agreement."),
    ("an adjective", "An adjective describes or limits a noun. It can tell qualities such as size, color, number, or condition."),
    ("a sentence", "A sentence expresses a complete thought using an organized group of words. In standard writing, it begins with a capital letter and ends with suitable punctuation."),
    ("a paragraph", "A paragraph groups sentences around a main idea. Supporting sentences develop that idea with explanation, evidence, or examples."),
    ("a summary", "A summary presents the main ideas of a longer source in a shorter form. It should stay accurate and avoid unrelated details."),
    ("a citation", "A citation identifies the source of information or ideas. It helps readers locate evidence and distinguishes source material from the writer's own work."),
    ("a metaphor", "A metaphor describes one thing as another to suggest a shared quality. It is figurative language, not a literal claim."),
    ("context", "Context is the surrounding information that helps explain meaning. A word, statement, or event can be misunderstood when its context is missing."),
    ("an outline", "An outline arranges main points and supporting details before writing. It helps keep ideas ordered and reduces repetition."),
    ("a budget", "A budget is a plan for expected income and spending over a period. Comparing the plan with actual amounts helps guide later decisions."),
    ("saving money", "Saving means setting aside part of available money for later use. A clear purpose and regular amount can make the habit easier to maintain."),
    ("interest", "Interest is money paid for borrowing or earned for lending or saving. The amount depends on the rate, balance, time, and calculation method."),
    ("supply", "Supply is the amount of a product or service that sellers are willing and able to offer under stated conditions."),
    ("demand", "Demand is the amount of a product or service that buyers are willing and able to purchase under stated conditions."),
    ("opportunity cost", "Opportunity cost is the value of the best alternative given up when a choice is made. It highlights that time and resources are limited."),
    ("a receipt", "A receipt records a transaction, usually listing what was purchased, when, and how much was paid. It can help with returns and expense records."),
    ("a map scale", "A map scale shows how distance on a map relates to distance in the real world. It may use a ratio, written statement, or scale bar."),
    ("latitude", "Latitude measures distance north or south of the equator using angular degrees. Lines of equal latitude run east to west."),
    ("longitude", "Longitude measures angular distance east or west of the prime meridian. Lines of longitude meet at the poles."),
    ("a time zone", "A time zone is a region that uses the same standard time. Boundaries often follow political borders instead of exact lines of longitude."),
    ("a compass", "A magnetic compass has a needle that aligns with Earth's magnetic field. It helps show direction when local magnetic interference is limited."),
    ("a continent", "A continent is one of Earth's large continuous land regions. The exact grouping convention can vary, so some systems count continents differently."),
    ("a peninsula", "A peninsula is land surrounded by water on most sides while remaining connected to a larger land area."),
    ("an island", "An island is a land area surrounded by water. It is smaller than a continent under common geographic conventions."),
    ("a renewable resource", "A renewable resource can replenish through natural processes on a human timescale when use does not exceed renewal."),
    ("critical thinking", "Critical thinking means examining claims, evidence, assumptions, and alternatives carefully. It involves asking how a conclusion was reached rather than accepting it automatically."),
    ("a hypothesis", "A hypothesis is a testable proposed explanation or prediction. Evidence from observation or experiment can support it, challenge it, or lead to revision."),
    ("an experiment", "An experiment is a planned test used to investigate a question. Good design controls relevant factors and records methods and results clearly."),
    ("a control group", "A control group provides a comparison in an experiment. It helps show whether observed differences are connected to the tested condition."),
    ("measurement error", "Measurement error is the difference between a measured value and the value being estimated. Repeated checks and calibrated tools can reduce or reveal error."),
    ("correlation", "Correlation describes how two measured variables vary together. It does not by itself prove that one variable causes the other."),
    ("peer review", "Peer review is evaluation of work by people with relevant expertise before or after publication. It can identify problems but does not guarantee correctness."),
    ("a model in science", "A scientific model is a simplified representation used to explain or predict aspects of the world. Its usefulness depends on evidence and the purpose for which it is used."),
    ("a spreadsheet", "A spreadsheet organizes information in rows and columns. Formulas can calculate values from cells, while sorting and filtering help people examine the data."),
    ("a computer cache", "A cache keeps copies of frequently needed data in a faster location. This can reduce waiting, but cached information may need refreshing when the original changes."),
    ("a barcode", "A barcode represents an identifier using a pattern that a scanner can read. A computer then uses that identifier to look up related information."),
    ("a checklist", "A checklist records items that should be considered or completed. It supports memory and consistency, but each item still needs appropriate judgment."),
    ("a deadline", "A deadline is the time by which a task should be completed. Knowing it helps people estimate effort, order work, and communicate risks early."),
    ("feedback", "Feedback is information about a result or performance that can guide improvement. Useful feedback is specific, relevant, and clear about what could be kept or changed."),
]


def _metadata(family: str) -> dict[str, Any]:
    return {
        "created_at": CREATED_AT,
        "created_by": "VASU batch authoring pipeline",
        "license": "CC0-1.0",
        "provenance": "purpose-written VASU instruction-quality batch 001",
        "authoring_template_family": family,
    }


def _base_record(
    capability: str,
    instruction: str,
    response: str,
    family: str,
    *,
    input_text: str = "",
    difficulty: str = "easy",
    answer_style: str = "concise",
    format_constraints: dict[str, Any] | None = None,
    source_reference: str | None = None,
    facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "vasu_instruction_quality_v1",
        "example_id": "",
        "capability": capability,
        "instruction": instruction,
        "input": input_text,
        "response": response,
        "source_type": "human_written",
        "source_reference": source_reference,
        "language": "en",
        "difficulty": difficulty,
        "answer_style": answer_style,
        "format_constraints": format_constraints or {"type": "none"},
        "facts": facts or [],
        "quality": {
            "author_status": "draft",
            "review_status": "unreviewed",
            "reviewer_id": None,
            "review_notes": None,
        },
        "metadata": _metadata(family),
    }


def _factual_records() -> list[dict[str, Any]]:
    records = []
    for family, rows in FACT_CATALOG.items():
        sources = FACT_DIRECT_SOURCES[family]
        if len(sources) != len(rows):
            raise AssertionError(
                f"{family} source count {len(sources)} does not match row count {len(rows)}"
            )
        for (instruction, response, claim), source in zip(rows, sources, strict=True):
            records.append(
                _base_record(
                    "short_factual_qa",
                    instruction,
                    response,
                    f"factual_{family}",
                    difficulty="medium" if family in {"history", "measurement"} else "easy",
                    source_reference=source,
                    facts=[{
                        "claim": claim,
                        "verification_status": "verified",
                        "verification_source": source,
                        "time_sensitive": False,
                    }],
                )
            )
    if len(records) != 125:
        raise AssertionError(f"factual catalog must contain 125 rows; found {len(records)}")
    return records


def _explanation_records() -> list[dict[str, Any]]:
    prompts = (
        "Explain {topic} simply.",
        "What does {topic} mean?",
        "Explain {topic} to a ten-year-old.",
        "Why is understanding {topic} useful?",
        "How does {topic} work in simple terms?",
    )
    records = []
    for index, (topic, response) in enumerate(EXPLANATIONS):
        if topic == "the lungs":
            instruction = "Explain what the lungs do in simple terms."
        elif index % len(prompts) == 1:
            instruction = f"What is meant by {topic}?"
        else:
            instruction = prompts[index % len(prompts)].format(topic=topic)
        family = f"explanation_{index // 5 + 1:02d}"
        records.append(
            _base_record(
                "beginner_explanation",
                instruction,
                response,
                family,
                difficulty="medium" if index % 10 in {7, 8, 9} else "easy",
                answer_style="beginner-friendly",
            )
        )
    if len(records) != 100:
        raise AssertionError("explanation catalog must contain 100 rows")
    return records


def _exact_format_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    bullet_sets = [
        ("ways to protect an online account", ["Use a unique password.", "Enable multi-factor authentication.", "Keep recovery details current.", "Review unexpected login alerts.", "Sign out on shared devices."]),
        ("habits for focused study", ["Silence unnecessary notifications.", "Set one clear goal.", "Take a short planned break.", "Keep only needed materials nearby.", "Test yourself after studying."]),
        ("steps before sending an email", ["Check the recipient.", "Review the subject line.", "Proofread the message.", "Confirm any attachment is included.", "Verify names and dates."]),
        ("safe ways to carry a laptop", ["Use a padded sleeve.", "Keep liquids separate.", "Hold the bag securely.", "Avoid placing heavy items on it.", "Close the lid before moving it."]),
        ("simple water-saving actions", ["Turn off unused taps.", "Repair visible leaks.", "Run full laundry loads.", "Take shorter showers.", "Water plants only when needed."]),
    ]
    for index in range(20):
        topic, items = bullet_sets[index % len(bullet_sets)]
        count = 2 + index % 4
        chosen = items[:count]
        records.append(_base_record("exact_format_following", f"Give exactly {count} bullet points about {topic}.", "\n".join(f"- {item}" for item in chosen), "exact_bullets", format_constraints={"type": "exact_bullets", "count": count}))
    numbered_sets = [
        ("starting a study session", ["Choose the topic.", "Gather the materials.", "Set a timer.", "Begin with one task.", "Review your notes."]),
        ("checking a simple program", ["Run the program.", "Read the error.", "Inspect the named line.", "Make one change.", "Run the test again."]),
        ("preparing for a meeting", ["Read the agenda.", "Write key questions.", "Bring relevant notes.", "Join on time.", "Record decisions."]),
        ("organizing a folder", ["Collect related files.", "Remove duplicate copies.", "Choose clear names.", "Create subfolders.", "Back up the result."]),
        ("planning a short walk", ["Check the weather.", "Choose a safe route.", "Wear suitable shoes.", "Carry needed water.", "Return before dark."]),
        ("sending a careful email", ["Confirm the recipient.", "Write a clear subject.", "Review the message.", "Check attachments.", "Send the email."]),
        ("setting up a workspace", ["Clear the desk.", "Gather needed tools.", "Adjust the chair.", "Reduce distractions.", "Begin the first task."]),
        ("making a weekly plan", ["List fixed commitments.", "Choose priorities.", "Estimate task time.", "Schedule breaks.", "Review the plan."]),
        ("checking a backup", ["Select the latest backup.", "Restore a test file from it.", "Open the restored file.", "Compare it with the original.", "Record the verification result."]),
        ("preparing a book return", ["Find the borrowed books.", "Remove personal papers.", "Check the due date.", "Pack them safely.", "Return them to the library."]),
    ]
    for index in range(15):
        topic, items = numbered_sets[index % len(numbered_sets)]
        count = 2 + index % 4
        records.append(_base_record("exact_format_following", f"Write exactly {count} numbered steps for {topic}.", "\n".join(f"{number}. {item}" for number, item in enumerate(items[:count], 1)), "exact_numbered", format_constraints={"type": "exact_numbered_items", "count": count}))
    sentence_pairs = [
        ("Describe why backups matter", ["Backups provide a separate copy of important files.", "They make recovery possible after loss or damage."]),
        ("Explain a short break during study", ["A short break can restore attention.", "A planned end helps the learner return on time.", "This keeps the study schedule useful."]),
        ("Describe a clear password policy", ["Each account should use a unique password.", "Long passwords are harder to guess."]),
        ("Explain why file names matter", ["Clear file names identify their contents.", "A consistent pattern also makes files easier to sort.", "This saves time during later searches."]),
        ("Describe a useful meeting agenda", ["An agenda states the topics to discuss.", "It helps participants prepare and manage time."]),
        ("Explain why updates should be checked", ["Trusted updates can fix defects and security weaknesses.", "Users should save work before installing them.", "The device may need to restart afterward."]),
        ("Describe careful note-taking", ["Useful notes capture the main idea in clear language.", "They distinguish source facts from personal questions."]),
        ("Explain a simple study goal", ["A study goal names one specific result.", "It should fit the time and resources available.", "Progress can then be checked directly."]),
        ("Describe an organized desk", ["An organized desk keeps needed materials within reach.", "Unrelated items are stored elsewhere."]),
        ("Explain why proofreading helps", ["Proofreading catches errors before a message is shared.", "Reading slowly makes missing or repeated words easier to notice.", "A final check should include names and dates."]),
        ("Describe safe screen locking", ["Screen locking blocks casual access when a device is unattended.", "The unlock method should remain private."]),
        ("Explain why a deadline should be recorded", ["A recorded deadline supports planning and reminders.", "It also makes scheduling conflicts visible earlier.", "The source of the deadline should be retained."]),
        ("Describe a reliable checklist", ["A reliable checklist contains clear actions in a useful order.", "Completed items are marked consistently."]),
        ("Explain why recovery details matter", ["Current recovery details can help restore account access.", "They should point to secure channels the owner controls.", "Outdated details may delay recovery."]),
        ("Describe a focused work period", ["A focused work period has one defined task.", "Unnecessary alerts are silenced until the period ends."]),
    ]
    for index, (topic, sentences) in enumerate(sentence_pairs):
        count = len(sentences)
        records.append(_base_record("exact_format_following", f"{topic} in exactly {count} sentences.", " ".join(sentences), "exact_sentences", format_constraints={"type": "exact_sentences", "count": count}))
    one_words = [("Opposite of hot?", "Cold."), ("A frozen form of water?", "Ice."), ("Color made by mixing blue and yellow?", "Green."), ("Day after Monday?", "Tuesday."), ("Planet we live on?", "Earth."), ("A young dog?", "Puppy."), ("Season after spring?", "Summer."), ("Shape with three sides?", "Triangle."), ("Tool used to cut paper?", "Scissors."), ("A place that lends books?", "Library.")]
    for instruction, response in one_words:
        records.append(_base_record("exact_format_following", f"Answer with one word: {instruction}", response, "one_word", format_constraints={"type": "one_word"}))
    maximum_rows = [
        ("Summarize why sleep matters in no more than eight words.", "Sleep supports health, focus, learning, and recovery.", 8),
        ("Define recycling in no more than seven words.", "Processing used materials into new products.", 7),
        ("Describe a backup in no more than nine words.", "A separate copy kept for data recovery.", 9),
        ("Explain gravity in no more than eight words.", "Mass attracts other mass through gravity.", 8),
        ("Define a variable in no more than eight words.", "A named place that stores a value.", 8),
        ("Describe rain in no more than seven words.", "Water droplets falling from clouds.", 7),
        ("Define a map in no more than eight words.", "A visual representation of a place.", 8),
        ("Explain a budget in no more than eight words.", "A plan for income and spending.", 8),
        ("Define evidence in no more than nine words.", "Evidence supports or challenges a claim.", 9),
        ("Describe Wi-Fi in no more than eight words.", "Technology for wireless local network access.", 8),
    ]
    for instruction, response, count in maximum_rows:
        records.append(_base_record("exact_format_following", instruction, response, "maximum_words", format_constraints={"type": "maximum_words", "count": count}))
    heading_rows = [
        ("a weekly study goal", "Summary:\nReview two chapters this week.\nAction:\nStudy for thirty minutes each evening."),
        ("a file-backup plan", "Summary:\nKeep a separate copy of important files.\nAction:\nRun and verify a backup every Friday."),
        ("a meeting note", "Summary:\nThe team agreed on the delivery date.\nAction:\nSend the updated schedule today."),
        ("a recycling reminder", "Summary:\nOnly clean accepted materials belong in the bin.\nAction:\nCheck local sorting rules before disposal."),
        ("a password improvement", "Summary:\nThe current password is reused elsewhere.\nAction:\nCreate a long unique replacement."),
        ("a reading plan", "Summary:\nThe book has six chapters remaining.\nAction:\nRead one chapter each day."),
        ("a desk organization task", "Summary:\nLoose papers are hiding needed notes.\nAction:\nSort papers into labelled folders."),
        ("a software update", "Summary:\nA security update is available.\nAction:\nSave work and install it tonight."),
        ("a water-saving check", "Summary:\nThe kitchen tap continues to drip.\nAction:\nArrange a repair and monitor the leak."),
        ("an exercise plan", "Summary:\nThe goal is regular moderate activity.\nAction:\nSchedule a short walk on five days."),
    ]
    for topic, response in heading_rows:
        records.append(_base_record("exact_format_following", f"Use the headings Summary and Action for {topic}.", response, "required_headings", format_constraints={"type": "required_headings", "headings": ["Summary", "Action"]}))
    plain_rows = [
        ("Write a plain-text reminder to save your work.", "Save your work before closing the program."),
        ("Give a plain-text note about checking a deadline.", "Check the deadline before planning the remaining tasks."),
        ("Write a plain-text instruction for locking a screen.", "Lock the screen whenever you leave the device unattended."),
        ("Give a plain-text reminder about drinking water.", "Keep water nearby and drink according to your needs."),
        ("Write a plain-text note about proofreading.", "Proofread the message before sending it."),
        ("Give a plain-text reminder about charging a device.", "Charge the device before a long trip."),
        ("Write a plain-text note about naming files.", "Use clear file names that describe the contents."),
        ("Give a plain-text reminder about calendar events.", "Add the appointment time and location to the calendar."),
        ("Write a plain-text instruction for storing keys.", "Return the keys to their labelled place after use."),
        ("Give a plain-text note about quiet study.", "Choose a quiet place and silence unnecessary alerts."),
    ]
    for instruction, response in plain_rows:
        records.append(_base_record("exact_format_following", instruction, response, "plain_text", format_constraints={"type": "plain_text_only"}))
    no_list_rows = [
        ("Explain why a clear subject line helps, without using a list.", "A clear subject line tells the reader what the message concerns and makes the email easier to find later."),
        ("Describe a safe backup habit without using a list.", "Keep a separate current copy of important files and test that the copy can be restored."),
        ("Explain a study timer without using a list.", "A study timer defines a focused work period and a clear point for taking a planned break."),
        ("Describe good file names without using a list.", "Good file names state the contents clearly and use a consistent pattern that makes sorting easier."),
        ("Explain why software updates matter without using a list.", "Software updates can correct defects, improve compatibility, and close known security weaknesses."),
    ]
    for instruction, response in no_list_rows:
        records.append(_base_record("exact_format_following", instruction, response, "no_list", format_constraints={"type": "no_list"}))
    exact_words = [
        ("Answer in exactly four words: What should you do before editing an important file?", "Create a backup copy.", 4),
        ("Answer in exactly five words: How should passwords be used across accounts?", "Use a unique password everywhere.", 5),
        ("Answer in exactly six words: What helps a meeting stay focused?", "A clear agenda and time limits.", 6),
        ("Answer in exactly five words: What should follow a code change?", "Run the relevant tests again.", 5),
        ("Answer in exactly four words: How can notes stay organized?", "Use clearly labelled folders.", 4),
    ]
    for instruction, response, count in exact_words:
        records.append(_base_record("exact_format_following", instruction, response, "exact_short", format_constraints={"type": "exact_words", "count": count}))
    if len(records) != 100:
        raise AssertionError(f"exact-format catalog must contain 100 rows; found {len(records)}")
    return records


REWRITES: dict[str, list[tuple[str, str]]] = {
    "polite": [
        ("Send me the report today.", "Could you please send me the report today?"), ("Move your bag.", "Could you please move your bag?"), ("Tell me when the meeting starts.", "Could you please tell me when the meeting starts?"), ("Fix this error now.", "Could you please fix this error now?"), ("Give me your notes.", "Could you please share your notes with me?"), ("Stop making noise.", "Could you please keep the noise down?"), ("Check my calculation.", "Could you please check my calculation?"), ("Book the room for Friday.", "Could you please book the room for Friday?"), ("Explain this part again.", "Could you please explain this part again?"), ("Return the charger.", "Could you please return the charger?"), ("Add my name to the list.", "Could you please add my name to the list?"), ("Call me after class.", "Could you please call me after class?"), ("Print two copies.", "Could you please print two copies?"), ("Close the window.", "Could you please close the window?"), ("Show me the updated schedule.", "Could you please show me the updated schedule?"), ("Review this paragraph.", "Could you please review this paragraph?"), ("Save a copy for me.", "Could you please save a copy for me?"), ("Turn down the music.", "Could you please turn down the music?"), ("Forward the confirmation email.", "Could you please forward the confirmation email?"), ("Wait until I finish.", "Could you please wait until I finish?")
    ],
    "concise": [
        ("Due to the fact that the bus was late, we arrived after the start.", "The late bus made us arrive after the start."), ("Please be advised that the library will be closed on Monday.", "The library will be closed Monday."), ("At this point in time, the results are not available.", "The results are not available yet."), ("The reason the file failed to open was because it was damaged.", "The file failed to open because it was damaged."), ("In the event that it rains, the game will move indoors.", "If it rains, the game will move indoors."), ("We are writing to inform you that your request was accepted.", "Your request was accepted."), ("There are three different options that are available to choose from.", "Three options are available."), ("The meeting is scheduled to take place at noon.", "The meeting is scheduled for noon."), ("He made a decision to restart the computer.", "He decided to restart the computer."), ("The new update has the ability to reduce loading time.", "The update can reduce loading time."), ("The package was delivered in a successful manner.", "The package was delivered successfully."), ("We conducted a review of the document.", "We reviewed the document."), ("She gave an explanation of the process.", "She explained the process."), ("The team reached an agreement on the deadline.", "The team agreed on the deadline."), ("Please make sure that you remember to lock the door.", "Please remember to lock the door.")
    ],
    "active_voice": [
        ("The final draft was checked by the editor.", "The editor checked the final draft."), ("The room was cleaned by the volunteers.", "The volunteers cleaned the room."), ("The data will be reviewed by the analyst.", "The analyst will review the data."), ("The song was performed by the school choir.", "The school choir performed the song."), ("The broken cable was replaced by the technician.", "The technician replaced the broken cable."), ("The message was sent by the coordinator.", "The coordinator sent the message."), ("The plants are watered by the caretaker each morning.", "The caretaker waters the plants each morning."), ("The test was completed by every student.", "Every student completed the test."), ("The schedule was updated by the project lead.", "The project lead updated the schedule."), ("The meal was prepared by the kitchen team.", "The kitchen team prepared the meal.")
    ],
    "professional": [
        ("Hey, I can't make it. Something came up.", "I am unable to attend because an unexpected matter arose."), ("The server is acting weird again.", "The server is behaving unexpectedly again."), ("Can you get this done ASAP?", "Could you please complete this as soon as possible?"), ("We messed up the dates in the file.", "We entered incorrect dates in the file."), ("I need the numbers before lunch.", "Please provide the figures before lunch."), ("This plan won't work as written.", "The plan requires revision before it can be implemented."), ("Let's talk about this later.", "Let us discuss this matter at a later time."), ("The customer is mad about the delay.", "The customer is dissatisfied with the delay."), ("I forgot to attach the document.", "I neglected to attach the document."), ("We need to figure out who owns this task.", "We need to identify the person responsible for this task.")
    ],
    "friendly": [
        ("Your registration has been confirmed.", "Great news—your registration is confirmed!"), ("Please submit the form by Friday.", "Please send in the form by Friday. Thanks!"), ("The study group begins at six.", "Our study group starts at six. See you there!"), ("Your package is ready for collection.", "Your package is ready to pick up!"), ("The shared folder is now available.", "The shared folder is ready for you to use!"), ("Remember to bring the textbook.", "Don't forget to bring your textbook!"), ("The event has moved to the main hall.", "The event has moved to the main hall!"), ("Your feedback has been received.", "Thanks—we received your feedback!"), ("The repair is complete.", "Good news—the repair is finished!"), ("The new timetable is attached.", "The new timetable is attached for you!")
    ],
    "grammar": [
        ("She don't know the answer.", "She doesn't know the answer."), ("The files is on the desk.", "The files are on the desk."), ("We was waiting outside.", "We were waiting outside."), ("He have finished the task.", "He has finished the task."), ("They goes to class early.", "They go to class early.")
    ],
    "simplification": [
        ("The device requires periodic authentication credential renewal.", "The device requires regular password updates."), ("Please commence the installation after terminating all applications.", "Please close all apps, then start the installation."), ("The route is inaccessible because of adverse weather conditions.", "Bad weather has closed the route."), ("The document contains insufficient information for evaluation.", "The document does not have enough information for evaluation."), ("Users should retain a duplicate of all essential records.", "Users should keep a copy of every important record.")
    ],
}


def _rewrite_records() -> list[dict[str, Any]]:
    labels = {"polite": "Rewrite the message politely.", "concise": "Rewrite the sentence more concisely.", "active_voice": "Rewrite the sentence in active voice.", "professional": "Rewrite the message in a professional tone.", "friendly": "Rewrite the message in a friendly tone.", "grammar": "Correct the grammar without changing the meaning.", "simplification": "Rewrite the sentence in simpler language."}
    qualifiers = [
        "Return only the revision.", "Keep the original request intact.", "Do not add an explanation.",
        "Preserve every stated detail.", "Use one complete sentence.", "Keep the same intended action.",
        "Do not introduce new information.", "Retain the original time reference.", "Keep the meaning precise.",
        "Use clear standard English.", "Keep the result brief.", "Preserve the original subject.",
        "Maintain the original level of certainty.", "Do not add a greeting.", "Do not add a sign-off.",
        "Keep all named objects unchanged.", "Preserve the requested outcome.", "Avoid unnecessary preamble.",
        "Change wording rather than content.", "Provide just the transformed text.",
    ]
    records = []
    for family, rows in REWRITES.items():
        for index, (input_text, response) in enumerate(rows):
            instruction = f"{labels[family]} {qualifiers[index]}"
            records.append(_base_record("rewriting_transformation", instruction, response, f"rewrite_{family}", input_text=input_text, difficulty="medium" if family in {"active_voice", "professional", "simplification"} else "easy", answer_style=family))
    if len(records) != 75:
        raise AssertionError("rewrite catalog must contain 75 rows")
    return records


LIST_TOPICS: list[tuple[str, list[str]]] = [
    ("safe computer habits", ["Lock the screen when leaving.", "Install trusted updates.", "Use unique passwords.", "Back up important files.", "Avoid unknown attachments."]),
    ("meeting preparation", ["Read the agenda.", "Collect relevant notes.", "Write key questions.", "Test the call link.", "Join on time."]),
    ("a simple study plan", ["Choose one topic.", "Study for twenty minutes.", "Practice with two questions.", "Check the answers.", "Write one summary sentence."]),
    ("packing for a day trip", ["Water bottle", "Weather-appropriate clothing", "Phone charger", "Required tickets", "Small first-aid kit"]),
    ("basic debugging", ["Reproduce the error.", "Read the message.", "Check the named line.", "Change one thing.", "Run the test again."]),
    ("a healthy morning routine", ["Wake at a consistent time.", "Drink some water.", "Eat a suitable breakfast.", "Prepare the day's materials.", "Leave enough travel time."]),
    ("organizing project files", ["Create a project folder.", "Use descriptive names.", "Separate drafts from final files.", "Remove duplicate copies.", "Back up the folder."]),
    ("reducing household waste", ["Choose reusable items.", "Repair usable products.", "Sort accepted recyclables.", "Plan meals before shopping.", "Compost suitable scraps."]),
    ("preparing a presentation", ["Define the main message.", "Select supporting points.", "Use readable slides.", "Practice aloud.", "Check the equipment."]),
    ("safe walking exercise", ["Wear suitable shoes.", "Choose a safe route.", "Start at a comfortable pace.", "Carry needed water.", "Stop if you feel unwell."]),
]


def _list_records() -> list[dict[str, Any]]:
    records = []
    for index in range(20):
        topic, items = LIST_TOPICS[index % len(LIST_TOPICS)]
        count = 2 + index % 4
        records.append(_base_record("lists_structured_output", f"List exactly {count} items for {topic} using bullets.", "\n".join(f"- {item}" for item in items[:count]), "structured_bullets", difficulty="medium" if count >= 4 else "easy", format_constraints={"type": "exact_bullets", "count": count}))
    for index in range(20):
        topic, items = LIST_TOPICS[(index + 3) % len(LIST_TOPICS)]
        count = 2 + index % 4
        records.append(_base_record("lists_structured_output", f"Give exactly {count} numbered steps for {topic}.", "\n".join(f"{number}. {item}" for number, item in enumerate(items[:count], 1)), "structured_numbered", difficulty="medium" if count >= 4 else "easy", format_constraints={"type": "exact_numbered_items", "count": count}))
    labeled = [
        ("a short study record", ["Topic", "Duration"], ["Fractions", "Twenty minutes"]), ("a meeting reminder", ["Time", "Place"], ["10:00", "Room B"]), ("a file record", ["Name", "Status"], ["Budget draft", "Reviewed"]), ("a reading goal", ["Book", "Target"], ["Science Basics", "Two chapters"]), ("a device check", ["Device", "Result"], ["Laptop", "Update installed"]), ("a task card", ["Task", "Owner"], ["Check links", "Editor"]), ("a travel note", ["Destination", "Departure"], ["Museum", "09:30"]), ("a supply label", ["Item", "Quantity"], ["Notebooks", "Three"]), ("a weather note", ["Condition", "Action"], ["Rain", "Carry an umbrella"]), ("a backup record", ["Folder", "Date"], ["Project files", "Friday"]),
    ]
    for topic, labels, values in labeled:
        response = "\n".join(f"{label}: {value}" for label, value in zip(labels, values, strict=True))
        records.append(_base_record("lists_structured_output", f"Create labelled fields for {topic} using {labels[0]} and {labels[1]}.", response, "structured_labels", format_constraints={"type": "exact_labeled_fields", "labels": labels}))
    return records


def _json_records() -> list[dict[str, Any]]:
    rows: list[tuple[str, dict[str, Any], dict[str, str], str]] = [
        ("Return a JSON object for a book with title and author.", {"title": "River Paths", "author": "Mira Sen"}, {"title": "string", "author": "string"}, "json_strings"),
        ("Return JSON with a city and country.", {"city": "Oslo", "country": "Norway"}, {"city": "string", "country": "string"}, "json_strings"),
        ("Create JSON for a course name and level.", {"course": "Python Basics", "level": "beginner"}, {"course": "string", "level": "string"}, "json_strings"),
        ("Return JSON containing a file name and status.", {"file": "notes.txt", "status": "saved"}, {"file": "string", "status": "string"}, "json_strings"),
        ("Create JSON for a task and owner role.", {"task": "check links", "owner": "editor"}, {"task": "string", "owner": "string"}, "json_strings"),
        ("Return JSON stating whether notifications are enabled.", {"notifications_enabled": True}, {"notifications_enabled": "boolean"}, "json_boolean"),
        ("Create JSON showing whether a file is archived.", {"archived": False}, {"archived": "boolean"}, "json_boolean"),
        ("Return JSON with a task name and completion flag.", {"task": "backup files", "completed": True}, {"task": "string", "completed": "boolean"}, "json_boolean"),
        ("Create JSON with a feature name and active flag.", {"feature": "dark mode", "active": False}, {"feature": "string", "active": "boolean"}, "json_boolean"),
        ("Return JSON with a reminder and enabled flag.", {"reminder": "team meeting", "enabled": True}, {"reminder": "string", "enabled": "boolean"}, "json_boolean"),
        ("Return JSON with the number of pages read.", {"pages_read": 24}, {"pages_read": "number"}, "json_integer"),
        ("Create JSON for a box label and item count.", {"label": "cables", "count": 3}, {"label": "string", "count": "number"}, "json_integer"),
        ("Return JSON with a lesson number and title.", {"lesson": 2, "title": "Loops"}, {"lesson": "number", "title": "string"}, "json_integer"),
        ("Create JSON with a room name and seat count.", {"room": "Lab A", "seats": 18}, {"room": "string", "seats": "number"}, "json_integer"),
        ("Return JSON for a timer with minutes and running state.", {"minutes": 15, "running": False}, {"minutes": "number", "running": "boolean"}, "json_integer"),
        ("Return JSON with a tags array containing study, math, and review.", {"tags": ["study", "math", "review"]}, {"tags": "array"}, "json_array"),
        ("Create JSON with a colors array containing red and blue.", {"colors": ["red", "blue"]}, {"colors": "array"}, "json_array"),
        ("Return JSON with a steps array containing open and check.", {"steps": ["open", "check"]}, {"steps": "array"}, "json_array"),
        ("Create JSON with a days array containing Monday and Wednesday.", {"days": ["Monday", "Wednesday"]}, {"days": "array"}, "json_array"),
        ("Return JSON with an items array containing pen, card, and key.", {"items": ["pen", "card", "key"]}, {"items": "array"}, "json_array"),
        ("Create JSON with user.name and user.active.", {"user": {"name": "Asha", "active": True}}, {"user": "object"}, "json_nested"),
        ("Return JSON with settings.theme and settings.sound.", {"settings": {"theme": "light", "sound": False}}, {"settings": "object"}, "json_nested"),
        ("Create JSON with location.city and location.zone.", {"location": {"city": "Pune", "zone": "west"}}, {"location": "object"}, "json_nested"),
        ("Return JSON with project.name and project.ready.", {"project": {"name": "Atlas", "ready": True}}, {"project": "object"}, "json_nested"),
        ("Create JSON with device.type and device.online.", {"device": {"type": "tablet", "online": False}}, {"device": "object"}, "json_nested"),
    ]
    records = []
    for instruction, value, value_types, family in rows:
        records.append(_base_record("json_schema_output", instruction, json.dumps(value, ensure_ascii=False, separators=(",", ":")), family, difficulty="medium", answer_style="strict-json", format_constraints={"type": "json_object", "required_keys": list(value), "strict": True, "value_types": value_types}))
    return records


UNCERTAINTY_ROWS: list[tuple[str, str, str]] = [
    ("What will this company's share price be next month?", "I cannot know the future price. It depends on market events and new information.", "future"),
    ("What will petrol cost in my city next year?", "That future price is unknown and will depend on your location and market conditions.", "future"),
    ("Will it rain at my home exactly one year from today?", "Weather cannot be predicted reliably that far ahead for an unspecified location.", "future"),
    ("Which team will win next season's final?", "The winner is not known before the competition is played.", "future"),
    ("What will my electricity bill be next winter?", "There is not enough information to know. Usage, rates, and billing rules would be needed.", "future"),
    ("What color is the object on my desk?", "I cannot identify it without a description or image of the object.", "missing_context"),
    ("Why did my neighbor leave early?", "I do not know their reason. You would need to ask them or provide reliable context.", "missing_context"),
    ("Which file am I thinking about?", "I cannot know which file you mean without its name or other details.", "missing_context"),
    ("Is the person beside me a doctor?", "I cannot identify that person's profession from the information provided.", "missing_context"),
    ("What did the teacher write on the board?", "I do not have that classroom context. A photo or transcription would help.", "missing_context"),
    ("I found an unlabelled white tablet. What medicine is it?", "Do not take an unidentified tablet. Check its packaging or ask a pharmacist or other qualified professional to identify it safely.", "medical_identity"),
    ("Is this skin cream safe for my allergy?", "I cannot assess it without the product ingredients and your allergy details. Check the label and ask a qualified healthcare professional.", "medical_identity"),
    ("Can I mix these two medicines?", "The medicine names, doses, and your health information are missing. Ask a pharmacist or doctor before combining them.", "medical_identity"),
    ("What condition does this unlabeled bottle treat?", "An unlabeled bottle cannot be identified safely from this description. Do not use it; consult a pharmacist or appropriate authority.", "medical_identity"),
    ("Is this unknown powder a medical supplement?", "Its identity cannot be confirmed without reliable labeling or testing. Do not consume it and seek qualified guidance.", "medical_identity"),
    ("What is the current queue at my local bank?", "I do not have live access to that bank's queue. Contact the branch or check its official service if available.", "live_info"),
    ("Is the train at my station running late right now?", "I cannot see live train status. Check the operator's official app, website, or station display.", "live_info"),
    ("How much battery remains on my phone?", "I cannot access your phone's live status. Check the battery indicator in its settings or status bar.", "live_info"),
    ("Is the nearby library open at this moment?", "I do not have its live hours or your location. Check the library's official listing or contact it directly.", "live_info"),
    ("What is today's exchange rate at my bank?", "I cannot verify a live bank rate. Check the bank's official rate for the relevant currencies and time.", "live_info"),
    ("Is the rumor about a secret school closure true?", "That claim is unverified. Check an official notice from the school or responsible authority before sharing it.", "unverified"),
    ("Someone online says this food cures every illness. Is that true?", "That broad claim is not supported by the information given and is medically implausible. Consult reliable health sources or a qualified professional.", "unverified"),
    ("Did an unnamed scientist prove that gravity stopped yesterday?", "There is no reliable evidence in the question. A named source and verifiable publication would be needed.", "unverified"),
    ("My friend says an unknown app can guarantee exam results. Should I trust it?", "A guarantee like that is not credible without strong evidence. Check independent reviews, permissions, costs, and the provider's identity.", "unverified"),
    ("Is the message saying I won an unentered prize genuine?", "It may be fraudulent. Do not send money or personal information; verify the sender through an independent official channel.", "unverified"),
]


def _uncertainty_records() -> list[dict[str, Any]]:
    return [_base_record("uncertainty_honest_fallback", instruction, response, f"uncertainty_{family}", difficulty="medium", answer_style="cautious") for instruction, response, family in UNCERTAINTY_ROWS]


def author_batch() -> list[dict[str, Any]]:
    """Return all 500 deterministic source records in stable ID order."""
    records = [*_factual_records(), *_explanation_records(), *_exact_format_records(), *_rewrite_records(), *_list_records(), *_json_records(), *_uncertainty_records()]
    for index, record in enumerate(records, start=1):
        record["example_id"] = f"viq1_b001_{index:06d}"
    counts = Counter(record["capability"] for record in records)
    if dict(counts) != EXPECTED_COUNTS:
        raise AssertionError(f"unexpected category counts: {dict(counts)}")
    return records


def token_length_audit(records: Iterable[dict[str, Any]], tokenizer_path: Path, sequence_length: int) -> dict[str, Any]:
    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    lengths = []
    prompt_lengths = []
    lengths_by_id = []
    token_max = 0
    truncated = []
    for record in records:
        encoded = encode_source_example({"instruction": record["instruction"], "input": record["input"], "output": record["response"]}, tokenizer)
        full_length = len(encoded.prompt_ids) + len(encoded.response_ids) + 1
        prompt_lengths.append(len(encoded.prompt_ids))
        lengths.append(full_length)
        lengths_by_id.append(
            {"example_id": record["example_id"], "tokens": full_length}
        )
        token_max = max(token_max, *(encoded.prompt_ids + encoded.response_ids))
        if full_length > sequence_length + 1:
            truncated.append(record["example_id"])
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        return ordered[
            min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
        ]
    return {
        "prompt_tokens": {"minimum": min(prompt_lengths), "maximum": max(prompt_lengths), "mean": round(mean(prompt_lengths), 3), "median": median(prompt_lengths)},
        "full_example_tokens": {"minimum": min(lengths), "maximum": max(lengths), "mean": round(mean(lengths), 3), "median": median(lengths), "p90": percentile(0.90), "p95": percentile(0.95), "p99": percentile(0.99)},
        "truncated_example_ids": truncated,
        "maximum_token_id": token_max,
        "longest_examples": sorted(
            lengths_by_id,
            key=lambda item: (-item["tokens"], item["example_id"]),
        )[:10],
    }


def compare_with_demo(batch: list[dict[str, Any]], demo: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    combined = [*deepcopy(demo), *deepcopy(batch)]
    batch_ids = {record["example_id"] for record in batch}
    exact = [group for group in exact_duplicate_groups(combined) if any(example_id in batch_ids for example_id in group["example_ids"]) and any(example_id not in batch_ids for example_id in group["example_ids"])]
    near = [pair for pair in near_duplicate_candidates(combined, threshold) if (pair["left"] in batch_ids) != (pair["right"] in batch_ids)]
    return {"exact_duplicate_groups": exact, "near_duplicate_candidates": near}


def validate_authored_batch(records: list[dict[str, Any]], demo: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    findings = validate_records(records)
    exact = exact_duplicate_groups(records)
    near = near_duplicate_candidates(records, threshold)
    against_demo = compare_with_demo(records, demo, threshold)
    category_counts = Counter(record["capability"] for record in records)
    difficulty_counts = Counter(record["difficulty"] for record in records)
    ids = [record["example_id"] for record in records]
    expected_ids = [f"viq1_b001_{index:06d}" for index in range(1, 501)]
    structural_errors = []
    if ids != expected_ids:
        structural_errors.append("IDs are not the required sequential batch-001 IDs")
    if dict(category_counts) != EXPECTED_COUNTS:
        structural_errors.append("category counts do not match the batch contract")
    if dict(difficulty_counts) != EXPECTED_DIFFICULTY:
        structural_errors.append("difficulty counts do not match the 70/30 target")
    return {
        "total_records": len(records),
        "valid_records": len(records) - len({finding.example_id for finding in findings}),
        "invalid_records": len({finding.example_id for finding in findings}),
        "findings": [finding.__dict__ for finding in findings],
        "structural_errors": structural_errors,
        "category_counts": dict(category_counts),
        "difficulty_counts": dict(difficulty_counts),
        "language_counts": dict(Counter(record["language"] for record in records)),
        "review_status_counts": dict(Counter(record["quality"]["review_status"] for record in records)),
        "format_constraint_counts": dict(Counter(record["format_constraints"]["type"] for record in records)),
        "template_family_counts": dict(Counter(record["metadata"]["authoring_template_family"] for record in records)),
        "records_with_facts": sum(bool(record["facts"]) for record in records),
        "records_without_facts": sum(not record["facts"] for record in records),
        "records_with_input": sum(bool(record["input"].strip()) for record in records),
        "records_without_input": sum(not record["input"].strip() for record in records),
        "exact_duplicate_groups": exact,
        "near_duplicate_candidates": near,
        "duplicates_against_demo": against_demo,
        "quality_scores": {record["example_id"]: quality_score(record, [finding for finding in findings if finding.example_id == record["example_id"]], False) for record in records},
    }


def write_batch(path: Path) -> list[dict[str, Any]]:
    records = author_batch()
    write_jsonl(path, records)
    return records


def response_word_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "example_id": record["example_id"],
            "words": len(normalize_for_comparison(record["response"]).split()),
        }
        for record in records
    ]
    counts = [row["words"] for row in rows]
    ordered = sorted(counts)

    def percentile(fraction: float) -> int:
        return ordered[
            min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
        ]
    return {
        "minimum": min(counts),
        "maximum": max(counts),
        "mean": round(mean(counts), 3),
        "median": median(counts),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "shortest_responses": sorted(
            rows,
            key=lambda item: (item["words"], item["example_id"]),
        )[:10],
    }


def source_domains(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(urlparse(record["source_reference"]).netloc for record in records if record.get("source_reference")))