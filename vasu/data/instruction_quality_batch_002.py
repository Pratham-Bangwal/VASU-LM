"""Deterministic, purpose-written authoring for instruction-quality batch 002."""

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


AUTHORING_VERSION = "vasu_instruction_quality_batch_002_authoring_v1"
CREATED_AT = "2026-07-20T00:00:00Z"

EXPECTED_COUNTS = {
    "short_factual_qa": 125,
    "beginner_explanation": 100,
    "exact_format_following": 100,
    "rewriting_transformation": 75,
    "lists_structured_output": 50,
    "json_schema_output": 25,
    "uncertainty_honest_fallback": 25,
}

EXPECTED_DIFFICULTY = {
    "easy": 300,
    "medium": 175,
    "hard": 25,
}


def _metadata(family: str) -> dict[str, Any]:
    return {
        "created_at": CREATED_AT,
        "created_by": "VASU batch authoring pipeline",
        "license": "CC0-1.0",
        "provenance": "purpose-written VASU instruction-quality batch 002",
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
    rows = [
        (
            "About how long does Mercury take to orbit the Sun?",
            "About 88 Earth days.",
            "Mercury completes one orbit around the Sun in about 88 Earth days.",
            "https://science.nasa.gov/mercury/",
        ),
        (
            "What useful gas is released during photosynthesis?",
            "Oxygen.",
            "Photosynthesis releases oxygen.",
            "https://www.britannica.com/science/photosynthesis",
        ),
        (
            "What gives an object weight near Earth?",
            "Earth's gravity.",
            "Earth's gravity gives nearby objects weight.",
            "https://www.britannica.com/science/gravity-physics",
        ),
        (
            "What two elements make up water?",
            "Hydrogen and oxygen.",
            "Water is made of hydrogen and oxygen.",
            "https://www.britannica.com/science/water",
        ),
        (
            "What type of astronomical object is the Sun?",
            "A star.",
            "The Sun is a star.",
            "https://science.nasa.gov/sun/",
        ),
        (
            "Why can people see the Moon from Earth?",
            "It reflects sunlight.",
            "The Moon is visible because it reflects sunlight.",
            "https://science.nasa.gov/moon/",
        ),
        (
            "What does a router examine when forwarding network traffic?",
            "Network addresses.",
            "A router uses network addresses to forward traffic.",
            "https://www.cloudflare.com/learning/network-layer/what-is-a-router/",
        ),
        (
            "What does a URL identify?",
            "The location of a web resource.",
            "A URL identifies the location of a web resource.",
            "https://developer.mozilla.org/en-US/docs/Glossary/URL",
        ),
        (
            "What kind of signals does Bluetooth use?",
            "Radio signals.",
            "Bluetooth uses radio signals for wireless communication.",
            "https://www.bluetooth.com/learn-about-bluetooth/tech-overview/",
        ),
        (
            "What does a GPS receiver use to estimate its position?",
            "Signals from satellites.",
            "A GPS receiver estimates position using satellite signals.",
            "https://oceanservice.noaa.gov/education/tutorial_geodesy/geo09_gps.html",
        ),
        (
            "Why is a separate backup copy useful?",
            "It allows data recovery after loss or damage.",
            "A separate backup can restore data after the original is lost or damaged.",
            "https://www.cisa.gov/news-events/news/data-backup-options",
        ),
        (
            "Why should passwords be unique across accounts?",
            "A stolen password then affects fewer accounts.",
            "Unique passwords limit the damage caused by one compromised password.",
            "https://www.cisa.gov/secure-our-world/use-strong-passwords",
        ),
        (
            "What medium does Wi-Fi commonly use to carry data?",
            "Radio waves.",
            "Wi-Fi commonly carries data using radio waves.",
            "https://www.wi-fi.org/discover-wi-fi",
        ),
        (
            "What is the main purpose of antivirus software?",
            "To detect and block malicious software.",
            "Antivirus software detects and blocks malicious software.",
            "https://www.cisa.gov/news-events/news/understanding-anti-virus-software",
        ),
        (
            "What can a hyperlink connect a user to?",
            "Another web resource.",
            "A hyperlink can connect a user to another web resource.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Hyperlink",
        ),
        (
            "What does a web browser interpret to display websites?",
            "Web content such as HTML.",
            "A browser interprets web content such as HTML for display.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Browser",
        ),
        (
            "Where are cloud-storage files kept?",
            "On remote servers.",
            "Cloud-storage files are kept on remote servers.",
            "https://www.ibm.com/think/topics/cloud-storage",
        ),
        (
            "What is the CPU mainly responsible for?",
            "Executing program instructions.",
            "A CPU executes program instructions.",
            "https://www.britannica.com/technology/central-processing-unit",
        ),
        (
            "Does RAM normally retain its contents after power is removed?",
            "No.",
            "RAM normally loses its stored contents when power is removed.",
            "https://www.britannica.com/technology/RAM-computing",
        ),
        (
            "What can a programming variable hold?",
            "A value.",
            "A programming variable can hold a value.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Variable",
        ),
        (
            "What does a programming loop do?",
            "It repeats instructions.",
            "A programming loop repeats instructions.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Loop",
        ),
        (
            "What does an algorithm describe?",
            "A sequence of steps for solving a problem.",
            "An algorithm describes steps for solving a problem.",
            "https://www.britannica.com/science/algorithm",
        ),
        (
            "What does an operating system manage?",
            "Computer hardware and software resources.",
            "An operating system manages hardware and software resources.",
            "https://www.britannica.com/technology/operating-system",
        ),
        (
            "What is a database designed to organize?",
            "Related data.",
            "A database organizes related data.",
            "https://www.britannica.com/technology/database",
        ),
        (
            "What does HTML primarily describe?",
            "The structure and content of a web page.",
            "HTML describes the structure and content of a web page.",
            "https://developer.mozilla.org/en-US/docs/Glossary/HTML",
        ),
        (
            "What does the kilogram measure in the SI system?",
            "Mass.",
            "The kilogram is the SI base unit used to measure mass.",
            "https://www.bipm.org/en/si-base-units/kilogram",
        ),
        (
            "How is the second defined in the SI system?",
            "Using the frequency of radiation from cesium-133 atoms.",
            "The SI second is defined using a specified transition frequency "
            "of the cesium-133 atom.",
            "https://www.bipm.org/en/si-base-units/second",
        ),
        (
            "What physical quantity is measured in amperes?",
            "Electric current.",
            "The ampere is the SI base unit for electric current.",
            "https://www.bipm.org/en/si-base-units/ampere",
        ),
        (
            "What physical quantity is measured in kelvins?",
            "Thermodynamic temperature.",
            "The kelvin is the SI base unit for thermodynamic temperature.",
            "https://www.bipm.org/en/si-base-units/kelvin",
        ),
        (
            "What quantity does the mole measure?",
            "Amount of substance.",
            "The mole is the SI base unit for amount of substance.",
            "https://www.bipm.org/en/si-base-units/mole",
        ),
        (
            "How many centimetres are in half a metre?",
            "Fifty centimetres.",
            "Half a metre equals fifty centimetres.",
            "https://www.nist.gov/pml/owm/si-units-length",
        ),
        (
            "How many years make one century?",
            "One hundred years.",
            "A century is a period of one hundred years.",
            "https://www.merriam-webster.com/dictionary/century",
        ),
        (
            "How many days are normally in a leap year?",
            "Three hundred sixty-six days.",
            "A leap year normally contains 366 days.",
            "https://www.britannica.com/science/year",
        ),
        (
            "What structure surrounds the brain inside the head?",
            "The cranium.",
            "The cranium is the part of the skull that surrounds the brain.",
            "https://www.britannica.com/science/skull",
        ),
        (
            "What part of hemoglobin binds oxygen?",
            "Iron-containing heme groups.",
            "Oxygen binds to iron-containing heme groups in hemoglobin.",
            "https://www.britannica.com/science/hemoglobin",
        ),
        (
            "What do the kidneys remove from the blood?",
            "Waste and excess water.",
            "The kidneys remove waste and excess water from the blood.",
            "https://www.niddk.nih.gov/health-information/"
            "kidney-disease/kidneys-how-they-work",
        ),
        (
            "What muscular movement pushes food through the esophagus?",
            "Peristalsis.",
            "Peristalsis pushes food through the esophagus toward the stomach.",
            "https://www.britannica.com/science/peristalsis",
        ),
        (
            "What is debugging used to find in software?",
            "Errors.",
            "Debugging is used to find and correct software errors.",
            "https://www.britannica.com/technology/debugging",
        ),
        (
            "What does a compiler process?",
            "Source code.",
            "A compiler processes source code and translates it into another form.",
            "https://www.britannica.com/technology/compiler",
        ),
        (
            "Why are functions useful in programs?",
            "They make reusable instructions possible.",
            "Functions allow program instructions to be reused.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Function",
        ),
        (
            "What two values can a binary bit represent?",
            "Zero and one.",
            "A binary bit can represent zero or one.",
            "https://www.merriam-webster.com/dictionary/bit",
        ),
        (
            "What language is commonly used to describe web-page structure?",
            "HTML.",
            "HTML is commonly used to describe web-page structure.",
            "https://developer.mozilla.org/en-US/docs/Glossary/HTML",
        ),
        (
            "What language is commonly used to style web pages?",
            "CSS.",
            "CSS is commonly used to control the presentation of web pages.",
            "https://developer.mozilla.org/en-US/docs/Glossary/CSS",
        ),
        (
            "What does encryption protect data from?",
            "Unauthorized reading.",
            "Encryption helps protect data from unauthorized reading.",
            "https://www.cisa.gov/news-events/news/"
            "what-encryption-and-how-does-it-work",
        ),
        (
            "What is phishing designed to steal?",
            "Sensitive information.",
            "Phishing attempts are designed to steal sensitive information.",
            "https://www.cisa.gov/secure-our-world/recognize-and-report-phishing",
        ),
        (
            "What happens during condensation?",
            "A gas changes into a liquid.",
            "Condensation is the change of a gas into a liquid.",
            "https://www.britannica.com/science/condensation-phase-change",
        ),
        (
            "What can move soil during erosion?",
            "Water, wind, ice, or gravity.",
            "Water, wind, ice, and gravity can move soil during erosion.",
            "https://www.nrcs.usda.gov/resources/"
            "education-and-teaching-materials/soil-erosion",
        ),
        (
            "What does biodiversity describe?",
            "The variety of living things.",
            "Biodiversity describes the variety of living things.",
            "https://www.epa.gov/enviroatlas/biodiversity-conservation",
        ),
        (
            "What makes solar energy renewable?",
            "Sunlight is naturally replenished.",
            "Solar energy is renewable because sunlight is naturally replenished.",
            "https://www.energy.gov/eere/solar/how-does-solar-work",
        ),
        (
            "What atmospheric process forms clouds from water vapor?",
            "Condensation.",
            "Condensation of water vapor helps form clouds.",
            "https://www.noaa.gov/education/resource-collections/"
            "weather-atmosphere/water-cycle",
        ),
        (
            "How many planets orbit the Sun in our solar system?",
            "Eight planets.",
            "Eight planets orbit the Sun in our solar system.",
            "https://science.nasa.gov/solar-system/planets/",
        ),
        (
            "Which planet has the shortest day in the solar system?",
            "Jupiter has the shortest day.",
            "Jupiter completes one rotation in about ten hours, giving it "
            "the shortest day of any planet.",
            "https://science.nasa.gov/jupiter/jupiter-facts/",
        ),
        (
            "Which planet is seventh from the Sun?",
            "Uranus.",
            "Uranus is the seventh planet from the Sun.",
            "https://science.nasa.gov/uranus/facts/",
        ),
        (
            "Which planet is farthest from the Sun?",
            "Neptune.",
            "Neptune is the farthest planet from the Sun.",
            "https://science.nasa.gov/neptune/neptune-facts/",
        ),
        (
            "Which planet rotates in the opposite direction from most planets?",
            "Venus.",
            "Venus rotates in the opposite direction from most planets.",
            "https://science.nasa.gov/venus/venus-facts/",
        ),
        (
            "What galaxy contains our solar system?",
            "The Milky Way.",
            "Our solar system is located in the Milky Way galaxy.",
            "https://science.nasa.gov/solar-system/solar-system-facts/",
        ),
        (
            "What is another name for a moon?",
            "A natural satellite.",
            "A moon is also called a natural satellite.",
            "https://science.nasa.gov/solar-system/moons/",
        ),
        (
            "Which terrestrial planet has the largest diameter?",
            "Earth has the largest diameter among the terrestrial planets.",
            "Earth is the largest terrestrial planet by diameter.",
            "https://science.nasa.gov/earth/facts/",
        ),
        (
            "What happens to liquid water during evaporation?",
            "It changes into water vapor.",
            "Evaporation changes liquid water into water vapor.",
            "https://www.nesdis.noaa.gov/about/k-12-education/"
            "atmosphere/what-humidity",
        ),
        (
            "What does relative humidity compare?",
            "The water vapor present with how much the air can hold.",
            "Relative humidity compares the water vapor present with the "
            "amount the air can hold at that temperature.",
            "https://www.nesdis.noaa.gov/about/k-12-education/"
            "atmosphere/what-humidity",
        ),
        (
            "What process releases water vapor from plants?",
            "Transpiration.",
            "Transpiration releases water vapor from plants.",
            "https://gml.noaa.gov/outreach/lesson_plans/"
            "The%20Hydrologic%20Cycle.pdf",
        ),
        (
            "Where can water be stored below Earth's surface?",
            "As groundwater.",
            "Water stored below Earth's surface is called groundwater.",
            "https://www.usgs.gov/water-science-school/water-cycle",
        ),
        (
            "In what three physical states does water naturally occur on Earth?",
            "Solid, liquid, and gas.",
            "Water naturally occurs on Earth as a solid, liquid, and gas.",
            "https://oceantoday.noaa.gov/watercycle/",
        ),
        (
            "What is the underground starting point of an earthquake called?",
            "The hypocenter.",
            "The hypocenter is the location below Earth's surface where an "
            "earthquake begins.",
            "https://www.usgs.gov/programs/earthquake-hazards/"
            "science-earthquakes",
        ),
        (
            "What is the surface point directly above an earthquake's "
            "hypocenter called?",
            "The epicenter.",
            "The epicenter is the point on Earth's surface directly above "
            "an earthquake's hypocenter.",
            "https://www.usgs.gov/programs/earthquake-hazards/"
            "science-earthquakes",
        ),
        (
            "What surface do blocks of Earth slip along during an earthquake?",
            "A fault.",
            "Blocks of Earth slip along a fault during an earthquake.",
            "https://www.usgs.gov/programs/earthquake-hazards/"
            "science-earthquakes",
        ),
        (
            "Where do most earthquakes and volcanic eruptions occur?",
            "Along tectonic plate boundaries.",
            "Most earthquakes and volcanic eruptions occur along tectonic "
            "plate boundaries.",
            "https://www.usgs.gov/programs/earthquake-hazards/"
            "cool-earthquake-facts",
        ),
        (
            "What can rising magma produce before a volcanic eruption?",
            "Detectable earthquakes.",
            "Rising magma commonly produces detectable earthquakes before "
            "a volcanic eruption.",
            "https://www.usgs.gov/faqs/"
            "how-can-we-tell-when-a-volcano-will-erupt",
        ),
        (
            "What instrument network records earthquakes?",
            "A seismographic network.",
            "Earthquakes are recorded by a seismographic network.",
            "https://www.usgs.gov/science/faqs/methods-and-analysis",
        ),
        (
            "Which organ stores bile until it is needed for digestion?",
            "The gallbladder stores bile.",
            "The gallbladder stores bile and releases it into the small "
            "intestine during digestion.",
            "https://www.niddk.nih.gov/health-information/"
            "digestive-diseases/digestive-system-how-it-works",
        ),
        (
            "What type of chemical messenger is insulin?",
            "A hormone.",
            "Insulin is a hormone produced by cells in the pancreas.",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8232639/",
        ),
        (
            "What body system contains glands such as the thyroid and pituitary?",
            "The endocrine system.",
            "The endocrine system includes glands such as the thyroid and "
            "pituitary glands.",
            "https://www.niddk.nih.gov/health-information/"
            "endocrine-diseases",
        ),
        (
            "What do hormones act as inside the body?",
            "Chemical messengers.",
            "Hormones act as chemical messengers that regulate body functions.",
            "https://www.ncbi.nlm.nih.gov/books/NBK538498/",
        ),
        (
            "Which digestive organ also produces important hormones?",
            "The pancreas.",
            "The pancreas produces hormones that are important to digestion.",
            "https://www.niddk.nih.gov/health-information/"
            "digestive-diseases/digestive-system-how-it-works",
        ),
        (
            "What can cells in the stomach and small intestine release to "
            "control digestion?",
            "Hormones.",
            "Cells in the stomach and small intestine release hormones that "
            "help control digestion.",
            "https://www.niddk.nih.gov/health-information/"
            "digestive-diseases/digestive-system-how-it-works",
        ),
        (
            "What does the Sun's gravity do for the solar system?",
            "It keeps solar-system objects in orbit.",
            "The Sun's gravity holds the solar system together and keeps "
            "objects in orbit.",
            "https://science.nasa.gov/sun/facts/",
        ),
        (
            "Approximately how long does the solar system take to orbit the "
            "center of the Milky Way?",
            "About 230 million years.",
            "The solar system takes about 230 million years to orbit the "
            "center of the Milky Way.",
            "https://science.nasa.gov/solar-system/solar-system-facts/",
        ),
        (
            "Which planet is the hottest in the solar system?",
            "Venus is the hottest planet.",
            "Venus is the hottest planet even though Mercury is closer "
            "to the Sun.",
            "https://science.nasa.gov/solar-system/solar-system-facts/",
        ),
        (
            "What is a planet outside our solar system called?",
            "An exoplanet.",
            "A planet outside our solar system is called an exoplanet.",
            "https://science.nasa.gov/universe/stars/planetary-system/",
        ),
        (
            "What are the four innermost planets collectively called?",
            "The terrestrial planets.",
            "Mercury, Venus, Earth, and Mars are collectively called the "
            "terrestrial planets.",
            "https://science.nasa.gov/learn/basics-of-space-flight/"
            "chapter1-2/",
        ),
        (
            "What is Pluto classified as?",
            "A dwarf planet.",
            "Pluto is classified as a dwarf planet.",
            "https://science.nasa.gov/dwarf-planets/pluto/",
        ),
        (
            "What is the SI base unit of length?",
            "The meter is the SI base unit of length.",
            "The meter is the SI base unit used to measure length.",
            "https://www.nist.gov/pml/owm/si-units-length",
        ),
        (
            "How many SI base units are there?",
            "There are seven SI base units.",
            "The International System of Units contains seven base units.",
            "https://www.nist.gov/pml/owm/metric-si/si-units",
        ),
        (
            "What symbol represents the SI unit of time?",
            "The symbol is s.",
            "The symbol for the SI second is a lowercase s.",
            "https://www.nist.gov/pml/owm/si-units-time",
        ),
        (
            "What is the SI symbol for the kelvin?",
            "The symbol is K.",
            "The SI symbol for the kelvin is an uppercase K.",
            "https://www.nist.gov/pml/owm/si-units-temperature",
        ),
        (
            "How many centimetres are in two metres?",
            "Two hundred centimetres.",
            "Two metres equal two hundred centimetres.",
            "https://www.nist.gov/pml/owm/si-units-length",
        ),
        (
            "How many millimetres are in half a centimetre?",
            "Five millimetres.",
            "Half a centimetre equals five millimetres.",
            "https://www.nist.gov/pml/owm/si-units-length",
        ),
        (
            "Where is groundwater found?",
            "Below Earth's surface.",
            "Groundwater is water located below Earth's surface.",
            "https://www.usgs.gov/special-topics/water-science-school/"
            "groundwater",
        ),
        (
            "What underground material can store and transmit groundwater?",
            "An aquifer.",
            "An aquifer is underground material that can store and transmit "
            "groundwater.",
            "https://www.usgs.gov/special-topics/water-science-school/"
            "science/aquifers-and-groundwater",
        ),
        (
            "What is a naturally flowing groundwater outlet called?",
            "A spring.",
            "A spring is a place where groundwater naturally flows onto "
            "Earth's surface.",
            "https://www.usgs.gov/special-topics/water-science-school/"
            "science/springs-and-water-cycle",
        ),
        (
            "What is the upper surface of groundwater called?",
            "The water table.",
            "The water table is the upper surface of the groundwater-saturated "
            "zone.",
            "https://www.usgs.gov/special-topics/water-science-school/"
            "science/groundwater-what-groundwater",
        ),
        (
            "What does DNS translate domain names into?",
            "IP addresses.",
            "DNS translates human-readable domain names into IP addresses.",
            "https://developer.mozilla.org/en-US/docs/Glossary/DNS",
        ),
        (
            "What does a web cache temporarily store?",
            "HTTP responses.",
            "A web cache temporarily stores HTTP responses for possible reuse.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Cache",
        ),
        (
            "What does a cache TTL specify?",
            "How long cached data may remain valid.",
            "A cache TTL specifies the amount of time for which cached data "
            "may be used.",
            "https://developer.mozilla.org/en-US/docs/Glossary/TTL",
        ),
        (
            "Which HTTP header allows a server to send a cookie to a browser?",
            "The Set-Cookie header.",
            "A server uses the Set-Cookie HTTP header to send a cookie to "
            "a browser.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Cookie",
        ),
        (
            "Which HTTP request header sends stored cookies back to a server?",
            "The Cookie header.",
            "The Cookie request header sends applicable stored cookies back "
            "to a server.",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/"
            "Reference/Headers/Cookie",
        ),
        (
            "What does the HTTP Cache-Control header manage?",
            "Caching behavior.",
            "The Cache-Control header contains directives that manage browser "
            "and shared-cache behavior.",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/"
            "Reference/Headers/Cache-Control",
        ),
        (
            "What type of protocol is HTTP?",
            "An application-layer protocol.",
            "HTTP is an application-layer protocol used for transferring "
            "resources on the web.",
            "https://developer.mozilla.org/en-US/docs/Glossary/HTTP",
        ),
        (
            "What does an HTTP response header provide?",
            "Information about the server's response.",
            "An HTTP response header provides metadata or instructions about "
            "a server's response.",
            "https://developer.mozilla.org/en-US/docs/Glossary/"
            "Response_header",
        ),
        (
            "Why can using a cached HTTP response improve performance?",
            "It can avoid downloading or generating the same response again.",
            "Using a cached response can avoid another network transfer or "
            "server computation.",
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/"
            "Guides/Caching",
        ),
        (
            "How many natural moons does Mars have?",
            "Mars has two natural moons.",
            "Mars has two moons named Phobos and Deimos.",
            "https://science.nasa.gov/mars/moons/facts/",
        ),
        (
            "What are the names of Mars's two moons?",
            "Phobos and Deimos.",
            "The two moons of Mars are named Phobos and Deimos.",
            "https://science.nasa.gov/mars/moons/facts/",
        ),
        (
            "Which of Mars's moons is larger?",
            "Phobos is larger.",
            "Phobos is the larger of Mars's two moons.",
            "https://science.nasa.gov/mars/moons/facts/",
        ),
        (
            "Why are seasons on Mars longer than seasons on Earth?",
            "Mars takes longer to orbit the Sun.",
            "Martian seasons are longer because Mars takes longer than Earth "
            "to orbit the Sun.",
            "https://science.nasa.gov/mars/facts/",
        ),
        (
            "What gas is the second most abundant in the Martian atmosphere?",
            "Nitrogen.",
            "Nitrogen is the second most abundant gas in the atmosphere "
            "of Mars.",
            "https://www.jpl.nasa.gov/news/press_kits/insight/"
            "landing/facts/mars-at-a-glance/",
        ),
        (
            "How does the atmosphere of Mars compare with Earth's atmosphere?",
            "It is much thinner.",
            "The atmosphere of Mars is much thinner than Earth's atmosphere.",
            "https://www.nasa.gov/learning-resources/for-kids-and-students/"
            "what-is-mars-grades-5-8/",
        ),
        (
            "What is the only moon known to have a thick atmosphere?",
            "Titan.",
            "Titan is the only moon known to have a thick atmosphere.",
            "https://science.nasa.gov/solar-system/moons/facts/",
        ),
        (
            "Does Earth's Moon have natural moons of its own?",
            "No, it does not.",
            "Earth's Moon has no natural moons of its own.",
            "https://science.nasa.gov/moon/facts/",
        ),
        (
            "What is an array in programming?",
            "An ordered collection of data.",
            "An array is an ordered collection of data values.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Array",
        ),
        (
            "At what index does a JavaScript array begin?",
            "Index zero.",
            "JavaScript array indexing begins at zero.",
            "https://developer.mozilla.org/en-US/docs/Learn_web_development/"
            "Core/Scripting/Arrays",
        ),
        (
            "Can a JavaScript array contain different types of values?",
            "Yes, it can contain values of different types.",
            "A JavaScript array can contain values of different data types.",
            "https://developer.mozilla.org/en-US/docs/Learn_web_development/"
            "Core/Scripting/Arrays",
        ),
        (
            "How are JavaScript array elements normally accessed?",
            "Using nonnegative integer indexes.",
            "JavaScript array elements are normally accessed using "
            "nonnegative integer indexes.",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript/"
            "Reference/Global_Objects/Array",
        ),
        (
            "What does the arguments object contain inside a JavaScript "
            "function?",
            "The values passed to the function.",
            "The arguments object contains values passed as arguments to "
            "a JavaScript function.",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript/"
            "Reference/Functions/arguments",
        ),
        (
            "Is a JavaScript function also an object?",
            "Yes, a JavaScript function is also an object.",
            "A JavaScript function is also an object.",
            "https://developer.mozilla.org/en-US/docs/Glossary/Function",
        ),
        (
            "What causes a glacier to move downhill?",
            "Its own weight and gravity.",
            "A glacier moves downhill under the influence of its weight "
            "and gravity.",
            "https://www.usgs.gov/water-science-school/science/"
            "glaciers-and-icecaps",
        ),
        (
            "Does a glacier originate on land or in the ocean?",
            "It originates on land.",
            "A glacier is a persistent accumulation of ice that originates "
            "on land.",
            "https://www.usgs.gov/water-science-school/science/"
            "glaciers-and-icecaps",
        ),
        (
            "How can a glacier change the landscape?",
            "By eroding it.",
            "Moving glaciers can erode and reshape the landscape.",
            "https://www.usgs.gov/water-science-school/science/"
            "glaciers-and-icecaps",
        ),
        (
            "What are smaller streams that join a larger river called?",
            "Tributaries.",
            "Smaller streams that join a larger river are called tributaries.",
            "https://www.usgs.gov/water-science-school/science/"
            "rivers-streams-and-creeks",
        ),
        (
            "Where does much of the water flowing in rivers come from?",
            "Precipitation runoff.",
            "Much river water comes from precipitation that runs off nearby "
            "land.",
            "https://www.usgs.gov/water-science-school/science/"
            "rivers-streams-and-creeks",
        ),
        (
            "Can groundwater contribute water to rivers?",
            "Yes, groundwater can contribute to river flow.",
            "Groundwater can seep into riverbeds and contribute to river flow.",
            "https://www.usgs.gov/water-science-school/science/"
            "how-much-water-there-earth",
        ),
        (
            "Where is most of Earth's water found?",
            "In the oceans.",
            "Most of Earth's water is stored in the oceans.",
            "https://www.usgs.gov/water-science-school/science/"
            "how-much-water-there-earth",
        ),
        (
            "What allows water to move upward through narrow plant tissues?",
            "Capillary action.",
            "Capillary action helps water move through narrow tissues in "
            "plant roots.",
            "https://www.usgs.gov/water-science-school/science/"
            "facts-about-water",
        ),
        (
            "What is sublimation in the water cycle?",
            "The direct change from ice to water vapor.",
            "Sublimation changes solid ice directly into water vapor.",
            "https://www.usgs.gov/water-science-school/science/"
            "ice-snow-and-glaciers-and-water-cycle",
        ),
        (
            "What is surface runoff?",
            "Water flowing over the land surface.",
            "Surface runoff is water that flows across the land surface.",
            "https://www.usgs.gov/water-science-school/water-cycle",
        ),
        (
            "What happens when snowmelt enters rivers and streams?",
            "It contributes to streamflow.",
            "Water from melting snow can contribute to river and stream flow.",
            "https://www.usgs.gov/water-science-school/science/"
            "ice-snow-and-glaciers-and-water-cycle",
        ),
    ]

    records: list[dict[str, Any]] = []

    for instruction, response, claim, source in rows:
        records.append(
            _base_record(
                capability="short_factual_qa",
                instruction=instruction,
                response=response,
                family="batch_002_factual_part_01",
                difficulty="easy",
                source_reference=source,
                facts=[
                    {
                        "claim": claim,
                        "verification_status": "verified",
                        "verification_source": source,
                        "time_sensitive": False,
                    }
                ],
            )
        )

    if len(records) != 125:
        raise AssertionError(
            "Batch 002 factual catalog must currently contain 125 rows"
        )

    return records


def _explanation_records() -> list[dict[str, Any]]:
    rows = [
        (
            "Explain what an operating system does in simple words.",
            "An operating system manages the computer's hardware and lets "
            "programs use it. It handles tasks such as files, memory, input "
            "devices, and running applications.",
            "operating_system_basics",
            "easy",
        ),
        (
            "Explain RAM to a beginner.",
            "RAM is temporary working space used by a computer while programs "
            "are running. More RAM allows the computer to keep more data ready "
            "for quick access, but its contents are normally lost when power "
            "is turned off.",
            "ram_basics",
            "easy",
        ),
        (
            "Explain what a computer file is.",
            "A file is a named collection of data stored on a device. A file "
            "may contain text, an image, a video, program code, or another "
            "kind of information.",
            "file_basics",
            "easy",
        ),
        (
            "Explain what a folder does on a computer.",
            "A folder groups files and other folders together. It helps keep "
            "stored information organized and easier to find.",
            "folder_basics",
            "easy",
        ),
        (
            "Explain what a programming function is.",
            "A function is a named group of instructions that performs a "
            "specific task. A program can call the function whenever that "
            "task is needed instead of repeating the same code.",
            "function_basics",
            "easy",
        ),
        (
            "Explain a variable in programming.",
            "A variable is a name that refers to a value used by a program. "
            "The value may change while the program runs.",
            "variable_basics",
            "easy",
        ),
        (
            "Explain what a loop does in programming.",
            "A loop repeats a group of instructions. It is useful when the "
            "same action must be performed several times.",
            "loop_basics",
            "easy",
        ),
        (
            "Explain what a conditional statement does.",
            "A conditional statement lets a program choose what to do based "
            "on whether a condition is true or false.",
            "conditional_basics",
            "easy",
        ),
        (
            "Explain what a list is in programming.",
            "A list stores several values in an ordered collection. Programs "
            "can access, add, remove, or change the values in the list.",
            "list_basics",
            "easy",
        ),
        (
            "Explain what debugging means.",
            "Debugging is the process of finding, understanding, and fixing "
            "problems in a program.",
            "debugging_basics",
            "easy",
        ),
        (
            "Explain what source code is.",
            "Source code is the human-readable set of instructions written "
            "by a programmer. It is later interpreted or translated so the "
            "computer can execute it.",
            "source_code_basics",
            "easy",
        ),
        (
            "Explain what a database is in simple words.",
            "A database is an organized collection of related information. "
            "It allows programs to store, search, update, and retrieve data.",
            "database_basics",
            "easy",
        ),
        (
            "Explain what the internet is.",
            "The internet is a worldwide network of connected computer "
            "networks. It allows devices to exchange data using agreed "
            "communication rules.",
            "internet_basics",
            "easy",
        ),
        (
            "Explain the difference between the internet and the web.",
            "The internet is the network that connects devices. The web is "
            "one service that uses the internet to deliver websites and "
            "linked pages.",
            "internet_web_difference",
            "medium",
        ),
        (
            "Explain what an IP address is.",
            "An IP address is a numerical address used to identify a device "
            "or network connection. It helps data reach the correct "
            "destination.",
            "ip_address_basics",
            "easy",
        ),
        (
            "Explain what DNS does.",
            "DNS converts readable domain names into IP addresses. This lets "
            "people use names such as example.com instead of remembering "
            "numerical addresses.",
            "dns_basics",
            "easy",
        ),
        (
            "Explain what a web server does.",
            "A web server receives requests from browsers and sends back web "
            "resources such as HTML pages, images, or data.",
            "web_server_basics",
            "easy",
        ),
        (
            "Explain what a web browser does.",
            "A web browser requests content from web servers and displays it "
            "as websites that people can read and interact with.",
            "browser_basics",
            "easy",
        ),
        (
            "Explain what encryption does.",
            "Encryption changes readable data into a protected form that "
            "cannot be understood without the correct key. It helps keep "
            "information private.",
            "encryption_basics",
            "easy",
        ),
        (
            "Explain what a backup is.",
            "A backup is an extra copy of important data. It can be used to "
            "restore the data if the original copy is deleted, damaged, or "
            "lost.",
            "backup_basics",
            "easy",
        ),
        (
            "Explain gravity to a child.",
            "Gravity is the pull that attracts objects toward one another. "
            "Earth's gravity keeps people on the ground and makes dropped "
            "objects fall.",
            "gravity_child",
            "easy",
        ),
        (
            "Explain why day and night happen.",
            "Earth rotates on its axis. The side facing the Sun experiences "
            "day, while the side turned away from the Sun experiences night.",
            "day_night_basics",
            "easy",
        ),
        (
            "Explain evaporation in simple words.",
            "Evaporation happens when liquid molecules gain enough energy to "
            "escape into the air as gas. For example, liquid water can become "
            "water vapor.",
            "evaporation_basics",
            "easy",
        ),
        (
            "Explain condensation in simple words.",
            "Condensation happens when a gas cools and changes into a liquid. "
            "Water droplets forming on a cold glass are an example.",
            "condensation_basics",
            "easy",
        ),
        (
            "Explain photosynthesis to a beginner.",
            "Photosynthesis is the process plants use to make food from light, "
            "water, and carbon dioxide. During the process, they also release "
            "oxygen.",
            "photosynthesis_basics",
            "easy",
        ),
        (
            "Explain what a CPU does.",
            "The CPU reads and executes program instructions. It performs "
            "calculations, controls operations, and coordinates work between "
            "different parts of the computer.",
            "cpu_basics",
            "easy",
        ),
        (
            "Explain what storage means in a computer.",
            "Storage keeps files and programs for later use. Unlike RAM, it "
            "normally keeps its contents even after the computer is turned off.",
            "storage_basics",
            "easy",
        ),
        (
            "Explain the difference between RAM and storage.",
            "RAM is temporary working memory used while programs run. Storage "
            "keeps files and applications for long-term use.",
            "ram_storage_difference",
            "medium",
        ),
        (
            "Explain what a computer network is.",
            "A computer network is a group of connected devices that exchange "
            "data and share resources such as files, printers, or internet "
            "access.",
            "network_basics",
            "easy",
        ),
        (
            "Explain what a router does.",
            "A router directs data between different networks. It examines "
            "network addresses and forwards traffic toward the correct "
            "destination.",
            "router_basics",
            "easy",
        ),
        (
            "Explain what Wi-Fi is.",
            "Wi-Fi is a wireless technology that lets devices connect to a "
            "network using radio waves instead of cables.",
            "wifi_basics",
            "easy",
        ),
        (
            "Explain what Bluetooth is.",
            "Bluetooth is a short-range wireless technology used to connect "
            "nearby devices such as headphones, keyboards, and phones.",
            "bluetooth_basics",
            "easy",
        ),
        (
            "Explain what a URL is.",
            "A URL is the address of a resource on the web. It tells a browser "
            "where to find a page, image, file, or other resource.",
            "url_basics",
            "easy",
        ),
        (
            "Explain what HTML does.",
            "HTML describes the structure and content of a web page. It defines "
            "elements such as headings, paragraphs, links, images, and forms.",
            "html_basics",
            "easy",
        ),
        (
            "Explain what CSS does.",
            "CSS controls how web-page content looks. It can change layout, "
            "spacing, fonts, sizes, and other visual properties.",
            "css_basics",
            "easy",
        ),
        (
            "Explain what an algorithm is.",
            "An algorithm is a clear sequence of steps for solving a problem "
            "or completing a task.",
            "algorithm_basics",
            "easy",
        ),
        (
            "Explain what an error message is.",
            "An error message reports that something went wrong. It often "
            "includes clues about the cause and where the problem occurred.",
            "error_message_basics",
            "easy",
        ),
        (
            "Explain what syntax means in programming.",
            "Syntax is the set of rules that defines how code must be written. "
            "Code with invalid syntax cannot be correctly understood by the "
            "language processor.",
            "syntax_basics",
            "easy",
        ),
        (
            "Explain what a compiler does.",
            "A compiler translates source code into another form that a "
            "computer can execute or process more directly.",
            "compiler_basics",
            "easy",
        ),
        (
            "Explain what an interpreter does.",
            "An interpreter reads and executes program instructions, often "
            "while the program is running instead of translating everything "
            "into a separate executable first.",
            "interpreter_basics",
            "medium",
        ),
        (
            "Explain what an API is in simple words.",
            "An API is a defined way for programs to communicate. It specifies "
            "what requests can be made and what responses are returned.",
            "api_basics",
            "easy",
        ),
        (
            "Explain what cloud storage is.",
            "Cloud storage keeps files on remote servers that are accessed "
            "through a network. This can make files available across multiple "
            "devices.",
            "cloud_storage_basics",
            "easy",
        ),
        (
            "Explain what malware is.",
            "Malware is software designed to harm systems, steal information, "
            "disrupt operations, or perform actions without permission.",
            "malware_basics",
            "easy",
        ),
        (
            "Explain what phishing is.",
            "Phishing is an attempt to trick someone into revealing sensitive "
            "information or opening a harmful link by pretending to be "
            "trustworthy.",
            "phishing_basics",
            "easy",
        ),
        (
            "Explain what two-factor authentication does.",
            "Two-factor authentication requires two different forms of proof "
            "before access is granted, such as a password and a temporary code.",
            "two_factor_authentication",
            "easy",
        ),
        (
            "Explain the water cycle in simple words.",
            "Water evaporates, forms clouds through condensation, falls as "
            "precipitation, and returns through runoff, groundwater, and "
            "collection.",
            "water_cycle_basics",
            "easy",
        ),
        (
            "Explain what an earthquake is.",
            "An earthquake is shaking caused by a sudden release of energy in "
            "Earth's crust, usually when rocks move along a fault.",
            "earthquake_basics",
            "easy",
        ),
        (
            "Explain what a volcano is.",
            "A volcano is an opening in Earth's crust through which magma, gas, "
            "and ash can reach the surface.",
            "volcano_basics",
            "easy",
        ),
        (
            "Explain what erosion is.",
            "Erosion is the movement of soil or rock by water, wind, ice, or "
            "gravity.",
            "erosion_basics",
            "easy",
        ),
        (
            "Explain what an ecosystem is.",
            "An ecosystem includes living organisms and the nonliving "
            "environment they interact with in a particular area.",
            "ecosystem_basics",
            "easy",
        ),
        (
            "Explain what a cell is in biology.",
            "A cell is the smallest basic unit of life. Living organisms are "
            "made of one or more cells that perform essential functions.",
            "cell_biology_basics",
            "easy",
        ),
        (
            "Explain what DNA does.",
            "DNA stores genetic instructions used for growth, development, "
            "function, and reproduction.",
            "dna_basics",
            "easy",
        ),
        (
            "Explain what a gene is.",
            "A gene is a section of DNA that contains instructions for a "
            "biological product or trait.",
            "gene_basics",
            "easy",
        ),
        (
            "Explain what a chromosome is.",
            "A chromosome is a tightly organized structure made mainly of DNA "
            "and proteins. It carries many genes.",
            "chromosome_basics",
            "easy",
        ),
        (
            "Explain what the immune system does.",
            "The immune system protects the body by identifying and responding "
            "to harmful organisms and abnormal cells.",
            "immune_system_basics",
            "easy",
        ),
        (
            "Explain what a vaccine does.",
            "A vaccine trains the immune system to recognize a particular "
            "disease-causing organism without requiring the full disease.",
            "vaccine_basics",
            "easy",
        ),
        (
            "Explain what blood pressure means.",
            "Blood pressure is the force of circulating blood pushing against "
            "the walls of blood vessels.",
            "blood_pressure_basics",
            "easy",
        ),
        (
            "Explain what the heart does.",
            "The heart is a muscular organ that pumps blood through the body. "
            "This delivers oxygen and nutrients and carries away waste.",
            "heart_basics",
            "easy",
        ),
        (
            "Explain what the lungs do.",
            "The lungs exchange gases between the air and blood. They bring "
            "oxygen into the body and help remove carbon dioxide.",
            "lungs_basics",
            "easy",
        ),
        (
            "Explain what digestion is.",
            "Digestion breaks food into smaller substances that the body can "
            "absorb and use for energy, growth, and repair.",
            "digestion_basics",
            "easy",
        ),
        (
            "Explain what a hormone is.",
            "A hormone is a chemical messenger produced by the body. It travels "
            "to target tissues and helps regulate body functions.",
            "hormone_basics",
            "easy",
        ),
        (
            "Explain what insulin does.",
            "Insulin helps cells take glucose from the blood and use or store "
            "it for energy.",
            "insulin_basics",
            "easy",
        ),
        (
            "Explain what metabolism means.",
            "Metabolism is the collection of chemical processes that keep the "
            "body alive, including releasing energy and building materials.",
            "metabolism_basics",
            "easy",
        ),
        (
            "Explain what a calorie measures.",
            "A calorie measures energy. In nutrition, it describes how much "
            "energy food can provide to the body.",
            "calorie_basics",
            "easy",
        ),
        (
            "Explain what protein does in the body.",
            "Protein provides building material for muscles, enzymes, hormones, "
            "and many other structures in the body.",
            "protein_basics",
            "easy",
        ),
        (
            "Explain what a planet is.",
            "A planet is a large body that orbits a star, has enough gravity to "
            "be nearly round, and has cleared most objects from its orbit.",
            "planet_basics",
            "easy",
        ),
        (
            "Explain what a star is.",
            "A star is a large ball of hot gas that produces energy through "
            "nuclear fusion in its core.",
            "star_basics",
            "easy",
        ),
        (
            "Explain what a galaxy is.",
            "A galaxy is a huge system of stars, gas, dust, and dark matter "
            "held together by gravity.",
            "galaxy_basics",
            "easy",
        ),
        (
            "Explain what an orbit is.",
            "An orbit is the curved path one object follows around another "
            "because of gravity and its motion.",
            "orbit_basics",
            "easy",
        ),
        (
            "Explain what a light-year measures.",
            "A light-year measures distance, not time. It is the distance light "
            "travels in one year.",
            "light_year_basics",
            "easy",
        ),
        (
            "Explain what renewable energy is.",
            "Renewable energy comes from sources that are naturally replaced, "
            "such as sunlight, wind, and flowing water.",
            "renewable_energy_basics",
            "easy",
        ),
        (
            "Explain what fossil fuels are.",
            "Fossil fuels are energy sources such as coal, oil, and natural gas "
            "formed from ancient organic material over very long periods.",
            "fossil_fuels_basics",
            "easy",
        ),
        (
            "Explain climate change in simple words.",
            "Climate change is a long-term shift in temperatures and weather "
            "patterns. Current warming is largely driven by human greenhouse "
            "gas emissions.",
            "climate_change_basics",
            "medium",
        ),
        (
            "Explain what the greenhouse effect is.",
            "The greenhouse effect happens when certain gases trap some heat "
            "in Earth's atmosphere instead of letting it escape to space.",
            "greenhouse_effect_basics",
            "easy",
        ),
        (
            "Explain what recycling does.",
            "Recycling collects and processes used materials so they can be "
            "turned into new products instead of being discarded.",
            "recycling_basics",
            "easy",
        ),
        (
            "Explain what machine learning is.",
            "Machine learning is a way for computers to learn patterns from "
            "data and use those patterns to make predictions or decisions.",
            "machine_learning_basics",
            "easy",
        ),
        (
            "Explain what training data is.",
            "Training data is the collection of examples a machine learning "
            "model studies to learn patterns.",
            "training_data_basics",
            "easy",
        ),
        (
            "Explain what a machine learning model is.",
            "A machine learning model is a learned mathematical system that "
            "uses patterns from data to produce predictions or outputs.",
            "ml_model_basics",
            "easy",
        ),
        (
            "Explain what overfitting means.",
            "Overfitting happens when a model memorizes training examples too "
            "closely and performs poorly on new data.",
            "overfitting_basics",
            "medium",
        ),
        (
            "Explain what validation data is.",
            "Validation data is separate data used to check how well a model "
            "works during development without training directly on it.",
            "validation_data_basics",
            "easy",
        ),
        (
            "Explain what an epoch is in model training.",
            "An epoch is one complete pass through the training dataset.",
            "epoch_basics",
            "easy",
        ),
        (
            "Explain what batch size means in training.",
            "Batch size is the number of training examples processed together "
            "before the model updates its parameters.",
            "batch_size_basics",
            "easy",
        ),
        (
            "Explain what a learning rate controls.",
            "The learning rate controls how large each model update is during "
            "training.",
            "learning_rate_basics",
            "easy",
        ),
        (
            "Explain what model parameters are.",
            "Model parameters are learned numerical values that control how a "
            "model transforms input into output.",
            "model_parameters_basics",
            "easy",
        ),
        (
            "Explain what a token is in a language model.",
            "A token is a small unit of text that a language model processes, "
            "such as a word, part of a word, or punctuation mark.",
            "token_basics",
            "easy",
        ),
        (
            "Explain what tokenization does.",
            "Tokenization converts text into smaller units called tokens so a "
            "language model can process it.",
            "tokenization_basics",
            "easy",
        ),
        (
            "Explain what a language model predicts.",
            "A language model predicts likely next tokens based on the tokens "
            "that came before them.",
            "language_model_prediction",
            "easy",
        ),
        (
            "Explain what inference means in machine learning.",
            "Inference is the process of using a trained model to produce an "
            "output for new input.",
            "inference_basics",
            "easy",
        ),
        (
            "Explain what a checkpoint is during training.",
            "A checkpoint is a saved copy of the model and training state that "
            "can be loaded later.",
            "checkpoint_basics",
            "easy",
        ),
        (
            "Explain what loss means during model training.",
            "Loss is a numerical measure of how far the model's predictions are "
            "from the expected answers.",
            "training_loss_basics",
            "easy",
        ),
        (
            "Explain what gradient descent does.",
            "Gradient descent updates model parameters step by step in a "
            "direction that aims to reduce loss.",
            "gradient_descent_basics",
            "medium",
        ),
        (
            "Explain what a GPU does in model training.",
            "A GPU performs many calculations in parallel, which can make "
            "machine learning training much faster.",
            "gpu_training_basics",
            "easy",
        ),
        (
            "Explain what a tensor is.",
            "A tensor is a structured collection of numbers arranged in one or "
            "more dimensions.",
            "tensor_basics",
            "easy",
        ),
        (
            "Explain what matrix multiplication does.",
            "Matrix multiplication combines rows and columns of numbers to "
            "produce a new matrix.",
            "matrix_multiplication_basics",
            "medium",
        ),
        (
            "Explain what an embedding is in a language model.",
            "An embedding represents a token as a list of numbers that the "
            "model can process.",
            "embedding_basics",
            "medium",
        ),
        (
            "Explain what attention does in a language model.",
            "Attention lets the model weigh which earlier tokens are most "
            "relevant when processing the current token.",
            "attention_basics",
            "medium",
        ),
        (
            "Explain what a neural network layer does.",
            "A neural network layer transforms incoming numbers into new "
            "representations using learned parameters.",
            "neural_network_layer_basics",
            "easy",
        ),
        (
            "Explain what activation functions do.",
            "Activation functions introduce nonlinear behavior, allowing a "
            "neural network to learn more complex patterns.",
            "activation_function_basics",
            "medium",
        ),
        (
            "Explain what fine-tuning means.",
            "Fine-tuning continues training an existing model on a more "
            "specific dataset or task.",
            "fine_tuning_basics",
            "easy",
        ),
        (
            "Explain what model evaluation means.",
            "Model evaluation measures how well a trained model performs on "
            "data or tasks that were not used for training.",
            "model_evaluation_basics",
            "easy",
        ),
    ]

    records: list[dict[str, Any]] = []

    for instruction, response, family, difficulty in rows:
        records.append(
            _base_record(
                capability="beginner_explanation",
                instruction=instruction,
                response=response,
                family=f"batch_002_{family}",
                difficulty=difficulty,
            )
        )

    if len(records) != 100:
        raise AssertionError(
            "Batch 002 explanation catalog must currently contain 100 rows"
        )

    return records



def _exact_format_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    topics = [
        ("preparing a coding session", ["Open the project.", "Read the current task.", "Run the baseline tests.", "Make one focused change.", "Record the result."]),
        ("checking a dataset file", ["Confirm the file path.", "Inspect the schema.", "Count the records.", "Check missing values.", "Save the audit notes."]),
        ("starting a revision block", ["Choose one chapter.", "Set a time limit.", "Review key definitions.", "Solve practice questions.", "Mark weak areas."]),
        ("reviewing a pull request", ["Read the description.", "Inspect changed files.", "Run relevant tests.", "Check edge cases.", "Write actionable feedback."]),
        ("protecting a shared account", ["Use a unique password.", "Enable two-step verification.", "Limit account access.", "Review login alerts.", "Remove unused sessions."]),
        ("planning a presentation", ["Define the audience.", "Choose one main message.", "Arrange supporting points.", "Simplify the slides.", "Practice aloud."]),
        ("verifying a model checkpoint", ["Load the checkpoint.", "Check configuration values.", "Run a fixed prompt set.", "Compare metrics.", "Record the hash."]),
        ("organizing experiment notes", ["Name the experiment.", "Record the seed.", "List hyperparameters.", "Attach metrics.", "Write the conclusion."]),
        ("preparing a lab computer", ["Connect the charger.", "Close unrelated programs.", "Check free storage.", "Verify dependencies.", "Start the required tool."]),
        ("submitting an assignment", ["Read the rubric.", "Check the file format.", "Verify the filename.", "Review the content.", "Submit before the deadline."]),
        ("testing a command-line tool", ["Read the help text.", "Use a small input.", "Check the exit code.", "Inspect the output.", "Test an invalid input."]),
        ("creating a safe archive", ["Collect the final files.", "Remove temporary files.", "Create the archive.", "Verify its contents.", "Store a checksum."]),
        ("checking an online form", ["Review required fields.", "Confirm spellings.", "Check dates.", "Verify uploaded files.", "Save a confirmation copy."]),
        ("planning a team handoff", ["Summarize the current state.", "List unfinished work.", "Identify blockers.", "Link important files.", "Name the next owner."]),
        ("preparing for an interview", ["Review the role.", "Study the organization.", "Prepare examples.", "Test the call setup.", "Choose questions to ask."]),
        ("checking a software release", ["Confirm the version.", "Read the changelog.", "Run smoke tests.", "Check rollback steps.", "Approve the release notes."]),
        ("setting up a study desk", ["Clear unrelated items.", "Place needed books nearby.", "Keep water available.", "Silence notifications.", "Start with the priority task."]),
        ("reviewing a budget sheet", ["Check category names.", "Verify formulas.", "Compare totals.", "Inspect unusual values.", "Save a reviewed copy."]),
        ("preparing travel documents", ["Check identity documents.", "Confirm reservations.", "Copy important numbers.", "Store digital backups.", "Pack documents securely."]),
        ("checking a written report", ["Read the title.", "Verify the structure.", "Check cited figures.", "Correct language errors.", "Confirm the conclusion."]),
        ("cleaning a data table", ["Identify duplicate rows.", "Standardize labels.", "Handle missing values.", "Validate data types.", "Export the cleaned table."]),
        ("starting a focus period", ["Choose one outcome.", "Close distracting tabs.", "Set a timer.", "Work without switching tasks.", "Review progress at the end."]),
        ("checking a backup restore", ["Choose a test file.", "Restore it separately.", "Open the restored copy.", "Compare its contents.", "Record the verification date."]),
        ("preparing a repository commit", ["Review the diff.", "Remove accidental changes.", "Run tests.", "Write a clear message.", "Commit the intended files."]),
        ("evaluating a model response", ["Read the prompt.", "Check correctness.", "Check instruction following.", "Note repetition.", "Record the final rating."]),
    ]

    sentence_overrides = {
        'preparing a coding session': 'Open the project and read the current task before changing anything. Confirm the environment and run a small check first.',
        'checking a dataset file': 'Confirm the dataset path and expected schema. Inspect representative records and count the full file. Save the findings with any record identifiers that need attention.',
        'starting a revision block': 'Choose one chapter and set a clear time limit. Review key ideas, then solve practice questions without notes.',
        'reviewing a pull request': 'Start by reading the pull request description and understanding its purpose. Inspect the changed files and run relevant tests. Finish by checking edge cases and writing actionable feedback.',
        'protecting a shared account': 'Use a unique password and enable two-step verification. Limit access to people who genuinely need the account.',
        'planning a presentation': 'Define the audience and one main message. Arrange supporting points in a clear sequence. Practise aloud and simplify any confusing section.',
        'verifying a model checkpoint': 'Load the checkpoint into the expected model configuration and confirm that all state keys match. Run a fixed evaluation sample and compare its outputs and metadata with the recorded baseline.',
        'organizing experiment notes': 'Name the experiment and record its seed and configuration. Attach metrics, observations, and relevant artifacts. Finish with a concise conclusion and next action.',
        'preparing a lab computer': 'Connect the charger and close unrelated programs. Verify the required software and files before beginning.',
        'submitting an assignment': 'Read the rubric and confirm the required file format. Check the filename and content before uploading. Submit before the deadline and save the confirmation.',
        'testing a command-line tool': 'Read the help text and test the command with a small input. Check the exit code and compare the output with the expected result.',
        'creating a safe archive': 'Collect only the final files and remove temporary material. Create the archive and inspect its contents. Store a checksum so later corruption can be detected.',
        'checking an online form': 'Review every required field and confirm names, dates, and spellings. Check attachments and the destination before submitting.',
        'planning a team handoff': 'Summarize the current state and unfinished work. Identify blockers, risks, and important context. Assign the next actions to a clear owner.',
        'preparing for an interview': 'Review the role and prepare relevant examples from your experience. Test the call setup and arrive with concise questions.',
        'checking a software release': 'Confirm the version and read the changelog. Run smoke tests and verify rollback instructions. Approve the release notes only after the checks pass.',
        'setting up a study desk': 'Clear unrelated items and place needed materials within reach. Adjust the chair, lighting, and screen for comfortable study.',
        'reviewing a budget sheet': 'Check category names and verify every formula. Compare subtotals with the overall total. Save a reviewed copy without overwriting the original.',
        'preparing travel documents': 'Check identity documents, reservations, and expiry dates. Keep secure digital copies and important contact numbers.',
        'checking a written report': 'Read the report for structure and a clear main argument. Verify cited figures and correct language errors. Confirm that the conclusion follows from the evidence.',
        'cleaning a data table': 'Identify duplicate rows and standardize labels and data types. Validate the cleaned table before replacing the original.',
        'starting a focus period': 'Choose one specific outcome and remove unrelated distractions. Work on that outcome until the planned interval ends. Review progress and decide the next action.',
        'checking a backup restore': 'Restore a representative file to a separate location. Open it and compare its contents with the original.',
        'preparing a repository commit': 'Review the diff and remove accidental changes. Run the relevant tests and write a clear commit message. Commit only the intended files.',
        'evaluating a model response': 'Read the prompt and check whether the response is correct and relevant. Then assess clarity, completeness, and adherence to requested formatting.',
    }
    exact_word_overrides = {
        'preparing a coding session': 'Plan goals, tools, tests.',
        'checking a dataset file': 'Verify its structure before processing.',
        'starting a revision block': 'Choose, review, practise, assess.',
        'reviewing a pull request': 'Inspect changes, test, then comment.',
        'protecting a shared account': 'Restrict access; rotate credentials.',
        'planning a presentation': 'Define audience, message, structure, rehearsal.',
        'verifying a model checkpoint': 'Load, inspect, test, compare.',
        'organizing experiment notes': 'Record seeds, settings, metrics, conclusions.',
        'preparing a lab computer': 'Charge, close, verify, begin.',
        'submitting an assignment': 'Check requirements, files, then submit.',
        'testing a command-line tool': 'Read, test, inspect, repeat.',
        'creating a safe archive': 'Collect, archive, inspect, checksum, store.',
        'checking an online form': 'Review fields, attachments, destination.',
        'planning a team handoff': 'Summarize, document, assign, confirm ownership.',
        'preparing for an interview': 'Research, practise, test, prepare.',
        'checking a software release': 'Verify version, tests, rollback, notes.',
        'setting up a study desk': 'Clear, arrange, adjust, begin.',
        'reviewing a budget sheet': 'Check labels, formulas, totals, copy.',
        'preparing travel documents': 'Verify names, dates, validity.',
        'checking a written report': 'Check structure, evidence, language, conclusion.',
        'cleaning a data table': 'Deduplicate, standardize, validate, preserve.',
        'starting a focus period': 'Choose one outcome and begin.',
        'checking a backup restore': 'Restore separately, open, compare.',
        'preparing a repository commit': 'Review, test, describe, commit carefully.',
        'evaluating a model response': 'Check correctness, relevance, clarity.',
    }

    for index, (topic, items) in enumerate(topics):
        count = 2 + (index % 4)
        selected = items[:count]
        records.append(
            _base_record(
                "exact_format_following",
                f"Give exactly {count} bullet points for {topic}.",
                "\n".join(f"- {item}" for item in selected),
                "batch_002_exact_bullets",
                format_constraints={"type": "exact_bullets", "count": count},
            )
        )
        records.append(
            _base_record(
                "exact_format_following",
                f"Write exactly {count} numbered steps for {topic}.",
                "\n".join(f"{number}. {item}" for number, item in enumerate(selected, 1)),
                "batch_002_exact_numbered",
                format_constraints={"type": "exact_numbered_items", "count": count},
            )
        )
        sentence_count = 2 + (index % 2)
        sentence_response = sentence_overrides[topic]
        records.append(
            _base_record(
                "exact_format_following",
                f"Describe {topic} in exactly {sentence_count} sentences.",
                sentence_response,
                "batch_002_exact_sentences",
                format_constraints={"type": "exact_sentences", "count": sentence_count},
            )
        )
        word_count = 4 + (index % 2)
        response = exact_word_overrides[topic]
        records.append(
            _base_record(
                "exact_format_following",
                f"Answer in exactly {word_count} words: How should someone approach {topic}?",
                response,
                "batch_002_exact_words",
                format_constraints={"type": "exact_words", "count": word_count},
            )
        )

    if len(records) != 100:
        raise AssertionError(f"exact-format catalog must contain 100 rows; found {len(records)}")
    return records


def _rewrite_records() -> list[dict[str, Any]]:
    tasks = [
        "send the revised timetable", "check the dataset columns", "share the meeting summary",
        "review the latest checkpoint", "confirm the room booking", "upload the signed form",
        "verify the experiment seed", "correct the broken link", "return the borrowed adapter",
        "update the dependency list", "check the calculation sheet", "forward the approval note",
        "rename the final archive", "test the login flow", "summarize the research notes",
        "prepare the release checklist", "inspect the error log", "confirm the submission receipt",
        "organize the shared folder", "review the model output", "save the configuration copy",
        "compare the validation scores", "check the file checksum", "clean the duplicate rows",
        "record the training settings",
    ]
    records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        urgent_input = f"Do this now: {task}."
        records.append(
            _base_record(
                "rewriting_transformation",
                f"Rewrite the request about {task} politely while preserving urgency. Return only the revision.",
                f"Could you please {task} now?",
                "batch_002_rewrite_polite",
                input_text=urgent_input,
                answer_style="polite",
            )
        )
        verbose_input = f"At this point in time, it is necessary for us to {task} before we continue."
        records.append(
            _base_record(
                "rewriting_transformation",
                f"Rewrite the sentence about {task} concisely without removing the required action.",
                f"We must {task} before continuing.",
                "batch_002_rewrite_concise",
                input_text=verbose_input,
                answer_style="concise",
            )
        )
        passive_input = f"The request to {task} was completed by the project assistant."
        records.append(
            _base_record(
                "rewriting_transformation",
                f"Rewrite the sentence about {task} in active voice. Return one complete sentence.",
                f"The project assistant completed the request to {task}.",
                "batch_002_rewrite_active",
                input_text=passive_input,
                answer_style="active_voice",
            )
        )
    if len(records) != 75:
        raise AssertionError("rewrite catalog must contain 75 rows")
    return records


def _list_records() -> list[dict[str, Any]]:
    rows = [
        (
            "a tokenizer audit",
            [
                "Confirm the tokenizer file and recorded hash.",
                "Encode representative multilingual and punctuation samples.",
                "Check token IDs against the configured vocabulary size.",
                "Document unknown-token and length anomalies.",
            ],
        ),
        (
            "a checkpoint review",
            [
                "Verify the checkpoint path and cryptographic hash.",
                "Inspect the saved model and optimizer metadata.",
                "Confirm architecture compatibility before loading.",
                "Record the validation metric and training step.",
            ],
        ),
        (
            "a dataset handoff",
            [
                "Provide the source file and its manifest.",
                "State the schema, license, and provenance.",
                "Include validation and duplicate-audit results.",
                "Keep training authorization disabled until approval.",
            ],
        ),
        (
            "a repository cleanup",
            [
                "Remove generated files that are not tracked intentionally.",
                "Archive obsolete scripts instead of leaving ambiguous copies.",
                "Check imports after moving or deleting modules.",
                "Run the full test suite before committing.",
            ],
        ),
        (
            "a validation run",
            [
                "Load the exact configuration under review.",
                "Execute schema, format, and duplicate checks.",
                "Capture failures with record identifiers.",
                "Save a deterministic validation report.",
            ],
        ),
        (
            "a release rehearsal",
            [
                "Build artifacts from a clean working tree.",
                "Verify hashes and manifest references.",
                "Test installation or loading in an isolated environment.",
                "Document rollback steps before promotion.",
            ],
        ),
        (
            "a model comparison",
            [
                "Use the same prompts and decoding settings.",
                "Measure quality, latency, and memory usage.",
                "Review failures rather than relying on averages alone.",
                "Record which checkpoint wins each criterion.",
            ],
        ),
        (
            "a training restart",
            [
                "Confirm the intended resume checkpoint.",
                "Restore optimizer and scheduler state when required.",
                "Verify the next global step before training.",
                "Run a short smoke test before the full job.",
            ],
        ),
        (
            "a code review",
            [
                "Read the change summary and affected interfaces.",
                "Check correctness, error handling, and tests.",
                "Look for duplicated logic or hidden compatibility breaks.",
                "Leave specific comments with actionable fixes.",
            ],
        ),
        (
            "a documentation update",
            [
                "Describe the user-visible behavior change.",
                "Update examples and configuration references.",
                "Remove statements that are no longer accurate.",
                "Link the change to the relevant implementation.",
            ],
        ),
        (
            "a benchmark session",
            [
                "Fix the hardware and software environment.",
                "Warm up the model before collecting timings.",
                "Repeat measurements and report their spread.",
                "Store raw results with the benchmark configuration.",
            ],
        ),
        (
            "a bug reproduction",
            [
                "Reduce the issue to the smallest failing input.",
                "Record the environment and exact command.",
                "Confirm the failure occurs consistently.",
                "Separate observed evidence from suspected causes.",
            ],
        ),
        (
            "a file migration",
            [
                "Inventory source files and destination paths.",
                "Preserve names, permissions, and hashes where required.",
                "Update every code and configuration reference.",
                "Verify the migrated files before deleting originals.",
            ],
        ),
        (
            "a configuration review",
            [
                "Check values against the documented schema.",
                "Confirm paths resolve from the expected working directory.",
                "Identify settings that affect compatibility or reproducibility.",
                "Reject contradictory or unused options.",
            ],
        ),
        (
            "a dependency upgrade",
            [
                "Read the upstream release and migration notes.",
                "Update the lockfile in a controlled environment.",
                "Run unit, integration, and compatibility tests.",
                "Document any required code or configuration changes.",
            ],
        ),
        (
            "a security check",
            [
                "Review untrusted inputs and file-loading paths.",
                "Check secrets are absent from source and logs.",
                "Inspect dependencies for known vulnerabilities.",
                "Record findings with severity and remediation steps.",
            ],
        ),
        (
            "a presentation rehearsal",
            [
                "Practice the full talk with a timer.",
                "Check that each slide supports one clear point.",
                "Simplify sections that are difficult to explain aloud.",
                "Prepare concise answers for likely questions.",
            ],
        ),
        (
            "a research reading session",
            [
                "Write the paper's main research question.",
                "Identify the method, assumptions, and evaluation setup.",
                "Note evidence that supports or weakens each claim.",
                "Summarize implications for the current project.",
            ],
        ),
        (
            "a study revision block",
            [
                "Choose one clearly bounded topic.",
                "Recall the key ideas without looking at notes.",
                "Solve representative questions under time limits.",
                "Review mistakes and schedule another check.",
            ],
        ),
        (
            "a team status update",
            [
                "State what was completed since the last update.",
                "Report current work with its owner.",
                "Highlight blockers, risks, and needed decisions.",
                "List the next concrete milestones.",
            ],
        ),
        (
            "a storage cleanup",
            [
                "Measure which folders consume the most space.",
                "Delete reproducible caches and temporary outputs.",
                "Archive important inactive material separately.",
                "Confirm backups before removing unique files.",
            ],
        ),
        (
            "a backup verification",
            [
                "Select a recent backup using its manifest.",
                "Restore representative files to a temporary location.",
                "Compare restored hashes with the originals.",
                "Record the verification date and outcome.",
            ],
        ),
        (
            "a form submission",
            [
                "Confirm every required field is complete.",
                "Check names, dates, identifiers, and attachments.",
                "Review the destination before submitting.",
                "Save the confirmation or reference number.",
            ],
        ),
        (
            "a meeting follow-up",
            [
                "Summarize decisions in neutral language.",
                "Assign each action to a named owner.",
                "Include deadlines and unresolved questions.",
                "Send the notes while the discussion is fresh.",
            ],
        ),
        (
            "an experiment summary",
            [
                "State the hypothesis and changed variable.",
                "Describe the data, controls, and evaluation method.",
                "Report results with uncertainty and limitations.",
                "Recommend the next experiment based on evidence.",
            ],
        ),
    ]

    records: list[dict[str, Any]] = []
    for context, items in rows:
        records.append(
            _base_record(
                "lists_structured_output",
                f"List exactly four bullet points for {context}.",
                "\n".join(f"- {item}" for item in items),
                "batch_002_structured_bullets",
                format_constraints={"type": "exact_bullets", "count": 4},
            )
        )
        records.append(
            _base_record(
                "lists_structured_output",
                f"Give exactly three numbered steps for {context}.",
                "\n".join(
                    f"{number}. {item}"
                    for number, item in enumerate(items[:3], start=1)
                ),
                "batch_002_structured_numbered",
                format_constraints={"type": "exact_numbered_items", "count": 3},
            )
        )

    if len(records) != 50:
        raise AssertionError("structured-list catalog must contain 50 rows")
    return records
def _json_records() -> list[dict[str, Any]]:
    rows = [
        ("experiment", {"experiment": "rope_ablation", "seed": 42}),
        ("checkpoint", {"checkpoint": "step_240000.pt", "verified": True}),
        ("dataset", {"dataset": "batch_002", "records": 500}),
        ("tokenizer", {"tokenizer": "vasu_bpe", "vocab_size": 32000}),
        ("training", {"training": "instruction_tuning", "authorized": False}),
        ("evaluation", {"evaluation": "format_gate", "passed": True}),
        ("artifact", {"artifact": "review_packet", "ready": False}),
        ("device", {"device": "cuda", "memory_gb": 6}),
        ("schedule", {"schedule": "night_run", "hours": 4}),
        ("metric", {"metric": "validation_loss", "value": 2.48}),
        ("sample", {"sample": "prompt_07", "accepted": True}),
        ("release", {"release": "candidate_v2", "promoted": False}),
        ("review", {"review": "human_gate", "status": "pending"}),
        ("branch", {"branch": "dataset-batch-002", "clean": True}),
        ("commit", {"commit": "dataset update", "files": 3}),
        ("run", {"run": "smoke_test", "errors": 0}),
        ("model", {"model": "vasu_60m", "layers": 10}),
        ("prompt", {"prompt": "explain gravity", "tokens": 9}),
        ("response", {"response": "accepted", "score": 4}),
        ("split", {"split": "validation", "ratio": 0.05}),
        ("mask", {"mask": "assistant_only", "aligned": True}),
        ("archive", {"archive": "batch_002.zip", "encrypted": False}),
        ("report", {"report": "duplicate_audit", "groups": 0}),
        ("config", {"config": "vasu_60m", "sequence_length": 256}),
        ("status", {"status": "review_required", "training_authorized": False}),
    ]
    records: list[dict[str, Any]] = []
    for family, value in rows:
        value_types = {
            key: "boolean" if isinstance(item, bool)
            else "number" if isinstance(item, (int, float))
            else "string"
            for key, item in value.items()
        }
        records.append(
            _base_record(
                "json_schema_output",
                f"Return strict JSON for this {family} record using exactly these keys: {', '.join(value)}.",
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                f"batch_002_json_{family}",
                answer_style="strict-json",
                format_constraints={
                    "type": "json_object",
                    "required_keys": list(value),
                    "strict": True,
                    "value_types": value_types,
                },
            )
        )
    return records


def _uncertainty_records() -> list[dict[str, Any]]:
    rows = [
        ("What exact loss will my next training run reach?", "That result cannot be known in advance. It depends on the data, initialization, hyperparameters, and runtime conditions."),
        ("Which checkpoint am I looking at on my screen?", "I cannot identify the checkpoint without its filename, path, or visible metadata."),
        ("Will this model definitely outperform every larger model?", "No reliable guarantee can be made. Performance depends on the tasks, data, training quality, and evaluation method."),
        ("What is the current GPU temperature on my laptop?", "I do not have live access to your laptop. Check the temperature with your local monitoring tool."),
        ("Did the training process finish while I was away?", "I cannot know without the latest log, process status, or checkpoint output."),
        ("Which dataset record caused the error?", "There is not enough information to identify it. The traceback, record ID, and failing input are needed."),
        ("Is this unknown checkpoint safe to load?", "Its safety is unverified. Confirm its trusted origin and inspect how it will be loaded before using it."),
        ("What will the validation loss be after ten more updates?", "The future validation loss is unknown and must be measured after those updates."),
        ("Did my teammate approve the release?", "I cannot know their decision without a recorded approval or direct confirmation."),
        ("Which hyperparameter am I thinking about?", "I cannot infer that without additional context such as its name or role."),
        ("Is the file on my desktop the newest version?", "I cannot inspect your desktop. Compare its timestamp, hash, or version metadata with the other copies."),
        ("Will increasing the batch size always improve the model?", "No. The effect depends on optimization, memory limits, learning rate, and the training setup."),
        ("Can this unlabeled binary file be used as training data?", "Its format and contents are unknown. Do not use it until the schema, provenance, and integrity are verified."),
        ("Is the evaluation score correct?", "It cannot be confirmed without the evaluation code, inputs, expected outputs, and calculation details."),
        ("What caused yesterday's unexplained system restart?", "The cause is unknown from this description. System logs and hardware or power events would be needed."),
        ("Is the online claim that one prompt fixes every model true?", "That claim is not credible without reproducible evidence across models and evaluations."),
        ("Can I delete this checkpoint without consequences?", "I cannot determine that without knowing whether it is referenced, backed up, or needed for recovery."),
        ("What learning rate should I use for an unspecified model?", "There is not enough information. Model size, optimizer, batch size, data, and prior runs are needed."),
        ("Is the latest package version compatible with my project?", "Compatibility is unverified without the package version, project dependencies, and release notes."),
        ("Did this response come from the base model or fine-tuned model?", "I cannot identify the source model without run metadata or checkpoint information."),
        ("Will a longer context always produce a better answer?", "No. A longer context can help or hurt depending on relevance, model capability, and prompt construction."),
        ("Is this benchmark result comparable to another report?", "Not necessarily. The dataset version, prompt format, decoding settings, and scoring method must match."),
        ("What is the password stored in this hidden file?", "I cannot access hidden files or reveal credentials. Inspect your authorized local configuration securely."),
        ("Can an unreviewed dataset be treated as error-free?", "No. An unreviewed dataset may contain factual, formatting, duplication, or labeling errors."),
        ("Has Batch 002 been approved for training?", "Training is not authorized unless the recorded review and release gate explicitly set training_authorized to true."),
    ]
    return [
        _base_record(
            "uncertainty_honest_fallback",
            instruction,
            response,
            f"batch_002_uncertainty_{index:02d}",
            answer_style="cautious",
        )
        for index, (instruction, response) in enumerate(rows, start=1)
    ]


def author_batch() -> list[dict[str, Any]]:
    """Return all 500 deterministic source records in stable ID order."""
    records = [*_factual_records(), *_explanation_records(), *_exact_format_records(), *_rewrite_records(), *_list_records(), *_json_records(), *_uncertainty_records()]
    difficulty_plan = {
        "short_factual_qa": ["easy"] * 125,
        "beginner_explanation": ["easy"] * 75 + ["medium"] * 25,
        "exact_format_following": ["easy"] * 50 + ["medium"] * 50,
        "rewriting_transformation": ["easy"] * 25 + ["medium"] * 50,
        "lists_structured_output": ["easy"] * 25 + ["medium"] * 25,
        "json_schema_output": ["medium"] * 25,
        "uncertainty_honest_fallback": ["hard"] * 25,
    }
    capability_offsets = Counter()
    for index, record in enumerate(records, start=1):
        record["example_id"] = f"viq1_b002_{index:06d}"
        capability = record["capability"]
        record["difficulty"] = difficulty_plan[capability][capability_offsets[capability]]
        capability_offsets[capability] += 1
    counts = Counter(record["capability"] for record in records)
    actual_counts = {
        capability: counts.get(capability, 0)
        for capability in EXPECTED_COUNTS
    }

    if actual_counts != EXPECTED_COUNTS:
        raise AssertionError(
            f"unexpected category counts: {actual_counts}"
        )
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
    expected_ids = [
        f"viq1_b002_{index:06d}"
        for index in range(1, 501)
    ]
    structural_errors = []
    if ids != expected_ids:
        structural_errors.append("IDs are not the required sequential batch-002 IDs")
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
