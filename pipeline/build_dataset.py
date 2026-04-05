"""
SpecCheck data pipeline.
Run once (or whenever you want to refresh product data):
    python pipeline/build_dataset.py

With SERPAPI_KEY + ANTHROPIC_API_KEY in .env, real specs are fetched from
European Google Shopping and extracted by Claude.
Without API keys the pipeline falls back to hardcoded specs (demo mode).

Results are cached in data/specs_cache.json — re-running the pipeline is
free after the first successful fetch.
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA      = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
from services.specs import get_product_specs


# ── Product lists: (name, brand, year, price) ─────────────────────────────────
# Prices are kept here because Icecat doesn't provide them.
# All other specs are fetched from Icecat (or from FALLBACK_SPECS below).

LAPTOPS = [
    # Premium / flagship
    ("MacBook Air M3",              "Apple",     2024, 1299),
    ("MacBook Pro M3 14\"",         "Apple",     2024, 1999),
    ("MacBook Pro M3 16\"",         "Apple",     2024, 2499),
    ("Dell XPS 15 9530",            "Dell",      2024, 1499),
    ("Dell XPS 13 9340",            "Dell",      2024, 1299),
    ("ThinkPad X1 Carbon Gen 11",   "Lenovo",    2024, 1649),
    ("ThinkPad X1 Extreme Gen 5",   "Lenovo",    2024, 1899),
    ("HP Spectre x360 14",          "HP",        2024, 1349),
    ("Razer Blade 15",              "Razer",     2024, 2499),
    ("Razer Blade 14",              "Razer",     2024, 1999),
    ("Microsoft Surface Laptop 5",  "Microsoft", 2023,  999),
    ("Microsoft Surface Pro 9",     "Microsoft", 2023, 1299),
    ("Samsung Galaxy Book3 Pro",    "Samsung",   2024, 1199),
    ("Samsung Galaxy Book3 Ultra",  "Samsung",   2024, 1799),
    # Mid-range
    ("ASUS ZenBook Pro 14",         "ASUS",      2024, 1199),
    ("ASUS ZenBook 14 OLED",        "ASUS",      2024,  799),
    ("ASUS ROG Zephyrus G14",       "ASUS",      2024, 1449),
    ("ASUS ROG Zephyrus G16",       "ASUS",      2024, 1599),
    ("Lenovo Legion 5 Pro",         "Lenovo",    2024, 1299),
    ("Lenovo Legion 7i",            "Lenovo",    2024, 1599),
    ("Lenovo IdeaPad 5 Pro 16",     "Lenovo",    2024,  899),
    ("HP Envy x360 15",             "HP",        2024,  849),
    ("HP OMEN 16",                  "HP",        2024, 1099),
    ("Acer Swift X 14",             "Acer",      2024,  899),
    ("Acer Predator Helios 16",     "Acer",      2024, 1299),
    ("LG Gram 16",                  "LG",        2024, 1099),
    ("LG Gram Pro 16",              "LG",        2024, 1699),
    # Budget
    ("Dell Inspiron 15 5530",       "Dell",      2024,  749),
    ("Acer Aspire 5 15",            "Acer",      2024,  549),
    ("HP Pavilion 15",              "HP",        2024,  599),
    ("Lenovo IdeaPad Slim 5",       "Lenovo",    2024,  649),
    ("ASUS VivoBook 15",            "ASUS",      2024,  499),
    ("Microsoft Surface Laptop Go 3","Microsoft",2024,  799),
]

SMARTPHONES = [
    # Premium
    ("iPhone 15 Pro Max",          "Apple",    2024, 1199),
    ("iPhone 15 Pro",              "Apple",    2024,  999),
    ("iPhone 15",                  "Apple",    2024,  799),
    ("iPhone 14",                  "Apple",    2023,  699),
    ("Samsung Galaxy S24 Ultra",   "Samsung",  2024, 1299),
    ("Samsung Galaxy S24+",        "Samsung",  2024,  999),
    ("Samsung Galaxy S24",         "Samsung",  2024,  799),
    ("Google Pixel 8 Pro",         "Google",   2024,  999),
    ("Google Pixel 8",             "Google",   2024,  699),
    ("OnePlus 12",                 "OnePlus",  2024,  799),
    ("Xiaomi 14 Pro",              "Xiaomi",   2024,  899),
    ("Xiaomi 14 Ultra",            "Xiaomi",   2024, 1399),
    ("Sony Xperia 1 V",            "Sony",     2023, 1299),
    # Mid-range
    ("Samsung Galaxy S23 FE",      "Samsung",  2024,  599),
    ("Samsung Galaxy A55",         "Samsung",  2024,  449),
    ("Samsung Galaxy A54",         "Samsung",  2024,  399),
    ("Google Pixel 7a",            "Google",   2023,  499),
    ("Nothing Phone 2",            "Nothing",  2024,  599),
    ("Nothing Phone 2a",           "Nothing",  2024,  349),
    ("OnePlus Nord 3",             "OnePlus",  2024,  449),
    ("Motorola Edge 40 Pro",       "Motorola", 2024,  599),
    ("Xiaomi Redmi Note 13 Pro",   "Xiaomi",   2024,  299),
    # Budget
    ("Samsung Galaxy A35",         "Samsung",  2024,  299),
    ("Motorola Moto G84",          "Motorola", 2024,  249),
    ("Xiaomi Redmi 13C",           "Xiaomi",   2024,  149),
    ("Nokia G42",                  "Nokia",    2024,  199),
]

HEADPHONES = [
    # Over-ear ANC
    ("WH-1000XM5",               "Sony",        2023,  349),
    ("WH-1000XM4",               "Sony",        2021,  249),
    ("QuietComfort 45",          "Bose",        2023,  279),
    ("QuietComfort Ultra",       "Bose",        2024,  429),
    ("Momentum 4 Wireless",      "Sennheiser",  2023,  299),
    ("Evolve2 85",               "Jabra",       2023,  449),
    ("Headphones 700",           "Bose",        2022,  229),
    ("LiveQ9f",                  "JBL",         2023,  199),
    ("Tune 770NC",               "JBL",         2023,  129),
    ("H95",                      "Logitech",    2023,  499),
    # In-ear / earbuds
    ("AirPods Pro 2",            "Apple",       2023,  249),
    ("AirPods 3",                "Apple",       2023,  169),
    ("Galaxy Buds2 Pro",         "Samsung",     2023,  199),
    ("Galaxy Buds FE",           "Samsung",     2024,   99),
    ("WF-1000XM5",               "Sony",        2024,  279),
    ("QuietComfort Earbuds II",  "Bose",        2023,  299),
    ("Freebuds Pro 3",           "Huawei",      2024,  199),
    # Open / on-ear
    ("Momentum On-Ear 2",        "Sennheiser",  2022,  149),
    ("Tune 510BT",               "JBL",         2023,   49),
]

MONITORS = [
    # 4K / professional
    ("UltraGear 27GP950-B",      "LG",      2023,  699),
    ("UltraGear 32GQ950-B",      "LG",      2023,  999),
    ("UltraSharp U2723QE",       "Dell",    2023,  649),
    ("UltraSharp U3223QE",       "Dell",    2023,  849),
    ("ProArt PA32UCG",           "ASUS",    2023, 1999),
    ("ProArt PA279CRV",          "ASUS",    2024,  649),
    ("DesignVue PD2705U",        "BenQ",    2023,  499),
    ("DesignVue PD3220U",        "BenQ",    2023,  999),
    # Gaming
    ("Odyssey G7 32\"",          "Samsung", 2023,  599),
    ("Odyssey G9 49\"",          "Samsung", 2023, 1299),
    ("Predator X34P",            "Acer",    2023,  799),
    ("Predator XB323UGX",        "Acer",    2024,  699),
    ("UltraGear 27GP850-B",      "LG",      2022,  349),
    ("TUF Gaming VG279QM",       "ASUS",    2023,  299),
    ("Optix MAG274QRF-QD",       "MSI",     2023,  399),
    # Budget / office
    ("P Series P2422H",          "Dell",    2023,  229),
    ("Evnia 27E1N3300A",         "Philips", 2023,  199),
    ("24G2U",                    "AOC",     2023,  179),
    ("IPS269Q",                  "Acer",    2023,  189),
]

CATEGORY_MAP = {
    "Laptops":     LAPTOPS,
    "Smartphones": SMARTPHONES,
    "Headphones":  HEADPHONES,
    "Monitors":    MONITORS,
}


# ── Hardcoded fallback specs ───────────────────────────────────────────────────
# Used when Icecat credentials are absent or a product isn't found.
# Schema: name → {cpu_score, ram_gb, battery_h, weight_kg, display_score,
#                  gpu_score, avg_rating, review_count, pos_pct,
#                  pos_topics, neg_topics}

FALLBACK_SPECS = {
    # Laptops
    "MacBook Air M3": {
        "cpu_score": 95, "ram_gb": 16, "battery_h": 18.0, "weight_kg": 1.24,
        "display_score": 78, "gpu_score": 72, "avg_rating": 4.7, "review_count": 4812,
        "pos_pct": 91, "pos_topics": "battery,performance,build quality,keyboard",
        "neg_topics": "port selection,no touchscreen",
    },
    "MacBook Pro M3 14\"": {
        "cpu_score": 99, "ram_gb": 18, "battery_h": 17.0, "weight_kg": 1.55,
        "display_score": 92, "gpu_score": 90, "avg_rating": 4.8, "review_count": 3201,
        "pos_pct": 94, "pos_topics": "performance,display,build quality",
        "neg_topics": "price,weight",
    },
    "Dell XPS 15 9530": {
        "cpu_score": 85, "ram_gb": 32, "battery_h": 6.5, "weight_kg": 1.86,
        "display_score": 98, "gpu_score": 80, "avg_rating": 4.1, "review_count": 5103,
        "pos_pct": 72, "pos_topics": "display,speakers,build quality",
        "neg_topics": "battery life,thermals,price",
    },
    "ThinkPad X1 Carbon Gen 11": {
        "cpu_score": 70, "ram_gb": 16, "battery_h": 12.0, "weight_kg": 1.12,
        "display_score": 62, "gpu_score": 45, "avg_rating": 4.4, "review_count": 2515,
        "pos_pct": 83, "pos_topics": "keyboard,durability,portability",
        "neg_topics": "display quality,price,gpu",
    },
    "ASUS ZenBook Pro 14": {
        "cpu_score": 78, "ram_gb": 16, "battery_h": 8.0, "weight_kg": 1.65,
        "display_score": 88, "gpu_score": 75, "avg_rating": 4.2, "review_count": 1872,
        "pos_pct": 76, "pos_topics": "display,value,performance",
        "neg_topics": "battery,fan noise",
    },
    "HP Spectre x360 14": {
        "cpu_score": 72, "ram_gb": 16, "battery_h": 13.0, "weight_kg": 1.36,
        "display_score": 85, "gpu_score": 48, "avg_rating": 4.3, "review_count": 2108,
        "pos_pct": 79, "pos_topics": "2-in-1 design,display,build",
        "neg_topics": "performance,price",
    },
    "Lenovo Legion 5 Pro": {
        "cpu_score": 88, "ram_gb": 16, "battery_h": 7.5, "weight_kg": 2.49,
        "display_score": 90, "gpu_score": 96, "avg_rating": 4.5, "review_count": 6721,
        "pos_pct": 85, "pos_topics": "gaming performance,display,value",
        "neg_topics": "weight,battery,thermals",
    },
    "ASUS ROG Zephyrus G14": {
        "cpu_score": 87, "ram_gb": 16, "battery_h": 10.0, "weight_kg": 1.65,
        "display_score": 90, "gpu_score": 95, "avg_rating": 4.6, "review_count": 4302,
        "pos_pct": 88, "pos_topics": "gaming,portability,display",
        "neg_topics": "battery under load,price",
    },
    "Microsoft Surface Laptop 5": {
        "cpu_score": 68, "ram_gb": 16, "battery_h": 14.0, "weight_kg": 1.27,
        "display_score": 80, "gpu_score": 40, "avg_rating": 4.2, "review_count": 1654,
        "pos_pct": 78, "pos_topics": "design,display,keyboard",
        "neg_topics": "performance,port selection,price",
    },
    "Acer Swift X 14": {
        "cpu_score": 75, "ram_gb": 16, "battery_h": 12.0, "weight_kg": 1.44,
        "display_score": 80, "gpu_score": 70, "avg_rating": 4.1, "review_count": 987,
        "pos_pct": 74, "pos_topics": "value,performance,battery",
        "neg_topics": "build quality,display brightness",
    },
    "Razer Blade 15": {
        "cpu_score": 90, "ram_gb": 16, "battery_h": 8.0, "weight_kg": 2.01,
        "display_score": 95, "gpu_score": 98, "avg_rating": 4.3, "review_count": 3109,
        "pos_pct": 80, "pos_topics": "display,build quality,gaming",
        "neg_topics": "price,battery,thermals",
    },
    "Samsung Galaxy Book3 Pro": {
        "cpu_score": 72, "ram_gb": 16, "battery_h": 12.0, "weight_kg": 1.17,
        "display_score": 90, "gpu_score": 45, "avg_rating": 4.2, "review_count": 1423,
        "pos_pct": 77, "pos_topics": "display,portability,design",
        "neg_topics": "performance,gpu,price",
    },
    "LG Gram 16": {
        "cpu_score": 70, "ram_gb": 16, "battery_h": 22.0, "weight_kg": 1.19,
        "display_score": 78, "gpu_score": 40, "avg_rating": 4.3, "review_count": 1876,
        "pos_pct": 80, "pos_topics": "battery,portability,weight",
        "neg_topics": "display brightness,gpu,build rigidity",
    },
    "Dell Inspiron 15 5530": {
        "cpu_score": 65, "ram_gb": 16, "battery_h": 8.0, "weight_kg": 1.83,
        "display_score": 68, "gpu_score": 40, "avg_rating": 4.0, "review_count": 3210,
        "pos_pct": 70, "pos_topics": "value,performance",
        "neg_topics": "display,build quality,battery",
    },
    "HP Envy x360 15": {
        "cpu_score": 67, "ram_gb": 16, "battery_h": 9.5, "weight_kg": 1.79,
        "display_score": 75, "gpu_score": 42, "avg_rating": 4.1, "review_count": 2341,
        "pos_pct": 73, "pos_topics": "value,2-in-1,battery",
        "neg_topics": "display,performance,build",
    },
    # Smartphones
    "iPhone 15 Pro": {
        "cpu_score": 98, "ram_gb": 8, "battery_h": 23.0, "weight_kg": 0.187,
        "display_score": 90, "gpu_score": 95, "avg_rating": 4.8, "review_count": 12450,
        "pos_pct": 93, "pos_topics": "performance,camera,build",
        "neg_topics": "price,battery life,heating",
    },
    "Samsung Galaxy S24 Ultra": {
        "cpu_score": 92, "ram_gb": 12, "battery_h": 27.0, "weight_kg": 0.232,
        "display_score": 98, "gpu_score": 88, "avg_rating": 4.7, "review_count": 9821,
        "pos_pct": 89, "pos_topics": "display,camera,s-pen",
        "neg_topics": "price,size,one ui",
    },
    "Google Pixel 8 Pro": {
        "cpu_score": 88, "ram_gb": 12, "battery_h": 24.0, "weight_kg": 0.213,
        "display_score": 92, "gpu_score": 82, "avg_rating": 4.6, "review_count": 6102,
        "pos_pct": 87, "pos_topics": "camera,software,updates",
        "neg_topics": "battery drain,price",
    },
    "OnePlus 12": {
        "cpu_score": 90, "ram_gb": 16, "battery_h": 24.0, "weight_kg": 0.220,
        "display_score": 88, "gpu_score": 85, "avg_rating": 4.5, "review_count": 4302,
        "pos_pct": 84, "pos_topics": "performance,charging speed,value",
        "neg_topics": "camera,software updates",
    },
    "iPhone 15": {
        "cpu_score": 88, "ram_gb": 6, "battery_h": 20.0, "weight_kg": 0.171,
        "display_score": 85, "gpu_score": 88, "avg_rating": 4.7, "review_count": 15230,
        "pos_pct": 91, "pos_topics": "performance,build,ecosystem",
        "neg_topics": "price,charger not included",
    },
    "Samsung Galaxy A54": {
        "cpu_score": 65, "ram_gb": 8, "battery_h": 28.0, "weight_kg": 0.202,
        "display_score": 80, "gpu_score": 58, "avg_rating": 4.4, "review_count": 8901,
        "pos_pct": 82, "pos_topics": "battery,value,display",
        "neg_topics": "performance,camera in low light",
    },
    "Xiaomi 14 Pro": {
        "cpu_score": 91, "ram_gb": 16, "battery_h": 25.0, "weight_kg": 0.223,
        "display_score": 94, "gpu_score": 86, "avg_rating": 4.5, "review_count": 2103,
        "pos_pct": 83, "pos_topics": "performance,display,charging",
        "neg_topics": "software,availability",
    },
    "Nothing Phone 2": {
        "cpu_score": 75, "ram_gb": 12, "battery_h": 22.0, "weight_kg": 0.201,
        "display_score": 82, "gpu_score": 68, "avg_rating": 4.3, "review_count": 1876,
        "pos_pct": 78, "pos_topics": "design,software,value",
        "neg_topics": "camera,performance vs price",
    },
    # Headphones
    "WH-1000XM5": {
        "cpu_score": 85, "ram_gb": 0, "battery_h": 30.0, "weight_kg": 0.250,
        "display_score": 90, "gpu_score": 0, "avg_rating": 4.7, "review_count": 18920,
        "pos_pct": 92, "pos_topics": "noise cancelling,sound quality,comfort",
        "neg_topics": "call quality,no multipoint initially",
    },
    "AirPods Pro 2": {
        "cpu_score": 88, "ram_gb": 0, "battery_h": 30.0, "weight_kg": 0.062,
        "display_score": 85, "gpu_score": 0, "avg_rating": 4.8, "review_count": 22103,
        "pos_pct": 94, "pos_topics": "anc,transparency,ecosystem",
        "neg_topics": "price,ear tip fit",
    },
    "QuietComfort 45": {
        "cpu_score": 80, "ram_gb": 0, "battery_h": 24.0, "weight_kg": 0.238,
        "display_score": 88, "gpu_score": 0, "avg_rating": 4.6, "review_count": 12301,
        "pos_pct": 89, "pos_topics": "comfort,noise cancelling,sound",
        "neg_topics": "no multipoint,app",
    },
    "Momentum 4": {
        "cpu_score": 82, "ram_gb": 0, "battery_h": 60.0, "weight_kg": 0.293,
        "display_score": 86, "gpu_score": 0, "avg_rating": 4.5, "review_count": 4201,
        "pos_pct": 86, "pos_topics": "battery,sound quality,comfort",
        "neg_topics": "anc vs sony,price",
    },
    "Galaxy Buds2 Pro": {
        "cpu_score": 78, "ram_gb": 0, "battery_h": 18.0, "weight_kg": 0.006,
        "display_score": 84, "gpu_score": 0, "avg_rating": 4.4, "review_count": 8930,
        "pos_pct": 82, "pos_topics": "anc,comfort,galaxy ecosystem",
        "neg_topics": "non-samsung use,call quality",
    },
    "Evolve2 85": {
        "cpu_score": 76, "ram_gb": 0, "battery_h": 37.0, "weight_kg": 0.340,
        "display_score": 80, "gpu_score": 0, "avg_rating": 4.5, "review_count": 2109,
        "pos_pct": 84, "pos_topics": "call quality,anc,comfort all day",
        "neg_topics": "price,consumer sound",
    },
    # New laptops
    "MacBook Pro M3 16\"": {
        "cpu_score": 99, "ram_gb": 18, "battery_h": 22.0, "weight_kg": 2.14,
        "display_score": 95, "gpu_score": 92, "avg_rating": 4.9, "review_count": 2100,
        "pos_pct": 95, "pos_topics": "performance,display,battery",
        "neg_topics": "price,weight",
    },
    "Dell XPS 13 9340": {
        "cpu_score": 78, "ram_gb": 16, "battery_h": 12.0, "weight_kg": 1.17,
        "display_score": 88, "gpu_score": 42, "avg_rating": 4.3, "review_count": 1800,
        "pos_pct": 80, "pos_topics": "portability,display,build quality",
        "neg_topics": "port selection,battery",
    },
    "ThinkPad X1 Extreme Gen 5": {
        "cpu_score": 92, "ram_gb": 32, "battery_h": 8.0, "weight_kg": 1.81,
        "display_score": 90, "gpu_score": 88, "avg_rating": 4.5, "review_count": 1200,
        "pos_pct": 84, "pos_topics": "performance,build,keyboard",
        "neg_topics": "battery,weight,price",
    },
    "Razer Blade 14": {
        "cpu_score": 89, "ram_gb": 16, "battery_h": 7.0, "weight_kg": 1.78,
        "display_score": 92, "gpu_score": 94, "avg_rating": 4.4, "review_count": 2300,
        "pos_pct": 82, "pos_topics": "compact gaming,display,build",
        "neg_topics": "battery,price,thermals",
    },
    "Microsoft Surface Pro 9": {
        "cpu_score": 74, "ram_gb": 16, "battery_h": 15.0, "weight_kg": 0.88,
        "display_score": 87, "gpu_score": 38, "avg_rating": 4.1, "review_count": 1500,
        "pos_pct": 76, "pos_topics": "portability,display,versatility",
        "neg_topics": "price,keyboard sold separately,performance",
    },
    "Samsung Galaxy Book3 Ultra": {
        "cpu_score": 88, "ram_gb": 16, "battery_h": 10.0, "weight_kg": 1.79,
        "display_score": 95, "gpu_score": 85, "avg_rating": 4.4, "review_count": 890,
        "pos_pct": 81, "pos_topics": "display,performance,design",
        "neg_topics": "battery,price,weight",
    },
    "ASUS ZenBook 14 OLED": {
        "cpu_score": 72, "ram_gb": 16, "battery_h": 10.0, "weight_kg": 1.39,
        "display_score": 94, "gpu_score": 40, "avg_rating": 4.4, "review_count": 2100,
        "pos_pct": 83, "pos_topics": "oled display,value,portability",
        "neg_topics": "battery,gpu,webcam",
    },
    "ASUS ROG Zephyrus G16": {
        "cpu_score": 90, "ram_gb": 16, "battery_h": 9.0, "weight_kg": 1.95,
        "display_score": 93, "gpu_score": 96, "avg_rating": 4.6, "review_count": 1400,
        "pos_pct": 87, "pos_topics": "display,gaming performance,design",
        "neg_topics": "battery under load,price",
    },
    "Lenovo Legion 7i": {
        "cpu_score": 92, "ram_gb": 16, "battery_h": 6.0, "weight_kg": 2.50,
        "display_score": 92, "gpu_score": 97, "avg_rating": 4.6, "review_count": 3100,
        "pos_pct": 86, "pos_topics": "gaming performance,display,cooling",
        "neg_topics": "battery,weight,price",
    },
    "Lenovo IdeaPad 5 Pro 16": {
        "cpu_score": 74, "ram_gb": 16, "battery_h": 11.0, "weight_kg": 1.90,
        "display_score": 86, "gpu_score": 58, "avg_rating": 4.2, "review_count": 1600,
        "pos_pct": 77, "pos_topics": "display,value,performance",
        "neg_topics": "build quality,battery,weight",
    },
    "HP OMEN 16": {
        "cpu_score": 86, "ram_gb": 16, "battery_h": 6.5, "weight_kg": 2.35,
        "display_score": 88, "gpu_score": 92, "avg_rating": 4.3, "review_count": 2800,
        "pos_pct": 80, "pos_topics": "gaming,value,display",
        "neg_topics": "battery,weight,thermals",
    },
    "Acer Predator Helios 16": {
        "cpu_score": 91, "ram_gb": 16, "battery_h": 5.0, "weight_kg": 2.60,
        "display_score": 91, "gpu_score": 96, "avg_rating": 4.4, "review_count": 1900,
        "pos_pct": 82, "pos_topics": "gaming,display,value",
        "neg_topics": "battery,weight,thermals",
    },
    "LG Gram Pro 16": {
        "cpu_score": 82, "ram_gb": 16, "battery_h": 17.0, "weight_kg": 1.19,
        "display_score": 88, "gpu_score": 55, "avg_rating": 4.5, "review_count": 980,
        "pos_pct": 85, "pos_topics": "weight,battery,performance",
        "neg_topics": "price,gpu,display brightness",
    },
    "Acer Aspire 5 15": {
        "cpu_score": 62, "ram_gb": 8, "battery_h": 8.0, "weight_kg": 1.80,
        "display_score": 65, "gpu_score": 35, "avg_rating": 4.0, "review_count": 8500,
        "pos_pct": 71, "pos_topics": "value,battery,everyday use",
        "neg_topics": "display,build quality,gpu",
    },
    "HP Pavilion 15": {
        "cpu_score": 60, "ram_gb": 8, "battery_h": 8.5, "weight_kg": 1.75,
        "display_score": 62, "gpu_score": 30, "avg_rating": 3.9, "review_count": 6200,
        "pos_pct": 68, "pos_topics": "value,everyday tasks",
        "neg_topics": "display,performance,build",
    },
    "Lenovo IdeaPad Slim 5": {
        "cpu_score": 65, "ram_gb": 16, "battery_h": 9.5, "weight_kg": 1.56,
        "display_score": 70, "gpu_score": 32, "avg_rating": 4.1, "review_count": 4300,
        "pos_pct": 74, "pos_topics": "value,battery,portability",
        "neg_topics": "display,gpu,build",
    },
    "ASUS VivoBook 15": {
        "cpu_score": 58, "ram_gb": 8, "battery_h": 7.5, "weight_kg": 1.70,
        "display_score": 60, "gpu_score": 28, "avg_rating": 3.9, "review_count": 9800,
        "pos_pct": 67, "pos_topics": "value,everyday use",
        "neg_topics": "display,performance,battery",
    },
    "Microsoft Surface Laptop Go 3": {
        "cpu_score": 60, "ram_gb": 8, "battery_h": 13.0, "weight_kg": 1.13,
        "display_score": 72, "gpu_score": 30, "avg_rating": 4.1, "review_count": 1100,
        "pos_pct": 75, "pos_topics": "portability,battery,design",
        "neg_topics": "performance,storage,price",
    },
    # New smartphones
    "iPhone 15 Pro Max": {
        "cpu_score": 99, "ram_gb": 8, "battery_h": 29.0, "weight_kg": 0.221,
        "display_score": 95, "gpu_score": 97, "avg_rating": 4.8, "review_count": 9800,
        "pos_pct": 92, "pos_topics": "performance,camera,battery",
        "neg_topics": "price,size,weight",
    },
    "iPhone 14": {
        "cpu_score": 85, "ram_gb": 6, "battery_h": 20.0, "weight_kg": 0.172,
        "display_score": 82, "gpu_score": 85, "avg_rating": 4.6, "review_count": 18200,
        "pos_pct": 88, "pos_topics": "performance,ecosystem,build",
        "neg_topics": "price,no usb-c,same design",
    },
    "Samsung Galaxy S24+": {
        "cpu_score": 91, "ram_gb": 12, "battery_h": 26.0, "weight_kg": 0.196,
        "display_score": 95, "gpu_score": 86, "avg_rating": 4.6, "review_count": 5400,
        "pos_pct": 87, "pos_topics": "display,performance,ai features",
        "neg_topics": "price,one ui bloat",
    },
    "Samsung Galaxy S24": {
        "cpu_score": 89, "ram_gb": 8, "battery_h": 22.0, "weight_kg": 0.167,
        "display_score": 90, "gpu_score": 84, "avg_rating": 4.5, "review_count": 7800,
        "pos_pct": 85, "pos_topics": "display,performance,compact",
        "neg_topics": "battery,price,one ui",
    },
    "Google Pixel 8": {
        "cpu_score": 84, "ram_gb": 8, "battery_h": 24.0, "weight_kg": 0.187,
        "display_score": 88, "gpu_score": 78, "avg_rating": 4.5, "review_count": 4100,
        "pos_pct": 85, "pos_topics": "camera,software,updates",
        "neg_topics": "battery drain,price",
    },
    "Xiaomi 14 Ultra": {
        "cpu_score": 97, "ram_gb": 16, "battery_h": 24.0, "weight_kg": 0.222,
        "display_score": 97, "gpu_score": 92, "avg_rating": 4.6, "review_count": 1500,
        "pos_pct": 86, "pos_topics": "camera,display,performance",
        "neg_topics": "software,availability,price",
    },
    "Sony Xperia 1 V": {
        "cpu_score": 93, "ram_gb": 12, "battery_h": 22.0, "weight_kg": 0.187,
        "display_score": 98, "gpu_score": 88, "avg_rating": 4.4, "review_count": 980,
        "pos_pct": 79, "pos_topics": "display,camera,build quality",
        "neg_topics": "price,battery,niche appeal",
    },
    "Samsung Galaxy S23 FE": {
        "cpu_score": 80, "ram_gb": 8, "battery_h": 24.0, "weight_kg": 0.209,
        "display_score": 85, "gpu_score": 75, "avg_rating": 4.3, "review_count": 4200,
        "pos_pct": 80, "pos_topics": "value,display,performance",
        "neg_topics": "camera vs s23,one ui",
    },
    "Samsung Galaxy A55": {
        "cpu_score": 68, "ram_gb": 8, "battery_h": 26.0, "weight_kg": 0.213,
        "display_score": 82, "gpu_score": 60, "avg_rating": 4.3, "review_count": 3100,
        "pos_pct": 79, "pos_topics": "battery,display,value",
        "neg_topics": "performance,camera",
    },
    "Google Pixel 7a": {
        "cpu_score": 80, "ram_gb": 8, "battery_h": 24.0, "weight_kg": 0.193,
        "display_score": 86, "gpu_score": 76, "avg_rating": 4.5, "review_count": 5800,
        "pos_pct": 86, "pos_topics": "camera,software,value",
        "neg_topics": "battery speed,plastic back",
    },
    "Nothing Phone 2a": {
        "cpu_score": 68, "ram_gb": 8, "battery_h": 26.0, "weight_kg": 0.190,
        "display_score": 80, "gpu_score": 60, "avg_rating": 4.3, "review_count": 2100,
        "pos_pct": 81, "pos_topics": "design,value,software",
        "neg_topics": "camera,performance ceiling",
    },
    "OnePlus Nord 3": {
        "cpu_score": 72, "ram_gb": 8, "battery_h": 28.0, "weight_kg": 0.193,
        "display_score": 84, "gpu_score": 65, "avg_rating": 4.3, "review_count": 2800,
        "pos_pct": 80, "pos_topics": "charging speed,value,display",
        "neg_topics": "camera,software updates",
    },
    "Motorola Edge 40 Pro": {
        "cpu_score": 88, "ram_gb": 12, "battery_h": 22.0, "weight_kg": 0.199,
        "display_score": 90, "gpu_score": 82, "avg_rating": 4.3, "review_count": 1600,
        "pos_pct": 79, "pos_topics": "display,performance,charging",
        "neg_topics": "camera,software support,availability",
    },
    "Xiaomi Redmi Note 13 Pro": {
        "cpu_score": 66, "ram_gb": 8, "battery_h": 28.0, "weight_kg": 0.187,
        "display_score": 86, "gpu_score": 58, "avg_rating": 4.4, "review_count": 6200,
        "pos_pct": 82, "pos_topics": "display,battery,value",
        "neg_topics": "software ads,camera inconsistency",
    },
    "Samsung Galaxy A35": {
        "cpu_score": 62, "ram_gb": 6, "battery_h": 26.0, "weight_kg": 0.210,
        "display_score": 78, "gpu_score": 55, "avg_rating": 4.2, "review_count": 3400,
        "pos_pct": 76, "pos_topics": "battery,value,display",
        "neg_topics": "performance,camera",
    },
    "Motorola Moto G84": {
        "cpu_score": 60, "ram_gb": 12, "battery_h": 28.0, "weight_kg": 0.167,
        "display_score": 82, "gpu_score": 50, "avg_rating": 4.2, "review_count": 2800,
        "pos_pct": 77, "pos_topics": "value,display,battery",
        "neg_topics": "camera,performance",
    },
    "Xiaomi Redmi 13C": {
        "cpu_score": 42, "ram_gb": 4, "battery_h": 25.0, "weight_kg": 0.192,
        "display_score": 62, "gpu_score": 35, "avg_rating": 4.0, "review_count": 4100,
        "pos_pct": 70, "pos_topics": "price,battery",
        "neg_topics": "performance,camera,software",
    },
    "Nokia G42": {
        "cpu_score": 50, "ram_gb": 6, "battery_h": 24.0, "weight_kg": 0.193,
        "display_score": 65, "gpu_score": 40, "avg_rating": 4.0, "review_count": 1100,
        "pos_pct": 69, "pos_topics": "value,clean software,repairability",
        "neg_topics": "performance,camera",
    },
    # New headphones
    "WH-1000XM4": {
        "cpu_score": 82, "ram_gb": 0, "battery_h": 30.0, "weight_kg": 0.254,
        "display_score": 88, "gpu_score": 0, "avg_rating": 4.7, "review_count": 32000,
        "pos_pct": 91, "pos_topics": "anc,sound quality,value",
        "neg_topics": "no USB-C initially,call quality",
    },
    "QuietComfort Ultra": {
        "cpu_score": 88, "ram_gb": 0, "battery_h": 24.0, "weight_kg": 0.250,
        "display_score": 92, "gpu_score": 0, "avg_rating": 4.6, "review_count": 3800,
        "pos_pct": 88, "pos_topics": "anc,immersive audio,comfort",
        "neg_topics": "price,battery vs xm5",
    },
    "Momentum 4 Wireless": {
        "cpu_score": 82, "ram_gb": 0, "battery_h": 60.0, "weight_kg": 0.293,
        "display_score": 86, "gpu_score": 0, "avg_rating": 4.5, "review_count": 4201,
        "pos_pct": 86, "pos_topics": "battery,sound quality,comfort",
        "neg_topics": "anc vs sony,price",
    },
    "Headphones 700": {
        "cpu_score": 78, "ram_gb": 0, "battery_h": 20.0, "weight_kg": 0.250,
        "display_score": 85, "gpu_score": 0, "avg_rating": 4.3, "review_count": 6800,
        "pos_pct": 80, "pos_topics": "anc,design,mic quality",
        "neg_topics": "price,no foldable,battery",
    },
    "LiveQ9f": {
        "cpu_score": 72, "ram_gb": 0, "battery_h": 25.0, "weight_kg": 0.220,
        "display_score": 80, "gpu_score": 0, "avg_rating": 4.2, "review_count": 3200,
        "pos_pct": 76, "pos_topics": "value,sound,anc",
        "neg_topics": "app,build quality",
    },
    "Tune 770NC": {
        "cpu_score": 60, "ram_gb": 0, "battery_h": 38.0, "weight_kg": 0.200,
        "display_score": 72, "gpu_score": 0, "avg_rating": 4.1, "review_count": 5100,
        "pos_pct": 74, "pos_topics": "value,battery,anc for price",
        "neg_topics": "sound quality,build,anc depth",
    },
    "H95": {
        "cpu_score": 90, "ram_gb": 0, "battery_h": 20.0, "weight_kg": 0.375,
        "display_score": 90, "gpu_score": 0, "avg_rating": 4.5, "review_count": 1200,
        "pos_pct": 85, "pos_topics": "sound quality,build,anc",
        "neg_topics": "price,weight,app",
    },
    "AirPods 3": {
        "cpu_score": 80, "ram_gb": 0, "battery_h": 30.0, "weight_kg": 0.037,
        "display_score": 78, "gpu_score": 0, "avg_rating": 4.5, "review_count": 14200,
        "pos_pct": 86, "pos_topics": "ecosystem,spatial audio,comfort",
        "neg_topics": "no anc,fit for some,price",
    },
    "Galaxy Buds FE": {
        "cpu_score": 62, "ram_gb": 0, "battery_h": 21.0, "weight_kg": 0.005,
        "display_score": 72, "gpu_score": 0, "avg_rating": 4.1, "review_count": 2800,
        "pos_pct": 73, "pos_topics": "value,anc,galaxy ecosystem",
        "neg_topics": "sound quality,non-samsung use",
    },
    "WF-1000XM5": {
        "cpu_score": 90, "ram_gb": 0, "battery_h": 24.0, "weight_kg": 0.005,
        "display_score": 92, "gpu_score": 0, "avg_rating": 4.7, "review_count": 7800,
        "pos_pct": 91, "pos_topics": "anc,sound quality,comfort",
        "neg_topics": "price,fit for some",
    },
    "QuietComfort Earbuds II": {
        "cpu_score": 87, "ram_gb": 0, "battery_h": 24.0, "weight_kg": 0.006,
        "display_score": 88, "gpu_score": 0, "avg_rating": 4.5, "review_count": 4900,
        "pos_pct": 87, "pos_topics": "anc,comfort,sound",
        "neg_topics": "price,no multipoint",
    },
    "Freebuds Pro 3": {
        "cpu_score": 78, "ram_gb": 0, "battery_h": 22.0, "weight_kg": 0.005,
        "display_score": 82, "gpu_score": 0, "avg_rating": 4.2, "review_count": 1400,
        "pos_pct": 77, "pos_topics": "anc,value,design",
        "neg_topics": "app,availability,non-huawei use",
    },
    "Momentum On-Ear 2": {
        "cpu_score": 65, "ram_gb": 0, "battery_h": 25.0, "weight_kg": 0.190,
        "display_score": 80, "gpu_score": 0, "avg_rating": 4.2, "review_count": 3100,
        "pos_pct": 75, "pos_topics": "sound quality,design,portability",
        "neg_topics": "anc,price,on-ear comfort",
    },
    "Tune 510BT": {
        "cpu_score": 45, "ram_gb": 0, "battery_h": 40.0, "weight_kg": 0.160,
        "display_score": 60, "gpu_score": 0, "avg_rating": 4.0, "review_count": 9200,
        "pos_pct": 72, "pos_topics": "price,battery,lightweight",
        "neg_topics": "sound quality,no anc,build",
    },
    # New monitors
    "32GQ950-B": {
        "cpu_score": 92, "ram_gb": 0, "battery_h": 0, "weight_kg": 8.1,
        "display_score": 97, "gpu_score": 0, "avg_rating": 4.7, "review_count": 1200,
        "pos_pct": 88, "pos_topics": "4k gaming,color accuracy,refresh rate",
        "neg_topics": "price,size",
    },
    "U3223QE": {
        "cpu_score": 86, "ram_gb": 0, "battery_h": 0, "weight_kg": 7.8,
        "display_score": 98, "gpu_score": 0, "avg_rating": 4.8, "review_count": 1400,
        "pos_pct": 92, "pos_topics": "color accuracy,usb-c,ergonomics",
        "neg_topics": "price,no speakers",
    },
    "ProArt PA279CRV": {
        "cpu_score": 84, "ram_gb": 0, "battery_h": 0, "weight_kg": 6.5,
        "display_score": 96, "gpu_score": 0, "avg_rating": 4.7, "review_count": 980,
        "pos_pct": 90, "pos_topics": "color accuracy,usb-c,value for pro",
        "neg_topics": "refresh rate,size",
    },
    "PD3220U": {
        "cpu_score": 87, "ram_gb": 0, "battery_h": 0, "weight_kg": 8.0,
        "display_score": 97, "gpu_score": 0, "avg_rating": 4.8, "review_count": 820,
        "pos_pct": 91, "pos_topics": "color accuracy,thunderbolt,professional",
        "neg_topics": "price,size",
    },
    "Odyssey G9 49\"": {
        "cpu_score": 88, "ram_gb": 0, "battery_h": 0, "weight_kg": 12.1,
        "display_score": 92, "gpu_score": 0, "avg_rating": 4.5, "review_count": 2800,
        "pos_pct": 83, "pos_topics": "ultrawide,immersion,gaming",
        "neg_topics": "price,size,gpu requirement",
    },
    "Predator XB323UGX": {
        "cpu_score": 90, "ram_gb": 0, "battery_h": 0, "weight_kg": 7.2,
        "display_score": 93, "gpu_score": 0, "avg_rating": 4.5, "review_count": 760,
        "pos_pct": 84, "pos_topics": "refresh rate,color,gaming",
        "neg_topics": "price,stand",
    },
    "27GP850-B": {
        "cpu_score": 85, "ram_gb": 0, "battery_h": 0, "weight_kg": 5.5,
        "display_score": 88, "gpu_score": 0, "avg_rating": 4.6, "review_count": 4100,
        "pos_pct": 87, "pos_topics": "gaming,refresh rate,value",
        "neg_topics": "color accuracy vs ips,stand",
    },
    "VG279QM": {
        "cpu_score": 82, "ram_gb": 0, "battery_h": 0, "weight_kg": 4.8,
        "display_score": 85, "gpu_score": 0, "avg_rating": 4.5, "review_count": 5200,
        "pos_pct": 85, "pos_topics": "refresh rate,value,gaming",
        "neg_topics": "color vs ips,stand quality",
    },
    "MAG274QRF-QD": {
        "cpu_score": 83, "ram_gb": 0, "battery_h": 0, "weight_kg": 5.3,
        "display_score": 90, "gpu_score": 0, "avg_rating": 4.5, "review_count": 2100,
        "pos_pct": 85, "pos_topics": "color accuracy,refresh rate,value",
        "neg_topics": "stand,price vs competition",
    },
    "P2422H": {
        "cpu_score": 60, "ram_gb": 0, "battery_h": 0, "weight_kg": 3.6,
        "display_score": 78, "gpu_score": 0, "avg_rating": 4.4, "review_count": 6800,
        "pos_pct": 82, "pos_topics": "value,color,ergonomics",
        "neg_topics": "refresh rate,bezels",
    },
    "27E1N3300A": {
        "cpu_score": 55, "ram_gb": 0, "battery_h": 0, "weight_kg": 3.4,
        "display_score": 75, "gpu_score": 0, "avg_rating": 4.2, "review_count": 2300,
        "pos_pct": 77, "pos_topics": "value,color,eye comfort",
        "neg_topics": "refresh rate,stand,no usb-c",
    },
    "24G2U": {
        "cpu_score": 58, "ram_gb": 0, "battery_h": 0, "weight_kg": 3.1,
        "display_score": 74, "gpu_score": 0, "avg_rating": 4.3, "review_count": 4100,
        "pos_pct": 79, "pos_topics": "value,refresh rate,gaming entry",
        "neg_topics": "color accuracy,stand",
    },
    "IPS269Q": {
        "cpu_score": 52, "ram_gb": 0, "battery_h": 0, "weight_kg": 3.2,
        "display_score": 72, "gpu_score": 0, "avg_rating": 4.1, "review_count": 1900,
        "pos_pct": 74, "pos_topics": "value,color,everyday use",
        "neg_topics": "refresh rate,stand,connectivity",
    },
    # Monitors
    "27GP950-B": {
        "cpu_score": 88, "ram_gb": 0, "battery_h": 0, "weight_kg": 6.2,
        "display_score": 95, "gpu_score": 0, "avg_rating": 4.7, "review_count": 3201,
        "pos_pct": 89, "pos_topics": "color accuracy,refresh rate,gaming",
        "neg_topics": "price,stand quality",
    },
    "U2723QE": {
        "cpu_score": 85, "ram_gb": 0, "battery_h": 0, "weight_kg": 5.8,
        "display_score": 97, "gpu_score": 0, "avg_rating": 4.8, "review_count": 2109,
        "pos_pct": 91, "pos_topics": "color accuracy,usb-c,ergonomics",
        "neg_topics": "price,no speakers",
    },
    "Odyssey G7": {
        "cpu_score": 80, "ram_gb": 0, "battery_h": 0, "weight_kg": 7.1,
        "display_score": 90, "gpu_score": 0, "avg_rating": 4.5, "review_count": 4312,
        "pos_pct": 85, "pos_topics": "refresh rate,curve,gaming",
        "neg_topics": "brightness uniformity,price",
    },
    "PA32UCG": {
        "cpu_score": 92, "ram_gb": 0, "battery_h": 0, "weight_kg": 9.3,
        "display_score": 99, "gpu_score": 0, "avg_rating": 4.9, "review_count": 891,
        "pos_pct": 95, "pos_topics": "color accuracy,hdr,professional",
        "neg_topics": "price,size",
    },
    "PD2705U": {
        "cpu_score": 78, "ram_gb": 0, "battery_h": 0, "weight_kg": 5.2,
        "display_score": 93, "gpu_score": 0, "avg_rating": 4.6, "review_count": 1654,
        "pos_pct": 87, "pos_topics": "color,usb-c,value",
        "neg_topics": "no hdr,stand",
    },
    "Predator X34P": {
        "cpu_score": 82, "ram_gb": 0, "battery_h": 0, "weight_kg": 8.9,
        "display_score": 88, "gpu_score": 0, "avg_rating": 4.4, "review_count": 2341,
        "pos_pct": 83, "pos_topics": "ultrawide,gaming,immersion",
        "neg_topics": "price,ips glow",
    },
}


USE_CASE_WEIGHTS = {
    # Keys MUST match the USE_CASES list in app.py exactly — scoring does exact-match lookup
    "Laptops": {
        "Work & productivity":  {"cpu_norm":0.25,"battery_norm":0.25,"weight_norm":0.15,"display_norm":0.15,"gpu_norm":0.05,"ram_norm":0.15},
        "Creative work":        {"cpu_norm":0.20,"display_norm":0.25,"gpu_norm":0.25,"ram_norm":0.15,"battery_norm":0.10,"weight_norm":0.05},
        "Gaming":               {"gpu_norm":0.35,"cpu_norm":0.25,"display_norm":0.20,"battery_norm":0.05,"ram_norm":0.15},
        "University":           {"battery_norm":0.25,"weight_norm":0.25,"price_norm":0.25,"cpu_norm":0.15,"display_norm":0.10},
        "Travel & portability": {"weight_norm":0.30,"battery_norm":0.30,"cpu_norm":0.15,"display_norm":0.10,"price_norm":0.15},
        "Programming":          {"cpu_norm":0.25,"display_norm":0.25,"ram_norm":0.20,"battery_norm":0.15,"gpu_norm":0.05,"weight_norm":0.10},
    },
    "Smartphones": {
        "Everyday use":      {"cpu_norm":0.20,"battery_norm":0.25,"display_norm":0.20,"gpu_norm":0.15,"price_norm":0.20},
        "Photography":       {"gpu_norm":0.40,"display_norm":0.25,"cpu_norm":0.20,"battery_norm":0.15},
        "Gaming":            {"cpu_norm":0.30,"gpu_norm":0.30,"display_norm":0.20,"battery_norm":0.10,"ram_norm":0.10},
        "Business":          {"cpu_norm":0.20,"battery_norm":0.25,"display_norm":0.20,"ram_norm":0.15,"price_norm":0.20},
        "Long battery life": {"battery_norm":0.50,"price_norm":0.20,"cpu_norm":0.15,"display_norm":0.15},
        "Value for money":   {"price_norm":0.35,"battery_norm":0.25,"cpu_norm":0.20,"display_norm":0.20},
    },
    "Headphones": {
        "Music & hi-fi":     {"display_norm":0.40,"cpu_norm":0.20,"battery_norm":0.20,"price_norm":0.20},
        "Office calls":      {"cpu_norm":0.35,"display_norm":0.25,"battery_norm":0.20,"weight_norm":0.20},
        "Travel & commute":  {"cpu_norm":0.30,"battery_norm":0.25,"weight_norm":0.25,"display_norm":0.20},
        "Gaming":            {"cpu_norm":0.30,"display_norm":0.30,"battery_norm":0.20,"price_norm":0.20},
        "Gym & sport":       {"weight_norm":0.35,"battery_norm":0.25,"cpu_norm":0.20,"price_norm":0.20},
        "Casual listening":  {"price_norm":0.35,"display_norm":0.30,"battery_norm":0.20,"cpu_norm":0.15},
    },
    "Monitors": {
        "Design & color work": {"display_norm":0.50,"cpu_norm":0.25,"price_norm":0.25},
        "Gaming":              {"cpu_norm":0.40,"display_norm":0.35,"price_norm":0.25},
        "Office & coding":     {"display_norm":0.35,"cpu_norm":0.25,"weight_norm":0.15,"price_norm":0.25},
        "Home cinema":         {"display_norm":0.45,"cpu_norm":0.20,"weight_norm":0.10,"price_norm":0.25},
        "Content creation":    {"display_norm":0.45,"cpu_norm":0.25,"ram_norm":0.05,"price_norm":0.25},
    },
}

BUDGET_RANGES = {
    "Laptops":     {"Under $800":(0,800),"$800–$1,400":(800,1400),"$1,400–$2,000":(1400,2000),"No limit":(0,99999)},
    "Smartphones": {"Under $400":(0,400),"$400–$800":(400,800),"$800–$1,200":(800,1200),"No limit":(0,99999)},
    "Headphones":  {"Under $150":(0,150),"$150–$300":(150,300),"$300+":(300,99999),"No limit":(0,99999)},
    "Monitors":    {"Under $400":(0,400),"$400–$800":(400,800),"$800+":(800,99999),"No limit":(0,99999)},
}


# ── Normalisation & model training ────────────────────────────────────────────

def normalise_col(series, invert=False):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([50.0] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn) * 100
    return 100 - norm if invert else norm


def build_df(category):
    rows = []
    products = CATEGORY_MAP[category]
    use_live = bool(os.getenv("SERPAPI_KEY") or os.getenv("ANTHROPIC_API_KEY"))

    for name, brand, year, price in products:
        specs = None
        if use_live:
            print(f"  Fetching: {brand} {name} …")
            specs = get_product_specs(brand, name, category, year, price)

        if specs is None:
            fb = FALLBACK_SPECS.get(name)
            if fb is None:
                print(f"  WARNING: no fallback data for '{name}', skipping.")
                continue
            specs = {"name": name, "brand": brand, "year": year, "price": price, **fb}
            if use_live:
                print(f"  → using fallback for {name}")

        rows.append(specs)

    df = pd.DataFrame(rows)
    df["category"] = category
    df.fillna({"ram_gb": 0, "gpu_score": 0, "battery_h": 0, "weight_kg": 0}, inplace=True)

    df["price_norm"]   = normalise_col(df["price"],      invert=True)
    df["weight_norm"]  = normalise_col(df["weight_kg"],  invert=True) if df["weight_kg"].sum() > 0 else 50
    df["battery_norm"] = normalise_col(df["battery_h"])  if df["battery_h"].sum() > 0 else 50
    df["cpu_norm"]     = normalise_col(df["cpu_score"])
    df["gpu_norm"]     = normalise_col(df["gpu_score"])
    df["display_norm"] = normalise_col(df["display_score"])
    df["ram_norm"]     = normalise_col(df["ram_gb"])
    df["base_score"]   = (df["avg_rating"] / 5.0 * 60 + df["pos_pct"] * 0.40).round(1)
    return df


def train_model(df, category):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    feature_cols = ["cpu_norm","gpu_norm","display_norm","battery_norm",
                    "weight_norm","ram_norm","price_norm","pos_pct"]
    X = df[feature_cols].fillna(50)
    y = df["base_score"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)),
    ])
    if len(X) > 5:
        scores = cross_val_score(model, X, y, cv=min(3, len(X)), scoring="r2")
        print(f"  [{category}] GBR CV R² = {scores.mean():.3f}")
    model.fit(X, y)
    return model


def save_metadata():
    meta = {
        "use_case_weights": USE_CASE_WEIGHTS,
        "budget_ranges":    BUDGET_RANGES,
        "feature_cols": ["cpu_norm","gpu_norm","display_norm","battery_norm",
                         "weight_norm","ram_norm","price_norm","pos_pct"],
    }
    with open(DATA / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  Saved metadata.json")


def main():
    use_live = bool(os.getenv("SERPAPI_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    mode = "SerpAPI + Claude" if use_live else "hardcoded fallback"
    print(f"Building SpecCheck dataset ({mode})…\n")

    for category in CATEGORY_MAP:
        print(f"Processing {category}…")
        df = build_df(category)
        out_path = DATA / f"products_{category.lower()}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} products → {out_path.name}")
        model = train_model(df, category)
        pkl_path = MODEL_DIR / f"model_{category.lower()}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved model → {pkl_path.name}")

    save_metadata()
    print("\nDone. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
