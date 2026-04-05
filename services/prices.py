"""
services/prices.py

Fetches live retailer prices using SerpAPI Google Shopping.
Results are cached per session so we don't burn API quota on rerenders.

Usage:
    from services.prices import get_prices
    retailers = get_prices("MacBook Air M3", "Laptops")
    # [{"retailer": "Amazon", "price": 1249.0, "link": "...", "is_lowest": True}, ...]

SerpAPI docs: https://serpapi.com/google-shopping-api
Get a free key at: https://serpapi.com/
"""

import os, requests, time
from typing import Optional

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

RETAILER_COLORS = {
    "amazon":    "#f90",
    "best buy":  "#003cb3",
    "walmart":   "#0071ce",
    "b&h":       "#333333",
    "lenovo":    "#e2001a",
    "dell":      "#007db8",
    "apple":     "#555555",
    "costco":    "#e31837",
    "newegg":    "#e85c24",
    "bhphotovideo": "#333333",
    "adorama":   "#c8102e",
    "target":    "#cc0000",
}

FALLBACK_PRICES = {
    # ── Laptops ───────────────────────────────────────────────────────────────
    "MacBook Air M3": [
        {"retailer": "Amazon",      "price": 1249.0, "logo_color": "#f90"},
        {"retailer": "Walmart",     "price": 1269.0, "logo_color": "#0071ce"},
        {"retailer": "Best Buy",    "price": 1299.0, "logo_color": "#003cb3"},
        {"retailer": "Apple Store", "price": 1299.0, "logo_color": "#555555"},
    ],
    "MacBook Pro M3 14\"": [
        {"retailer": "Amazon",      "price": 1899.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1999.0, "logo_color": "#003cb3"},
        {"retailer": "Apple Store", "price": 1999.0, "logo_color": "#555555"},
        {"retailer": "B&H Photo",   "price": 1979.0, "logo_color": "#333333"},
    ],
    "MacBook Pro M3 16\"": [
        {"retailer": "Amazon",      "price": 2399.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 2499.0, "logo_color": "#003cb3"},
        {"retailer": "Apple Store", "price": 2499.0, "logo_color": "#555555"},
        {"retailer": "B&H Photo",   "price": 2449.0, "logo_color": "#333333"},
    ],
    "Dell XPS 15 9530": [
        {"retailer": "Amazon",      "price": 1449.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1499.0, "logo_color": "#003cb3"},
        {"retailer": "Dell.com",    "price": 1549.0, "logo_color": "#007db8"},
        {"retailer": "B&H Photo",   "price": 1499.0, "logo_color": "#333333"},
    ],
    "Dell XPS 13 9340": [
        {"retailer": "Amazon",      "price": 1199.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1299.0, "logo_color": "#003cb3"},
        {"retailer": "Dell.com",    "price": 1349.0, "logo_color": "#007db8"},
        {"retailer": "Walmart",     "price": 1249.0, "logo_color": "#0071ce"},
    ],
    "ThinkPad X1 Carbon Gen 11": [
        {"retailer": "Best Buy",    "price": 1549.0, "logo_color": "#003cb3"},
        {"retailer": "Amazon",      "price": 1599.0, "logo_color": "#f90"},
        {"retailer": "Lenovo.com",  "price": 1649.0, "logo_color": "#e2001a"},
        {"retailer": "B&H Photo",   "price": 1589.0, "logo_color": "#333333"},
    ],
    "ThinkPad X1 Extreme Gen 5": [
        {"retailer": "Lenovo.com",  "price": 1899.0, "logo_color": "#e2001a"},
        {"retailer": "Amazon",      "price": 1849.0, "logo_color": "#f90"},
        {"retailer": "B&H Photo",   "price": 1879.0, "logo_color": "#333333"},
        {"retailer": "Best Buy",    "price": 1949.0, "logo_color": "#003cb3"},
    ],
    "HP Spectre x360 14": [
        {"retailer": "HP.com",      "price": 1399.0, "logo_color": "#0096d6"},
        {"retailer": "Best Buy",    "price": 1349.0, "logo_color": "#003cb3"},
        {"retailer": "Amazon",      "price": 1299.0, "logo_color": "#f90"},
        {"retailer": "Walmart",     "price": 1319.0, "logo_color": "#0071ce"},
    ],
    "Razer Blade 15": [
        {"retailer": "Amazon",      "price": 2399.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 2499.0, "logo_color": "#003cb3"},
        {"retailer": "Razer.com",   "price": 2499.0, "logo_color": "#00d600"},
        {"retailer": "Newegg",      "price": 2449.0, "logo_color": "#e85c24"},
    ],
    "Razer Blade 14": [
        {"retailer": "Amazon",      "price": 1899.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1999.0, "logo_color": "#003cb3"},
        {"retailer": "Razer.com",   "price": 1999.0, "logo_color": "#00d600"},
        {"retailer": "Newegg",      "price": 1949.0, "logo_color": "#e85c24"},
    ],
    "Microsoft Surface Laptop 5": [
        {"retailer": "Microsoft",   "price": 999.0,  "logo_color": "#737373"},
        {"retailer": "Amazon",      "price": 949.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 999.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 969.0,  "logo_color": "#0071ce"},
    ],
    "Microsoft Surface Pro 9": [
        {"retailer": "Microsoft",   "price": 1299.0, "logo_color": "#737373"},
        {"retailer": "Amazon",      "price": 1199.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1299.0, "logo_color": "#003cb3"},
        {"retailer": "Costco",      "price": 1249.0, "logo_color": "#e31837"},
    ],
    "Samsung Galaxy Book3 Pro": [
        {"retailer": "Samsung.com", "price": 1199.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",      "price": 1099.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1199.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 1149.0, "logo_color": "#0071ce"},
    ],
    "Samsung Galaxy Book3 Ultra": [
        {"retailer": "Samsung.com", "price": 1799.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",      "price": 1699.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1799.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",   "price": 1749.0, "logo_color": "#333333"},
    ],
    "ASUS ZenBook Pro 14": [
        {"retailer": "Amazon",      "price": 1099.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1199.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",      "price": 1149.0, "logo_color": "#e85c24"},
        {"retailer": "Walmart",     "price": 1129.0, "logo_color": "#0071ce"},
    ],
    "ASUS ZenBook 14 OLED": [
        {"retailer": "Amazon",      "price": 749.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 799.0,  "logo_color": "#003cb3"},
        {"retailer": "Newegg",      "price": 769.0,  "logo_color": "#e85c24"},
        {"retailer": "Walmart",     "price": 759.0,  "logo_color": "#0071ce"},
    ],
    "ASUS ROG Zephyrus G14": [
        {"retailer": "Amazon",      "price": 1399.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1449.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",      "price": 1419.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",   "price": 1429.0, "logo_color": "#333333"},
    ],
    "ASUS ROG Zephyrus G16": [
        {"retailer": "Amazon",      "price": 1549.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1599.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",      "price": 1569.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",   "price": 1579.0, "logo_color": "#333333"},
    ],
    "Lenovo Legion 5 Pro": [
        {"retailer": "Amazon",      "price": 1199.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1299.0, "logo_color": "#003cb3"},
        {"retailer": "Lenovo.com",  "price": 1299.0, "logo_color": "#e2001a"},
        {"retailer": "Newegg",      "price": 1249.0, "logo_color": "#e85c24"},
    ],
    "Lenovo Legion 7i": [
        {"retailer": "Amazon",      "price": 1499.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1599.0, "logo_color": "#003cb3"},
        {"retailer": "Lenovo.com",  "price": 1599.0, "logo_color": "#e2001a"},
        {"retailer": "Newegg",      "price": 1549.0, "logo_color": "#e85c24"},
    ],
    "Lenovo IdeaPad 5 Pro 16": [
        {"retailer": "Amazon",      "price": 849.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 899.0,  "logo_color": "#003cb3"},
        {"retailer": "Lenovo.com",  "price": 899.0,  "logo_color": "#e2001a"},
        {"retailer": "Walmart",     "price": 869.0,  "logo_color": "#0071ce"},
    ],
    "HP Envy x360 15": [
        {"retailer": "HP.com",      "price": 849.0,  "logo_color": "#0096d6"},
        {"retailer": "Best Buy",    "price": 849.0,  "logo_color": "#003cb3"},
        {"retailer": "Amazon",      "price": 799.0,  "logo_color": "#f90"},
        {"retailer": "Walmart",     "price": 819.0,  "logo_color": "#0071ce"},
    ],
    "HP OMEN 16": [
        {"retailer": "HP.com",      "price": 1099.0, "logo_color": "#0096d6"},
        {"retailer": "Best Buy",    "price": 1099.0, "logo_color": "#003cb3"},
        {"retailer": "Amazon",      "price": 999.0,  "logo_color": "#f90"},
        {"retailer": "Newegg",      "price": 1049.0, "logo_color": "#e85c24"},
    ],
    "Acer Swift X 14": [
        {"retailer": "Amazon",      "price": 849.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 899.0,  "logo_color": "#003cb3"},
        {"retailer": "Newegg",      "price": 869.0,  "logo_color": "#e85c24"},
        {"retailer": "Walmart",     "price": 859.0,  "logo_color": "#0071ce"},
    ],
    "Acer Predator Helios 16": [
        {"retailer": "Amazon",      "price": 1249.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1299.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",      "price": 1269.0, "logo_color": "#e85c24"},
        {"retailer": "Walmart",     "price": 1279.0, "logo_color": "#0071ce"},
    ],
    "LG Gram 16": [
        {"retailer": "Amazon",      "price": 1049.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1099.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",   "price": 1079.0, "logo_color": "#333333"},
        {"retailer": "Walmart",     "price": 1059.0, "logo_color": "#0071ce"},
    ],
    "LG Gram Pro 16": [
        {"retailer": "Amazon",      "price": 1599.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1699.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",   "price": 1649.0, "logo_color": "#333333"},
        {"retailer": "Newegg",      "price": 1629.0, "logo_color": "#e85c24"},
    ],
    "Dell Inspiron 15 5530": [
        {"retailer": "Dell.com",    "price": 749.0,  "logo_color": "#007db8"},
        {"retailer": "Amazon",      "price": 699.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 749.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 719.0,  "logo_color": "#0071ce"},
    ],
    "Acer Aspire 5 15": [
        {"retailer": "Amazon",      "price": 499.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 549.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 519.0,  "logo_color": "#0071ce"},
        {"retailer": "Newegg",      "price": 509.0,  "logo_color": "#e85c24"},
    ],
    "HP Pavilion 15": [
        {"retailer": "HP.com",      "price": 599.0,  "logo_color": "#0096d6"},
        {"retailer": "Amazon",      "price": 549.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 599.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 569.0,  "logo_color": "#0071ce"},
    ],
    "Lenovo IdeaPad Slim 5": [
        {"retailer": "Amazon",      "price": 599.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 649.0,  "logo_color": "#003cb3"},
        {"retailer": "Lenovo.com",  "price": 649.0,  "logo_color": "#e2001a"},
        {"retailer": "Walmart",     "price": 619.0,  "logo_color": "#0071ce"},
    ],
    "ASUS VivoBook 15": [
        {"retailer": "Amazon",      "price": 449.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 499.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 469.0,  "logo_color": "#0071ce"},
        {"retailer": "Newegg",      "price": 459.0,  "logo_color": "#e85c24"},
    ],
    "Microsoft Surface Laptop Go 3": [
        {"retailer": "Microsoft",   "price": 799.0,  "logo_color": "#737373"},
        {"retailer": "Amazon",      "price": 749.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 799.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 769.0,  "logo_color": "#0071ce"},
    ],
    # ── Smartphones ───────────────────────────────────────────────────────────
    "iPhone 15 Pro Max": [
        {"retailer": "Apple Store", "price": 1199.0, "logo_color": "#555555"},
        {"retailer": "Amazon",      "price": 1149.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1199.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 1179.0, "logo_color": "#0071ce"},
    ],
    "iPhone 15 Pro": [
        {"retailer": "Apple Store", "price": 999.0,  "logo_color": "#555555"},
        {"retailer": "Amazon",      "price": 949.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 999.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 979.0,  "logo_color": "#0071ce"},
    ],
    "iPhone 15": [
        {"retailer": "Apple Store", "price": 799.0,  "logo_color": "#555555"},
        {"retailer": "Amazon",      "price": 749.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 799.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 769.0,  "logo_color": "#0071ce"},
    ],
    "iPhone 14": [
        {"retailer": "Apple Store", "price": 699.0,  "logo_color": "#555555"},
        {"retailer": "Amazon",      "price": 599.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 699.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 649.0,  "logo_color": "#0071ce"},
    ],
    "Samsung Galaxy S24 Ultra": [
        {"retailer": "Samsung.com", "price": 1299.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",      "price": 1199.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 1299.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 1249.0, "logo_color": "#0071ce"},
    ],
    "Samsung Galaxy S24+": [
        {"retailer": "Samsung.com", "price": 999.0,  "logo_color": "#1428a0"},
        {"retailer": "Amazon",      "price": 899.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 999.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 949.0,  "logo_color": "#0071ce"},
    ],
    "Samsung Galaxy S24": [
        {"retailer": "Samsung.com", "price": 799.0,  "logo_color": "#1428a0"},
        {"retailer": "Amazon",      "price": 699.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",    "price": 799.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",     "price": 749.0,  "logo_color": "#0071ce"},
    ],
    "Google Pixel 8 Pro": [
        {"retailer": "Google Store", "price": 999.0, "logo_color": "#4285f4"},
        {"retailer": "Amazon",       "price": 899.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 999.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 949.0, "logo_color": "#0071ce"},
    ],
    "Google Pixel 8": [
        {"retailer": "Google Store", "price": 699.0, "logo_color": "#4285f4"},
        {"retailer": "Amazon",       "price": 599.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 699.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 649.0, "logo_color": "#0071ce"},
    ],
    "OnePlus 12": [
        {"retailer": "Amazon",       "price": 749.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 799.0, "logo_color": "#003cb3"},
        {"retailer": "OnePlus.com",  "price": 799.0, "logo_color": "#eb0029"},
        {"retailer": "Walmart",      "price": 769.0, "logo_color": "#0071ce"},
    ],
    "Xiaomi 14 Pro": [
        {"retailer": "Amazon",       "price": 849.0, "logo_color": "#f90"},
        {"retailer": "Newegg",       "price": 879.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 869.0, "logo_color": "#333333"},
        {"retailer": "eBay",         "price": 829.0, "logo_color": "#e53238"},
    ],
    "Xiaomi 14 Ultra": [
        {"retailer": "Amazon",       "price": 1349.0,"logo_color": "#f90"},
        {"retailer": "Newegg",       "price": 1399.0,"logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 1379.0,"logo_color": "#333333"},
        {"retailer": "eBay",         "price": 1329.0,"logo_color": "#e53238"},
    ],
    "Sony Xperia 1 V": [
        {"retailer": "Amazon",       "price": 1199.0,"logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 1299.0,"logo_color": "#003cb3"},
        {"retailer": "B&H Photo",    "price": 1249.0,"logo_color": "#333333"},
        {"retailer": "Newegg",       "price": 1229.0,"logo_color": "#e85c24"},
    ],
    "Samsung Galaxy S23 FE": [
        {"retailer": "Samsung.com",  "price": 599.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 499.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 599.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 549.0, "logo_color": "#0071ce"},
    ],
    "Samsung Galaxy A55": [
        {"retailer": "Samsung.com",  "price": 449.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 399.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 449.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 419.0, "logo_color": "#0071ce"},
    ],
    "Samsung Galaxy A54": [
        {"retailer": "Samsung.com",  "price": 399.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 329.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 399.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 349.0, "logo_color": "#0071ce"},
    ],
    "Google Pixel 7a": [
        {"retailer": "Google Store", "price": 499.0, "logo_color": "#4285f4"},
        {"retailer": "Amazon",       "price": 399.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 499.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 449.0, "logo_color": "#0071ce"},
    ],
    "Nothing Phone 2": [
        {"retailer": "Amazon",       "price": 549.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 599.0, "logo_color": "#003cb3"},
        {"retailer": "Nothing.tech", "price": 599.0, "logo_color": "#333333"},
        {"retailer": "Newegg",       "price": 569.0, "logo_color": "#e85c24"},
    ],
    "Nothing Phone 2a": [
        {"retailer": "Amazon",       "price": 319.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 349.0, "logo_color": "#003cb3"},
        {"retailer": "Nothing.tech", "price": 349.0, "logo_color": "#333333"},
        {"retailer": "Newegg",       "price": 329.0, "logo_color": "#e85c24"},
    ],
    "OnePlus Nord 3": [
        {"retailer": "Amazon",       "price": 429.0, "logo_color": "#f90"},
        {"retailer": "OnePlus.com",  "price": 449.0, "logo_color": "#eb0029"},
        {"retailer": "Newegg",       "price": 439.0, "logo_color": "#e85c24"},
        {"retailer": "eBay",         "price": 419.0, "logo_color": "#e53238"},
    ],
    "Motorola Edge 40 Pro": [
        {"retailer": "Amazon",       "price": 549.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 599.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 569.0, "logo_color": "#0071ce"},
        {"retailer": "Newegg",       "price": 559.0, "logo_color": "#e85c24"},
    ],
    "Xiaomi Redmi Note 13 Pro": [
        {"retailer": "Amazon",       "price": 279.0, "logo_color": "#f90"},
        {"retailer": "Newegg",       "price": 299.0, "logo_color": "#e85c24"},
        {"retailer": "eBay",         "price": 269.0, "logo_color": "#e53238"},
        {"retailer": "B&H Photo",    "price": 289.0, "logo_color": "#333333"},
    ],
    "Samsung Galaxy A35": [
        {"retailer": "Samsung.com",  "price": 299.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 259.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 299.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 279.0, "logo_color": "#0071ce"},
    ],
    "Motorola Moto G84": [
        {"retailer": "Amazon",       "price": 229.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 249.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 239.0, "logo_color": "#0071ce"},
        {"retailer": "Newegg",       "price": 234.0, "logo_color": "#e85c24"},
    ],
    "Xiaomi Redmi 13C": [
        {"retailer": "Amazon",       "price": 139.0, "logo_color": "#f90"},
        {"retailer": "Newegg",       "price": 149.0, "logo_color": "#e85c24"},
        {"retailer": "eBay",         "price": 134.0, "logo_color": "#e53238"},
        {"retailer": "Walmart",      "price": 144.0, "logo_color": "#0071ce"},
    ],
    "Nokia G42": [
        {"retailer": "Amazon",       "price": 189.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 199.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 194.0, "logo_color": "#0071ce"},
        {"retailer": "eBay",         "price": 179.0, "logo_color": "#e53238"},
    ],
    # ── Headphones ────────────────────────────────────────────────────────────
    "WH-1000XM5": [
        {"retailer": "Amazon",       "price": 279.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 299.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 289.0, "logo_color": "#0071ce"},
        {"retailer": "Sony.com",     "price": 349.0, "logo_color": "#000000"},
    ],
    "WH-1000XM4": [
        {"retailer": "Amazon",       "price": 199.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 229.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 219.0, "logo_color": "#0071ce"},
        {"retailer": "Sony.com",     "price": 249.0, "logo_color": "#000000"},
    ],
    "AirPods Pro 2": [
        {"retailer": "Apple Store",  "price": 249.0, "logo_color": "#555555"},
        {"retailer": "Amazon",       "price": 189.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 249.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 219.0, "logo_color": "#0071ce"},
    ],
    "AirPods 3": [
        {"retailer": "Apple Store",  "price": 169.0, "logo_color": "#555555"},
        {"retailer": "Amazon",       "price": 139.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 169.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 149.0, "logo_color": "#0071ce"},
    ],
    "QuietComfort 45": [
        {"retailer": "Amazon",       "price": 229.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 279.0, "logo_color": "#003cb3"},
        {"retailer": "Bose.com",     "price": 279.0, "logo_color": "#000000"},
        {"retailer": "Walmart",      "price": 249.0, "logo_color": "#0071ce"},
    ],
    "QuietComfort Ultra": [
        {"retailer": "Amazon",       "price": 349.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 429.0, "logo_color": "#003cb3"},
        {"retailer": "Bose.com",     "price": 429.0, "logo_color": "#000000"},
        {"retailer": "B&H Photo",    "price": 399.0, "logo_color": "#333333"},
    ],
    "Momentum 4 Wireless": [
        {"retailer": "Amazon",       "price": 249.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 299.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",    "price": 279.0, "logo_color": "#333333"},
        {"retailer": "Walmart",      "price": 269.0, "logo_color": "#0071ce"},
    ],
    "Evolve2 85": [
        {"retailer": "Amazon",       "price": 399.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 449.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",    "price": 429.0, "logo_color": "#333333"},
        {"retailer": "Jabra.com",    "price": 449.0, "logo_color": "#002d6e"},
    ],
    "Headphones 700": [
        {"retailer": "Amazon",       "price": 189.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 229.0, "logo_color": "#003cb3"},
        {"retailer": "Bose.com",     "price": 229.0, "logo_color": "#000000"},
        {"retailer": "Walmart",      "price": 209.0, "logo_color": "#0071ce"},
    ],
    "LiveQ9f": [
        {"retailer": "Amazon",       "price": 169.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 199.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 179.0, "logo_color": "#0071ce"},
        {"retailer": "Newegg",       "price": 174.0, "logo_color": "#e85c24"},
    ],
    "Tune 770NC": [
        {"retailer": "Amazon",       "price": 99.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 129.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 109.0, "logo_color": "#0071ce"},
        {"retailer": "Target",       "price": 119.0, "logo_color": "#cc0000"},
    ],
    "H95": [
        {"retailer": "Amazon",       "price": 449.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 499.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",    "price": 469.0, "logo_color": "#333333"},
        {"retailer": "Newegg",       "price": 459.0, "logo_color": "#e85c24"},
    ],
    "Galaxy Buds2 Pro": [
        {"retailer": "Samsung.com",  "price": 199.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 149.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 199.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 169.0, "logo_color": "#0071ce"},
    ],
    "Galaxy Buds FE": [
        {"retailer": "Samsung.com",  "price": 99.0,  "logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 79.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 99.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 89.0,  "logo_color": "#0071ce"},
    ],
    "WF-1000XM5": [
        {"retailer": "Amazon",       "price": 229.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 279.0, "logo_color": "#003cb3"},
        {"retailer": "Sony.com",     "price": 279.0, "logo_color": "#000000"},
        {"retailer": "Walmart",      "price": 249.0, "logo_color": "#0071ce"},
    ],
    "QuietComfort Earbuds II": [
        {"retailer": "Amazon",       "price": 249.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 299.0, "logo_color": "#003cb3"},
        {"retailer": "Bose.com",     "price": 299.0, "logo_color": "#000000"},
        {"retailer": "B&H Photo",    "price": 279.0, "logo_color": "#333333"},
    ],
    "Freebuds Pro 3": [
        {"retailer": "Amazon",       "price": 169.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 199.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 179.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 189.0, "logo_color": "#333333"},
    ],
    "Momentum On-Ear 2": [
        {"retailer": "Amazon",       "price": 129.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 149.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",    "price": 139.0, "logo_color": "#333333"},
        {"retailer": "Walmart",      "price": 134.0, "logo_color": "#0071ce"},
    ],
    "Tune 510BT": [
        {"retailer": "Amazon",       "price": 39.0,  "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 49.0,  "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 44.0,  "logo_color": "#0071ce"},
        {"retailer": "Target",       "price": 46.0,  "logo_color": "#cc0000"},
    ],
    # ── Monitors ──────────────────────────────────────────────────────────────
    # Keys must match dataset names (full names with brand prefix)
    "UltraGear 27GP950-B": [
        {"retailer": "Amazon",       "price": 649.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 699.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 669.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 679.0, "logo_color": "#333333"},
    ],
    "UltraGear 32GQ950-B": [
        {"retailer": "Amazon",       "price": 899.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 999.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 949.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 969.0, "logo_color": "#333333"},
    ],
    "UltraSharp U2723QE": [
        {"retailer": "Amazon",       "price": 599.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 649.0, "logo_color": "#003cb3"},
        {"retailer": "Dell.com",     "price": 649.0, "logo_color": "#007db8"},
        {"retailer": "B&H Photo",    "price": 629.0, "logo_color": "#333333"},
    ],
    "UltraSharp U3223QE": [
        {"retailer": "Amazon",       "price": 799.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 849.0, "logo_color": "#003cb3"},
        {"retailer": "Dell.com",     "price": 849.0, "logo_color": "#007db8"},
        {"retailer": "B&H Photo",    "price": 829.0, "logo_color": "#333333"},
    ],
    "ProArt PA32UCG": [
        {"retailer": "Amazon",       "price": 1799.0,"logo_color": "#f90"},
        {"retailer": "B&H Photo",    "price": 1999.0,"logo_color": "#333333"},
        {"retailer": "Newegg",       "price": 1899.0,"logo_color": "#e85c24"},
        {"retailer": "Adorama",      "price": 1949.0,"logo_color": "#c8102e"},
    ],
    "ProArt PA279CRV": [
        {"retailer": "Amazon",       "price": 599.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 649.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 619.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 629.0, "logo_color": "#333333"},
    ],
    "DesignVue PD2705U": [
        {"retailer": "Amazon",       "price": 449.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 499.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",    "price": 479.0, "logo_color": "#333333"},
        {"retailer": "Newegg",       "price": 469.0, "logo_color": "#e85c24"},
    ],
    "DesignVue PD3220U": [
        {"retailer": "Amazon",       "price": 899.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 999.0, "logo_color": "#003cb3"},
        {"retailer": "B&H Photo",    "price": 949.0, "logo_color": "#333333"},
        {"retailer": "Newegg",       "price": 929.0, "logo_color": "#e85c24"},
    ],
    "Odyssey G7 32\"": [
        {"retailer": "Samsung.com",  "price": 599.0, "logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 529.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 599.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 559.0, "logo_color": "#e85c24"},
    ],
    "Odyssey G9 49\"": [
        {"retailer": "Samsung.com",  "price": 1299.0,"logo_color": "#1428a0"},
        {"retailer": "Amazon",       "price": 1149.0,"logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 1299.0,"logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 1199.0,"logo_color": "#e85c24"},
    ],
    "Predator X34P": [
        {"retailer": "Amazon",       "price": 749.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 799.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 769.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 779.0, "logo_color": "#333333"},
    ],
    "Predator XB323UGX": [
        {"retailer": "Amazon",       "price": 649.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 699.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 669.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 679.0, "logo_color": "#333333"},
    ],
    "UltraGear 27GP850-B": [
        {"retailer": "Amazon",       "price": 299.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 349.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 319.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 329.0, "logo_color": "#333333"},
    ],
    "TUF Gaming VG279QM": [
        {"retailer": "Amazon",       "price": 249.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 299.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 269.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 279.0, "logo_color": "#333333"},
    ],
    "Optix MAG274QRF-QD": [
        {"retailer": "Amazon",       "price": 349.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 399.0, "logo_color": "#003cb3"},
        {"retailer": "Newegg",       "price": 369.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 379.0, "logo_color": "#333333"},
    ],
    "P Series P2422H": [
        {"retailer": "Amazon",       "price": 199.0, "logo_color": "#f90"},
        {"retailer": "Best Buy",     "price": 229.0, "logo_color": "#003cb3"},
        {"retailer": "Dell.com",     "price": 229.0, "logo_color": "#007db8"},
        {"retailer": "Walmart",      "price": 209.0, "logo_color": "#0071ce"},
    ],
    "Evnia 27E1N3300A": [
        {"retailer": "Amazon",       "price": 179.0, "logo_color": "#f90"},
        {"retailer": "Newegg",       "price": 199.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 189.0, "logo_color": "#333333"},
        {"retailer": "Walmart",      "price": 184.0, "logo_color": "#0071ce"},
    ],
    "24G2U": [
        {"retailer": "Amazon",       "price": 159.0, "logo_color": "#f90"},
        {"retailer": "Newegg",       "price": 179.0, "logo_color": "#e85c24"},
        {"retailer": "B&H Photo",    "price": 169.0, "logo_color": "#333333"},
        {"retailer": "Walmart",      "price": 164.0, "logo_color": "#0071ce"},
    ],
    "IPS269Q": [
        {"retailer": "Amazon",       "price": 169.0, "logo_color": "#f90"},
        {"retailer": "Newegg",       "price": 189.0, "logo_color": "#e85c24"},
        {"retailer": "Best Buy",     "price": 189.0, "logo_color": "#003cb3"},
        {"retailer": "Walmart",      "price": 174.0, "logo_color": "#0071ce"},
    ],
}


RETAILER_SEARCH_URLS = {
    "amazon":        "https://www.amazon.com/s?k={q}",
    "best buy":      "https://www.bestbuy.com/site/searchpage.jsp?st={q}",
    "walmart":       "https://www.walmart.com/search?q={q}",
    "ebay":          "https://www.ebay.com/sch/i.html?_nkw={q}",
    "target":        "https://www.target.com/s?searchTerm={q}",
    "newegg":        "https://www.newegg.com/p/pl?d={q}",
    "b&h":           "https://www.bhphotovideo.com/c/search?q={q}",
    "bhphotovideo":  "https://www.bhphotovideo.com/c/search?q={q}",
    "adorama":       "https://www.adorama.com/l/?searchinfo={q}",
    "costco":        "https://www.costco.com/CatalogSearch?keyword={q}",
    "sam's club":    "https://www.samsclub.com/s/{q}",
    "apple":         "https://www.apple.com/shop/buy-mac?q={q}",
    "lenovo":        "https://www.lenovo.com/us/en/search?q={q}",
    "dell":          "https://www.dell.com/en-us/search/{q}",
    "back market":   "https://www.backmarket.com/en-us/search?q={q}",
    "backmarket":    "https://www.backmarket.com/en-us/search?q={q}",
    "gazelle":       "https://www.gazelle.com/search?q={q}",
    "verizon":       "https://www.verizon.com/search-results/?query={q}",
    "at&t":          "https://www.att.com/buy/broadband/phones.html?search={q}",
    "t-mobile":      "https://www.t-mobile.com/search?q={q}",
    "mercari":       "https://www.mercari.com/search/?keyword={q}",
    "swappa":        "https://swappa.com/search?q={q}",
    "offerup":       "https://offerup.com/search?q={q}",
    "gamestop":      "https://www.gamestop.com/search/?q={q}",
    "cdw":           "https://www.cdw.com/search/?key={q}",
    "micro center":  "https://www.microcenter.com/search/search_results.aspx?N=0&SearchTerm={q}",
    "microcenter":   "https://www.microcenter.com/search/search_results.aspx?N=0&SearchTerm={q}",
    "antonline":     "https://www.antonline.com/search?q={q}",
    "b&h photo":     "https://www.bhphotovideo.com/c/search?q={q}",
    "hp":            "https://www.hp.com/us-en/shop/slp/search-results?query={q}",
    "samsung":       "https://www.samsung.com/us/search/searchMain/index/?searchTerm={q}",
    "sony":              "https://electronics.sony.com/search?text={q}",
    "unclaimed baggage": "https://www.unclaimedbaggage.com/search?q={q}",
    "swappa":            "https://swappa.com/search?q={q}",
    "decluttr":          "https://www.decluttr.com/search?q={q}",
    "phonepower":        "https://www.phonepower.com/search?q={q}",
    "expansys":          "https://www.expansys.com/search/?q={q}",
    "rakuten":           "https://www.rakuten.com/search/{q}/",
}

def _get_retailer_color(name: str) -> str:
    name_lower = name.lower()
    for key, color in RETAILER_COLORS.items():
        if key in name_lower:
            return color
    return "#888888"

def _retailer_search_url(retailer_name: str, product_name: str) -> str:
    """Return a direct retailer search URL for the product."""
    import urllib.parse, re
    q = urllib.parse.quote_plus(product_name)
    name_lower = retailer_name.lower()

    # Check known retailers first
    for key, template in RETAILER_SEARCH_URLS.items():
        if key in name_lower:
            return template.format(q=q)

    # Strip marketplace seller suffixes like "eBay - wafuu" or "Amazon - seller"
    base_name = re.split(r"\s*[-–|]\s*", retailer_name)[0].strip()
    base_lower = base_name.lower()

    # Re-check after stripping (catches "eBay - wafuu" → "eBay")
    for key, template in RETAILER_SEARCH_URLS.items():
        if key in base_lower:
            return template.format(q=q)

    # If the name itself is a domain (e.g. "wafuu.com"), link directly to it
    domain_match = re.search(r"^[\w-]+\.(com|co|net|org|io|shop)$", base_lower)
    if domain_match:
        return f"https://www.{base_lower}/search?q={q}"

    # Unknown retailer — Google search is more reliable than a guessed URL
    return f"https://www.google.com/search?q={urllib.parse.quote_plus(product_name + ' buy ' + base_name)}"


def _parse_serpapi_response(data: dict, product_name: str = "") -> list[dict]:
    results = []
    shopping_results = data.get("shopping_results", [])
    inline = data.get("inline_shopping_results", [])
    all_results = shopping_results + inline

    seen = set()
    for item in all_results:
        source = item.get("source", "").strip()
        price_str = item.get("price", "")
        link = item.get("link", item.get("product_link", ""))

        if not source or not price_str:
            continue
        if source in seen:
            continue
        seen.add(source)

        try:
            price = float(
                price_str.replace("$", "").replace(",", "").strip().split()[0]
            )
        except (ValueError, IndexError):
            continue

        results.append({
            "retailer":    source,
            "price":       price,
            "link":        _retailer_search_url(source, product_name),
            "logo_color":  _get_retailer_color(source),
        })

        if len(results) >= 5:
            break

    results.sort(key=lambda x: x["price"])
    return results


def get_prices(product_name: str, category: str) -> list[dict]:
    """
    Returns a list of retailer dicts sorted by price ascending.
    Falls back to hardcoded data if SerpAPI key is missing or call fails.
    Each dict: {retailer, price, link, logo_color, is_lowest}
    """
    results = []

    if SERPAPI_KEY and SERPAPI_KEY != "your_serpapi_key_here":
        try:
            params = {
                "engine":    "google_shopping",
                "q":         f"{product_name} {category}",
                "api_key":   SERPAPI_KEY,
                "num":       10,
                "gl":        "us",
                "hl":        "en",
            }
            resp = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=8,
            )
            if resp.status_code == 200:
                results = _parse_serpapi_response(resp.json(), product_name)
        except Exception:
            results = []

    if not results:
        raw = FALLBACK_PRICES.get(product_name, [])
        results = [
            {**r, "link": _retailer_search_url(r["retailer"], product_name)}
            for r in raw
        ]

    # Strip zero/negative prices — never show $0 in the UI
    results = [r for r in results if r.get("price", 0) > 0]
    results = sorted(results, key=lambda x: x["price"])

    for i, r in enumerate(results):
        r["is_lowest"] = (i == 0)

    return results


def get_prices_batch(products: list[dict], category: str) -> dict:
    """
    Fetches prices for multiple products. Adds a small delay between
    SerpAPI calls to respect rate limits.
    Returns {product_name: [retailer_dicts]}
    """
    out = {}
    for i, prod in enumerate(products):
        name = prod["name"]
        out[name] = get_prices(name, category)
        if SERPAPI_KEY and i < len(products) - 1:
            time.sleep(0.5)
    return out
