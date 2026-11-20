ALIASES = {
    "USA": "United States of America",
    "United States": "United States of America",
    "United Kingdom": "The United Kingdom",
    "United Kingdom of Great Britain and Northern Ireland": "The United Kingdom",
    "Korea Republic of": "Republic of Korea",
    "Korea, Rep.": "Republic of Korea",
    "South Korea": "Republic of Korea",
    "Iran Islamic Republic of": "Iran (Islamic Republic of)",
    "Iran, Islamic Rep.": "Iran (Islamic Republic of)",
    "Russian Federation (the)": "Russian Federation",
    "Egypt, Arab Rep.": "Egypt",
    "Lao PDR": "Lao People's Democratic Republic",
    "Vietnam": "Viet Nam",
    "Taiwan": "Chinese Taipei",
    "Taiwan, China": "Chinese Taipei",
    "Macao SAR, China": "Macao (sar) China",
}


def canonical_country(value: str) -> str:
    name = str(value).strip()
    return ALIASES.get(name, name)
