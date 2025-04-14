import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent="process_modelling",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)


# wiki = wikipediaapi.Wikipedia('en')
page = wiki.page("Rabbit")

text = page.text.lower()
count = text.count("rabbit")

print(f"Frequency of 'rabbit' in the Rabbit article: {count}")
