# import wikipediaapi

# # Initialize Wikipedia API
# wiki = wikipediaapi.Wikipedia(
#     user_agent="process_modelling",
#     language="en",
#     extract_format=wikipediaapi.ExtractFormat.WIKI,
# )


# def get_summary_and_links(page_title):
#     # Get the Wikipedia page
#     page = wiki.page(page_title)

#     if not page.exists():
#         print(f"Page '{page_title}' does not exist.")
#         return

#     # Extract summary
#     summary = page.summary

#     # Extract links in the summary
#     linked_pages = [link for link in page.links.keys() if link in summary]

#     return summary, linked_pages


# # Example usage
# page_title = "animal"
# summary, linked_pages = get_summary_and_links(page_title)

# print("Summary:")
# print(summary)
# print("\nLinked pages in summary:")
# for link in linked_pages:
#     print(link)


import requests
from bs4 import BeautifulSoup


def get_summary_and_links(page_title):
    # Set up the Wikipedia API URL
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"

    # Send a request to the Wikipedia API
    response = requests.get(url)

    if response.status_code != 200:
        print(
            f"Error fetching page '{page_title}'. Status code: {response.status_code}"
        )
        return None, None

    data = response.json()

    # Get the summary in raw HTML format
    summary_html = data.get("extract_html")
    if not summary_html:
        print(f"No summary found for page '{page_title}'.")
        return None, None

    # Use BeautifulSoup to parse the HTML and extract linked pages
    soup = BeautifulSoup(summary_html, "html.parser")

    # Extract the plain text summary
    summary_text = soup.get_text()

    # Extract all hyperlinks in the summary
    linked_pages = []
    for link in soup.find_all("a", href=True):
        linked_pages.append(
            {
                "link_text": link.get_text(),
                "page_url": link["href"],
                "linked_page_title": link["href"].split("/")[
                    -1
                ],  # extract the actual page title from the URL
            }
        )

    return summary_text, linked_pages


# Example usage
page_title = "Python_(programming_language)"
summary_text, linked_pages = get_summary_and_links(page_title)

print("Summary Text:")
print(summary_text)
print("\nLinked Pages in Summary:")
for link in linked_pages:
    print(
        f"Link Text: {link['link_text']}, Page Title: {link['linked_page_title']}, URL: {link['page_url']}"
    )
