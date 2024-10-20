# import wikipediaapi
# import pickle as pk

# wiki = wikipediaapi.Wikipedia(
#     user_agent="process_modelling",
#     language="en",
#     extract_format=wikipediaapi.ExtractFormat.WIKI,
# )

# contexts = ["brick", "paperclip", "animal"]
# context_to_contexttext = dict(
#     zip(contexts, [wiki.page(context).summary for context in contexts])
# )

# pk.dump(context_to_contexttext, open("../pickle/wikipedia_contexts.pk", "wb"))


import wikipediaapi
from bs4 import BeautifulSoup
import requests
import pickle as pk

wiki = wikipediaapi.Wikipedia(user_agent="process_modelling", language="en")

# Function to get the number of summary paragraphs and extract links
def get_summary_links(page_name, num_paras):
    page = wiki.page(page_name)
    
    if page.exists():
        # summary_html = page.summary
        response = requests.get(page.fullurl)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Only select the summary part from the page content
        summary_paragraphs = soup.find_all('p', limit=num_paras)
        
        links = []
        for paragraph in summary_paragraphs:
            for link in paragraph.find_all('a', href=True):
                links.append(link['href'])
        
        return links
    else:
        return f"Page '{page_name}' does not exist."

# Example usage
task_to_contexts = {"autbrick": {}, "autpaperclip": {}, "vf": {}}
pagename_to_numparas = {"animal": 7, "brick": 5, "paperclip": 4}
pagename_to_task = {"brick": "autbrick", "paperclip": "autpaperclip", "animal": "vf"}

for page_name in ["brick", "paperclip", "animal"]:
    task_to_contexts[pagename_to_task[page_name]]["nocontext"] = ""
    task_to_contexts[pagename_to_task[page_name]][page_name] = wiki.page(page_name).summary
    summary_links = get_summary_links(page_name, pagename_to_numparas[page_name])
    for link in summary_links:
        if "/wiki" in link:
            page_title = link.replace("/wiki/", "")
            task_to_contexts[pagename_to_task[page_name]][page_title] = wiki.page(page_title).summary

pk.dump(task_to_contexts, open("../pickle/wikipedia_contexts.pk", "wb"))