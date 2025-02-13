from pandas import DataFrame
from utils import read_bibtex, bibtex_to_dataframe
import requests

# path to the bibtex file
bibtech_file_path = 'data/all results_relevant-only.bib'

# number of papers to analyze from all the papers, -1 for all papers
num_papers = -1

def analyze_paper(bibtex_entry_data):

# # old version of system_prompt
#     system_prompt = """
# Read the following papers title and abstract.
# Analyze if the abstract specifies how results were evaluated.
# If so, summarize the evaluation, else answer with only the word "NONE".
#     """

    system_prompt = """
The user will provide you with the title and the abstract of a paper.
You analyze if the abstract explicitly specifies how results of the paper were evaluated.
If the abstract does not explicitly specify this, answer with the word "no".
If the abstract does specify this, answer with the word "yes".
Always respond with the word "yes" or "no" first, then provide further information if necessary.
    """

    user_prompt = f"{bibtex_entry_data.title}\n\n{bibtex_entry_data.abstract}"

    # create request (works wit LM Studio API, similar to OpenAI API)
    headers = { "Content-Type": "application/json" }
    body = {
        "model": "llama-3.2-3b-instruct",
        "messages": [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ],
        "temperature": 0,
        "max_tokens": 1000,
        "stream": False
    }
    
    # send the request
    response = requests.post("http://localhost:1234/v1/chat/completions", headers=headers, json=body)
    return response.json().get("choices")[0].get("message").get("content")

if __name__ == "__main__":

    # read the data
    data = read_bibtex(bibtech_file_path)
    bibtex_data = bibtex_to_dataframe(data)

    # create empty DataFrame for the results
    results = DataFrame(columns=["ID", "title", "abstract", "evaluation_methods"])

    # analyze the first n papers
    for i, row in enumerate(bibtex_data.itertuples(), start=1):
        if num_papers != -1 and i > num_papers:
            break
        result = analyze_paper(row)
        # # remove all line breaks from result
        # res_nl = result.replace("\n", " ")
        print(f"Analyzed paper {i:04} with ID {row.ID.ljust(20)} --- {result.replace("\n", " ")[:20]} ...")
        # append the paper ID and the result to the DataFrame
        results = results._append({"ID": row.ID, "title": row.title, "abstract": row.abstract, "evaluation_methods": result}, ignore_index=True)

    # save the results to a csv file seperated by semicolon
    results.to_csv("data/evaluation_results.csv", sep=";", index=False)