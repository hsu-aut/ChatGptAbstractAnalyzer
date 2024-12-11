import configparser
import openai
from pandas import DataFrame
from pydantic import BaseModel
from utils import read_bibtex, bibtex_to_dataframe
import instructor

# Load the API key from the config file
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['API']['api_key']

# Path to the BibTeX file
bibtech_file_path = 'data/data.bib'

# Number of papers to analyze from all the papers
num_papers = 20

# Define the data models for structured output
class EvaluationType(BaseModel):
    name: str
    used_in_paper_id: list[int]
class EvaluationTypeAnalysis(BaseModel):
    evaluation_types: list[EvaluationType]

if __name__ == "__main__":
    # read the data
    data = read_bibtex(bibtech_file_path)
    bibtex_data = bibtex_to_dataframe(data)
    # select 20 random papers 
    sample = bibtex_data.sample(n=num_papers, random_state=1)
    # print(sample.head(10))
    # generate prompts for the selected papers
    paper_list = ""
    for i, row in enumerate(sample.itertuples(), start=1):
        paper_list += f"-- Paper ID {i} --\n"
        paper_list += f"Title: {row.title}\n"
        paper_list += f"Abstract: {row.abstract}\n\n"
    # remove the extra newlines at the end
    paper_list = paper_list.strip()
    # print(paper_list)
    # analyze the abstracts
    base_prompt = """
Read the following list of papers, containing the paper's title and abstract. 
Try to analyze how the papers evaluate their work. 
If a paper does not specify this in the abstract, do not provide any information or make assumptions.
As a response, provide a list of the used types evalaution, each with the IDs of the papers it was used in.
    """
    prompt = f"{base_prompt}\n{paper_list}"
    # print(prompt)
    client = instructor.from_openai(
        openai.OpenAI(api_key=api_key),
        mode=instructor.Mode.JSON,
    )
    evaluation_types = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=EvaluationTypeAnalysis,
        messages=[
            {"role": "system", "content": "You read scientific paper abstracts and analyze them."},
            {"role": "user", "content": prompt}
        ],
    )
    print(evaluation_types.model_dump_json(indent=2))