import configparser
from textwrap import indent

import openai
from pandas import DataFrame

from utils import read_bibtex, bibtex_to_dataframe, iterate_over_entries

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['API']['api_key']
openai.api_key = api_key


def random_sample_analysis(bibtex_data: DataFrame, sample_size: int, seed: int):
    """
    Take a fixed seed random sample from the bibtex file and analyze the abstracts.
    """
    sample = bibtex_data.sample(n=sample_size, random_state=seed)
    print(sample.head())
    return iterate_over_entries(sample, api_key)


if __name__ == "__main__":
    data = read_bibtex('data/scopus_unknown.bib')
    data = bibtex_to_dataframe(data)
    result = random_sample_analysis(data, 100, 1)
    print(result.model_dump_json(indent=2))
    with open("results.json", "w") as result_file:
        result_file.write(result.model_dump_json(indent=2))
