import csv
import datetime
import sys
from collections import Counter, defaultdict
from typing import List

from pydantic_core import from_json
from rapidfuzz import fuzz

from utils import PaperInfo

# The recursion limit needs to extended for this script
sys.setrecursionlimit(10000)


def frequency_analysis(tech_lst: List[tuple[str, int]]):
    clusters = defaultdict(list)
    return fuzzy_matching_loop(tech_lst, clusters)


def fuzzy_matching_loop(tech_lst: List[tuple[str, int]], cluster: dict,
                        cutoff_ratio=90.0):
    """
    This is a recursive implementation of the fuzzy synonym matching. For finding synonyms.
    Rappid fuzz with the using the "partial ratio" is used for this purpose.
    The first element of the tech_list, gets compared to all other elements.
    If it matches within the cutoff_ratio it will get added the synonym cluster.
    The function calls itself with the matched synonyms removed from the tech_list.
    It returns the synonym cluster.
    """
    if tech_lst == []:
        return cluster

    # tech_a gets compared to tech_b. tech_a is then used as generic term.
    tech_a, _ = tech_lst[0]
    matched_synonyms = []
    for tech_b, frequency in tech_lst:
        r = fuzz.partial_ratio(tech_a, tech_b)
        if r >= cutoff_ratio:
            # If the tech_a and tech_b match within the given ratio tech_b gets added to
            # cluster, under the key of tech_a. tech_a is used as a generic term.
            cluster[tech_a].append((tech_b, frequency))
            matched_synonyms.append(tech_b)

    next_tech_lst = [(x, f) for x, f in tech_lst if x not in matched_synonyms]

    # The generic term, tech_a may not exist in the next iteration.
    # Per definition, tech_a will always have a 100.0 ratio with itself.
    assert tech_a not in next_tech_lst
    # Assert that the progress is made.
    assert len(next_tech_lst) < len(tech_lst)

    return fuzzy_matching_loop(next_tech_lst, cluster)


if __name__ == "__main__":
    # Open the output file from the abstract analyzer.
    results = open("results_2000.json", "r")
    paper_list = from_json(results.read())

    counter = Counter()

    for paper in paper_list:
        parsed_paper = PaperInfo(**paper)
        # Clean the technology names.
        # Add them to a counter set; this will be used for the frequency analysis.
        counter += Counter(
            [x.replace("-", " ").removeprefix(" ").removesuffix(" ") for x in
             parsed_paper.technology_used])

    tech_list = [(tech, freq) for tech, freq in counter.most_common()]
    analysis_clusters = frequency_analysis(tech_list)

    cut_of_freq = 5
    csv_data = []
    for tech, syn in sorted(analysis_clusters.items()):
        syn_names = [x for x, _ in syn]
        total_freq = sum([x for _, x in syn])

        if total_freq >= cut_of_freq:
            csv_data.append({"Frequency": total_freq, "Technology Name": tech,
                             "Matched Synonyms": syn_names})
            print(f"Freq: {total_freq}, Technology: {tech}: {syn_names}")

    with open(f"analyzed_technology_{cut_of_freq}_{datetime.datetime.now()}.csv", "w",
              newline='') as csv_file:
        fieldnames = ["Frequency", "Technology Name", "Matched Synonyms"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(csv_data)
