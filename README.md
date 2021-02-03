# WebNLG Challenge 2020: Text-to-RDF Evaluation Script

The evaluation script for the Text-to-RDF task for the [WebNLG Challenge 2020](https://webnlg-challenge.loria.fr/challenge_2020/). 

## About
This script will link the candidate triples to the reference triples (based on what gives the highest average F1 score), and calculate the Precision, Recall, and F1 score metrics based on the metrics used for the [SemEval 2013 task](https://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/semeval_2013-task-9_1-evaluation-metrics.pdf) (see also [this page](https://github.com/ivyleavedtoadflax/nervaluate) for an explanation of the scoring types). Additionally, the Precision, Recall, and F1 score of the full triple will be calculated (based on [Liu et al., 2018](https://arxiv.org/abs/1807.01763)).

The script will also try to link every candidate attribute as good as possible to the reference attribute. Variations of the reference attribute will be interpreted as a longer string (if there are no other non-matching words before or after the matched reference), or as a separate guess (if there are).

There are four types of matching:
 * Strict: for each element of the triple, exact match of the candidate string with the reference is required, and the element type (subject, predicate, object) should match with the reference.
 * Exact: for each element of the triple, exact match of the candidate string with the reference is required, and the element type (subject, predicate, object) is irrelevant.
 * Partial: for each element of the triple, the candidate string should match at least partially with the reference string, and the element type (subject, predicate, object) is irrelevant.
 * Type: for each element of the triple, the candidate string should match at least partially with the reference string, and the element type (subject, predicate, object) should match with the reference.


## Format

The candidates xml should be formatted as:

```
<benchmark>
  <entries>
    <entry category="Airport" eid="Id19">
      <generatedtripleset>
        <gtriple>Aarhus | leaderName | Jacob_Bundsgaard</gtriple>
      </generatedtripleset>
    </entry>
    <entry category="Airport" eid="Id18">
      <generatedtripleset>
        <gtriple>Antwerp_International_Airport | operatedBy | Government_of_Flanders</gtriple>
        <gtriple>Antwerp_International_Airport | cityServed | Antwerp</gtriple>
      </generatedtripleset>
    </entry>
  </entries>
</benchmark>
```

## Requirements

To run this script, you need the following libraries:

- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/)
- [Regex](https://pypi.org/project/regex/)
- [NERvaluate](https://github.com/ivyleavedtoadflax/nervaluate)
- [NLTK](https://www.nltk.org/install.html)
- [Scikit-learn](https://scikit-learn.org/stable/install.html)

These libraries can also be installed by running ```pip3 install -r requirements.txt```

## Execution
The command to use the script is: ```python3 Evaluation_script.py <reference xml> <candidates xml>```

And to save the results as a json file: ```python3 Evaluation_script_json.py <reference xml> <candidates xml> <results json>```
