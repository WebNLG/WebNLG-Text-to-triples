# WebNLG Text-to-triples

The evaluation script for the Text-to-triples task for WebNLG. This script will link the candidate triples to the reference triples (based on what gives the highest average F1 score), and calculate the Precision, Recall, and F1 score metrics based on the metrics used for the [SemEval 2013 task](https://www.cs.york.ac.uk/semeval-2013/task9/data/uploads/semeval_2013-task-9_1-evaluation-metrics.pdf) (see also [this page](https://github.com/ivyleavedtoadflax/nervaluate) for an explanation of the scoring types). Additionally, the Precsion, Recall, and F1 score for the full triple will be calculated (based on [Liu et al., 2018](https://arxiv.org/abs/1807.01763)).

The script will also try to link every candidate attribute as good as possible to the reference attribute. Variations of the reference attribute will be interpreted as a longer string (if there are no other non-matching words before or after the reference), or as a separate guess.

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
      </generatedtripleset>
    </entry>
  </entries>
</benchmark>
```

To run this script you need the following libraries:

- [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/)
- [Regex](https://pypi.org/project/regex/)
- [NERvaluate](https://github.com/ivyleavedtoadflax/nervaluate)
- [NLTK](https://www.nltk.org/install.html)
- [Scikit-learn](https://scikit-learn.org/stable/install.html)

These libraries can also be installed by running ```pip3 install -r requirements.txt```


The command to use the script is: ```python3 Evaluation_script.py <reference xml> <candidates xml>```
