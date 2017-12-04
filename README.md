# NSF_Awards
Exploring National Science Foundation Awards and getting insights from historical data back to 1960.<br><br>
The NSF defines seven areas of science called directorate. This study aims to determine the optimal number of directorates. Does each diretorate operate independently or is there overlap between them? <br><br>
On another hand, an attempt to predict award type was conducted. Classifying award type automatically allows for faster processing and reduces delay due to bureaucracy.

The National Science Foundation is a federal agency that supports the most promising research ideas from all sciences and engineering fields except medical sciences. This is an end-to-end project were the following pipeline was designed:

1. Download NSF Award, historical data is available on the NSF website and open to the public. Each fiscal year has one zip file and each zip file contains all awards for that year in an xml format.
  Python modules :<br>
          - requests<br>
          - zipfile<br>

2. Read each award xml file and create a structured data file
  Python modules :<br>
          - BeautifulSoup<br>
          - nltk<br>
          - multiprocessing<br>

3. Cleaning and wrangling. Text data has abbreviations and contents such as web link that are not desirable in this study.
  Python modules :<br>
          - pandas<br>
          - sklearn<br>
          - re<br>
          - itertools<br>

4. Award Type Classification. Prediction of award type.
  Python modules :<br>
          - utilsvectorizer (custom module for vectorization, Bag of words, N-grams)<br>
          - Abstract_transformation (custom  module to clean Abstract data)<br>
          - AwardInstr_transformation (custom  module to clean Award type data)<br>
          - sklearn (Multinomial Naive Bayes)<br>
          - nltk<br>

5. Abstract clustering. We will use abstracts to identify silos and areas of overlap between directorates.
  Python modules :<br>
          - utilsvectorizer (custom module for vectorization, Bag of words, N-grams)<br>
          - Abstract_transformation (custom  module to clean Abstract data)<br>
          - Directorate_transformation (custom  module to clean Directorate data)<br>
          - sklearn (K-means)<br>
          - nltk <br>
