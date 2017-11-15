# NSF_Awards
Explore awards and get insights from historical data back to 1960

The National Science Foundation is a federal agency that supports the most promising research ideas from all sciences and engineering fields except medical sciences.

1. Download NSF Award
  Python modules :
          - requests
          - zipfile

2. Read each award xml file and create a structured data file
  Python modules :
          - BeautifulSoup
          - nltk
          - multiprocessing

3. Cleaning and wrangling
  Python modules :
          - pandas
          - sklearn
          - re
          - itertools

4. Award Type Classification
          - utilsvectorizer (custom module for vectorization, Bag of words, N-grams)
          - Abstract_transformation (custom  module to clean Abstract data)
          - AwardInstr_transformation (custom  module to clean Award type data)
          - sklearn (Multinomial Naive Bayes)
          - nltk

5. Abstract clustering
          - utilsvectorizer (custom module for vectorization, Bag of words, N-grams)
          - Abstract_transformation (custom  module to clean Abstract data)
          - Directorate_transformation (custom  module to clean Directorate data)
          - sklearn (K-means)
          - nltk
