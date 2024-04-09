

This README.txt file describes the test set of the dataset "Don't Patronize Me! (DPM!), An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities", an annotated corpus with PCL (Patronizing and Condescending Language) in the newswire domain. The test set of DPM! consists of the following files:

-- README_test.txt
-- PCL_test.tsv *
-- PCL_testlabels_t1.txt
-- PCL_testlabels_t2.txt

* Please note that this file has a Disclaimer at the top. The actual data starts at line 5.

where

-- README_test.txt is this file.

-- PCL_test.tsv contains paragraphs towards vulnerable communities extracted from media sources.
It contains one instance per line with the following format:

	- <par_id> <tab> <art_id> <tab> <keyword> <tab> <country_code> <tab> <text> <tab> <label>

	where
	- <par_id> is a unique id for each one of the paragraphs in the corpus.
	- <art_id> is the document id in the original NOW corpus (News on Web: https://www.english-corpora.org/now/).
	- <keyword> is the search term used to retrieve texts about a target community.
	- <country_code> is a two-letter ISO Alpha-2 country code for the source media outlet.
	- <text> is the paragraph containing the keyword.



-- PCL_testlabels_t1.txt contains labels for Task 1: Binary Classification. Each line contains a numeric label 0 or 1 where:
	- 0 means a negative case of PCL.
	- 1 means a positive case of PCL.


-- PCL_testlabels_t2.txt contains labels for Task 2: Multilabel Classification. For each paragraph, the labels identify which category or categories of PCL are present in the text, if any. Each line contains a set of comma-separated labels, either 0 or 1, which denote de presence (1) or absense (0) of each one of the 7 categories of PCL**. Each position on the list represents the following categories:

	- 'Unbalanced_power_relations': 0, 
	- 'Shallow_solution': 1,
	- 'Presupposition': 2,
	- 'Authority_voice': 3,
	- 'Metaphors': 4,
	- 'Compassion': 5,
	- 'The_poorer_the_merrier': 6


	###################################################################################################
	** For more information about the categories or the dataset, please see our papers:

	--- Pérez-Almendros, Carla, Luis Espinosa Anke, and Steven Schockaert. "Don’t Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities." Proceedings of the 28th International Conference on Computational Linguistics. 2020. ---

	--- Pérez-Almendros, Carla, Luis Espinosa Anke, and Steven Schockaert. "SemEval-2022 task 4: Patronizing and condescending language detection." Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022). 2022. ---

	###################################################################################################



    ###################################################################################################
	
	For more information and code related to the DPM! dataset, please see 
	https://github.com/Perez-AlmendrosC/dontpatronizeme 
	
	###################################################################################################