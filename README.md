# Multimodal Meme Classification: Identifying Offensive Content in Image and Text

Code for the paper

@inproceedings{suryawanshi-etal-2020-MultiOFF,
    title = "Multimodal Meme Dataset (MultiOFF) for Identifying Offensive Content in Image and Text",
    author = "Suryawanshi, Shardul and Chakravarthi, Bharathi Raja and Arcan, Mihael and Buitelaar, Paul,
    booktitle = "Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying ({TRAC}-2020)",
    month = May,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    abstract = "A meme is a form of media that spreads an idea or emotion across the internet. As posting meme has become a new form of communication of the web,  due to the multimodal nature of memes, postings of hateful memes or related events like trolling, cyberbullying are increasing day by day. Hate speech, offensive content and aggression content detection have been extensively explored in a single modality such as text or image.  However, combining two modalities to detect offensive content is still a developing area. Memes make it even more challenging since they express humour and sarcasm in an implicit way, because of which the meme may not be offensive if we only consider the text or the image. Therefore, it is necessary to combine both modalities to identify whether a given meme is offensive or not. Since there was no publicly available dataset for multimodal offensive meme content detection, we leveraged the memes related to the 2016 U.S. presidential election and created the MultiOFF multimodal meme dataset for offensive content detection dataset. We subsequently developed a classifier for this task using the MultiOFF dataset. We use an early fusion technique to combine the image and text modality and compare it with a text- and an image-only baseline to investigate its effectiveness.  Our results show improvements in terms of Precision, Recall, and F-Score.",
}

This is a document that involves step by step instructions to execute the code

(Pre-requisite: Conda/python environment should have packages mentioned in requirement.txt file before execution
		Glove embedding of 50d has been used can be dowloaded from "http://nlp.stanford.edu/data/glove.6B.zip")

--> Use google drive link to access the data "https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6?usp=sharing"

--> Split dataset has train, test and validation data

--> Labelled Image has memes belonging to each of the above dataset

--> This data needs to be placed and directory location needs to be changed while reading the data in main code

--> Once done with the setup mentioned above, one can execute the code in the sequence mentioned as below:
	--> Stacked_LSTM_VGG16.ipynb
	--> BiLSTM_VGG16.ipynb
	--> CNN_VGG16.ipynb
	--> LR_NB_DNN.ipynb
