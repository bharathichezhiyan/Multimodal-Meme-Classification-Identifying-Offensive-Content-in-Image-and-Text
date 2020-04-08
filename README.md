# Multimodal Meme Classification: Identifying Offensive Content in Image and Text

If you are using the code or the dataset for your research work then please cite our paper below:

@inproceedings{suryawanshi-etal-2020-MultiOFF,

    title = "Multimodal Meme Dataset (MultiOFF) for Identifying Offensive Content in Image and Text",
    
    author = "Suryawanshi, Shardul and Chakravarthi, Bharathi Raja and Arcan, Mihael and Buitelaar, Paul,
    
    booktitle = "Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying ({TRAC}-2020)",
    
    month = May,
    
    year = "2020",
    
    publisher = "Association for Computational Linguistics",
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
